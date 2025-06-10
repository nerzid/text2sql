import os
import asyncio
import logging
from datetime import datetime, timedelta, date

import asyncpg
from datasets import Dataset, load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------
# 1. Load feedback from PostgreSQL
# ----------------------------
async def fetch_all_feedback_from_db(dsn: str, task: str = "text2sql"):
    query = """
        SELECT input, prediction, correct_output, is_correct
        FROM feedback
        WHERE task = $1
    """
    conn = await asyncpg.connect(dsn)
    rows = await conn.fetch(query, task)
    await conn.close()

    feedback = []
    for row in rows:
        correct = row["correct_output"] if not row["is_correct"] else row["prediction"]
        if not correct:
            continue
        try:
            parsed_input = json.loads(row["input"])
            feedback.append(
                {
                    "question": parsed_input["question"],
                    "sql": correct,
                    "table_str": parsed_input["table_str"],
                }
            )
        except Exception as e:
            logger.warning(f"Skipping malformed feedback row: {e}")
    return feedback


# ----------------------------
# 2. Preprocessing: formatting_func
# ----------------------------
def build_formatting_func(tokenizer):
    def formatting_func(batch):
        prompts = []
        for question, sql_str, table_str in zip(
            batch["question"], batch["sql"], batch["table_str"]
        ):
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": f"""You are a SQL expert.

                        Given the question, original query, generate a SQL query to answer the question. Follow the response format and guidelines strictly. Do not include any additional text outside the specified format.

                        Use the table schema below!
                        ===Tables===
                        {table_str}

                        ===Response Guidelines===
                        1. Ensure the SQL is properly formatted.
                        2. Always return a valid JSON object using the structure below.

                        ===Response Format===
                        query=<SQL query if sufficient context is available>,

                        ===Question===
                        {question}
                        """,
                    },
                    {"role": "assistant", "content": f"query={sql_str}"},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompts.append(prompt)
        return prompts

    return formatting_func


def improve_lm_performance(model_id: str, db_url: str) -> None:
    """
    Improve the language model's performance by fine-tuning on feedback data and curated base data.

    This function performs the following steps:
    1. Fetches feedback data from the database
    2. Loads and samples curated base data
    3. Merges feedback and base data
    4. Loads the pre-trained model and tokenizer
    5. Sets up and runs the trainer for fine-tuning
    6. Pushes the fine-tuned model to the Hugging Face Hub

    Args:
        model_id (str): The identifier of the pre-trained model to use
        db_url (str): The URL of the PostgreSQL database containing feedback data

    Returns:
        None
    """
    # 1. Fetch all feedback data
    all_feedback_data = asyncio.run(fetch_all_feedback_from_db(db_url))
    if not all_feedback_data:
        logger.info("No feedback available for fine-tuning.")
        return

    feedback_dataset = Dataset.from_list(all_feedback_data)

    # 2. Load and sample curated WikiSQL base
    base_data = load_dataset("json", data_files="data/curated_base.jsonl")["train"]
    base_data = base_data.select_columns(["question", "sql", "table_str"])
    base_data = base_data.shuffle(seed=42).select(range(int(len(base_data) * 0.3)))

    # 3. Merge everything
    full_dataset = concatenate_datasets([feedback_dataset, base_data]).shuffle(seed=42)

    # 4. Load model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "o_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # 5. Setup trainer
    version_tag = f"v{date.today().isoformat()}"
    # output_dir = f"./output/qwen2.5-3B-4bit-text2sql-{version_tag}"
    formatting_func = build_formatting_func(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=full_dataset,
        formatting_func=formatting_func,
        args=SFTConfig(
            per_device_train_batch_size=2,
            dataset_num_proc=1,  # this can be commented out
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            push_to_hub=True,
            hub_model_id="nerzid/qwen2.5-3B-4bit-text2sql",
            hub_private_repo=False,
            eval_strategy="steps",
        ),
    )

    trainer.train()
    logger.info(f"Model pushed to 🤗 Hub with tag: {version_tag}")


# ----------------------------
# 4. Run
# ----------------------------
MODEL = os.getenv("HF_MODEL", "nerzid/qwen2.5-3B-4bit-text2sql")
DB_URL = os.getenv("POSTGRES_URL")

if not DB_URL:
    raise ValueError("Missing POSTGRES_URL in environment")

improve_lm_performance(MODEL, DB_URL)
