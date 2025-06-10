from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
import logging
import json
from datetime import datetime
from app.core.dependencies import get_redis_client
from app.schemas.base_models import (
    FeedbackRequest,
    PreprocessTextRequest,
    QueryRequest,
    TextRequest,
)
from app.services.ai_detector import is_ai_generated
from app.services.text_to_sql import preprocess_text, text2sql


app = FastAPI()


# Mount the /static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve favicon.ico to avoid favicon not found errors
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


@app.post("/is_ai_generated")
async def create_ai_gen_detection_service(request: TextRequest):
    """
    Endpoint to detect if the given text is AI-generated.

    Args:
        request (TextRequest): The request object containing the text to be analyzed.

    Returns:
        dict: A dictionary containing the truncated input text and the AI detection result.

    Raises:
        HTTPException: If the input text is empty or if an error occurs during processing.
    """
    try:
        text = request.text.strip()
        logging.info(
            f"Received text for AI detection: '{text[:100]}{'...' if len(text) > 100 else ''}'"
        )

        if not text:
            logging.warning("Received empty text input for AI detection.")
            raise HTTPException(status_code=400, detail="Empty text input")

        prediction = is_ai_generated(text)
        logging.info(
            f"AI detection result: {prediction} for text snippet: '{text[:100]}{'...' if len(text) > 100 else ''}'"
        )

        return {"text": text[:100] + "...", "ai_generated": prediction}
    except Exception as e:
        logging.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text2sql")
async def text_to_sql(query: QueryRequest):
    """
    Endpoint to convert natural language text to SQL query.

    Args:
        query (QueryRequest): The request object containing the question to be converted to SQL.

    Returns:
        JSONResponse: A JSON response containing the generated SQL query.

    Raises:
        HTTPException: If an error occurs during the text-to-SQL conversion.
    """
    logging.info(f"Received query for text2sql: {query.question}")
    try:
        result = text2sql(query.question)
        logging.info(f"Text2SQL successful. Result: {result}")
        return JSONResponse(content={"result": result}, status_code=200)
    except Exception as e:
        logging.error(f"Text2SQL failed: {e}")
        raise HTTPException(status_code=500, detail="Text2SQL failed.")


@app.post("/feedback")
async def feedback(data: FeedbackRequest):
    """
    Endpoint to handle feedback for AI-generated content.

    Args:
        data (FeedbackRequest): The request object containing feedback information.

    Returns:
        JSONResponse: A JSON response indicating the feedback was successfully enqueued.

    Raises:
        HTTPException: If required fields are missing or if an error occurs during processing.
    """
    try:
        logging.info(f"Received feedback: {data}")
        if not data.input:
            raise HTTPException(status_code=400, detail="Missing input")
        if not data.task:
            raise HTTPException(status_code=400, detail="Missing task")
        if not data.prediction:
            raise HTTPException(status_code=400, detail="Missing prediction")

        if data.task == "text2sql":
            question = data.input["question"]
            table_str = data.input["table_str"]
            full_input = {"question": question, "table_str": table_str}
        elif data.task == "ai-detector":
            full_input = data.input
        else:
            raise HTTPException(status_code=400, detail="Invalid task")

        # Normalize correct SQL
        correct_output = data.prediction if data.is_correct else data.correct_output
        if not correct_output:
            raise HTTPException(
                status_code=400,
                detail="Missing correct_output for incorrect prediction",
            )

        feedback_dict = {
            "input": full_input,
            "prediction": data.prediction,
            "is_correct": data.is_correct,
            "correct_output": correct_output,
            "task": data.task,
            "model": data.model,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Enqueue feedback to Redis
        r = get_redis_client()
        r.rpush("feedback_queue", json.dumps(feedback_dict))
        return JSONResponse(content={"status": "Feedback enqueued"}, status_code=200)
    except Exception as e:
        logging.error(f"Feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess_text")
async def preprocess(data: PreprocessTextRequest):
    """
    Endpoint to preprocess the given text.

    Args:
        data (PreprocessTextRequest): The request object containing the text to be preprocessed.

    Returns:
        JSONResponse: A JSON response containing the preprocessed text.

    Raises:
        HTTPException: If an error occurs during preprocessing.
    """
    logging.info(f"Received request for preprocessing: {data.text}")
    try:
        preprocessed_text = preprocess_text(data.text)
        logging.info(f"Preprocessing successful. Result: {preprocessed_text}")
        result_dict = {
            "result": preprocessed_text["disambiguated_text"],
            "is_too_vague": preprocessed_text["is_too_vague"],
        }

        return JSONResponse(content=result_dict, status_code=200)
    except Exception as e:
        logging.error(f"Preprocess failed: {e}")
        raise HTTPException(status_code=500, detail="Preprocessing failed.")


@app.get("/health")
def health_check():
    """
    Endpoint for health check.

    Returns:
        dict: A dictionary containing the status of the service.
    """
    logging.info("Received health check request")
    return {"status": "ok"}
