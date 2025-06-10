import dspy


class AITextFromText(dspy.Signature):
    """Rephrase the text to sound more natural and fluent but keep the sentence's structure same, i.e., if it is a question, it should stay as a question

    /no_think
    """

    text: str = dspy.InputField(desc="The input sentence")
    ai_text: str = dspy.OutputField(desc="The AI generated sentence")


class AITextFromSQL(dspy.Signature):
    """Convert the sql to text. The text should be in a question format or an instruction.

    /no_think
    """

    sql: str = dspy.InputField(desc="The input sql query")
    text: str = dspy.OutputField(desc="The text that represents the sql query")


class DisambiguateTextForSQL(dspy.Signature):
    """Disambiguate the text to prepare it to be used to generate an SQL query by

    1- fixing its grammar
    2- using the relevant table headers if necessary
    3- making sure the text is in a question format or an instruction
    4- changing the wording to be sql friendly, e.g., replacing "show me" with "select"
    5- clarifying any ambiguity in the text without adding any extra information

    Even if the text is too ambiguous to be corrected, try your best to rewrite it in a question format or an instruction.

    /no_think
    """

    text: str = dspy.InputField(desc="The input text")
    relevant_headers: str = dspy.InputField(desc="Relevant headers from the database")
    disambiguated_text: str = dspy.OutputField(
        desc="The text that is ready to be used to generate an SQL query"
    )
    is_too_vague: bool = dspy.OutputField(
        desc="True if the text is too ambiguous to the point it cannot be used to generate an sql query without additional context, False otherwise."
    )
