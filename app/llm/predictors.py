import dspy
from app.llm.signatures import (
    AITextFromText,
    AITextFromSQL,
    DisambiguateTextForSQL,
)
from app.llm.config import lm

# Load LM
dspy.configure(lm=lm)

get_AI_text_from_text = dspy.Predict(
    AITextFromText
)  # uses approach 1 model to detect if text is AI-generated
get_AI_text_from_sql = dspy.Predict(
    AITextFromSQL
)  # uses approach 2 model to detect if text is AI-generated
get_disambiguated_text = dspy.Predict(
    DisambiguateTextForSQL
)  # disambiguates the text to make it easier to generate an SQL query from it
