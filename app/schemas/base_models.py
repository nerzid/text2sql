from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class TextRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    input: dict
    prediction: str
    is_correct: bool
    correct_output: Optional[str] = None
    task: Optional[str] = None
    model: Optional[str] = None


class PreprocessTextRequest(BaseModel):
    text: str
