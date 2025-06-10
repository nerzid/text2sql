import dspy
from app.core.config import settings

provider = settings.LLM_PROVIDER
model = settings.LLM_MODEL
api_url = settings.LLM_API_URL
lm = dspy.LM(
    provider + "/" + model,
    api_base=api_url,
    api_key="1",
    max_completion_tokens=20000,
    cache=True,
)
dspy.configure(lm=lm)
