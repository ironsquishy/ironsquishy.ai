from fastapi import FastAPI
from pydantic import BaseModel

from app.prompting import build_prompt

app = FastAPI(title="ironsquishy-phi-agent")


class PromptRequest(BaseModel):
    prompt: str


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/prompt-preview")
def prompt_preview(req: PromptRequest) -> dict:
    return {"prompt": build_prompt(req.prompt)}