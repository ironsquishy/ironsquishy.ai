from pydantic import BaseModel, Field


class InfraAnswer(BaseModel):
    summary: str = Field(..., description="Short high-level answer")
    risks: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)