from pydantic import BaseModel, EmailStr, Field


class RegisterIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=60)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class IngestIn(BaseModel):
    youtube_url: str = Field(..., description="Any YouTube URL (watch, youtu.be, shorts, embed)")


class AskIn(BaseModel):
    conversation_id: str
    question: str = Field(..., min_length=1, max_length=2000)
