from pydantic import BaseModel

class TranslationRequest(BaseModel):
    text: str
    src: str
    tgt: str

class TTSRequest(BaseModel):
    lang: str
    text: str