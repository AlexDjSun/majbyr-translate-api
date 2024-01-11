from pydantic import BaseModel

class TranslationRequest(BaseModel):
    text: str = 'Чолӧм!'
    src: str = 'kpv'
    tgt: str = 'fin'

class TTSRequest(BaseModel):
    lang: str = 'kpv'
    text: str = 'Чолӧм!'