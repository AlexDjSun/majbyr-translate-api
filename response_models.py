from pydantic import BaseModel
from typing import List

class TranslationResponse(BaseModel):
    result: str = 'Hei!'
    alternatives: List[str] = ['Hei!', 'Terve!', 'Moi!']

class TTSResponse(BaseModel):
    audio: bytes = b''

class ListResponse(BaseModel):
    languages: List[str] = ['eng', 'fin', 'kpv']

