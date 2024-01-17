from pydantic import BaseModel
from typing import List

class TranslationResponse(BaseModel):
    result: str = 'Hei!'
    alternatives: List[str] = ['Hei!', 'Terve!', 'Moi!']

class ComplexTranslationResponse(BaseModel):
    result: str = 'Hei! Tervetuloa! Miten voit?'
    sentences: List[List[str]] = [['Hei!', 'Terve!', 'Moi!'], ['Tervetuloa!', 'Toivottaa tervetulleeksi!'], ['Miten voit?', 'Miten menee?', 'Mitä kuuluu?']]

class TTSResponse(BaseModel):
    audio: bytes = b''

class ListResponse(BaseModel):
    languages: List[str] = ['eng', 'fin', 'kpv']

