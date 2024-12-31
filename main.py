# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from config import LANGUAGES
from request_models import TranslationRequest, TTSRequest
from response_models import TranslationResponse, TTSResponse, ListResponse, ComplexTranslationResponse
from utils import audio_streamer, process_text_for_tts, replace_unsupported_chars
from model_manager import initialize_models

import re
import nltk

import asyncio

nltk.download('punkt', download_dir='/tmp')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator, sp_processor, tts_languages, tts_models, langid = initialize_models()

language_dict = {lang.split('_')[0]: lang for lang in LANGUAGES}

@app.get("/translation_languages/", response_model=ListResponse)
async def get_translation_languages():
    """
    Endpoint to get a list of supported translation languages.
    """
    return {"languages": list(language_dict.keys())}

@app.get("/tts_languages/")
async def get_text_to_speech_languages():
    """
    Endpoint to get a list of supported TTS languages.
    """
    return {"languages": tts_languages}

@app.post("/translate/", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Endpoint for text translation all at once.
    """
    src_lang_tag = language_dict.get(request.src)
    tgt_lang_tag = language_dict.get(request.tgt)

    if not src_lang_tag or not tgt_lang_tag:
        raise HTTPException(status_code=400, detail="Unsupported language")

    text = replace_unsupported_chars(request.text)
    tokenized_source = [f'__{src_lang_tag}__'] + sp_processor.EncodeAsPieces(text)

    results = await asyncio.to_thread(translator.translate_batch,
        [tokenized_source], 
        target_prefix=[[f'__{tgt_lang_tag}__']], 
        num_hypotheses=4,
        beam_size=4,
    )

    translations = [sp_processor.DecodePieces(hypothesis[1:]).replace('⁇', '').replace('<unk>', '') for hypothesis in results[0].hypotheses]
    
    return {"result": translations[0],
            "alternatives": translations[1:]}

@app.post("/translate_by_sentences_v1/")#, response_model=ComplexTranslationResponse)
async def translate_by_sentences(request: TranslationRequest):
    """
    Endpoint for text translation by sentences.
    """
    src_lang_tag = language_dict.get(request.src)
    tgt_lang_tag = language_dict.get(request.tgt)

    if not src_lang_tag or not tgt_lang_tag:
        raise HTTPException(status_code=400, detail="Unsupported language")

    text = replace_unsupported_chars(request.text)

    paragraphs = [p for p in text.split('\n')]

    translation_lists = []
    for paragraph in paragraphs:
        paragraph_translations = []
        if not paragraph.strip():
            translation_lists.append([['']])
            continue
        sentences = nltk.sent_tokenize(paragraph)
        tokenized_sentences = [[f'__{src_lang_tag}__'] + sp_processor.EncodeAsPieces(sentence) for sentence in sentences]
        tgt_lang_tag_list = [[f'__{tgt_lang_tag}__'] for _ in tokenized_sentences]
        tokenized_translations = await asyncio.to_thread(translator.translate_batch,
                                                            tokenized_sentences,
                                                            target_prefix=tgt_lang_tag_list,
                                                            num_hypotheses=4,
                                                            beam_size=4)

        for i, translation in enumerate(tokenized_translations):
            sentence_translations = []
            for hypothesis in translation.hypotheses:
                translated_sentence = sp_processor.DecodePieces(hypothesis[1:]).replace('⁇', '').replace('<unk>', '')
                sentence_translations.append(translated_sentence)
            paragraph_translations.append(sentence_translations)
        translation_lists.append(paragraph_translations)

    reconstructed_text = ''
    for paragraph_sentences in translation_lists:
        paragraph = ' '.join([sentences[0] for sentences in paragraph_sentences])
        reconstructed_text += paragraph + '\n'

    return {"result": reconstructed_text.strip(),
            "translations": translation_lists}

@app.post("/translate_by_sentences/")#, response_model=ComplexTranslationResponse)
async def translate_v2(request: TranslationRequest):
    """
    Endpoint for text translation by sentences.
    """
    src_lang_tag = language_dict.get(request.src)
    tgt_lang_tag = language_dict.get(request.tgt)

    if not src_lang_tag or not tgt_lang_tag:
        raise HTTPException(status_code=400, detail="Unsupported language")

    text = replace_unsupported_chars(request.text)
    paragraphs = text.split('\n')

    src_sentences = []
    src_dict = {}
    for n, paragraph in enumerate(paragraphs):
        sentences = nltk.sent_tokenize(paragraph)
        if sentences:
            src_dict[f"p{n}"] = sentences
            src_sentences.extend(sentences)

    tokenized_sentences = [[f'__{src_lang_tag}__'] + sp_processor.EncodeAsPieces(sentence) for sentence in src_sentences]
    tgt_lang_tag_list = [[f'__{tgt_lang_tag}__'] for _ in tokenized_sentences]

    tokenized_translations = await asyncio.to_thread(translator.translate_batch,
                                                     tokenized_sentences,
                                                     target_prefix=tgt_lang_tag_list,
                                                     num_hypotheses=4,
                                                     beam_size=4,
                                                     disable_unk=True)

    translated_sentences = [
        [sp_processor.DecodePieces(hypothesis[1:]).replace('⁇', '') for hypothesis in translation.hypotheses]
        for translation in tokenized_translations
    ]

    sentence_index = 0
    translation_lists = []
    for n in range(len(paragraphs)):
        if f"p{n}" in src_dict:
            num_sentences = len(src_dict[f"p{n}"])
            translation_lists.append(translated_sentences[sentence_index:sentence_index + num_sentences])
            sentence_index += num_sentences
        else:
            translation_lists.append([[""]])

    reconstructed_text = '\n'.join(' '.join(sentences[0] for sentences in paragraph_sentences)
                                   for paragraph_sentences in translation_lists)

    return {"result": reconstructed_text.strip(),
            "translations": translation_lists}

@app.get("/tts/", response_class=StreamingResponse)
def text_to_speech(lang: str, text: str):
    """
    Endpoint for text-to-speech conversion.
    """
    if lang not in tts_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    tts = tts_models[lang]
    text = process_text_for_tts(lang, text)
    speech_data = tts.synthesis(text)

    return StreamingResponse(audio_streamer(speech_data["x"], speech_data["sampling_rate"]), media_type="audio/wav")

@app.get('/detect_language/')
async def detect_language(text: str):
    """
    Endpoint for language detection.
    """
    lang = langid.predict(text)[0][0].split('__')[-1]
    return lang