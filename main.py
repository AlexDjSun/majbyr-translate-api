# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from config import LANGUAGES
from request_models import TranslationRequest, TTSRequest
from response_models import TranslationResponse, TTSResponse, ListResponse, ComplexTranslationResponse
from utils import audio_streamer, process_text
from model_manager import initialize_models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator, sp_processor, tts_languages, tts_models = initialize_models()

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
    Endpoint for text translation.
    """
    src_lang_tag = language_dict.get(request.src)
    tgt_lang_tag = language_dict.get(request.tgt)

    if not src_lang_tag or not tgt_lang_tag:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # Prepend the source language tag and tokenize
    tokenized_source = [f'__{src_lang_tag}__'] + sp_processor.EncodeAsPieces(request.text)

    # Translate the tokenized text
    results = translator.translate_batch(
        [tokenized_source], 
        target_prefix=[[f'__{tgt_lang_tag}__']], 
        num_hypotheses=4,
        beam_size=4,
    )

    translations = [sp_processor.DecodePieces(hypothesis[1:]) for hypothesis in results[0].hypotheses]
    
    return {"result": translations[0],
            "alternatives": translations[1:]}

@app.post("/translate_complex/", response_model=ComplexTranslationResponse)
async def complex_translate(request: TranslationRequest):
    """
    Endpoint for text translation.
    """
    src_lang_tag = language_dict.get(request.src)
    tgt_lang_tag = language_dict.get(request.tgt)

    if not src_lang_tag or not tgt_lang_tag:
        raise HTTPException(status_code=400, detail="Unsupported language")

    text = request.text
    EOS_chars = ['.', '!', '?']

    if text[-1] not in EOS_chars:
        text += '.'

    for EOS_char in EOS_chars:
        text = text.replace(EOS_char, EOS_char + '\n')

    sentences = text.strip().split('\n')
    tokenized_sentences = []
    for sentence in sentences:
        if sentence.strip() == '':
            tokenized_sentences.append(['\n'])
            continue
        tokenized_sentences.append([f'__{src_lang_tag}__'] + sp_processor.EncodeAsPieces(sentence))

    translations = []
    for sentence in tokenized_sentences:   
        if sentence == ['\n']:
            translations.append(['\n'])
            continue 
        # Translate the tokenized text
        translations.append(translator.translate_batch(
            [sentence], 
            target_prefix=[[f'__{tgt_lang_tag}__']], 
            num_hypotheses=4,
            beam_size=4,
        )[0].hypotheses)
    
    translated_sentences = []
    for translation in translations:
        if translation == ['\n']:
            translated_sentences.append('\n')
            continue
        translated_sentences.append([sp_processor.DecodePieces(alt[1:]) for alt in translation])
    
    return {'result': ' '.join(sentence[0] for sentence in translated_sentences),
            'sentences': translated_sentences
            }

@app.get("/tts/", response_class=StreamingResponse)
def text_to_speech(lang: str, text: str):
    """
    Endpoint for text-to-speech conversion.
    """
    if lang not in tts_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    tts = tts_models[lang]
    text = process_text(lang, text)
    speech_data = tts.synthesis(text)

    return StreamingResponse(audio_streamer(speech_data["x"], speech_data["sampling_rate"]), media_type="audio/wav")