# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

# Importing configurations and models
from config import LANGUAGES
from request_models import TTSRequest, TranslationRequest
from utils import audio_streamer, process_text
from model_manager import initialize_models

app = FastAPI()

# Initialize models
translator, sp_processor, tts_languages, tts_models = initialize_models()

language_dict = {lang.split('_')[0]: lang for lang in LANGUAGES}

@app.get("/translation_languages/")
async def get_translation_languages():
    """
    Endpoint to get a list of supported translation languages.
    """
    return {"languages": LANGUAGES}

@app.get("/tts_languages/")
async def get_tts_languages():
    """
    Endpoint to get a list of supported TTS languages.
    """
    return {"languages": tts_languages}

@app.post("/translate/")
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
    
    return {"translations": translations}

@app.post("/tts/")
async def text_to_speech(request: TTSRequest):
    """
    Endpoint for text-to-speech conversion.
    """
    lang = request.lang
    text = request.text
    text = process_text(lang, text)

    if lang not in tts_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    tts_model = tts_models.get(lang)
    if not tts_model:
        raise HTTPException(status_code=500, detail="TTS model loading error")

    speech_data = tts_model.synthesis(text)
    return StreamingResponse(audio_streamer(speech_data["x"], speech_data["sampling_rate"]), media_type="audio/wav")

@app.get("/tts/")
def tts(lang: str, text: str):
    if lang not in tts_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    tts = tts_models[lang]
    text = process_text(lang, text)
    speech_data = tts.synthesis(text)

    return StreamingResponse(audio_streamer(speech_data["x"], speech_data["sampling_rate"]), media_type="audio/wav")