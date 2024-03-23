# model_manager.py
import os
import torch
from scipy.io.wavfile import write

import ctranslate2
import sentencepiece as spm

from ttsmms import download
from tts_interface import TTS

import fasttext

from config import TRANSLATION_MODEL_PATH, TTS_MODELS_PATH, SP_MODEL, LANGID_MODEL_PATH, LANGUAGES

 

language_dict = {lang.split('_')[0]: lang for lang in LANGUAGES}

def load_translation_model():
    """
    Loads the translation model.
    """
    translator = ctranslate2.Translator(TRANSLATION_MODEL_PATH, device="auto")
    print("Translation model is loaded")
    return translator

def load_sentencepiece_model():
    """
    Loads the SentencePiece model.
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(SP_MODEL)
    return sp

def download_and_load_tts_models():
    """
    Downloads (if necessary) and loads TTS models for the supported languages.
    """
    # Create TTS directory if it doesn't exist
    if not os.path.exists(TTS_MODELS_PATH): 
        os.makedirs(TTS_MODELS_PATH)

    tts_languages = []
    tts_models = {}

    for lang in language_dict.keys():
        model_path = f"{TTS_MODELS_PATH}/{lang}"
        if not os.path.exists(model_path):
            try:
                download(lang, TTS_MODELS_PATH)
                print(f"Downloaded {lang} TTS model")
            except Exception as e:
                print(f"Failed to download {lang} TTS model: {e}")
                continue

        tts_languages.append(lang)
        tts_models[lang] = TTS(model_path)
        print(f"Loaded {lang} TTS model")

    return tts_languages, tts_models

def load_langid_model():
    """
    Loads the langid model.
    """
    langid_model = fasttext.load_model(LANGID_MODEL_PATH)
    return langid_model

def initialize_models():
    """
    Initializes all models used in the application.
    """

    translator = load_translation_model()
    sp_processor = load_sentencepiece_model()
    tts_languages, tts_models = download_and_load_tts_models()
    langid = load_langid_model()

    return translator, sp_processor, tts_languages, tts_models, langid

