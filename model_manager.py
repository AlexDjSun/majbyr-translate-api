# model_manager.py
import os
import torch
from scipy.io.wavfile import write

import ctranslate2
import sentencepiece as spm

from ttsmms import TTS, download
from config import TRANSLATION_MODEL_PATH, TTS_MODELS_PATH, SP_MODEL, LANGUAGES

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

def _synthesis(self, txt, wav_path=None):
    txt = self._use_uroman(txt)
    txt = self.text_mapper.filter_oov(txt)
    stn_tst = self.text_mapper.get_text(txt, self.hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).cuda()
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        hyp = self.net_g.infer(
            x_tst, x_tst_lengths,  
            noise_scale= 0.667, # ?
            noise_scale_w=0.8, # ?
            length_scale=1.0 # speech speed
        )[0][0,0].cpu().float().numpy()
    if wav_path != None:
        write(wav_path, self.hps.data.sampling_rate, hyp)
        return wav_path
    return {"x":hyp,"sampling_rate":self.sampling_rate}

def initialize_models():
    """
    Initializes all models used in the application.
    """
    TTS.synthesis = _synthesis
    translator = load_translation_model()
    sp_processor = load_sentencepiece_model()
    tts_languages, tts_models = download_and_load_tts_models()

    return translator, sp_processor, tts_languages, tts_models

