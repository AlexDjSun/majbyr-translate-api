import os

MODELS_PATH = "models/"
if not os.path.exists(MODELS_PATH):
    # stop the program if the models directory does not exist
    raise Exception("Models directory does not exist")

TRANSLATION_MODEL_PATH = "models/converted-smugri-v4/"
TTS_MODELS_PATH = "models/tts/"
SP_MODEL = "models/spm/flores200_sacrebleu_tokenizer_spm.ext.model"
LANGID_MODEL_PATH = "models/langid/langdetect_v0.multi.bin"

LANGUAGES = [
    'kpv_Cyrl', 
    'mhr_Cyrl', 
    'mrj_Cyrl', 
    'udm_Cyrl', 
    'eng_Latn', 
    'est_Latn', 
    'fin_Latn', 
    'rus_Cyrl',
    'hun_Latn', 
    'kca_Cyrl', 
    'koi_Cyrl', 
    'krl_Latn', 
    'myv_Cyrl', 
    'liv_Latn', 
    'lud_Latn', 
    'lvs_Latn', 
    'mdf_Cyrl', 
    'mns_Cyrl', 
    'nob_Latn', 
    'olo_Latn', 
    'sma_Latn', 
    'sme_Latn', 
    'smj_Latn', 
    'smn_Latn', 
    'sms_Latn', 
    'vep_Latn', 
    'vro_Latn'
    ]
