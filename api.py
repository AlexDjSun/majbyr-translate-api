from fastapi import FastAPI, HTTPException
import ctranslate2
import sentencepiece as spm

from pydantic import BaseModel

class TranslationRequest(BaseModel):
    text: str
    src: str
    tgt: str

TRANSLATION_MODEL_PATH = "models/converted-smugri-v4"
SP_MODEL = "models/spm/flores200_sacrebleu_tokenizer_spm.ext.model"
LANGUAGES = ['kpv_Cyrl', 'mhr_Cyrl', 'udm_Cyrl', 'eng_Latn', 'est_Latn', 'fin_Latn', 'rus_Cyrl','hun_Latn', 'kca_Cyrl', 'koi_Cyrl', 'krl_Latn', 'liv_Latn', 'lud_Latn', 'lvs_Latn', 'mdf_Cyrl', 'mns_Cyrl', 'mrj_Cyrl', 'myv_Cyrl', 'nob_Latn', 'olo_Latn', 'sma_Latn', 'sme_Latn', 'smj_Latn', 'smn_Latn', 'sms_Latn', 'vep_Latn', 'vro_Latn']

app = FastAPI()

# Load models
translator = ctranslate2.Translator(TRANSLATION_MODEL_PATH, device="auto")
sp = spm.SentencePieceProcessor()
sp.Load(SP_MODEL)
print("Models loaded")

language_dict = {lang.split('_')[0]: lang for lang in LANGUAGES}

@app.post("/translate/")
def translate_text(request: TranslationRequest):
    src_lang_tag = language_dict.get(request.src)
    tgt_lang_tag = language_dict.get(request.tgt)

    if not src_lang_tag or not tgt_lang_tag:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # Prepend the source language tag and tokenize
    tokenized_source = [f'__{src_lang_tag}__'] + sp.EncodeAsPieces(request.text)

    # Translate the tokenized text
    results = translator.translate_batch(
        [tokenized_source], 
        target_prefix=[[f'__{tgt_lang_tag}__']], 
        num_hypotheses=4,
        beam_size=4,
    )

    translations = [sp.DecodePieces(hypothesis[1:]) for hypothesis in results[0].hypotheses]
    
    return {"translations": translations}