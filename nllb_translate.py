# -*- coding: utf-8 -*-
"""NLLB-200 离线翻译：俄语 -> 中文（Qwen 不可用时的兜底）。"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "rus_Cyrl"
TGT_LANG = "zho_Hans"

_tokenizer = None
_model = None


def load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def translate(text):
    """俄语 -> 中文。失败返回 None。"""
    try:
        load()
    except Exception:
        return None
    _tokenizer.src_lang = SRC_LANG
    inputs = _tokenizer(text, return_tensors="pt")
    forced = _tokenizer.convert_tokens_to_ids(TGT_LANG)
    try:
        outputs = _model.generate(**inputs, forced_bos_token_id=forced, max_new_tokens=128)
    except Exception:
        return None
    return _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
