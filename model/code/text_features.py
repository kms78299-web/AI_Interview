"""
텍스트 특징 추출 (BERT 임베딩과 텍스트 통계). 모델은 extract_features.py에서 한 번 로드한 것을 넘겨받아 재사용한다.
"""

import re

import numpy as np

from config import EMBED_DIM, FILLER_WORDS, TEXT_BERT_DIM

_WORD_RE = re.compile(r"[A-Za-z가-힣]+")
_HANGUL_RE = re.compile(r"[가-힣]")
_FILLER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b", re.IGNORECASE
)


def is_korean(text):
    return bool(_HANGUL_RE.search(text))


def load_bert(model_name, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def _embed_bert(text, tokenizer, model, device, max_length=512):
    import torch

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state[0]  # (seq_len, hidden)
    attn_mask = inputs["attention_mask"][0].unsqueeze(-1).float()  # (seq_len, 1)
    # attention mask 기반 mean pooling (padding 토큰 제외)
    summed = (last_hidden * attn_mask).sum(dim=0)
    count = attn_mask.sum(dim=0).clamp(min=1e-9)
    mean_pooled = (summed / count).cpu().numpy()
    return mean_pooled.astype(np.float32)


def extract_text_embedding(clean_text, tokenizer, model, device, max_length=512):
    """BERT류 모델로 텍스트 임베딩(768)을 만든다. 텍스트가 비면 0벡터를 반환한다."""
    if not clean_text:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    return _embed_bert(clean_text, tokenizer, model, device, max_length=max_length)


# 흔한 기능어는 "반복 단어"로 잡아봐야 의미가 없어서 repeated_words 탐지에서 제외.
_STOPWORDS = {
    "i", "a", "the", "to", "and", "is", "it", "was", "we", "you", "of", "in",
    "that", "this", "my", "me", "so", "for", "on", "at", "as", "be", "am",
}


def extract_text_stats_for_prompt(clean_text):
    """LLM 프롬프트용 텍스트 통계(단어 수, 군더더기 말, 반복 단어, 어휘 다양성). BERT 없이 계산한다."""
    if not clean_text:
        return {
            "transcript": "",
            "filler_words": [],
            "filler_count": 0,
            "repeated_words": [],
            "answer_length": 0,
            "vocab_diversity": 0.0,
        }

    words = _WORD_RE.findall(clean_text.lower())
    answer_length = len(words)
    vocab_diversity = round(len(set(words)) / len(words), 2) if words else 0.0

    filler_matches = [m.lower() for m in _FILLER_RE.findall(clean_text)]
    filler_words = sorted(set(filler_matches))
    filler_count = len(filler_matches)

    word_counts = {}
    for w in words:
        if w in _STOPWORDS or w in FILLER_WORDS:
            continue
        word_counts[w] = word_counts.get(w, 0) + 1
    repeated_words = sorted([w for w, c in word_counts.items() if c >= 2], key=lambda w: -word_counts[w])[:5]

    return {
        "transcript": clean_text,
        "filler_words": filler_words,
        "filler_count": filler_count,
        "repeated_words": repeated_words,
        "answer_length": answer_length,
        "vocab_diversity": vocab_diversity,
    }


def extract_text_features(clean_text, get_bert_fn, device):
    """clean_text(타임스탬프 제거됨)로 임베딩과 통계를 함께 만든다. get_bert_fn은 한글 여부에 따라 (tokenizer, model)을 돌려준다."""
    if not clean_text:
        # transcript가 비어있는 샘플: 에러로 취급하지 않고 0벡터로 채운다.
        bert_vec = np.zeros(TEXT_BERT_DIM, dtype=np.float32)
        filler_count, vocab_diversity, answer_length = 0.0, 0.0, 0.0
        return np.concatenate([bert_vec, [filler_count, vocab_diversity, answer_length]]).astype(np.float32)

    tokenizer, model = get_bert_fn(is_korean(clean_text))
    bert_vec = _embed_bert(clean_text, tokenizer, model, device)

    words = _WORD_RE.findall(clean_text.lower())
    answer_length = float(len(words))
    vocab_diversity = float(len(set(words)) / len(words)) if words else 0.0
    filler_count = float(len(_FILLER_RE.findall(clean_text)))

    return np.concatenate([bert_vec, [filler_count, vocab_diversity, answer_length]]).astype(np.float32)
