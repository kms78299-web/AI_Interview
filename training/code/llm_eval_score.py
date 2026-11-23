"""
LLM 프롬프트 검증 3단계: Gemini 응답(llm_eval_gemini.py)을 규칙 준수와 평가자 라벨 일치로 채점한다. API 호출 없음, CPU.

판정 기준 (결과를 보기 전에 정함. 결과를 보고 바꾸지 않는다):
  1. 규칙 준수가 1순위다. 서비스 조건 C의 위반률이 A보다 높으면(차이의 사람 bootstrap 95% 구간 하한 > 0) 모델 정보가 해롭다고 본다.
  2. 평가자 라벨 일치(세션 overall_band vs 사람 평균 라벨)는 보조 지표다. 라벨은 '짧을수록 높음' 편향이 있으므로
     길이 보정 편상관을 함께 보고, '길이만으로 맞히기' 기준선보다 높은지 확인한다.
  3. C가 A보다 라벨 일치가 높아도(구간 하한 > 0) 그 차이가 모델 등급을 그대로 옮긴 결과라면(C의 overall_band = 모델 등급 비율이 A보다 높음)
     모델 정보의 효과가 아니라 복사로 본다.
  val은 비교·수정용, test는 최종 프롬프트로 한 번만 채점한다.

python code/llm_eval_score.py --split val      -> llm_eval/val/score.json, 표 출력
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAND = {"상": 2, "중": 1, "하": 0}
LABELS = ["overall_performance", "answer_score", "interview_score"]
WORD_RE = re.compile(r"[A-Za-z가-힣']+")
N_BOOT, SEED = 1000, 0

# 감정·태도 추측 (프롬프트 규칙 1에서 금지한 표현)
EMOTION_RE = re.compile(r"긴장|불안|떨리|떨림|당황|자신감|자신 있|자신있|준비가? ?부족|준비되지 않|성의|열정|소극적|적극적인 태도")
# 개인차가 큰 값을 바꾸라는 조언 (규칙 7(c))
TRAIT_ADVICE_RE = re.compile(r"(목소리|음량|볼륨|소리)[^.。\n]{0,15}(작게|줄이|낮추)|(음높이|톤|피치)[^.。\n]{0,15}(낮추|내리|바꾸)|깜빡임[^.。\n]{0,15}(줄이|참)")
# 빈 대본을 '답변 안 함'으로 단정
NO_ANSWER_RE = re.compile(r"답변(을|이)? ?(전혀 )?(하지 않|없었|이루어지지 않|안 했|하지 못)|침묵으로 일관|아무 말")
MODEL_RE = re.compile(r"모델|예측")
# 대본에 없는 의도·상태로 답변 성격을 바꿔 씀
OVERCLAIM_RE = re.compile(r"거부|거절|준비하지 못|준비되지 않|준비를 못|회피")
# 좋은 점의 근거로 쓰지 말라고 한 값 (프롬프트 규칙 3)
STRENGTH_BASIS_RE = re.compile(r"음량|볼륨|음높이|톤|목소리|발성|시선|응시|정면|표정|깜빡|군더더기")


def texts(obj):
    """응답 안의 모든 문자열 (금지 표현 검사용)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from texts(v)


def check(resp, analysis, cond):
    """응답 하나의 규칙 위반 목록. analysis: llm_eval_prepare 결과."""
    videos = analysis["result"]["videos"]
    n = len(videos)
    ids = [f"Q{k + 1}" for k in range(n)]
    seg = {f"Q{k + 1}": {g["id"]: g for g in (v.get("segments") or [])} for k, v in enumerate(videos)}
    empty = {f"Q{k + 1}" for k, v in enumerate(videos) if not (v.get("transcript") or "").strip()}
    v = {}
    if not isinstance(resp, dict):
        return {"json_object": True}
    af = resp.get("answer_feedback") or []
    v["answer_count"] = [a.get("question_id") for a in af if isinstance(a, dict)] != ids
    st = {a.get("question_id"): a.get("status") for a in af if isinstance(a, dict)}
    ins = [q for q in ids if st.get(q) == "insufficient_data"]
    expect = "ready" if not ins else ("insufficient_data" if len(ins) == n else "partial")
    v["session_status"] = resp.get("status") != expect
    v["empty_marked_evaluated"] = any(st.get(q) == "evaluated" for q in empty)          # 빈 대본인데 평가함
    v["text_marked_insufficient"] = any(q not in empty for q in ins)                    # 대본이 있는데 평가 불가
    bad_null = False
    for a in af:
        if isinstance(a, dict) and a.get("status") == "insufficient_data":
            bad_null |= any(a.get(k) is not None for k in ("content_band", "strength", "improvement"))
            bad_null |= (a.get("relevance") or {}).get("rating") != "평가불가"
    v["insufficient_fields"] = bad_null
    band_ok = resp.get("overall_band") in BAND or (resp.get("overall_band") is None and expect == "insufficient_data")
    v["overall_band_value"] = not band_ok
    df = resp.get("delivery_feedback") or {}
    bad_id = strength_issue = 0
    for kind in ("strengths", "improvements"):
        for item in df.get(kind) or []:
            for e in (item.get("evidence") or []) if isinstance(item, dict) else []:
                g = seg.get(e.get("question_id"), {}).get(e.get("segment_id")) if isinstance(e, dict) else None
                if g is None:
                    bad_id += 1
                elif kind == "strengths" and g.get("rule_flag") == "issue":
                    strength_issue += 1
    v["evidence_unknown_id"] = bad_id > 0
    v["strength_on_issue_segment"] = strength_issue > 0
    all_text = " ".join(texts(resp))
    v["emotion_words"] = bool(EMOTION_RE.search(all_text))
    v["trait_advice"] = bool(TRAIT_ADVICE_RE.search(all_text))
    empty_text = " ".join(" ".join(texts(a)) for a in af if isinstance(a, dict) and a.get("question_id") in empty)
    v["no_answer_claim"] = bool(empty and NO_ANSWER_RE.search(empty_text + " " + str(resp.get("band_reason") or "")
                                                             + " " + str(resp.get("summary") or "")))
    v["model_mentioned_in_A"] = cond == "A" and bool(MODEL_RE.search(all_text))
    v["overclaim_words"] = bool(OVERCLAIM_RE.search(all_text))
    v["strength_forbidden_basis"] = any(
        STRENGTH_BASIS_RE.search(str(item.get("feedback") or "")) for item in (df.get("strengths") or []) if isinstance(item, dict))
    return v


def partial_spearman(a, b, c):
    m = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    if m.sum() < 8:
        return np.nan
    ra, rb, rc = rankdata(a[m]), rankdata(b[m]), rankdata(c[m])
    res = lambda y: y - np.c_[np.ones_like(rc), rc] @ np.linalg.lstsq(np.c_[np.ones_like(rc), rc], y, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def sp(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 8 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(spearmanr(a[m], b[m])[0])


def boot_ci(fn, n, rng):
    vals = [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    vals = [x for x in vals if not np.isnan(x)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [np.nan, np.nan]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "llm_eval"))
    args = ap.parse_args()
    base = os.path.join(args.out_dir, args.split)
    rng = np.random.default_rng(SEED)

    analyses = {}
    for f in glob.glob(os.path.join(base, "analysis", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        analyses[d["user_no"]] = d
    recs = [json.load(open(f, encoding="utf-8")) for f in glob.glob(os.path.join(base, "gemini", "*.json"))]

    # 사람 평균 라벨과 3등분 기준(train 사람 평균으로 정함)
    meta = pd.read_csv(os.path.join(ROOT, "features/meta.csv"), dtype={"id": str, "user_no": str})
    split = json.load(open(os.path.join(ROOT, "splits/split_seed42.json")))
    pm = meta.groupby("user_no")[LABELS].mean()
    train_users = meta[meta["id"].isin(split["train"])]["user_no"].unique()
    cuts = {lab: np.percentile(pm.loc[train_users, lab], [100 / 3, 200 / 3]) for lab in LABELS}

    # 응답별 위반과 등급
    rows = []
    for r in recs:
        a = analyses.get(r["user_no"])
        if a is None:
            continue
        resp = r.get("parsed")
        viol = check(resp, a, r["condition"]) if r.get("ok") else {}
        model_band = ((a["result"]["session"].get("delivery") or {}).get("scores", {}).get("overall_performance") or {}).get("band")
        rows.append({"user_no": r["user_no"], "cond": r["condition"], "rep": r["repeat"], "ok": bool(r.get("ok")),
                     "band": BAND.get((resp or {}).get("overall_band")) if r.get("ok") else None,
                     "content": [BAND.get(x.get("content_band")) for x in (resp or {}).get("answer_feedback") or []
                                 if isinstance(x, dict)] if r.get("ok") else None,
                     "model_band": BAND.get(model_band), **{f"v_{k}": bool(x) for k, x in viol.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("채점할 응답이 없음")
    vcols = sorted(c for c in df.columns if c.startswith("v_"))
    users = sorted(df["user_no"].unique(), key=int)
    conds = [c for c in "ABC" if c in set(df["cond"])]
    out = {"split": args.split, "n_persons": len(users), "n_responses": len(df), "conditions": {}}

    # 사람별 지표 (반복 평균)
    words = {u: np.mean([len(WORD_RE.findall(v.get("transcript") or "")) for v in analyses[u]["result"]["videos"]]) for u in users}
    loglen = np.array([np.log1p(words[u]) for u in users])
    lab = {k: pm.loc[users, k].values for k in LABELS}

    per_user_viol = {}
    for c in conds:
        d = df[df["cond"] == c]
        ok = d[d["ok"]]
        vio = ok[vcols].fillna(False)
        any_v = vio.any(axis=1)
        pu = ok.assign(any=any_v.values).groupby("user_no")["any"].mean().reindex(users).values
        per_user_viol[c] = pu
        band_mean = ok.groupby("user_no")["band"].mean().reindex(users).values.astype(float)
        mb = ok.groupby("user_no")["model_band"].first().reindex(users).values.astype(float)
        same_as_model = ok.dropna(subset=["band", "model_band"])
        res = {
            "responses": int(len(d)), "json_ok": float(d["ok"].mean()),
            "any_violation_rate": float(any_v.mean()) if len(vio) else np.nan,
            "violation_rates": {k[2:]: float(vio[k].mean()) for k in vcols},
            "band_distribution": {b: int((ok["band"] == v).sum()) for b, v in BAND.items()} | {"null": int(ok["band"].isna().sum())},
            "band_equals_model_band": float((same_as_model["band"] == same_as_model["model_band"]).mean()) if len(same_as_model) else np.nan,
            "repeat_identical_band": float(ok.groupby("user_no")["band"].nunique(dropna=False).eq(1).mean()),
            "repeat_identical_content": float(ok.groupby("user_no")["content"].apply(lambda s: len({json.dumps(x) for x in s}) == 1).mean()),
            "label_agreement": {}}
        for k in LABELS:
            y = lab[k]
            tri = np.digitize(y, cuts[k])  # 0=하, 1=중, 2=상
            m = ~np.isnan(band_mean)
            res["label_agreement"][k] = {
                "spearman": sp(band_mean, y),
                "spearman_ci": boot_ci(lambda i: sp(band_mean[i], y[i]), len(users), rng),
                "partial_spearman_length": partial_spearman(band_mean, y, loglen),
                "tertile_agreement": float(np.mean(np.round(band_mean[m]) == tri[m])),
                "baseline_length_only": sp(-loglen, y),
                "baseline_model_band": sp(mb, y)}
        out["conditions"][c] = res

    # 조건 차이 (사람 bootstrap): 위반률, 라벨 일치
    diffs = {}
    for a_, b_ in [("C", "A"), ("B", "A"), ("C", "B")]:
        if a_ in conds and b_ in conds:
            x, y = per_user_viol[a_], per_user_viol[b_]
            fn = lambda i: float(np.nanmean(x[i] - y[i]))  # noqa: E731
            diffs[f"{a_}-{b_}_violation"] = {"diff": fn(np.arange(len(users))), "ci": boot_ci(fn, len(users), rng)}
            ba = df[(df["cond"] == a_) & df["ok"]].groupby("user_no")["band"].mean().reindex(users).values.astype(float)
            bb = df[(df["cond"] == b_) & df["ok"]].groupby("user_no")["band"].mean().reindex(users).values.astype(float)
            yv = lab["overall_performance"]
            fn2 = lambda i: sp(ba[i], yv[i]) - sp(bb[i], yv[i])  # noqa: E731
            diffs[f"{a_}-{b_}_spearman_overall"] = {"diff": fn2(np.arange(len(users))), "ci": boot_ci(fn2, len(users), rng)}
    out["differences"] = diffs

    with open(os.path.join(base, "score.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    fmt = lambda v: "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"  # noqa: E731
    print(f"[{args.split}] 사람 {len(users)}명, 응답 {len(df)}개\n")
    print("## 규칙 위반률 (응답 기준)")
    print("항목".ljust(28) + "".join(c.rjust(8) for c in conds))
    for k in ["any_violation"] + [k[2:] for k in vcols]:
        vals = [out["conditions"][c]["any_violation_rate"] if k == "any_violation" else out["conditions"][c]["violation_rates"].get(k)
                for c in conds]
        print(k.ljust(28) + "".join(fmt(v).rjust(8) for v in vals))
    print("\n## 일관성·모델 등급")
    for k in ["json_ok", "repeat_identical_band", "repeat_identical_content", "band_equals_model_band"]:
        print(k.ljust(28) + "".join(fmt(out["conditions"][c][k]).rjust(8) for c in conds))
    print("band 분포".ljust(28) + "  ".join(f"{c}:{out['conditions'][c]['band_distribution']}" for c in conds))
    print("\n## 평가자 라벨 일치 (세션 등급 반복 평균 vs 사람 평균 라벨)")
    for k in LABELS:
        print(f"[{k}]")
        for c in conds:
            la = out["conditions"][c]["label_agreement"][k]
            print(f"  {c}: Spearman {fmt(la['spearman'])} [{fmt(la['spearman_ci'][0])}, {fmt(la['spearman_ci'][1])}]"
                  f"  길이 보정 {fmt(la['partial_spearman_length'])}  3등분 일치 {fmt(la['tertile_agreement'])}")
        la = out["conditions"][conds[0]]["label_agreement"][k]
        print(f"  기준선: 길이만 {fmt(la['baseline_length_only'])}, 모델 등급 {fmt(la['baseline_model_band'])}")
    print("\n## 조건 차이 (사람 bootstrap 95% 구간)")
    for k, d in diffs.items():
        print(f"  {k:28s} {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    print(f"\n저장: {os.path.join(base, 'score.json')}")


if __name__ == "__main__":
    main()
