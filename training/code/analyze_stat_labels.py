"""
LLM 프롬프트용 측정값 해석 기준: RecruitView에서 각 측정값과 평가자 라벨의 관계를 본다.

기준은 train에서만 정하고 val로 방향이 유지되는지 확인한다. test는 쓰지 않는다.
라벨은 답변 길이와 반비례하므로(짧을수록 높음), 길이 영향을 뺀 사람 단위 편상관을 주 지표로 쓴다.

  answer        답변 단위 Spearman (길이 미보정, 참고용)
  person        사람 평균끼리 Spearman
  person_ctrl   사람 평균끼리, 사람 평균 단어 수(log)를 뺀 편 Spearman  <- 주 지표
  within_ctrl   같은 사람 안 (사람 평균을 빼고) 단어 수를 뺀 편 Spearman

판정 (미리 정한 기준, 결과를 보고 바꾸지 않는다):
  경향 있음   train person_ctrl의 사람 bootstrap 95% 구간이 0을 포함하지 않고 |r| >= 0.15, val에서 같은 부호
  관계 약함   그 외
  판단 불가   데이터셋에서 값이 거의 변하지 않음 (표준편차 0 또는 고유값 3개 이하)

python code/analyze_stat_labels.py            -> features/stat_label_relations.json, 표 출력
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from config import INTERVIEW_LABEL_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = {  # extract_stats.STAT_LABELS와 같은 항목
    "audio": ["speaking_rate_wpm", "silence_ratio", "pitch_mean", "rms_mean", "rms_std", "duration_sec"],
    "text": ["answer_length", "filler_count", "vocab_diversity"],
    "video": ["gaze_away_ratio", "gaze_x_std", "head_motion", "head_yaw_mean_abs", "head_roll_std",
              "blink_rate_per_min", "mouth_width_std", "face_detected_ratio"],
}
LENGTH = "text.answer_length"
MIN_ABS_R, N_BOOT, SEED = 0.15, 1000, 0


def load(split_ids):
    meta = pd.read_csv(os.path.join(ROOT, "features/meta.csv"), dtype={"id": str})
    meta = meta[meta["id"].isin(split_ids)]
    rows = []
    for i in meta["id"]:
        p = os.path.join(ROOT, f"features/stats/{i}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        row = {"id": i}
        for g, keys in STATS.items():
            for k in keys:
                v = d.get(g, {}).get(k)
                row[f"{g}.{k}"] = float(v) if v is not None else np.nan
        rows.append(row)
    return meta.merge(pd.DataFrame(rows), on="id")


def _resid(y, x):
    """순위끼리 선형 회귀 잔차 (편 Spearman용)."""
    A = np.c_[np.ones_like(x), x]
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def partial_spearman(a, b, c):
    m = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    if m.sum() < 10:
        return np.nan
    ra, rb, rc = rankdata(a[m]), rankdata(b[m]), rankdata(c[m])
    return float(np.corrcoef(_resid(ra, rc), _resid(rb, rc))[0, 1])


def sp(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(spearmanr(a[m], b[m])[0]) if m.sum() >= 10 else np.nan


def person_table(df, cols):
    out = df.groupby("user_no")[list(dict.fromkeys(cols))].mean()  # stat이 LENGTH 자체일 때 열이 겹치지 않게
    out["log_len"] = np.log1p(df.groupby("user_no")[LENGTH].mean())
    return out


def relations(df, stat, label, rng=None):
    persons = person_table(df, [stat, label, LENGTH])
    x, y, c = persons[stat].values, persons[label].values, persons["log_len"].values
    res = {"answer": sp(df[stat].values, df[label].values), "person": sp(x, y)}
    is_length = stat in (LENGTH, "audio.duration_sec")
    res["person_ctrl"] = np.nan if is_length else partial_spearman(x, y, c)
    # 같은 사람 안: 사람 평균을 뺀 값끼리, 단어 수(log, 사람 평균 뺀 것)를 통제
    g = df.groupby("user_no")
    dm = lambda s: (s - g[s.name].transform("mean")).values  # noqa: E731
    loglen = np.log1p(df[LENGTH])
    loglen_dm = (loglen - loglen.groupby(df["user_no"]).transform("mean")).values
    res["within_ctrl"] = np.nan if is_length else partial_spearman(dm(df[stat]), dm(df[label]), loglen_dm)
    if rng is not None and not is_length:
        n, boots = len(persons), []
        for _ in range(N_BOOT):
            k = rng.integers(0, n, n)
            boots.append(partial_spearman(x[k], y[k], c[k]))
        res["person_ctrl_ci"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    return res


def degenerate(series):
    s = series.dropna()
    return s.std() == 0 or s.nunique() <= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "features/stat_label_relations.json"))
    args = ap.parse_args()

    split = json.load(open(os.path.join(ROOT, "splits/split_seed42.json")))
    train, val = load(split["train"]), load(split["val"])
    print(f"train {len(train)}개 답변 / {train['user_no'].nunique()}명, val {len(val)}개 / {val['user_no'].nunique()}명 (test 미사용)")
    rng = np.random.default_rng(SEED)

    stat_cols = [f"{g}.{k}" for g, ks in STATS.items() for k in ks]
    out = {"criteria": {"main": "person_ctrl", "min_abs_r": MIN_ABS_R, "n_boot": N_BOOT,
                        "rule": "train 95% CI excludes 0, |r|>=min_abs_r, same sign on val"},
           "n": {"train_answers": len(train), "train_persons": int(train["user_no"].nunique()),
                 "val_answers": len(val), "val_persons": int(val["user_no"].nunique())},
           "stats": {}}
    for stat in stat_cols:
        entry = {"label_relations": {}}
        if degenerate(train[stat]):
            entry["verdict"] = "판단 불가"
            entry["reason"] = f"값이 거의 변하지 않음 (고유값 {train[stat].nunique()}개)"
            out["stats"][stat] = entry
            continue
        p = person_table(train, [stat, LENGTH])
        entry["length_corr_person"] = sp(p[stat].values, p["log_len"].values)
        for label in INTERVIEW_LABEL_NAMES:
            tr = relations(train, stat, label, rng)
            va = relations(val, stat, label)
            r, ci = tr["person_ctrl"], tr.get("person_ctrl_ci")
            if ci is None:
                verdict = "길이 자체"
            elif (ci[0] > 0 or ci[1] < 0) and abs(r) >= MIN_ABS_R and np.sign(va["person_ctrl"]) == np.sign(r):
                verdict = "높을수록 높게 평가" if r > 0 else "높을수록 낮게 평가"
            else:
                verdict = "관계 약함"
            entry["label_relations"][label] = {"train": tr, "val": va, "verdict": verdict}
        out["stats"][stat] = entry

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"저장: {args.out}\n")

    fmt = lambda v: "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2f}"  # noqa: E731
    for label in INTERVIEW_LABEL_NAMES:
        print(f"## {label}  (train person_ctrl [95% CI] / val person_ctrl / train within_ctrl / 판정)")
        for stat, e in out["stats"].items():
            if "verdict" in e:
                print(f"  {stat:28s} {e['verdict']} ({e['reason']})")
                continue
            rel = e["label_relations"][label]
            ci = rel["train"].get("person_ctrl_ci")
            ci_s = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "        -       "
            print(f"  {stat:28s} {fmt(rel['train']['person_ctrl'])} {ci_s}  {fmt(rel['val']['person_ctrl'])}"
                  f"  {fmt(rel['train']['within_ctrl'])}  {rel['verdict']}"
                  f"   (보정 전 사람 단위 {fmt(rel['train']['person'])}, 답변 단위 {fmt(rel['train']['answer'])})")
        print()


if __name__ == "__main__":
    main()
