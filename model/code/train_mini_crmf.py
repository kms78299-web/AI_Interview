"""
RECRUITVIEW 원논문(CRMF, arXiv:2512.00450)을 geometry/manifold/router 없이
단순화한 경량 fusion MLP — "mini CRMF".

입력: features/embed/{encoder}/{id}.npy (Track 1 임베딩) + features/meta.csv
(build_meta.py로 생성, id/user_no/라벨 12개). 선택한 --encoders 전부가 있는
id만 사용한다.

사용 예:
    python train_mini_crmf.py --encoders wav2vec2-base bert-base-uncased --fusion prefusion
    python train_mini_crmf.py --encoders wav2vec2-base bert-base-uncased --fusion concat
    python train_mini_crmf.py --encoders wav2vec2-base bert-base-uncased videomae --smoke_test

금지(요청받은 범위 밖): manifold/router 구현, encoder fine-tuning, 결과 숫자 조정.
"""

import argparse
import copy
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_FEATURES_DIR,
    DEFAULT_META_PATH,
    DEFAULT_SPLITS_DIR,
    ENCODER_TO_MODALITY,
    INTERVIEW_LABEL_NAMES,
    SPLIT_SEED,
)

# [text, audio, video] 고정 순서 — --encoders를 어떤 순서로 줘도 이 순서로 토큰을 쌓는다.
MODALITY_ORDER = ["text", "audio", "video"]


# ─────────────────────────────────────────────
# 0. 공통 유틸
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_winsorize(x, threshold=1.5):
    """|x|<=threshold면 그대로, 아니면 sign(x)*(threshold + threshold*tanh((|x|-threshold)/threshold)).
    x: np.ndarray. 순위(rank)를 보존하는 단조 압축이라 Spearman에는 영향 없음."""
    abs_x = np.abs(x)
    squashed = threshold + threshold * np.tanh((abs_x - threshold) / threshold)
    return np.where(abs_x <= threshold, x, np.sign(x) * squashed)


def batch_pearson_loss(pred, target):
    """--corr_loss용: 배치 안에서 타깃별 Pearson 상관을 구해 1 - mean(|r|)를 반환 (미분 가능)."""
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    num = (pred_c * target_c).sum(dim=0)
    denom = torch.sqrt((pred_c**2).sum(dim=0) * (target_c**2).sum(dim=0) + 1e-8)
    r = num / denom
    return 1.0 - r.abs().mean()


def assert_finite(arr, what):
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{what}에 NaN/inf 있음 (첫 번째 위치: {bad[0].tolist()})")


# ─────────────────────────────────────────────
# 1. user_no 기준 split (70/15/15, 전 실험 공유)
# ─────────────────────────────────────────────
def build_or_load_split(meta_df, splits_dir, seed=SPLIT_SEED):
    split_path = os.path.join(splits_dir, f"split_seed{seed}.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        print(f"[INFO] 기존 split 재사용: {split_path} "
              f"(train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])})")
        return split

    meta_df = meta_df[meta_df["user_no"].astype(str).str.len() > 0]
    user_nos = sorted(meta_df["user_no"].astype(str).unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(user_nos)
    n = len(user_nos)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_users = set(user_nos[:n_train])
    val_users = set(user_nos[n_train:n_train + n_val])
    test_users = set(user_nos[n_train + n_val:])

    id_to_user = dict(zip(meta_df["id"].astype(str), meta_df["user_no"].astype(str)))
    split = {
        "seed": seed,
        "train": sorted(i for i, u in id_to_user.items() if u in train_users),
        "val": sorted(i for i, u in id_to_user.items() if u in val_users),
        "test": sorted(i for i, u in id_to_user.items() if u in test_users),
    }
    os.makedirs(splits_dir, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[INFO] 새 split 생성: {split_path} "
          f"(user_no train={len(train_users)}, val={len(val_users)}, test={len(test_users)})")
    return split


def check_no_user_overlap(meta_df, split):
    id_to_user = dict(zip(meta_df["id"].astype(str), meta_df["user_no"].astype(str)))
    train_u = {id_to_user[i] for i in split["train"] if i in id_to_user}
    val_u = {id_to_user[i] for i in split["val"] if i in id_to_user}
    test_u = {id_to_user[i] for i in split["test"] if i in id_to_user}
    assert not (train_u & val_u), f"train/val에 겹치는 user_no: {train_u & val_u}"
    assert not (train_u & test_u), f"train/test에 겹치는 user_no: {train_u & test_u}"
    assert not (val_u & test_u), f"val/test에 겹치는 user_no: {val_u & test_u}"


# ─────────────────────────────────────────────
# 2. 데이터 로딩
# ─────────────────────────────────────────────
def resolve_modality_dirs(encoders, features_dir):
    modality_dirs = {}
    for enc in encoders:
        modality = ENCODER_TO_MODALITY.get(enc)
        if modality is None:
            raise ValueError(
                f"알 수 없는 encoder: '{enc}'. config.py의 AUDIO/TEXT/VIDEO_EMBED_ENCODERS에 등록된 "
                f"이름만 가능: {sorted(ENCODER_TO_MODALITY.keys())}"
            )
        if modality in modality_dirs:
            raise ValueError(f"modality={modality}에 encoder가 2개 이상 지정됨 (--encoders 확인)")
        modality_dirs[modality] = os.path.join(features_dir, "embed", enc)
    # [text, audio, video] 고정 순서로 재정렬
    return {m: modality_dirs[m] for m in MODALITY_ORDER if m in modality_dirs}


def find_available_ids(meta_df, modality_dirs):
    ids = []
    for id_ in meta_df["id"].astype(str):
        if all(os.path.exists(os.path.join(d, f"{id_}.npy")) for d in modality_dirs.values()):
            ids.append(id_)
    return ids


def load_embeddings(ids, modality_dirs):
    """modality -> (N, dim) float32 배열. NaN/inf 있으면 id와 함께 에러."""
    data = {}
    for modality, d in modality_dirs.items():
        vecs = []
        for id_ in ids:
            v = np.load(os.path.join(d, f"{id_}.npy")).astype(np.float32)
            if not np.all(np.isfinite(v)):
                raise ValueError(f"NaN/inf 발견: modality={modality}, id={id_}, path={d}/{id_}.npy")
            vecs.append(v)
        data[modality] = np.stack(vecs)
    return data


def load_labels(meta_df, ids):
    sub = meta_df.set_index(meta_df["id"].astype(str)).loc[ids]
    y = sub[INTERVIEW_LABEL_NAMES].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(y)):
        bad_rows = np.argwhere(~np.all(np.isfinite(y), axis=1)).ravel()
        bad_ids = [ids[i] for i in bad_rows]
        raise ValueError(f"라벨에 NaN/inf: {bad_ids}")
    return y


# ─────────────────────────────────────────────
# 3. 모델
# ─────────────────────────────────────────────
class ModalityProjector(nn.Module):
    def __init__(self, in_dim, d_model, dropout):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(self.linear(x)))


class PrefusionBlock(nn.Module):
    """토큰별 modality embedding + TransformerEncoder + learned attention pooling."""

    def __init__(self, num_modalities, d_model, n_layers=1, n_heads=4, dropout=0.1):
        super().__init__()
        self.modality_embed = nn.Embedding(num_modalities, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool_proj = nn.Linear(d_model, 1)

    def forward(self, tokens):  # (B, M, D)
        B, M, _ = tokens.shape
        modality_ids = torch.arange(M, device=tokens.device).unsqueeze(0).expand(B, M)
        tokens = tokens + self.modality_embed(modality_ids)
        h = self.transformer(tokens)  # (B, M, D)
        scores = self.pool_proj(h).squeeze(-1)  # (B, M)
        alpha = torch.softmax(scores, dim=1)  # (B, M)
        return (alpha.unsqueeze(-1) * h).sum(dim=1)  # (B, D)


class ConcatFusionBlock(nn.Module):
    """논문 ablation의 simple concatenation 비교용."""

    def __init__(self, num_modalities, d_model):
        super().__init__()
        self.proj = nn.Linear(num_modalities * d_model, d_model)

    def forward(self, tokens):  # (B, M, D)
        B, M, D = tokens.shape
        return self.proj(tokens.reshape(B, M * D))


class EuclideanExpert(nn.Module):
    """Linear+ReLU 2층 + residual (x_out = x_L + x_0)."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_model)
        self.l2 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0):
        h = self.dropout(self.act(self.l1(x0)))
        h = self.dropout(self.act(self.l2(h)))
        return h + x0


class MultiTaskHead(nn.Module):
    def __init__(self, d_model, target_names, hidden=256, adapter_dim=64, dropout=0.1):
        super().__init__()
        self.target_names = list(target_names)
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(hidden, adapter_dim), nn.GELU(), nn.Linear(adapter_dim, 1))
            for name in self.target_names
        })

    def forward(self, z):
        h = self.shared(z)
        return torch.cat([self.adapters[name](h) for name in self.target_names], dim=1)


class MiniCRMF(nn.Module):
    def __init__(self, modality_dims, d_model, fusion, target_names,
                 dropout=0.1, n_transformer_layers=1, n_heads=4):
        super().__init__()
        self.modalities = [m for m in MODALITY_ORDER if m in modality_dims]
        self.projectors = nn.ModuleDict({
            m: ModalityProjector(modality_dims[m], d_model, dropout) for m in self.modalities
        })
        if fusion == "prefusion":
            self.fusion = PrefusionBlock(len(self.modalities), d_model, n_transformer_layers, n_heads, dropout)
        elif fusion == "concat":
            self.fusion = ConcatFusionBlock(len(self.modalities), d_model)
        else:
            raise ValueError(f"알 수 없는 --fusion: {fusion}")
        self.expert = EuclideanExpert(d_model, dropout)
        self.head = MultiTaskHead(d_model, target_names, dropout=dropout)

    def forward(self, batch):  # batch: {modality: (B, in_dim)}
        tokens = torch.stack([self.projectors[m](batch[m]) for m in self.modalities], dim=1)
        z = self.expert(self.fusion(tokens))
        return self.head(z)


# ─────────────────────────────────────────────
# 4. Dataset
# ─────────────────────────────────────────────
class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = {m: torch.from_numpy(v) for m, v in embeddings.items()}
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        x = {m: v[idx] for m, v in self.embeddings.items()}
        return x, self.targets[idx]


# ─────────────────────────────────────────────
# 5. 평가
# ─────────────────────────────────────────────
def evaluate(model, loader, device, target_names, original_labels):
    """original_labels: loader와 같은 순서로 정렬된 (N, T) 원본(비가공) 라벨.
    Spearman은 단조 변환(winsorization)에 불변이라 모델 출력(윈저라이즈 공간)을
    원본 라벨과 바로 비교해도 된다 (스펙: "평가는 원본 라벨과 비교")."""
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = {k: v.to(device) for k, v in xb.items()}
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    metrics, spearmans, pearsons = {}, [], []
    for i, name in enumerate(target_names):
        sp = spearmanr(preds[:, i], original_labels[:, i]).statistic
        pr = pearsonr(preds[:, i], original_labels[:, i]).statistic
        metrics[f"spearman_{name}"] = float(sp)
        metrics[f"pearson_{name}"] = float(pr)
        spearmans.append(sp)
        pearsons.append(pr)
    metrics["spearman_macro"] = float(np.mean(spearmans))
    metrics["pearson_macro"] = float(np.mean(pearsons))
    return metrics, preds


# ─────────────────────────────────────────────
# 6. 학습 (seed 1회)
# ─────────────────────────────────────────────
def train_one_seed(seed, args, modality_dims, train_ds, val_loader, val_labels_raw, device):
    set_seed(seed)
    model = MiniCRMF(
        modality_dims, args.d_model, args.fusion, INTERVIEW_LABEL_NAMES,
        dropout=args.dropout, n_transformer_layers=args.transformer_layers, n_heads=args.n_heads,
    ).to(device)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size
    )
    steps_per_epoch = max(1, len(train_loader))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps_per_epoch * args.max_epochs, pct_start=0.15,
    )
    huber = nn.HuberLoss(delta=1.0)

    best_val_spearman = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            pred = model(xb)
            loss = huber(pred, yb)
            if args.corr_loss:
                loss = loss + args.corr_lambda * batch_pearson_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * yb.shape[0]
        epoch_loss /= len(train_ds)

        val_metrics, _ = evaluate(model, val_loader, device, INTERVIEW_LABEL_NAMES, val_labels_raw)
        print(f"  [seed={seed}] epoch={epoch:02d} train_loss={epoch_loss:.4f} "
              f"val_spearman_macro={val_metrics['spearman_macro']:.4f}")

        if val_metrics["spearman_macro"] > best_val_spearman:
            best_val_spearman = val_metrics["spearman_macro"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  [seed={seed}] early stopping (epoch {epoch}, patience={args.patience})")
                break

    model.load_state_dict(best_state)
    return model, best_val_spearman


# ─────────────────────────────────────────────
# 7. main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="mini CRMF — 경량 fusion MLP 학습")
    p.add_argument("--encoders", nargs="+", required=True,
                    help="예: wav2vec2-base bert-base-uncased videomae (config.py 레지스트리 이름)")
    p.add_argument("--fusion", choices=["prefusion", "concat"], default="prefusion")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--transformer_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--corr_loss", action="store_true")
    p.add_argument("--corr_lambda", type=float, default=0.1)
    p.add_argument("--winsor_threshold", type=float, default=1.5)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR)
    p.add_argument("--meta_path", default=DEFAULT_META_PATH)
    p.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    p.add_argument("--checkpoints_dir", default=DEFAULT_CHECKPOINTS_DIR)
    p.add_argument("--exp_name", default=None, help="기본값: {fusion}_{encoder1-encoder2-...}")
    p.add_argument("--smoke_test", action="store_true",
                    help="샘플 20개(또는 그보다 적으면 가용한 만큼)로 1 seed, 적은 epoch만 돌려서 loss 감소만 확인")
    return p.parse_args()


def main():
    args = parse_args()
    if args.exp_name is None:
        args.exp_name = f"{args.fusion}_{'-'.join(args.encoders)}"
    if args.smoke_test:
        args.seeds = args.seeds[:1]
        args.max_epochs = min(args.max_epochs, 3)
        print("[INFO] --smoke_test: seed 1개, max_epochs<=3, 샘플 수 제한해서 loss 감소만 확인")

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("[WARN] CUDA 사용 불가 -> CPU로 진행 (드라이버/torch 빌드 버전 확인 필요할 수 있음)")
    print(f"[INFO] device={device}  encoders={args.encoders}  fusion={args.fusion}  exp_name={args.exp_name}")

    meta_df = pd.read_csv(args.meta_path, dtype={"id": str, "user_no": str})
    modality_dirs = resolve_modality_dirs(args.encoders, args.features_dir)
    print(f"[INFO] modality 순서: {list(modality_dirs.keys())}")

    available_ids = find_available_ids(meta_df, modality_dirs)
    print(f"[INFO] 선택한 encoder가 전부 있는 샘플 수: {len(available_ids)} / meta.csv 전체 {len(meta_df)}")
    if len(available_ids) == 0:
        raise RuntimeError("선택한 --encoders 조합으로 임베딩이 존재하는 샘플이 하나도 없음. "
                            "extract_features.py --track embed 를 먼저 돌렸는지 확인할 것.")

    split = build_or_load_split(meta_df, args.splits_dir, seed=SPLIT_SEED)
    check_no_user_overlap(meta_df, split)
    available_set = set(available_ids)
    train_ids = [i for i in split["train"] if i in available_set]
    val_ids = [i for i in split["val"] if i in available_set]
    test_ids = [i for i in split["test"] if i in available_set]

    if args.smoke_test:
        # available_ids를 id 순서로 그냥 앞에서 20개 자르면(0001~0020 식) 그 구간에
        # val/test 샘플이 우연히 하나도 없어서 0개가 될 수 있음(실제로 겪은 문제) —
        # train/val/test로 나눈 "후"에 각각 따로 잘라야 세 split 모두 비지 않음.
        train_ids = train_ids[: min(14, len(train_ids))]
        val_ids = val_ids[: min(3, len(val_ids))]
        test_ids = test_ids[: min(3, len(test_ids))]

    print(f"[INFO] split 적용 후 샘플 수: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    if min(len(train_ids), len(val_ids), len(test_ids)) == 0:
        raise RuntimeError("train/val/test 중 하나가 0개임 — 샘플이 너무 적거나 "
                            "(smoke_test 포함) split이 쏠려서 생긴 문제일 수 있음.")

    train_emb = load_embeddings(train_ids, modality_dirs)
    val_emb = load_embeddings(val_ids, modality_dirs)
    test_emb = load_embeddings(test_ids, modality_dirs)
    train_y_raw = load_labels(meta_df, train_ids)
    val_y_raw = load_labels(meta_df, val_ids)
    test_y_raw = load_labels(meta_df, test_ids)

    # 입력 표준화 (train 통계로 fit, scaler 저장)
    scalers = {}
    for m in modality_dirs:
        scaler = StandardScaler().fit(train_emb[m])
        scalers[m] = scaler
        train_emb[m] = scaler.transform(train_emb[m]).astype(np.float32)
        val_emb[m] = scaler.transform(val_emb[m]).astype(np.float32)
        test_emb[m] = scaler.transform(test_emb[m]).astype(np.float32)
        assert_finite(train_emb[m], f"표준화 후 train[{m}]")

    # 라벨 soft winsorization (학습 타깃 전용, train std 기준 sigma 환산)
    label_mean = train_y_raw.mean(axis=0)
    label_std = train_y_raw.std(axis=0)
    label_std_safe = np.where(label_std < 1e-6, 1.0, label_std)
    train_y_sigma = (train_y_raw - label_mean) / label_std_safe
    train_y_target = soft_winsorize(train_y_sigma, threshold=args.winsor_threshold).astype(np.float32)
    assert_finite(train_y_target, "winsorize된 train 라벨")

    modality_dims = {m: train_emb[m].shape[1] for m in modality_dirs}
    print(f"[INFO] modality별 입력 차원: {modality_dims}")

    train_ds = EmbedDataset(train_emb, train_y_target)
    val_ds = EmbedDataset(val_emb, np.zeros_like(val_y_raw, dtype=np.float32))  # eval은 raw로 비교하므로 target 미사용
    test_ds = EmbedDataset(test_emb, np.zeros_like(test_y_raw, dtype=np.float32))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    seed_results = []
    best_overall = {"val_spearman": -float("inf"), "model": None, "seed": None}
    for seed in args.seeds:
        print(f"[INFO] === seed={seed} 학습 시작 ===")
        model, best_val_spearman = train_one_seed(
            seed, args, modality_dims, train_ds, val_loader, val_y_raw, device
        )
        test_metrics, _ = evaluate(model, test_loader, device, INTERVIEW_LABEL_NAMES, test_y_raw)
        val_metrics, _ = evaluate(model, val_loader, device, INTERVIEW_LABEL_NAMES, val_y_raw)
        print(f"[INFO] seed={seed} 최종 test_spearman_macro={test_metrics['spearman_macro']:.4f} "
              f"test_pearson_macro={test_metrics['pearson_macro']:.4f}")
        seed_results.append({"seed": seed, "val": val_metrics, "test": test_metrics})
        if best_val_spearman > best_overall["val_spearman"]:
            best_overall = {"val_spearman": best_val_spearman, "model": model, "seed": seed}

    # 3 seed 평균 ± 표준편차 리포트
    def agg(key_prefix, split_name):
        keys = [f"{key_prefix}_{name}" for name in INTERVIEW_LABEL_NAMES] + [f"{key_prefix}_macro"]
        out = {}
        for k in keys:
            vals = [r[split_name][k] for r in seed_results]
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return out

    report = {
        "exp_name": args.exp_name,
        "encoders": args.encoders,
        "fusion": args.fusion,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "seeds": args.seeds,
        "per_seed": seed_results,
        "val_spearman_agg": agg("spearman", "val"),
        "val_pearson_agg": agg("pearson", "val"),
        "test_spearman_agg": agg("spearman", "test"),
        "test_pearson_agg": agg("pearson", "test"),
        "best_seed": best_overall["seed"],
    }

    print("=" * 70)
    print(f"[RESULT] {args.exp_name} — test spearman_macro = "
          f"{report['test_spearman_agg']['spearman_macro']['mean']:.4f} "
          f"± {report['test_spearman_agg']['spearman_macro']['std']:.4f} "
          f"({len(args.seeds)} seeds)")
    print("=" * 70)

    out_dir = os.path.join(args.checkpoints_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_overall["model"].state_dict(), os.path.join(out_dir, "mlp_best.pt"))
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    with open(os.path.join(out_dir, "label_stats.json"), "w") as f:
        json.dump({
            "mean": dict(zip(INTERVIEW_LABEL_NAMES, label_mean.tolist())),
            "std": dict(zip(INTERVIEW_LABEL_NAMES, label_std.tolist())),
            "winsor_threshold": args.winsor_threshold,
        }, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "encoders": args.encoders, "modality_dims": modality_dims, "fusion": args.fusion,
            "d_model": args.d_model, "transformer_layers": args.transformer_layers,
            "n_heads": args.n_heads, "dropout": args.dropout, "lr": args.lr,
            "weight_decay": args.weight_decay, "batch_size": args.batch_size,
            "max_epochs": args.max_epochs, "patience": args.patience, "seeds": args.seeds,
            "corr_loss": args.corr_loss, "corr_lambda": args.corr_lambda,
            "winsor_threshold": args.winsor_threshold, "best_seed": best_overall["seed"],
            # 논문에 명시 안 돼서 우리가 고른 하이퍼파라미터들 (보고에 포함하라고 한 부분)
            "_not_from_paper": ["d_model", "transformer_layers", "n_heads", "dropout", "lr",
                                 "weight_decay", "adapter_dim=64(고정)", "hidden=256(고정)"],
        }, f, indent=2)
    with open(os.path.join(out_dir, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] 저장 완료: {out_dir}")


if __name__ == "__main__":
    main()
