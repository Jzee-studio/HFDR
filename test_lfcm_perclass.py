#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LFCM vs AT 逐类别 OOD 诊断（per-class corruption accuracy analysis）
====================================================================
对每个模型（AT / HFDR / LFCM 多个 K）× 每种腐蚀 × 每个类别计算精度，
构造 Δ = acc(LFCM) − acc(AT) 的 (腐蚀 × 类别) 差值矩阵，并回答：

    LFCM 相对 AT 的收益和损失，是随类别有结构的（某些类别稳定受益、
    某些类别稳定受损），还是无结构的噪声（腐蚀×类别交互的随机涨落）？

检验方法（详见报告 "方法" 一节）：
  1. 方差分解：把 Δ 矩阵的总方差分解为 腐蚀主效应 / 类别主效应 / 残差
     （腐蚀×类别交互），类别主效应占比 η²_class 越大 → 结构越强。
  2. 置换检验：在每行（腐蚀）内打乱类别标签，得到"无类别结构"零分布，
     比较观测 η²_class 与零分布 → p 值。
  3. 跨腐蚀一致性：不同腐蚀的逐类差值向量两两相关系数均值 ρ̄，
     同样与置换零分布比较。结构真实 → 各类别在腐蚀间保持一致涨落。

用法：
    python test_lfcm_perclass.py                          # 全量评测 + 分析 + 出图
    python test_lfcm_perclass.py --dataset CIFAR10        # 只跑 CIFAR-10
    python test_lfcm_perclass.py --models AT,LFCM_K64     # 只选部分模型（逗号分隔）
    python test_lfcm_perclass.py --analyze-only           # 只用缓存重算分析/重画图
    python test_lfcm_perclass.py --no-cache               # 忽略缓存强制重测

说明：
  - 评测协议与 test_ood.py 完全一致（模型内 Norm 约定、无归一化数据管线、
    LFCM 构建时只传 codebook_size/code_dim/hidden_dim），逐类数字与既有
    *_ood_test.log 总体精度可直接对齐。
  - 每个模型×腐蚀的逐类精度缓存在 --cache-dir（.npz），重复运行免费。
  - checkpoint 位置：{checkpoint-root}/{Dataset}/{Prefix}/model_best.pth.tar
  - 全量跑一遍约 2-3 小时（19 腐蚀 × 11 模型 × 2 数据集，受 GPU 影响）。
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models import build_model
from utils import CIFAR10C, CIFAR100C

# ---------------------------------------------------------------------------
# 常量与注册表（沿用 test_ood.py 的约定）
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "/data/xujiazhao"

DATASET_CONFIG = {
    "CIFAR10": {
        "cls": CIFAR10C,
        "data_subdir": "CIFAR-10-C",
        "num_class": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2471, 0.2435, 0.2616),
    },
    "CIFAR100": {
        "cls": CIFAR100C,
        "data_subdir": "CIFAR-100-C",
        "num_class": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}

# CIFAR-10 类别名
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# CIFAR-100 fine 类别名（标准顺序，共 100 个）
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

def get_class_names(dataset_name):
    if dataset_name == "CIFAR10":
        return CIFAR10_CLASSES
    return CIFAR100_CLASSES

# CIFAR-C 官方腐蚀分组（noise/blur/weather/digital）
CORRUPTION_GROUPS = {
    "gaussian_noise": "noise", "shot_noise": "noise",
    "impulse_noise": "noise", "speckle_noise": "noise",
    "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur",
    "zoom_blur": "blur", "gaussian_blur": "blur",
    "snow": "weather", "frost": "weather", "fog": "weather",
    "spatter": "weather", "brightness": "weather",
    "contrast": "digital", "elastic_transform": "digital",
    "pixelate": "digital", "jpeg_compression": "digital", "saturate": "digital",
}
GROUP_ORDER = ["noise", "blur", "weather", "digital"]
GROUP_PALETTE = {"noise": "#8e44ad", "blur": "#2980b9",
                 "weather": "#16a085", "digital": "#e67e22"}

# 可用模型清单（与 result/ 下 *_ood_test.log 的 checkpoint 前缀一致）
# lfcm 键为空 → 非 LFCM 模型；有 → 构建时传给 build_model 的 lfcm_cfg
MODEL_MANIFEST = {
    "CIFAR10": [
        {"name": "AT",        "method": "AT",   "prefix": "WRN34_10_F",          "lfcm": None},
        {"name": "HFDR",      "method": "HFDR", "prefix": "WRN34_10_F_HFDR",     "lfcm": None},
        {"name": "LFCM_K32",  "method": "LFCM", "prefix": "WRN34_10_LFCM_K32",   "lfcm": {"codebook_size": 32, "code_dim": 32, "hidden_dim": 64}},
        {"name": "LFCM_K64",  "method": "LFCM", "prefix": "WRN34_10_LFCM",       "lfcm": {"codebook_size": 64, "code_dim": 32, "hidden_dim": 64}},
        {"name": "LFCM_K128", "method": "LFCM", "prefix": "WRN34_10_LFCM_K128",  "lfcm": {"codebook_size": 128, "code_dim": 32, "hidden_dim": 64}},
    ],
    "CIFAR100": [
        {"name": "AT",        "method": "AT",   "prefix": "WRN34_100_AT",        "lfcm": None},
        {"name": "HFDR",      "method": "HFDR", "prefix": "WRN34_100_HFDR",      "lfcm": None},
        {"name": "LFCM_K64",  "method": "LFCM", "prefix": "WRN34_100_LFCM",      "lfcm": {"codebook_size": 64, "code_dim": 32, "hidden_dim": 64}},
        {"name": "LFCM_K128", "method": "LFCM", "prefix": "WRN34_100_LFCM_K128", "lfcm": {"codebook_size": 128, "code_dim": 32, "hidden_dim": 64}},
        {"name": "LFCM_K256", "method": "LFCM", "prefix": "WRN34_100_LFCM_K256", "lfcm": {"codebook_size": 256, "code_dim": 32, "hidden_dim": 64}},
        {"name": "LFCM_K512", "method": "LFCM", "prefix": "WRN34_100_LFCM_K512", "lfcm": {"codebook_size": 512, "code_dim": 32, "hidden_dim": 64}},
    ],
}

DEFAULT_REFERENCE = "AT"


# ---------------------------------------------------------------------------
# 数据与模型构建（复刻 test_ood.py 的评测协议）
# ---------------------------------------------------------------------------
def get_corruptions(dataset_name, data_root):
    """发现数据目录下可用的腐蚀 .npy 文件（排序与 test_ood.py 一致）。"""
    subdir = DATASET_CONFIG[dataset_name]["data_subdir"]
    root = os.path.join(data_root, subdir)
    if os.path.isdir(root):
        names = []
        for fname in sorted(os.listdir(root)):
            if fname.endswith(".npy") and fname != "labels.npy":
                names.append(os.path.splitext(fname)[0])
        return names
    return []


def build_corruption_loader(dataset_name, corruption, data_root, batch_size, workers):
    cfg = DATASET_CONFIG[dataset_name]
    # 与 test_ood.py 一致：AT/HFDR/LFCM 的 data_norm=False，仅 ToTensor
    transform = transforms.Compose([transforms.ToTensor()])
    root = os.path.join(data_root, cfg["data_subdir"])
    dataset = cfg["cls"](root=root, name=corruption, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def build_and_load_model(dataset_name, entry, checkpoint_root, checkpoint_name):
    """按 test_ood.py 的协议构建模型并加载 checkpoint。"""
    cfg = DATASET_CONFIG[dataset_name]
    num_class = cfg["num_class"]

    # 只透传架构键：与 test_ood.py 一致，tau 保持模块默认 1.0
    # 注意丢弃 None 值：build_model 里 .get(key, default) 只在键缺失时才用默认值，
    # 键存在但值为 None 会把 None 传给 nn.Linear 导致构建失败
    lfcm_cfg = entry.get("lfcm")
    if lfcm_cfg is not None:
        lfcm_cfg = {k: v for k, v in lfcm_cfg.items()
                    if k in ("codebook_size", "code_dim", "hidden_dim") and v is not None}

    net = build_model(
        backbone="WRN34",
        method=entry["method"],
        num_class=num_class,
        lfcm_cfg=lfcm_cfg,
    )
    net.Num_class = num_class
    norm_mean = torch.tensor(cfg["mean"]).to(device)
    norm_std = torch.tensor(cfg["std"]).to(device)
    if entry["method"] in {"AT", "HFDR", "TRADES", "LFCM"}:
        net.Norm = True
        net.norm_mean = norm_mean
        net.norm_std = norm_std

    net = net.to(device)
    net = torch.nn.DataParallel(net)

    checkpoint_path = os.path.join(checkpoint_root, dataset_name, entry["prefix"], checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net


# ---------------------------------------------------------------------------
# 逐类别评测
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_per_class(net, loader, num_class, desc=""):
    """返回逐类别 (acc, loss) 与总体 (acc, loss)。"""
    net.eval()
    correct = np.zeros(num_class, dtype=np.float64)
    total = np.zeros(num_class, dtype=np.float64)
    loss_sum = np.zeros(num_class, dtype=np.float64)
    classes = torch.arange(num_class, device=device)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        preds = outputs.argmax(1)

        mask = targets[:, None] == classes[None, :]                    # B x K
        hit = (preds[:, None] == classes[None, :]) & mask
        total += mask.sum(0).cpu().numpy()
        correct += hit.sum(0).cpu().numpy()
        loss_sum += (loss[:, None] * mask.float()).sum(0).cpu().numpy()

    safe = np.maximum(total, 1.0)
    per_cls_acc = 100.0 * correct / safe
    per_cls_loss = loss_sum / safe
    overall_acc = 100.0 * correct.sum() / total.sum()
    overall_loss = loss_sum.sum() / total.sum()
    return per_cls_acc, per_cls_loss, overall_acc, overall_loss


# ---------------------------------------------------------------------------
# 缓存：每个模型一个 npz，存 (腐蚀 × 类别) 精度矩阵
# ---------------------------------------------------------------------------
def cache_path_of(args, dataset_name, model_name):
    return os.path.join(args.cache_dir, f"{dataset_name}__{model_name}.npz")


def load_cached_matrix(cache_path, corruptions, num_class):
    """缓存命中且覆盖所需腐蚀、类别数一致时返回 (corruptions, acc, loss)，否则 None。"""
    if not os.path.isfile(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=True)
    cached_corr = list(data["corruptions"])
    if data["acc"].shape[1] != num_class:
        return None
    if not all(c in cached_corr for c in corruptions):
        return None
    order = [cached_corr.index(c) for c in corruptions]
    return corruptions, data["acc"][order], data["loss"][order]


def save_cached_matrix(cache_path, corruptions, acc, loss):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, corruptions=np.array(corruptions), acc=acc, loss=loss)


def evaluate_model(args, dataset_name, entry, corruptions, logger):
    """评测单个模型的所有腐蚀，命中缓存直接返回。"""
    model_name = entry["name"]
    cache_path = cache_path_of(args, dataset_name, model_name)
    num_class = DATASET_CONFIG[dataset_name]["num_class"]

    cached = None if args.no_cache else load_cached_matrix(cache_path, corruptions, num_class)
    if cached is not None:
        logger.info("[%s] %s: cache hit (%d corruptions)", dataset_name, model_name, len(corruptions))
        return cached

    net = build_and_load_model(dataset_name, entry, args.checkpoint_root, args.checkpoint_name)
    num_class = DATASET_CONFIG[dataset_name]["num_class"]
    acc_all, loss_all = [], []
    for corruption in corruptions:
        loader = build_corruption_loader(dataset_name, corruption,
                                         args.data_root, args.batch_size, args.workers)
        acc, loss, overall_acc, overall_loss = evaluate_per_class(
            net, loader, num_class, desc=f"{dataset_name}/{model_name}/{corruption}")
        acc_all.append(acc)
        loss_all.append(loss)
        logger.info("[%s] %s | %-18s | overall Acc: %.2f | Loss: %.4f",
                    dataset_name, model_name, corruption, overall_acc, overall_loss)
    acc_mat = np.stack(acc_all)   # (C, K)
    loss_mat = np.stack(loss_all)  # (C, K)
    save_cached_matrix(cache_path, corruptions, acc_mat, loss_mat)
    logger.info("[%s] %s: saved cache -> %s", dataset_name, model_name, cache_path)
    return corruptions, acc_mat, loss_mat


# ---------------------------------------------------------------------------
# 结构分析：方差分解 + 置换零分布
# ---------------------------------------------------------------------------
def delta_analysis(delta, n_perm=1000, seed=0, logger=None):
    """对差值矩阵 Δ (腐蚀 × 类别) 做结构检验。

    返回 dict：
      grand, eta_corr, eta_cls, eta_res        —— 方差分解占比
      col_mean, col_std, row_mean              —— 类别剖面 / 腐蚀剖面
      p_eta, null_eta, null_eta_mean           —— 类别主效应置换检验
      rho_mean, p_rho, null_rho, null_rho_mean —— 跨腐蚀一致性检验
    """
    rng = np.random.default_rng(seed)
    C, K = delta.shape

    grand = float(delta.mean())
    row_mean = delta.mean(axis=1)
    col_mean = delta.mean(axis=0)
    col_std = delta.std(axis=0)

    ss_total = float(((delta - grand) ** 2).sum())
    ss_corr = float(K * ((row_mean - grand) ** 2).sum())
    ss_cls = float(C * ((col_mean - grand) ** 2).sum())
    ss_res = max(ss_total - ss_corr - ss_cls, 0.0)
    eta_corr = ss_corr / ss_total if ss_total > 0 else 0.0
    eta_cls = ss_cls / ss_total if ss_total > 0 else 0.0
    eta_res = ss_res / ss_total if ss_total > 0 else 0.0

    def permute_within_rows(d):
        """在每行内打乱类别标签 → 摧毁类别结构、保留每腐蚀的整体偏移与散布。"""
        out = d.copy()
        for c in range(C):
            out[c] = rng.permutation(out[c])
        return out

    def eta_cls_of(d):
        cm = d.mean(axis=0)
        ss_t = float(((d - d.mean()) ** 2).sum())
        ss_c = float(C * ((cm - d.mean()) ** 2).sum())
        return ss_c / ss_t if ss_t > 0 else 0.0

    # 类别主效应零分布
    null_eta = np.array([eta_cls_of(permute_within_rows(delta)) for _ in range(n_perm)])
    p_eta = (1 + int((null_eta >= eta_cls).sum())) / (1 + n_perm)

    # 跨腐蚀一致性：两两腐蚀的逐类差值向量 Pearson r 均值
    rmat = np.corrcoef(delta)
    triu_idx = np.triu_indices(C, 1)
    rho_mean = float(np.nan_to_num(rmat[triu_idx]).mean())

    null_rho = np.empty(n_perm)
    for i in range(n_perm):
        d = permute_within_rows(delta)
        rm = np.corrcoef(d)
        null_rho[i] = float(np.nan_to_num(rm[triu_idx]).mean())
    p_rho = (1 + int((null_rho >= rho_mean).sum())) / (1 + n_perm)

    if logger:
        logger.info("  Δ 均值=%.3f | η²_corr=%.3f η²_cls=%.3f η²_res=%.3f | "
                    "p(η²_cls)=%s | ρ̄=%.3f p(ρ̄)=%s",
                    grand, eta_corr, eta_cls, eta_res,
                    f"{p_eta:.4f}" if p_eta > 1.0 / n_perm else f"<{1.0 / n_perm:.4f}",
                    rho_mean,
                    f"{p_rho:.4f}" if p_rho > 1.0 / n_perm else f"<{1.0 / n_perm:.4f}")

    return {
        "grand": grand, "row_mean": row_mean, "col_mean": col_mean, "col_std": col_std,
        "eta_corr": eta_corr, "eta_cls": eta_cls, "eta_res": eta_res,
        "p_eta": p_eta, "null_eta": null_eta,
        "rho_mean": rho_mean, "p_rho": p_rho, "null_rho": null_rho,
        "null_eta_mean": float(null_eta.mean()), "null_rho_mean": float(null_rho.mean()),
        "corr_mat": rmat,
    }


def verdict_of(res, alpha=0.05):
    """由检验结果给出结构判定。"""
    eta_sig = res["p_eta"] < alpha
    rho_sig = res["p_rho"] < alpha
    if eta_sig and rho_sig:
        return "有结构（类别主效应与跨腐蚀一致性均显著）"
    if eta_sig or rho_sig:
        return "弱/部分结构（两个检验中仅一个显著）"
    return "无结构（与类别标签置换后的噪声不可区分）"


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.35,
    })
    return plt


def _diverging_norm(delta):
    from matplotlib.colors import TwoSlopeNorm
    vmax = float(np.nanmax(np.abs(delta)))
    if vmax == 0:
        vmax = 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def plot_delta_heatmap(plt, delta, row_labels, class_names, title, out_path):
    """Δ 矩阵热力图（腐蚀 × 类别）+ 行/列均值边际条。"""
    C, K = delta.shape
    norm = _diverging_norm(delta)
    cmap = plt.cm.RdBu_r
    vmax = norm.vmax

    fig = plt.figure(figsize=(10.5, 6.0) if K <= 10 else (20.0, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[24, 1.4], height_ratios=[1.0, 12],
                          wspace=0.04, hspace=0.04, left=0.14, right=0.93,
                          bottom=0.14, top=0.88)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(delta, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(C))
    ax.set_yticklabels(row_labels, fontsize=8)
    if K <= 10:
        ax.set_xticks(range(K))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticks(range(0, K, 5))
        ax.set_xticklabels([str(i) for i in range(0, K, 5)], fontsize=6.5)
        ax.set_xticks(range(K), minor=True)
    ax.set_title(title, fontsize=11)
    ax.grid(False)

    # 顶部边际：类别均值
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    col_mean = delta.mean(axis=0)
    ax_top.bar(range(K), col_mean, width=1.0, color=[cmap(norm(v)) for v in col_mean])
    ax_top.set_ylim(-vmax, vmax)
    ax_top.set_ylabel("mean", fontsize=7)
    ax_top.set_title("class mean (pp)", fontsize=7, loc="left", pad=2)
    ax_top.tick_params(axis="y", labelsize=6)
    ax_top.grid(axis="y", alpha=0.35)

    # 右侧边际：腐蚀均值
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    row_mean = delta.mean(axis=1)
    ax_right.barh(range(C), row_mean, height=1.0, color=[cmap(norm(v)) for v in row_mean])
    ax_right.set_xlim(-vmax, vmax)
    ax_right.set_title("corruption\nmean", fontsize=7, loc="left", pad=2)
    ax_right.tick_params(axis="x", labelsize=6)
    ax_right.grid(axis="x", alpha=0.35)

    cbar_ax = fig.add_axes([0.945, 0.14, 0.012, 0.74])
    fig.colorbar(im, cax=cbar_ax, label="Δ acc (pp)")
    fig.savefig(out_path)
    plt.close(fig)


def plot_class_profile(plt, delta, row_labels, class_names, title, out_path):
    """类别剖面：每类 Δ 的均值 ± std（按均值降序），叠加各腐蚀散点（按腐蚀分组着色）。"""
    K = delta.shape[1]
    col_mean = delta.mean(axis=0)
    col_std = delta.std(axis=0)
    order = np.argsort(-col_mean)
    xs = np.arange(K)

    fig, ax = plt.subplots(figsize=(7.5, 4.6) if K <= 10 else (17.0, 5.2))
    ax.axhline(0, color="#7f8c8d", lw=1.0, zorder=1)

    seen_groups = set()
    for c in range(delta.shape[0]):
        grp = CORRUPTION_GROUPS.get(row_labels[c], "blur")
        label = grp if grp not in seen_groups else None
        seen_groups.add(grp)
        ax.scatter(xs, delta[c][order], s=12, color=GROUP_PALETTE[grp],
                   alpha=0.5, zorder=2, label=label, edgecolors="none")
    ax.errorbar(xs, col_mean[order], yerr=col_std[order], fmt="o",
                color="#2c3e50", markersize=3.5, lw=1.2, capsize=2, zorder=3)

    if K <= 10:
        ax.set_xticks(xs)
        ax.set_xticklabels([class_names[i] for i in order], rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticks(range(0, K, 10))
        ax.set_xticklabels([str(i) for i in range(0, K, 10)], fontsize=7)
        ax.set_xlabel("class (ranked by mean Δ)", fontsize=9)
        # 标注收益/损失最大的类（交替纵向偏移防重叠）
        top = order[:8]
        bot = order[-8:]
        for rank_i, i in enumerate(top):
            ax.annotate(class_names[i], (rank_i, col_mean[i]),
                        textcoords="offset points", xytext=(0, 13 if rank_i % 2 else 6),
                        fontsize=6, color="#c0392b", rotation=70, ha="left")
        for rank_i, i in enumerate(bot):
            ax.annotate(class_names[i], (K - 8 + rank_i, col_mean[i]),
                        textcoords="offset points", xytext=(0, -13 if rank_i % 2 else -6),
                        fontsize=6, color="#2980b9", rotation=70, ha="right")

    ax.set_ylabel("Δ acc (pp)")
    ax.set_title(title, fontsize=11)
    ax.legend(title="corruption group", loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_ylim(delta.min() - 1.0, delta.max() + 1.0)
    fig.savefig(out_path)
    plt.close(fig)


def plot_corr_heatmap(plt, rmat, row_labels, rho_mean, p_rho, title, out_path):
    """腐蚀间逐类差值向量相关系数矩阵：结构真实 → 普遍为正。"""
    C = rmat.shape[0]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(rmat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(row_labels, rotation=90, fontsize=6.5)
    ax.set_yticklabels(row_labels, fontsize=6.5)
    for i in range(C):
        for j in range(C):
            if i != j:
                ax.text(j, i, f"{rmat[i, j]:.2f}", ha="center", va="center",
                        fontsize=4.5, color="#2c3e50")
    ax.grid(False)
    ax.set_title(f"{title}\nmean pairwise r = {rho_mean:.3f}  (null p = {p_rho:.4f})",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.savefig(out_path)
    plt.close(fig)


def plot_null_test(plt, res, title, out_path):
    """置换零分布对比：左 = η²_class，右 = ρ̄。观测值红竖线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    ax1.hist(res["null_eta"], bins=40, color="#bdc3c7", edgecolor="white", linewidth=0.4)
    ax1.axvline(res["eta_cls"], color="#e74c3c", lw=2.0,
                label=f"observed η² = {res['eta_cls']:.3f}\np = {res['p_eta']:.4f}")
    ax1.set_xlabel("η² (class main-effect share)")
    ax1.set_ylabel("permutations")
    ax1.legend(fontsize=8)

    ax2.hist(res["null_rho"], bins=40, color="#bdc3c7", edgecolor="white", linewidth=0.4)
    ax2.axvline(res["rho_mean"], color="#e74c3c", lw=2.0,
                label=f"observed ρ̄ = {res['rho_mean']:.3f}\np = {res['p_rho']:.4f}")
    ax2.set_xlabel("ρ̄ (mean pairwise corr. across corruptions)")
    ax2.set_ylabel("permutations")
    ax2.legend(fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path)
    plt.close(fig)


def plot_k_sweep(plt, ks, mean_deltas, etas, rhos, null_eta_mean, null_rho_mean,
                 dataset_name, out_path):
    """K 扫描：平均 Δ、类别主效应占比、跨腐蚀一致性 随 K 变化（虚线 = 零分布均值）。"""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].axhline(0, color="#7f8c8d", lw=1.0)
    axes[0].plot(ks, mean_deltas, "o-", color="#2c3e50", lw=1.4, markersize=5)
    axes[0].set_xlabel("codebook size K"); axes[0].set_ylabel("mean Δ acc (pp)")
    axes[0].set_title("overall gain vs AT")

    axes[1].axhline(null_eta_mean, color="#95a5a6", lw=1.2, ls="--",
                    label=f"null mean = {null_eta_mean:.3f}")
    axes[1].plot(ks, etas, "o-", color="#2980b9", lw=1.4, markersize=5)
    axes[1].set_xlabel("codebook size K"); axes[1].set_ylabel("η² (class share)")
    axes[1].set_title("class-structure share")
    axes[1].legend(fontsize=7)

    axes[2].axhline(null_rho_mean, color="#95a5a6", lw=1.2, ls="--",
                    label=f"null mean = {null_rho_mean:.3f}")
    axes[2].plot(ks, rhos, "o-", color="#16a085", lw=1.4, markersize=5)
    axes[2].set_xlabel("codebook size K"); axes[2].set_ylabel("ρ̄")
    axes[2].set_title("cross-corruption consistency")
    axes[2].legend(fontsize=7)

    fig.suptitle(f"{dataset_name}: LFCM−AT structure vs codebook size K", fontsize=11)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def fmt_p(p, n_perm):
    return f"{p:.4f}" if p > 1.0 / n_perm else f"<{1.0 / n_perm:.4f}"


def build_report(args, dataset_name, eval_records, delta_results, out_fig_dir, logger):
    """生成逐数据集 Markdown 报告片段；全数据集汇总在主函数拼接。"""
    class_names = get_class_names(dataset_name)
    lines = []
    lines.append(f"## {dataset_name}")
    lines.append("")

    # 1) 总体精度（sanity：与 *_ood_test.log 对齐）
    lines.append("### 总体 OOD 精度（均值 ± 跨腐蚀 std，与既有 ood log 可对照）")
    lines.append("")
    lines.append("| model | mean acc (%) |")
    lines.append("|:------|-------------:|")
    for name, (_corr, acc, _loss) in eval_records.items():
        lines.append(f"| {name} | {acc.mean():.2f} ± {acc.std():.2f} |")
    lines.append("")

    # 2) 结构检验表（各 LFCM K 及 HFDR 对照 − AT）
    lines.append("### Δ = model − AT 的逐类结构检验")
    lines.append("")
    lines.append("| model | mean Δ (pp) | η²_corr | η²_cls | η²_res | p(η²_cls) | ρ̄ | p(ρ̄) | 判定 |")
    lines.append("|:------|------------:|--------:|-------:|-------:|----------:|----:|------:|:-----|")
    for name, res in delta_results.items():
        lines.append(
            f"| {name} | {res['grand']:+.2f} | {res['eta_corr']:.3f} | {res['eta_cls']:.3f} | "
            f"{res['eta_res']:.3f} | {fmt_p(res['p_eta'], args.n_perm)} | "
            f"{res['rho_mean']:.3f} | {fmt_p(res['p_rho'], args.n_perm)} | {verdict_of(res)} |")
    lines.append("")

    # 3) 每模型：类别剖面 TOP/BOTTOM 与图链接
    for name, res in delta_results.items():
        order = np.argsort(-res["col_mean"])
        top = ", ".join(f"{class_names[i]} {res['col_mean'][i]:+.1f}" for i in order[:5])
        bot = ", ".join(f"{class_names[i]} {res['col_mean'][i]:+.1f}" for i in order[-5:])
        lines.append(f"**{name} − AT**  · 前5收益: {top}  · 前5损失: {bot}")
        lines.append("")
        lines.append(f"![{name} heatmap](figures/per_class/{dataset_name}/{dataset_name}__{name}__delta_heatmap.png)")
        lines.append(f"![{name} profile](figures/per_class/{dataset_name}/{dataset_name}__{name}__class_profile.png)")
        if name.startswith("LFCM"):
            lines.append(f"![{name} corr](figures/per_class/{dataset_name}/{dataset_name}__{name}__corr_consistency.png)")
            lines.append(f"![{name} null](figures/per_class/{dataset_name}/{dataset_name}__{name}__null_test.png)")
        lines.append("")

    # 4) HFDR 与 LFCM 类别剖面的相似度（结构是否 LFCM 特有）
    if "HFDR" in delta_results:
        hfdr_col = delta_results["HFDR"]["col_mean"]
        sim_lines = []
        for name, res in delta_results.items():
            if name.startswith("LFCM"):
                r = float(np.corrcoef(hfdr_col, res["col_mean"])[0, 1])
                sim_lines.append(f"| {name} | {r:.3f} |")
        if sim_lines:
            lines.append("### 类别剖面相似度：corr(col_mean(HFDR−AT), col_mean(LFCM−AT))")
            lines.append("")
            lines.append("| LFCM variant | r |")
            lines.append("|:-------------|---:|")
            lines.extend(sim_lines)
            lines.append("")
            lines.append("> r 高 → LFCM 的类别结构主要继承自 HFDR 框架；r 低 → 结构是 LFCM 特有。")
            lines.append("")

    # 5) K 扫描图
    ks, md, etas, rhos = [], [], [], []
    for name in sorted(delta_results, key=lambda n: int(n.split("_K")[1]) if "_K" in n else 10**6):
        if name.startswith("LFCM_K"):
            res = delta_results[name]
            ks.append(int(name.split("_K")[1]))
            md.append(res["grand"])
            etas.append(res["eta_cls"])
            rhos.append(res["rho_mean"])
    if ks:
        lines.append(f"![K sweep](figures/per_class/{dataset_name}/{dataset_name}__K_sweep.png)")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LFCM vs AT 逐类别 OOD 诊断（per-class corruption accuracy & structure test）")
    parser.add_argument("--dataset", nargs="*", default=None,
                        choices=["CIFAR10", "CIFAR100"],
                        help="要分析的 dataset（默认：两个都跑）")
    parser.add_argument("--models", default=None,
                        help="逗号分隔的模型名（默认：清单内全部）；如 AT,LFCM_K64,HFDR")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="差值基准模型名（默认 AT）")
    parser.add_argument("--checkpoint-root", default="./checkpoint")
    parser.add_argument("--checkpoint-name", default="model_best.pth.tar")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--corruptions", nargs="*", default=None,
                        help="只测部分腐蚀（默认全部 19 种）")
    parser.add_argument("--cache-dir", default="./result/per_class_cache")
    parser.add_argument("--no-cache", action="store_true",
                        help="忽略缓存强制重测（结果仍写回缓存）")
    parser.add_argument("--analyze-only", action="store_true",
                        help="跳过评测，只用缓存做分析/出图")
    parser.add_argument("--output-dir", default="./result/figures/per_class")
    parser.add_argument("--report-path", default="./result/per_class_analysis_report.md")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="置换检验次数（默认 1000）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", default="./result/per_class_analysis.log")
    args = parser.parse_args()

    datasets = args.dataset or list(MODEL_MANIFEST.keys())

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
    logger = logging.getLogger("lfcm_perclass")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] - %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
        fh = logging.FileHandler(args.log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    logger.info("========== LFCM vs AT per-class OOD diagnostic ==========")
    logger.info("datasets: %s | n_perm: %d | seed: %d", datasets, args.n_perm, args.seed)

    report_parts = []

    for dataset_name in datasets:
        cfg = DATASET_CONFIG[dataset_name]
        num_class = cfg["num_class"]
        assert len(get_class_names(dataset_name)) == num_class, \
            f"class-name list mismatch for {dataset_name}"

        manifest = MODEL_MANIFEST[dataset_name]
        if args.models:
            wanted = [m.strip() for m in args.models.split(",") if m.strip()]
            unknown = [m for m in wanted if m not in {e["name"] for e in manifest}]
            if unknown:
                logger.warning("[%s] unknown model names ignored: %s (available: %s)",
                               dataset_name, unknown, [e["name"] for e in manifest])
            manifest = [e for e in manifest if e["name"] in wanted]
        if not any(e["name"] == args.reference for e in manifest):
            raise ValueError(f"reference model '{args.reference}' not in selected models for {dataset_name}")

        corruptions = args.corruptions or get_corruptions(dataset_name, args.data_root)
        if not corruptions:
            raise FileNotFoundError(
                f"No corruption files under {os.path.join(args.data_root, DATASET_CONFIG[dataset_name]['data_subdir'])}")
        logger.info("[%s] corruptions (%d): %s", dataset_name, len(corruptions), ", ".join(corruptions))

        # ---- 逐模型评测（或读缓存）----
        eval_records = {}
        for entry in manifest:
            name = entry["name"]
            try:
                if args.analyze_only:
                    cached = load_cached_matrix(cache_path_of(args, dataset_name, name),
                                                corruptions, num_class)
                    if cached is None:
                        raise FileNotFoundError(f"cache missing: {cache_path_of(args, dataset_name, name)}")
                    logger.info("[%s] %s: cache loaded", dataset_name, name)
                else:
                    cached = evaluate_model(args, dataset_name, entry, corruptions, logger)
                eval_records[name] = cached
            except FileNotFoundError as e:
                logger.warning("[%s] %s: SKIP (%s)", dataset_name, name, e)

        if args.reference not in eval_records:
            raise FileNotFoundError(
                f"reference model '{args.reference}' unavailable for {dataset_name} "
                f"(checkpoint/cache missing); got: {list(eval_records)}")
        ref_corr, ref_acc, _ = eval_records[args.reference]

        # ---- 差值矩阵 + 结构检验 ----
        delta_results = {}
        delta_matrices = {}
        for name, (corr, acc, _loss) in eval_records.items():
            if name == args.reference:
                continue
            if corr != ref_corr:  # 保险：腐蚀顺序对齐
                idx = [corr.index(c) for c in ref_corr]
                acc = acc[idx]
            delta = acc - ref_acc
            logger.info("[%s] %s − %s :", dataset_name, name, args.reference)
            res = delta_analysis(delta, n_perm=args.n_perm, seed=args.seed, logger=logger)
            delta_results[name] = res
            delta_matrices[name] = delta

            # 导出逐类差值 CSV
            class_names = get_class_names(dataset_name)
            with open(os.path.join(args.cache_dir, f"{dataset_name}__{name}__delta_vs_{args.reference}.csv"),
                      "w", encoding="utf-8") as f:
                f.write("class_idx,class_name,mean_delta,std_delta," + ",".join(ref_corr) + "\n")
                for k in range(num_class):
                    f.write(f"{k},{class_names[k]},{res['col_mean'][k]:.3f},{res['col_std'][k]:.3f},"
                            + ",".join(f"{delta[c, k]:.3f}" for c in range(delta.shape[0])) + "\n")
            logger.info("[%s] %s: delta CSV -> %s", dataset_name, name,
                        os.path.join(args.cache_dir, f"{dataset_name}__{name}__delta_vs_{args.reference}.csv"))

        # ---- 出图 ----
        plt = setup_matplotlib()
        out_fig_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(out_fig_dir, exist_ok=True)
        class_names = get_class_names(dataset_name)

        for name, res in delta_results.items():
            tag = f"{dataset_name} — {name} − {args.reference} per-class Δ acc (pp)"
            delta = delta_matrices[name]
            if name.startswith("LFCM"):
                plot_delta_heatmap(plt, delta, ref_corr, class_names, tag,
                                   os.path.join(out_fig_dir, f"{dataset_name}__{name}__delta_heatmap.png"))
                plot_class_profile(plt, delta, ref_corr, class_names, tag,
                                   os.path.join(out_fig_dir, f"{dataset_name}__{name}__class_profile.png"))
                plot_corr_heatmap(plt, res["corr_mat"], ref_corr, res["rho_mean"], res["p_rho"], tag,
                                  os.path.join(out_fig_dir, f"{dataset_name}__{name}__corr_consistency.png"))
                plot_null_test(plt, res, tag,
                               os.path.join(out_fig_dir, f"{dataset_name}__{name}__null_test.png"))
                logger.info("[%s] %s: figures done", dataset_name, name)
            elif name == "HFDR":  # HFDR 作对照：只出热力图 + 类别剖面
                plot_delta_heatmap(plt, delta, ref_corr, class_names, tag,
                                   os.path.join(out_fig_dir, f"{dataset_name}__{name}__delta_heatmap.png"))
                plot_class_profile(plt, delta, ref_corr, class_names, tag,
                                   os.path.join(out_fig_dir, f"{dataset_name}__{name}__class_profile.png"))

        # K 扫描图
        ks, md, etas, rhos = [], [], [], []
        for name, res in delta_results.items():
            if name.startswith("LFCM_K"):
                ks.append(int(name.split("_K")[1]))
                md.append(res["grand"])
                etas.append(res["eta_cls"])
                rhos.append(res["rho_mean"])
        if ks:
            order = np.argsort(ks)
            ks = [ks[i] for i in order]; md = [md[i] for i in order]
            etas = [etas[i] for i in order]; rhos = [rhos[i] for i in order]
            null_eta_mean = float(np.mean([delta_results[n]["null_eta_mean"]
                                           for n in delta_results if n.startswith("LFCM_K")]))
            null_rho_mean = float(np.mean([delta_results[n]["null_rho_mean"]
                                           for n in delta_results if n.startswith("LFCM_K")]))
            plot_k_sweep(plt, ks, md, etas, rhos, null_eta_mean, null_rho_mean,
                         dataset_name, os.path.join(out_fig_dir, f"{dataset_name}__K_sweep.png"))
            logger.info("[%s] K sweep figure done", dataset_name)

        report_parts.append(build_report(args, dataset_name, eval_records,
                                         delta_results, out_fig_dir, logger))

    # ---- 汇总报告 ----
    header = []
    header.append("# LFCM vs AT 逐类别 OOD 诊断报告")
    header.append("")
    header.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header.append(f"- 数据集: {', '.join(datasets)}  ·  差值基准: {args.reference}")
    header.append(f"- 置换检验次数: {args.n_perm}  ·  腐蚀数: 19（CIFAR-C）")
    header.append(f"- 逐类差值 CSV: `result/per_class_cache/`（`{dataset}__<model>__delta_vs_AT.csv`）")
    header.append("")
    header.append("## 结论：LFCM 相对 AT 的逐类收益/损失是否有结构？")
    header.append("")
    header.append(
        "判据：把 Δ(腐蚀×类别) 总方差分解为 腐蚀主效应 / 类别主效应 / 残差（交互）。"
        "若类别主效应占比 η²_cls 显著高于「每行内打乱类别标签」的置换零分布（p<0.05），"
        "且不同腐蚀的逐类差值向量普遍正相关（ρ̄ 显著），则收益/损失**随类别有稳定结构**；"
        "反之则与噪声不可区分（无结构）。HFDR−AT 作对照：若 HFDR 无结构而 LFCM 有，"
        "结构来自 LFCM 的 canonicalization；若两者都有，结构可能继承自 HFDR 滤波框架。")
    header.append("")
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + report_parts))
    logger.info("Report -> %s", args.report_path)
    logger.info("========== done ==========")


if __name__ == "__main__":
    cudnn.benchmark = True
    main()
