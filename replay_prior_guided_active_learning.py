# -*- coding: utf-8 -*-
"""
基于 ALL2962 oracle 的“三路径 + signed prior”离线回放脚本。

这份脚本对应新版主动学习流程，因此不再比较旧版的：
- no_prior
- positive_prior

而是比较：
- tripath_no_prior
- tripath_signed_prior

这里的“离线回放”不是让模型自己给自己打标签，
而是：
1. 用当前训练集训练 committee；
2. 在 oracle 剩余样本里构造三路径候选池；
3. 用模型从候选池里选出一批序列；
4. 直接读取 oracle 里的真实标签，模拟真实实验反馈；
5. 将真实标签样本加入训练集，进入下一轮。

它的用途是：
- 在真正开始新流程迭代前，先比较“不开先验”和“开 signed prior”哪种更合适；
- 并生成 prior_gate_decision.json，供真实推荐脚本在运行时自动读取。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import time

# ============================================================
# 0. 动态导入新版主脚本
# ============================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR
RUN_SCRIPT_PATH = PROJECT_ROOT / "run_dna_agn_active_learning_tripath.py"


def load_al_core():
    if not RUN_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"找不到主脚本：{RUN_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("al_tripath", RUN_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


al = load_al_core()


# ============================================================
# 1. 回放配置
# ============================================================
@dataclass
class ReplayConfig:
    oracle_csv: str = str(PROJECT_ROOT / "ALL2998.csv")
    training_csv: str = str(PROJECT_ROOT / "Iteration3_156.csv")
    output_root: str = str(PROJECT_ROOT / "offline_replay_outputs_tripath")
    signed_prior_csv: Optional[str] = None
    # training_csv 是 replay 的当前起始训练集；
    # initial_training_size 表示这条新流程最初是从多少条样本起步，
    # 它主要用于和主流程保持轮次口径一致；
    # recommend_k 表示 replay 每一轮模拟会“选多少条并读 oracle 真值回填”，
    # 这个值应与真实推荐脚本保持一致。
    n_rounds: int = 10
    recommend_k: int = 12
    n_repeats: int = 30
    seed_start: int = 42
    initial_training_size: int = 120

    # --- 与新版主流程一致的核心超参数 ---
    committee_size: int = 25
    n_jobs: int = -1
    use_gpu_backend: bool = False
    gpu_device: str = "cuda"
    torch_logreg_max_iter: int = 120
    torch_reg_max_iter: int = 120
    feature_subsample_ratio: float = 0.80
    logreg_C: float = 0.30
    logreg_l1_ratio: float = 0.30
    logreg_max_iter: int = 6000

    w_target: float = 0.375
    w_uncertainty: float = 0.20
    w_diversity: float = 0.30
    w_motif: float = 0.125
    diversity_hamming_weight: float = 0.50
    diversity_staple_weight: float = 0.50
    staple_diversity_top_k: int = 30

    motif_top_k: int = 15
    motif_negative_scale: float = 0.70
    # --- 波长回归分支 ---
    use_wavelength_branch: bool = True
    wavelength_col: str = "main_peak_lambda_nm"
    wavelength_reg_alpha: float = 0.01
    wavelength_reg_l1_ratio: float = 0.30
    wavelength_reg_max_iter: int = 20000
    wavelength_boundary_nm: float = 780.0
    wavelength_sigmoid_tau: float = 15.0
    w_wavelength: float = 0.10
    # --- 亮度回归分支 ---
    use_brightness_branch: bool = True
    brightness_col: str = "main_peak_lli_raw"
    brightness_reg_alpha: float = 0.01
    brightness_reg_l1_ratio: float = 0.30
    brightness_reg_max_iter: int = 20000
    brightness_floor_quantile: float = 0.20
    brightness_good_quantile: float = 0.80
    brightness_sigmoid_tau: float = 0.35
    w_brightness: float = 0.10
    brightness_hard_risk_threshold: float = 0.85

    # --- 三路径回放参数 ---
    # replay 不能像真实流程那样从 4^10 全空间生成候选再去做实验，
    # 因为离线回放只能在“已有真实标签的 oracle 样本集合”内部取点。
    # 当前这份脚本使用的是 ALL2962 作为 oracle，
    # 因此这里的三路径不是在全空间里生成新序列，
    # 而是在“剩余 oracle 样本”内部近似实现三条候选路径：
    # 路径 A：围绕当前训练集中的 NIR / Far Red seeds 做局部开发；
    # 路径 B：在剩余 oracle 里挑更远、更 novel、且先验不太差的样本；
    # 路径 C：在剩余 oracle 里随机抽样，模拟真实流程中的随机覆盖池。
    path_a_hamming_radius: int = 2
    path_a_include_farred_seeds: bool = True
    path_a_pool_size: int = 800
    path_b_pool_size: int = 1200
    path_b_min_cg_count: int = 4
    # --- 路径 C：随机覆盖池（在剩余 oracle 内部随机抽样）
    path_c_enabled: bool = True
    path_c_pool_size: int = 300

    gate_auc_improvement_threshold: float = 0.05


# ============================================================
# 2. 基础读写
# ============================================================
def resolve_existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("以下路径都不存在：\n" + "\n".join(str(x) for x in candidates))


# 给 DataFrame 增加一列 _digits，
# 用整数编码后的 10-mer 表示序列，便于后面快速计算 Hamming 距离。
# 这一列只在 replay 内部使用，不参与导出，也不作为模型输入特征。
def add_digits_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_digits"] = out["Sequence"].map(lambda s: [al.BASE_TO_INT[ch] for ch in al.normalize_sequence(s)])
    return out


# 把 ReplayConfig 映射成主脚本可直接使用的 al.Config。
# 这样 replay 在 committee、打分、波长/亮度分支、greedy 选批等核心行为上，
# 能尽量复用真实推荐脚本的实现，避免出现“replay 和真实流程不是同一套逻辑”的问题。
# 注意：这里有意把 enable_candidate_pruning 关掉，
# 因为 replay 的候选池本来就只是 oracle 剩余样本的子集，
# 一般不需要再做真实流程里那种较强的候选池压缩。
def build_runtime_cfg(cfg: ReplayConfig, use_motif_prior: bool) -> al.Config:
    return al.Config(
        input_csv="",
        output_root="",
        banned_sequences_path="",
        initial_training_size=cfg.initial_training_size,
        recommend_k=cfg.recommend_k,
        min_hamming_distance=0,
        # candidate_mode:
        # - "tripath"：使用 A/B/C 三路径候选生成流程
        # - "full_space"：使用全空间扫描，仅用于旧流程对照
        candidate_mode="tripath",
        path_a_enabled=True,
        path_a_include_farred_seeds=cfg.path_a_include_farred_seeds,
        path_a_pool_size=cfg.path_a_pool_size,
        path_b_enabled=True,
        path_b_pool_size=cfg.path_b_pool_size,
        path_b_min_cg_count=cfg.path_b_min_cg_count,
        enable_candidate_pruning=False,  # replay 中候选池已经只有 oracle 子集，一般不再额外 prune
        base_model_name="elasticnet_logreg",
        committee_size=cfg.committee_size,
        n_jobs=cfg.n_jobs,
        use_gpu_backend=cfg.use_gpu_backend,
        gpu_device=cfg.gpu_device,
        torch_logreg_max_iter=cfg.torch_logreg_max_iter,
        torch_reg_max_iter=cfg.torch_reg_max_iter,
        random_seed=cfg.seed_start,
        per_class_sample_size=None,
        feature_subsample_ratio=cfg.feature_subsample_ratio,
        logreg_C=cfg.logreg_C,
        logreg_l1_ratio=cfg.logreg_l1_ratio,
        logreg_max_iter=cfg.logreg_max_iter,
        w_target=cfg.w_target,
        w_uncertainty=cfg.w_uncertainty,
        w_diversity=cfg.w_diversity,
        w_motif=cfg.w_motif,
        diversity_hamming_weight=cfg.diversity_hamming_weight,
        diversity_staple_weight=cfg.diversity_staple_weight,
        staple_diversity_top_k=cfg.staple_diversity_top_k,
        use_motif_prior=use_motif_prior,
        motif_prior_csv=None,
        motif_prior_top_k=cfg.motif_top_k,
        motif_negative_scale=cfg.motif_negative_scale,
        path_c_enabled=cfg.path_c_enabled,
        path_c_pool_size=cfg.path_c_pool_size,

        use_wavelength_branch=cfg.use_wavelength_branch,
        wavelength_col=cfg.wavelength_col,
        wavelength_reg_alpha=cfg.wavelength_reg_alpha,
        wavelength_reg_l1_ratio=cfg.wavelength_reg_l1_ratio,
        wavelength_reg_max_iter=cfg.wavelength_reg_max_iter,
        wavelength_boundary_nm=cfg.wavelength_boundary_nm,
        wavelength_sigmoid_tau=cfg.wavelength_sigmoid_tau,
        w_wavelength=cfg.w_wavelength,
        use_brightness_branch=cfg.use_brightness_branch,
        brightness_col=cfg.brightness_col,
        brightness_reg_alpha=cfg.brightness_reg_alpha,
        brightness_reg_l1_ratio=cfg.brightness_reg_l1_ratio,
        brightness_reg_max_iter=cfg.brightness_reg_max_iter,
        brightness_floor_quantile=cfg.brightness_floor_quantile,
        brightness_good_quantile=cfg.brightness_good_quantile,
        brightness_sigmoid_tau=cfg.brightness_sigmoid_tau,
        w_brightness=cfg.w_brightness,
        brightness_hard_risk_threshold=cfg.brightness_hard_risk_threshold,
    )

# ============================================================
# 3. 回放用的三路径候选池（在 oracle 剩余样本内部构造）
# ============================================================
# 从 oracle 中扣掉当前训练集已经包含的序列，
# 得到“本轮仍可被模拟挑选”的剩余样本池。
# replay 的所有候选路径，都是在这个剩余 oracle 池上构造的。
def build_remaining_oracle_pool(oracle_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    train_set = set(train_df["Sequence"])
    return oracle_df.loc[~oracle_df["Sequence"].isin(train_set)].copy().reset_index(drop=True)

def build_oracle_path_A_pool(
    remaining_df: pd.DataFrame,
    train_df: pd.DataFrame,
    cfg: ReplayConfig,
    pos_weights: Dict[str, float],
    neg_weights: Dict[str, float],
    runtime_cfg: al.Config,
) -> pd.DataFrame:
    """
    路径 A：局部开发池（仅在剩余 oracle 内部找“距离 seed 比较近”的样本）。
    这一步的目的，是让 replay 中仍然保留 exploitation 路径。
    """
    seed_mask = train_df["class_label"].eq("NIR")
    if cfg.path_a_include_farred_seeds:
        seed_mask = seed_mask | train_df["class_label"].eq("Far Red")

    seed_df = train_df.loc[seed_mask].copy()
    if seed_df.empty:
        seed_df = train_df.loc[train_df["class_label"] != "Dark"].copy()

    seed_digits = np.asarray(seed_df["_digits"].tolist(), dtype=np.uint8)
    rem_digits = np.asarray(remaining_df["_digits"].tolist(), dtype=np.uint8)
    min_dist = al.compute_min_hamming_to_reference(rem_digits, seed_digits)
    path_a_df = remaining_df.loc[min_dist <= cfg.path_a_hamming_radius].copy().reset_index(drop=True)
    if path_a_df.empty:
        return path_a_df

    X = path_a_df[al.STANDARD_FEATURES].to_numpy(dtype=np.float32)
    path_a_df["signed_prior_seed_score"] = al.signed_motif_prior_score(X, pos_weights, neg_weights, runtime_cfg)
    path_a_df = path_a_df.sort_values("signed_prior_seed_score", ascending=False).head(cfg.path_a_pool_size).reset_index(drop=True)
    path_a_df["candidate_path"] = "A_local_oracle"
    return path_a_df



def build_oracle_path_B_pool(
    remaining_df: pd.DataFrame,
    path_a_df: pd.DataFrame,
    train_df: pd.DataFrame,
    cfg: ReplayConfig,
    pos_weights: Dict[str, float],
    neg_weights: Dict[str, float],
    full_signed_prior_df: pd.DataFrame,
    runtime_cfg: al.Config,
) -> pd.DataFrame:
    """
    路径 B：探索池。
    这里不再看是否接近 NIR seed，而是从剩余 oracle 样本里挑“更远、更 novel、且先验不太差”的样本。
    这样 replay 中仍然保留 exploration 路径。
    """
    exclude = set(path_a_df["Sequence"]) if not path_a_df.empty else set()
    work = remaining_df.loc[~remaining_df["Sequence"].isin(exclude)].copy().reset_index(drop=True)
    if work.empty:
        return work

    # 与真实主流程保持一致：路径 B 在 replay 中也施加最低 C+G 数硬约束。
    work = work.loc[
        work["Sequence"].map(lambda s: sum(ch in {"C", "G"} for ch in str(s)) >= cfg.path_b_min_cg_count)
    ].copy().reset_index(drop=True)
    if work.empty:
        return work

    if full_signed_prior_df.empty:
        important_features = al.STANDARD_FEATURES[: cfg.staple_diversity_top_k]
    else:
        important_features = full_signed_prior_df["feature"].tolist()[: cfg.staple_diversity_top_k]

    X_train_imp = train_df[important_features].to_numpy(dtype=np.float32)
    X_work_imp = work[important_features].to_numpy(dtype=np.float32)
    max_vec = al.build_feature_theoretical_max_vector(important_features)
    novelty = al.normalized_l1_distance_to_reference(X_work_imp, X_train_imp, max_vec)

    X_full = work[al.STANDARD_FEATURES].to_numpy(dtype=np.float32)
    signed_prior = al.signed_motif_prior_score(X_full, pos_weights, neg_weights, runtime_cfg)

    # 这里故意不用模型概率，避免路径 B 变成“提前用最终分数排序”。
    # 只用 novelty + signed prior 形成一个便宜探索分，保持与真实流程中的“候选生成层”分工一致。
    work["pathB_seed_score"] = 0.70 * novelty + 0.30 * np.clip(signed_prior, 0.0, 1.0)
    work = work.sort_values("pathB_seed_score", ascending=False).head(cfg.path_b_pool_size).reset_index(drop=True)
    work["candidate_path"] = "B_explore_oracle"
    return work

def build_oracle_path_C_pool(
    remaining_df: pd.DataFrame,
    path_a_df: pd.DataFrame,
    path_b_df: pd.DataFrame,
    cfg: ReplayConfig,
    repeat_seed: int,
) -> pd.DataFrame:
    """
    路径 C：在剩余 oracle 内部做随机抽样，模拟真实流程中的随机覆盖池。

    注意：
    replay 里不能像真实流程那样从 4^10 全空间生成新序列，
    因为离线回放只能在“已有真实标签的 oracle 样本集合”内部取点。
    当前这份脚本使用的是 ALL2962 作为 oracle，
    因此这里的路径 C，是从“尚未进入训练集、也未被 A/B 占用的 oracle 剩余样本”
    中做随机抽样，作为随机覆盖的近似版本。
    """
    exclude = set()
    if not path_a_df.empty:
        exclude.update(path_a_df["Sequence"].tolist())
    if not path_b_df.empty:
        exclude.update(path_b_df["Sequence"].tolist())

    work = remaining_df.loc[~remaining_df["Sequence"].isin(exclude)].copy().reset_index(drop=True)
    if work.empty:
        return work

    rng = np.random.RandomState(repeat_seed + 2024)
    n = min(cfg.path_c_pool_size, len(work))
    sampled_idx = rng.choice(np.arange(len(work)), size=n, replace=False)
    out = work.iloc[sampled_idx].copy().reset_index(drop=True)
    out["candidate_path"] = "C_random_oracle"
    return out

# 把 A / B / C 三条 oracle 内部候选路径合并成一个统一候选池。
# 注意这里的去重只是在 oracle 已有样本内部做 Sequence 去重，
# 不涉及真实流程里对全空间候选做的 candidate pruning。
# 这样 replay 可以尽量保留“在 oracle 中本来存在的可选样本”。
def build_oracle_tripath_pool(
    oracle_df: pd.DataFrame,
    train_df: pd.DataFrame,
    cfg: ReplayConfig,
    pos_weights: Dict[str, float],
    neg_weights: Dict[str, float],
    full_signed_prior_df: pd.DataFrame,
    runtime_cfg: al.Config,
) -> pd.DataFrame:
    remaining_df = build_remaining_oracle_pool(oracle_df, train_df)
    path_a_df = build_oracle_path_A_pool(remaining_df, train_df, cfg, pos_weights, neg_weights, runtime_cfg)
    path_b_df = build_oracle_path_B_pool(remaining_df, path_a_df, train_df, cfg, pos_weights, neg_weights, full_signed_prior_df, runtime_cfg)
    if cfg.path_c_enabled:
        path_c_df = build_oracle_path_C_pool(
            remaining_df=remaining_df,
            path_a_df=path_a_df,
            path_b_df=path_b_df,
            cfg=cfg,
            repeat_seed=runtime_cfg.random_seed,
        )
    else:
        path_c_df = pd.DataFrame()
    parts = [x for x in [path_a_df, path_b_df, path_c_df] if not x.empty]
    if not parts:
        raise RuntimeError("oracle 三路径候选池为空")

    candidate_df = pd.concat(parts, axis=0, ignore_index=True)
    candidate_df = candidate_df.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    return candidate_df

# ============================================================
# 4. 单条 replay 轨迹
# ============================================================
# 执行单条 replay 轨迹。
# 对于给定 arm（tripath_no_prior 或 tripath_signed_prior），
# 这条函数会重复执行：
# 训练 -> 构造 oracle 候选池 -> 模型选批 -> 读取 oracle 真值 -> 回填训练集
# 从而模拟真实主动学习流程在多轮实验中的表现。
def run_single_arm_replay(
    oracle_df: pd.DataFrame,
    training_df: pd.DataFrame,
    cfg: ReplayConfig,
    repeat_id: int,
    arm_name: str,
    use_motif_prior: bool,
    pos_weights: Dict[str, float],
    neg_weights: Dict[str, float],
    full_signed_prior_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = training_df.copy().reset_index(drop=True)
    round_rows = []
    selection_rows = []
    initial_nir = int((train_df["class_label"] == "NIR").sum())
    cumulative_new_nir_hits = 0

    for round_idx in range(1, cfg.n_rounds + 1):
        runtime_cfg = build_runtime_cfg(cfg, use_motif_prior=use_motif_prior)
        runtime_cfg.random_seed = cfg.seed_start + repeat_id * 1000 + round_idx * 10 + (0 if arm_name == "tripath_no_prior" else 1)

        X_train = train_df[al.STANDARD_FEATURES].copy()
        committee = al.train_committee_model(X_train, train_df["class_label"], runtime_cfg)

        # 注意：replay 里的候选池不是从 4^10 全空间现生成，
        # 而是从“当前训练集之外的 oracle 剩余样本”中近似构造三路径候选，
        # 以保证后续每个被选点都能查到真实标签。
        candidate_df = build_oracle_tripath_pool(
            oracle_df=oracle_df,
            train_df=train_df,
            cfg=cfg,
            pos_weights=pos_weights,
            neg_weights=neg_weights,
            full_signed_prior_df=full_signed_prior_df,
            runtime_cfg=runtime_cfg,
        )
        if len(candidate_df) < cfg.recommend_k:
            raise RuntimeError(f"第 {round_idx} 轮 oracle 候选池只有 {len(candidate_df)} 条，不够选 {cfg.recommend_k} 条")
        # --------------------------------------------------------
        # 初始化本轮两个辅助回归分支的容器：
        # 1. 波长回归分支
        # 2. 亮度回归分支
        # 若某个分支本轮没有足够可用样本，则保持为 None。
        # --------------------------------------------------------
        wavelength_models = None
        brightness_models = None
        brightness_train_log = None
        # --------------------------------------------------------
        # 亮度回归分支：只用非 Dark 且有真实亮度的样本。
        # replay 里这里与真实主脚本保持一致，
        # 目的是让“别选到很暗样本”的约束在离线比较中也能体现。
        # --------------------------------------------------------
        if cfg.use_brightness_branch and (cfg.brightness_col in train_df.columns):
            bright_mask = (
                train_df["class_label"].ne("Dark")
                & pd.to_numeric(train_df[cfg.brightness_col], errors="coerce").notna()
            )
            if int(bright_mask.sum()) >= 10:
                brightness_train_raw = pd.to_numeric(
                    train_df.loc[bright_mask, cfg.brightness_col],
                    errors="coerce"
                ).to_numpy(dtype=np.float32)
                brightness_train_log = np.log1p(
                    np.clip(brightness_train_raw, a_min=0.0, a_max=None)
                )

                brightness_models = al.train_brightness_committee_model(
                    X=X_train.loc[bright_mask],
                    y_lli=train_df.loc[bright_mask, cfg.brightness_col],
                    cfg=runtime_cfg,
                )

        # --------------------------------------------------------
        # 波长回归分支：只用非 Dark 且有真实波长的样本。
        # replay 中保留这一路辅助分支，是为了让“靠近 780 nm 边界”的偏好
        # 在离线比较里也与真实流程一致。
        # --------------------------------------------------------
        if cfg.use_wavelength_branch and (cfg.wavelength_col in train_df.columns):
            lambda_mask = (
                train_df["class_label"].ne("Dark")
                & pd.to_numeric(train_df[cfg.wavelength_col], errors="coerce").notna()
            )
            if int(lambda_mask.sum()) >= 10:
                wavelength_models = al.train_wavelength_committee_model(
                    X=X_train.loc[lambda_mask],
                    y_lambda=train_df.loc[lambda_mask, cfg.wavelength_col],
                    cfg=runtime_cfg,
                )

        screened = al.screen_candidate_space(
            committee=committee,
            wavelength_models=wavelength_models,
            brightness_models=brightness_models,
            brightness_train_log=brightness_train_log,
            train_df=train_df,
            candidate_df=candidate_df,
            cfg=runtime_cfg,
            pos_weights=pos_weights,
            neg_weights=neg_weights,
            full_signed_prior_df=full_signed_prior_df,
        )
        selected_df = al.greedy_select_batch(screened, train_df, runtime_cfg)
        selected_df["true_is_nir"] = (selected_df["class_label"] == "NIR").astype(int)

        batch_nir_hits = int(selected_df["true_is_nir"].sum())
        batch_farred_fp = int((selected_df["class_label"] == "Far Red").sum())
        batch_dark_fp = int((selected_df["class_label"] == "Dark").sum())
        cumulative_new_nir_hits += batch_nir_hits
        cumulative_selected = round_idx * cfg.recommend_k
        cumulative_hit_rate = cumulative_new_nir_hits / float(cumulative_selected)
        nir_purity = batch_nir_hits / float(max(batch_nir_hits + batch_farred_fp + batch_dark_fp, 1))

        round_rows.append({
            "repeat_id": repeat_id,
            "arm": arm_name,
            "round": round_idx,
            "training_size_before": len(train_df),
            "candidate_pool_size": len(candidate_df),
            "batch_nir_hits": batch_nir_hits,
            "batch_farred_false_positive": batch_farred_fp,
            "batch_dark_false_positive": batch_dark_fp,
            "batch_nir_purity": nir_purity,
            "cumulative_new_nir_hits": cumulative_new_nir_hits,
            "cumulative_hit_rate": cumulative_hit_rate,
            "initial_nir_count": initial_nir,
        })

        export_df = selected_df.copy()
        export_df["repeat_id"] = repeat_id
        export_df["arm"] = arm_name
        export_df["round"] = round_idx
        export_df["training_size_before"] = len(train_df)
        selection_rows.append(export_df)

        # 这里不使用模型预测标签作为反馈，
        # 而是直接回到 oracle 中读取本轮被选序列的真实标签与真实属性，
        # 用来模拟真实实验完成后的“真值回填”。
        revealed = oracle_df.loc[oracle_df["Sequence"].isin(selected_df["Sequence"])].copy()
        # 将本轮 oracle 真值样本并回训练集。
        # 若某条序列此前已存在，则保留最后一条，保证最新回填记录覆盖旧记录。
        train_df = pd.concat([train_df, revealed], axis=0, ignore_index=True)
        train_df = train_df.drop_duplicates(subset=["Sequence"], keep="last").reset_index(drop=True)

    return pd.DataFrame(round_rows), pd.concat(selection_rows, axis=0, ignore_index=True)

# ============================================================
# 5. 汇总与 gating
# ============================================================
def compute_learning_curve_auc(values: Sequence[float]) -> float:
    y = np.asarray(values, dtype=float)
    if len(y) == 0:
        return 0.0
    if len(y) == 1:
        return float(y[0])
    return float(np.sum((y[:-1] + y[1:]) * 0.5))


def summarize_round_metrics(round_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (repeat_id, arm), sub in round_metrics_df.groupby(["repeat_id", "arm"], sort=True):
        sub = sub.sort_values("round")
        final = sub.iloc[-1]
        rows.append({
            "repeat_id": repeat_id,
            "arm": arm,
            "n_rounds_completed": int(len(sub)),
            "final_cumulative_new_nir_hits": int(final["cumulative_new_nir_hits"]),
            "final_cumulative_hit_rate": float(final["cumulative_hit_rate"]),
            "mean_batch_nir_purity": float(sub["batch_nir_purity"].mean()),
            "learning_curve_auc": compute_learning_curve_auc(sub["cumulative_new_nir_hits"].tolist()),
        })
    return pd.DataFrame(rows)

def summarize_across_repeats(per_repeat_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "final_cumulative_new_nir_hits",
        "final_cumulative_hit_rate",
        "mean_batch_nir_purity",
        "learning_curve_auc",
    ]
    for arm, sub in per_repeat_summary_df.groupby("arm", sort=True):
        row = {"arm": arm, "n_repeats": int(len(sub))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

# 根据 replay 的总体结果，自动生成“下一轮真实流程是否建议打开 signed prior”的 gating 结论。
# 当前规则很明确：
# 1. signed prior 的 learning_curve_auc_mean 相对 no_prior 提升达到阈值；
# 2. 且最终累计 NIR 命中不更差；
# 同时满足时，recommended_use_prior 才会被写成 True。
def make_prior_gate_decision(overall_summary_df: pd.DataFrame, cfg: ReplayConfig) -> Dict[str, object]:
    arm_to_row = {row["arm"]: row for _, row in overall_summary_df.iterrows()}
    if "tripath_no_prior" not in arm_to_row or "tripath_signed_prior" not in arm_to_row:
        return {
            "gate_metric": "learning_curve_auc_mean",
            "recommended_use_prior": False,
            "reason": "缺少必要的 arm，无法做 gating。",
        }

    no_row = arm_to_row["tripath_no_prior"]
    sg_row = arm_to_row["tripath_signed_prior"]
    no_auc = float(no_row["learning_curve_auc_mean"])
    sg_auc = float(sg_row["learning_curve_auc_mean"])
    no_hits = float(no_row["final_cumulative_new_nir_hits_mean"])
    sg_hits = float(sg_row["final_cumulative_new_nir_hits_mean"])
    auc_improve_ratio = (sg_auc - no_auc) / max(abs(no_auc), 1e-12)
    hits_not_worse = sg_hits >= no_hits
    recommended = (auc_improve_ratio >= cfg.gate_auc_improvement_threshold) and hits_not_worse
    return {
        "gate_metric": "learning_curve_auc_mean",
        "gate_auc_improvement_threshold": cfg.gate_auc_improvement_threshold,
        "tripath_no_prior_auc_mean": no_auc,
        "tripath_signed_prior_auc_mean": sg_auc,
        "auc_improve_ratio": float(auc_improve_ratio),
        "tripath_no_prior_final_hits_mean": no_hits,
        "tripath_signed_prior_final_hits_mean": sg_hits,
        "hits_not_worse": bool(hits_not_worse),
        "recommended_use_prior": bool(recommended),
        "rule_explanation": "当 tripath_signed_prior 在 learning_curve_auc_mean 上相对 tripath_no_prior 提升达到阈值，且最终累计 NIR 命中不更差时，推荐真实流程打开 signed prior。",
    }

# ============================================================
# 6. 导出
# ============================================================
def export_outputs(
    out_dir: Path,
    cfg: ReplayConfig,
    round_metrics_df: pd.DataFrame,
    selections_df: pd.DataFrame,
    per_repeat_summary_df: pd.DataFrame,
    overall_summary_df: pd.DataFrame,
    gate_decision: Dict[str, object],
    signed_prior_path: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    round_metrics_df.to_csv(out_dir / "round_metrics.csv", index=False, encoding="utf-8-sig")
    selections_df.to_csv(out_dir / "selected_sequences_by_round.csv", index=False, encoding="utf-8-sig")
    per_repeat_summary_df.to_csv(out_dir / "per_repeat_summary.csv", index=False, encoding="utf-8-sig")
    overall_summary_df.to_csv(out_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "prior_gate_decision.json").write_text(json.dumps(gate_decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "replay_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""三路径 signed prior 离线回放说明
=============================

当前训练集：{Path(cfg.training_csv).name}
当前 oracle：{Path(cfg.oracle_csv).name}
当前 signed prior 文件：{Path(signed_prior_path).name}

本目录记录的是离线回放结果，不是直接用于实验上板的真实推荐结果。

本次对照 arms：
- tripath_no_prior
- tripath_signed_prior

主要指标：
- final_cumulative_new_nir_hits
- learning_curve_auc
- mean_batch_nir_purity

prior_gate_decision.json 会被真实推荐脚本读取，
用来自动决定是否在下一轮真实实验中打开 signed prior。
"""
    (out_dir / "README_results.txt").write_text(readme, encoding="utf-8")


# ============================================================
# 7. 主流程
# ============================================================
def main(cfg: ReplayConfig):
    oracle_path = resolve_existing_path(Path(cfg.oracle_csv), PROJECT_ROOT / Path(cfg.oracle_csv).name)
    training_path = resolve_existing_path(Path(cfg.training_csv), PROJECT_ROOT / Path(cfg.training_csv).name)

    print("=" * 72)
    print("开始执行：ALL2962 oracle 上的三路径 signed prior 离线回放")
    print("=" * 72)

    print("[1/6] 读取 oracle 与当前训练集 ...")
    oracle_df = add_digits_column(al.load_and_clean_data(str(oracle_path)))
    training_df = add_digits_column(al.load_and_clean_data(str(training_path)))
    _ = al.select_standard_144_features(oracle_df)
    _ = al.select_standard_144_features(training_df)

    missing = set(training_df["Sequence"]) - set(oracle_df["Sequence"])
    if missing:
        raise ValueError(f"当前训练集里有序列不在 ALL2962 oracle 中，示例：{list(sorted(missing))[:10]}")

    print(f"    oracle 样本数：{len(oracle_df)}")
    print(f"    当前训练集样本数：{len(training_df)}")
    print(f"    当前标签分布：{training_df['class_label'].value_counts().to_dict()}")

    print("[2/6] 读取 signed prior ...")
    if cfg.signed_prior_csv:
        signed_prior_path = resolve_existing_path(Path(cfg.signed_prior_csv), PROJECT_ROOT / Path(cfg.signed_prior_csv).name)
    else:
        signed_prior_path = resolve_existing_path(Path(al.infer_signed_importance_csv_by_training_size(len(training_df), build_runtime_cfg(cfg, True))))
    pos_weights, neg_weights, full_signed_prior_df = al.load_signed_motif_prior_from_csv(str(signed_prior_path), cfg.motif_top_k)
    print(f"    signed prior 文件：{signed_prior_path}")
    print(f"    promote_NIR motifs：{len(pos_weights)}")
    print(f"    suppress_NIR motifs：{len(neg_weights)}")

    print("[3/6] 开始 replay ...")
    all_round_metrics = []
    all_selections = []
    # 两个对照臂：
    # 1. tripath_no_prior：三路径候选生成，但最终打分不启用 signed motif prior
    # 2. tripath_signed_prior：三路径候选生成，并启用 signed motif prior
    arm_specs = [
        ("tripath_no_prior", False),
        ("tripath_signed_prior", True),
    ]
    for repeat_id in range(cfg.n_repeats):
        print(f"    >>> repeat {repeat_id + 1}/{cfg.n_repeats}")
        for arm_name, use_prior in arm_specs:
            round_df, select_df = run_single_arm_replay(
                oracle_df=oracle_df,
                training_df=training_df,
                cfg=cfg,
                repeat_id=repeat_id,
                arm_name=arm_name,
                use_motif_prior=use_prior,
                pos_weights=pos_weights,
                neg_weights=neg_weights,
                full_signed_prior_df=full_signed_prior_df,
            )
            all_round_metrics.append(round_df)
            all_selections.append(select_df)

    round_metrics_df = pd.concat(all_round_metrics, axis=0, ignore_index=True)
    selections_df = pd.concat(all_selections, axis=0, ignore_index=True)

    print("[4/6] 汇总指标 ...")
    per_repeat_summary_df = summarize_round_metrics(round_metrics_df)
    overall_summary_df = summarize_across_repeats(per_repeat_summary_df)

    print("[5/6] 生成 gating 决策并导出 ...")
    gate_decision = make_prior_gate_decision(overall_summary_df, cfg)
    out_dir = Path(cfg.output_root)
    export_outputs(
        out_dir=out_dir,
        cfg=cfg,
        round_metrics_df=round_metrics_df,
        selections_df=selections_df,
        per_repeat_summary_df=per_repeat_summary_df,
        overall_summary_df=overall_summary_df,
        gate_decision=gate_decision,
        signed_prior_path=str(signed_prior_path),
    )

    print("[6/6] 运行完成")
    print(overall_summary_df.to_string(index=False))
    print("prior_gate_decision.json 内容：")
    print(json.dumps(gate_decision, ensure_ascii=False, indent=2))

# 下面这组参数是“直接运行本脚本”时使用的默认 replay 配置。
# 如果你只是临时做对照实验，通常只需要改 training_csv、n_rounds、recommend_k、n_repeats 这几项。
if __name__ == "__main__":
    start_time = time.time()
    print("开始执行 tripath signed prior 离线回放 ...")

    cfg = ReplayConfig(
        oracle_csv=str(PROJECT_ROOT / "ALL2998.csv"),
        training_csv=str(PROJECT_ROOT / "Iteration3_156.csv"),
        output_root=str(PROJECT_ROOT / "offline_replay_outputs_tripath_iter4_156"),
        n_rounds=10,
        recommend_k=12,
        n_repeats=30,
        seed_start=42,
        initial_training_size=120,
        committee_size=25,
        n_jobs=-1,
        use_gpu_backend=False,
        gpu_device="cuda",
        torch_logreg_max_iter=120,
        torch_reg_max_iter=120,
        feature_subsample_ratio=0.80,
        logreg_C=0.30,
        logreg_l1_ratio=0.30,
        logreg_max_iter=6000,
        w_target=0.375,
        w_uncertainty=0.20,
        w_diversity=0.30,
        w_motif=0.125,
        diversity_hamming_weight=0.50,
        diversity_staple_weight=0.50,
        staple_diversity_top_k=30,
        motif_top_k=15,
        motif_negative_scale=0.70,
        path_a_hamming_radius=2,
        path_a_include_farred_seeds=True,
        path_a_pool_size=800,
        path_b_pool_size=1200,
        path_b_min_cg_count=4,
        path_c_enabled=True,
        path_c_pool_size=300,

        use_wavelength_branch=True,
        wavelength_col="main_peak_lambda_nm",
        wavelength_reg_alpha=0.01,
        wavelength_reg_l1_ratio=0.30,
        wavelength_reg_max_iter=20000,
        wavelength_boundary_nm=780.0,
        wavelength_sigmoid_tau=15.0,
        w_wavelength=0.10,
        use_brightness_branch=True,
        brightness_col="main_peak_lli_raw",
        brightness_reg_alpha=0.01,
        brightness_reg_l1_ratio=0.30,
        brightness_reg_max_iter=20000,
        brightness_floor_quantile=0.20,
        brightness_good_quantile=0.80,
        brightness_sigmoid_tau=0.35,
        w_brightness=0.10,
        brightness_hard_risk_threshold=0.85,
        gate_auc_improvement_threshold=0.05,
    )

    main(cfg)

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"离线回放完成，总耗时：{elapsed:.2f} 秒")
    print("-" * 70)
