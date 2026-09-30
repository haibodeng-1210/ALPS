from __future__ import annotations

import json
import shutil
import time
import warnings
from dataclasses import dataclass, asdict
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. 全局常量
# ============================================================
DNA_BASES = "ACGT"
BASE_TO_INT = {b: i for i, b in enumerate(DNA_BASES)}
INT_TO_BASE = np.array(list(DNA_BASES))
SEQ_LEN = 10
TOTAL_SEQUENCE_SPACE = 4 ** SEQ_LEN
CLASS_ORDER = ["Green", "Red", "Far Red", "NIR", "Dark"]

# 144 个标准 staple features：X_mY, m=0..8
STANDARD_FEATURES = [
    f"{x}_{m}{y}"
    for m in range(SEQ_LEN - 1)
    for x in DNA_BASES
    for y in DNA_BASES
]

# ============================================================
# 1. 参数配置
# ============================================================
@dataclass
class Config:
    # ---------- 输入输出 ----------
    # input_csv：当前这一轮真正用于训练和推荐的训练集。
    #            第一轮通常是 Initial120.csv，后续轮次应改成最新 verified 训练集。
    # output_root：主流程输出目录，推荐结果、候选池、README 和 run_summary 都会放这里。
    # banned_sequences_path：禁用序列表，用来排除已测过、不想重复推荐或人工屏蔽的序列。
    # initial_training_size：这条新流程最初起步时的样本数。
    #                        它用于推断“第几次迭代”和 signed prior 文件名，
    #                        后续轮次不要随着训练集变大而改动。
    input_csv: str = "Initial120.csv"
    output_root: str = "active_learning_outputs_tripath"
    banned_sequences_path: str = "../Old_workflow/merged_unique_sequences.csv"
    initial_training_size: int = 120

    # ---------- 主动学习设置 ----------
    # recommend_k：每轮最终推荐多少条序列进入实验验证。
    # min_hamming_distance：最终 batch 内部额外的最小 Hamming 距离硬约束。
    # 当前设为 0，表示不额外启用这层硬限制，主要依赖 diversity 项去分散候选。
    recommend_k: int = 14
    min_hamming_distance: int = 0

    # ---------- 候选池模式 ----------
    # 当前三路径主流程使用 tripath。
    # full_space 仅保留用于与旧式全空间扫描做对照。
    candidate_mode: str = "tripath"  # "tripath" or "full_space"

    # 路径 A：局部开发池（从已知 NIR / Far Red 好样本附近做局部突变）
    # path_a_include_farred_seeds=True 表示允许把 Far Red 也作为边界邻域 seed，
    # 有助于从 Far Red 往 NIR 边界继续推进。
    # path_a_max_three_mutations_per_seed 控制每个 seed 随机补多少个 3 位突变，
    # 用来限制 3 位组合爆炸。
    # path_a_pool_size 是路径 A 在预筛后“最多保留多少条”的上限，
    # 不是保证一定会生成/保留到这么多条。
    path_a_enabled: bool = True
    path_a_include_farred_seeds: bool = True
    path_a_max_three_mutations_per_seed: int = 8
    path_a_pool_size: int = 20000

    # 路径 B：de novo 探索池（随机起点 + signed prior hill climbing）
    # path_b_random_starts 控制会生成多少条随机起点轨迹；
    # path_b_hillclimb_steps 控制每条轨迹沿 signed prior 走多少步；
    # path_b_min_cg_count 是路径 B 的 chemistry hard constraint：
     # 不仅随机起点要满足它，hill-climbing 过程和最终进入 B 池的候选也要满足它。
     # 这样路径 B 才真正代表“受化学经验约束的 de novo 探索”，而不是纯随机探索。
    # path_b_pool_size 同样只是预筛后的保留上限，不保证最终 unique 候选一定达到这个数。
    path_b_enabled: bool = True
    path_b_random_starts: int = 10000
    path_b_hillclimb_steps: int = 12
    path_b_min_cg_count: int = 4
    path_b_pool_size: int = 20000

    # 路径 C：随机覆盖池（从 4^10 空间中均匀随机抽样）
    # 这一路径不带局部突变，也不做 hill climbing，
    # 作用是给整个候选池补充更无偏的随机覆盖，
    # 防止路径 A 和路径 B 过度围绕现有先验与高分局部区域打转。
    # path_c_pool_size 也是“最多尝试保留多少条”的上限。
    path_c_enabled: bool = True
    path_c_pool_size: int = 5000

    # 三条路径合并后做 staple-space 去近邻压缩，避免最终候选里出现大量机制上近重复的序列
    # enable_candidate_pruning=True 时，会在“重要 staple 特征子空间”里做 greedy pruning。
    # candidate_prune_top_k_features 表示只看 prior 排名前多少个重要特征；
    # candidate_prune_min_staple_distance 表示在这个子空间里的最小归一化 L1 距离阈值。
    # 这一层可能会显著缩小最终候选池数量，因此这里的参数会直接影响候选池大小。
    enable_candidate_pruning: bool = True
    candidate_prune_top_k_features: int = 40
    candidate_prune_min_staple_distance: float = 0.05

    # ---------- 五折 LDA 委员会设置 ----------
    base_model_name: str = "anova16_shrinkage_lda"
    committee_size: int = 5
    committee_inner_folds: int = 4
    classifier_anova_k: int = 16
    classifier_lda_shrinkage: float = 0.50
    temperature_min: float = 0.05
    temperature_max: float = 20.0
    n_jobs: int = -1
    use_gpu_backend: bool = False
    # Legacy configuration fields retained for old JSON compatibility; GPU execution is rejected.
    gpu_device: str = "cuda"
    torch_logreg_max_iter: int = 120
    torch_reg_max_iter: int = 120
    random_seed: int = 42
    per_class_sample_size: Optional[int] = None
    feature_subsample_ratio: float = 0.80

    # 旧分类器参数保留用于读取已有配置；五折 LDA 委员会不使用这些字段。
    logreg_C: float = 0.30
    logreg_l1_ratio: float = 0.30
    logreg_max_iter: int = 6000

    # ---------- 全空间扫描时的 batch 大小 ----------
    batch_size: int = 50000

    # ---------- 新采集函数权重 ----------
    # 最终总分不是单一的 P(NIR)，而是多项加权：
    #   w_target      * safe_nir_score
    # + w_uncertainty * uncertainty_norm
    # + w_diversity   * diversity_norm
    # + w_motif       * motif_prior_signed
    # + w_wavelength  * wavelength_bonus
    # + w_brightness  * brightness_bonus
    #
    # 其中 safe_nir_score 不是简单的 P_NIR_mean，
    # 而是“像 NIR，同时又明确不像 Far Red 和 Dark”的安全版本目标分。

    w_target: float = 0.375
    w_uncertainty: float = 0.20
    w_diversity: float = 0.30
    w_motif: float = 0.125

    # ---------- diversity 混合权重 ----------
    # 新版 diversity 不再只看 Hamming，而是：
    #   mixed_diversity = α * hamming_div + (1-α) * staple_div
    diversity_hamming_weight: float = 0.50
    diversity_staple_weight: float = 0.50
    staple_diversity_top_k: int = 30

    # ---------- motif prior ----------
    # use_motif_prior：是否启用 signed motif prior。
    # motif_prior_csv：若手工指定 prior 文件，则优先使用它；否则按当前训练集大小自动推断。
    # motif_prior_top_k：正向/负向各取前多少个 motif 进入 prior。
    # motif_negative_scale：负向 motif 的惩罚强度，越大表示越强调“坏 motif 要避开”。
    # motif_prior_features：预留接口，当前主流程不依赖它作为核心输入。
    use_motif_prior: bool = True
    motif_prior_csv: Optional[str] = None
    motif_prior_top_k: int = 15
    motif_negative_scale: float = 0.70
    motif_prior_features: Optional[Union[List[str], Dict[str, float]]] = None


    # ---------- 波长回归分支 ----------
    # 这个分支不替代五分类主模型，而是作为边界细化辅助项：
    # 它回答“这条序列预测离 780 nm 的 NIR 边界还有多远”。
    use_wavelength_branch: bool = True
    wavelength_col: str = "main_peak_lambda_nm"

    # 回归模型参数：先用 ElasticNet，保持和当前主模型风格统一、修改成本最低
    wavelength_reg_alpha: float = 0.01
    wavelength_reg_l1_ratio: float = 0.30
    wavelength_reg_max_iter: int = 20000

    # 780 nm 作为 NIR 边界
    wavelength_boundary_nm: float = 780.0

    # 用 sigmoid 把“离边界多近”映射成 [0,1] 加成项。
    # tau 越小，边界越陡；tau 越大，边界越平缓。
    wavelength_sigmoid_tau: float = 15.0

    # 最终总分中，波长边界加成项所占的权重
    w_wavelength: float = 0.10

    # ---------- 亮度回归分支 ----------
    # 这里的亮度列使用主峰 LLI（或主峰面积/积分强度），允许缺失。
    # 亮度分支不替代五分类中的 Dark，而是做两件事：
    # 1. 给出 brightness_bonus：更亮的样本得分更高；
    # 2. 给出 brightness_dim_risk：预测很暗的样本在最终 batch 里被过滤。
    use_brightness_branch: bool = True
    brightness_col: str = "main_peak_lli_raw"

    # 亮度分支同样先用 ElasticNet，保持和波长分支风格一致
    brightness_reg_alpha: float = 0.01
    brightness_reg_l1_ratio: float = 0.30
    brightness_reg_max_iter: int = 20000

    # 亮度奖励/暗风险都基于 log1p(LLI) 后的训练集分位数来定义，
    # 避免直接用原始长尾亮度值导致不稳定。
    brightness_floor_quantile: float = 0.20
    brightness_good_quantile: float = 0.80
    brightness_sigmoid_tau: float = 0.35

    # 最终总分中亮度奖励项的权重
    w_brightness: float = 0.10

    # 如果 dim_risk 高于这个阈值，则认为“太暗”，不进入最终 batch。
    brightness_hard_risk_threshold: float = 0.85

    # ---------- 离线 gating ----------
    # prior_gate_json：若提供，则优先读取离线 replay 生成的 gating 结论，
    # 决定本轮真实推荐是否启用 motif prior。
    # fallback_use_motif_prior_when_gate_missing=True 表示：
    # 若 gating 文件不存在，则退回 use_motif_prior 这个手工默认值。
    prior_gate_json: Optional[str] = None
    fallback_use_motif_prior_when_gate_missing: bool = True

    # ---------- 导出 ----------
    export_top_n_pool: int = 200


# ============================================================
# 2. 通用工具函数
# ============================================================
def normalize_sequence(seq: str) -> str:
    return str(seq).strip().upper().replace(" ", "")

def is_valid_dna_10mer(seq: str) -> bool:
    seq = normalize_sequence(seq)
    return len(seq) == SEQ_LEN and all(ch in DNA_BASES for ch in seq)

def normalize_feature_name(name: str) -> str:
    name = str(name).strip()
    left, right = name.split("_", 1)
    left = left.upper()
    middle = right[:-1]
    last = right[-1].upper()
    feat = f"{left}_{middle}{last}"
    if feat not in STANDARD_FEATURES:
        raise ValueError(f"非法特征名：{name} -> {feat}")
    return feat

def chinese_iteration_name(iter_idx: int) -> str:
    table = {
        1: "第一次迭代", 2: "第二次迭代", 3: "第三次迭代", 4: "第四次迭代",
        5: "第五次迭代", 6: "第六次迭代", 7: "第七次迭代", 8: "第八次迭代",
        9: "第九次迭代", 10: "第十次迭代",
    }
    return table.get(iter_idx, f"第{iter_idx}次迭代")

# 根据“当前训练集大小 - 初始训练集大小”，推断这是第几次迭代。
# 这里默认每一轮都会新加入 recommend_k 条已验证样本，
# 因此 recommend_k 必须与真实每轮实验条数保持一致。
def infer_iteration_index(n_samples: int, cfg: Config) -> int:
    delta = max(0, n_samples - cfg.initial_training_size)
    return delta // cfg.recommend_k + 1

# 根据当前训练集大小，自动推断本轮应该读取哪一份 signed importance 文件。
# 若当前样本数仍等于初始训练集大小，则读取 small；
# 否则读取 train{当前样本数} 对应的文件。
def infer_signed_importance_csv_by_training_size(n_samples: int, cfg: Config) -> str:
    base_dir = Path("全集重要性分析")
    if n_samples == cfg.initial_training_size:
        return str(base_dir / "nir_feature_importance_signed_small.csv")
    return str(base_dir / f"nir_feature_importance_signed_train{n_samples}.csv")


def resolve_effective_use_motif_prior(cfg: Config) -> bool:
    if not cfg.prior_gate_json:
        return bool(cfg.use_motif_prior)

    gate_path = Path(cfg.prior_gate_json)
    if not gate_path.exists():
        if cfg.fallback_use_motif_prior_when_gate_missing:
            print(f"[提示] 未找到 prior gate 文件：{gate_path}，退回到 cfg.use_motif_prior={cfg.use_motif_prior}")
            return bool(cfg.use_motif_prior)
        raise FileNotFoundError(f"未找到 prior gate 文件：{gate_path}")

    gate_info = json.loads(gate_path.read_text(encoding="utf-8"))
    decision = bool(gate_info.get("recommended_use_prior", cfg.use_motif_prior))
    print(f"[提示] 根据离线 replay 的 gating 结果，本轮 use_motif_prior = {decision}")
    return decision


def encode_sequence_to_int(seq: str) -> int:
    code = 0
    for ch in normalize_sequence(seq):
        code = code * 4 + BASE_TO_INT[ch]
    return code


def decode_int_to_sequence(code: int) -> str:
    digits = []
    x = int(code)
    for _ in range(SEQ_LEN):
        digits.append(x % 4)
        x //= 4
    digits = digits[::-1]
    return "".join(INT_TO_BASE[np.asarray(digits, dtype=np.uint8)])


def sequence_to_digits(seq: str) -> List[int]:
    """
    把一条 10-mer DNA 序列转换成长度为 10 的整数列表。

    编码规则与全脚本其它地方保持一致：
    A=0, C=1, G=2, T=3。

    这个函数的主要用途是：
    1. 在候选池打分阶段，把 DataFrame 中的字符串序列快速转成整数表示；
    2. 后续直接拿这些整数位表示去计算 Hamming 距离，避免重复写列表推导式；
    3. 让主脚本和 offline replay 脚本在“序列 -> digits”的处理中保持统一。

    """
    seq = normalize_sequence(seq)
    if not is_valid_dna_10mer(seq):
        raise ValueError(f"非法 10-mer 序列：{seq}")
    return [BASE_TO_INT[ch] for ch in seq]


def decode_codes_to_sequences(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint32)
    powers = (4 ** np.arange(SEQ_LEN - 1, -1, -1, dtype=np.uint32))
    digits = ((codes[:, None] // powers[None, :]) % 4).astype(np.uint8)
    return np.array(["".join(INT_TO_BASE[row]) for row in digits], dtype=object)

def make_output_dir(cfg: Config, n_samples: int) -> Path:
    iteration_idx = infer_iteration_index(n_samples, cfg)
    folder_name = f"{chinese_iteration_name(iteration_idx)}_训练集{n_samples}条_三路径"
    out_dir = Path(cfg.output_root) / folder_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# ============================================================
# 3. 数据读取与清洗
# ============================================================
def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    alias_map = {}
    if "Sequence" not in df.columns and "sequence" in df.columns:
        alias_map["sequence"] = "Sequence"
    if "class_label" not in df.columns and "target" in df.columns:
        alias_map["target"] = "class_label"
    if alias_map:
        df = df.rename(columns=alias_map)

    required = {"Sequence", "class_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列：{missing}")

    df = df.copy()
    df["Sequence"] = df["Sequence"].astype(str).map(normalize_sequence)
    df["class_label"] = df["class_label"].astype(str).str.strip()

    mask_valid = df["Sequence"].map(is_valid_dna_10mer)
    if (~mask_valid).any():
        print(f"[警告] 删除非法序列 {int((~mask_valid).sum())} 条")
    df = df.loc[mask_valid].copy()

    mask_label = df["class_label"].isin(CLASS_ORDER)
    if (~mask_label).any():
        print(f"[警告] 删除非法标签 {int((~mask_label).sum())} 条")
    df = df.loc[mask_label].copy()

    if df.duplicated(subset=["Sequence"], keep="last").any():
        print(f"[警告] 按 Sequence 去重，保留最后一条")
    df = df.drop_duplicates(subset=["Sequence"], keep="last").reset_index(drop=True)

    present_features = [feature for feature in STANDARD_FEATURES if feature in df.columns]
    if not present_features:
        generated = pd.DataFrame(
            [sequence_to_144_feature_dict(sequence) for sequence in df["Sequence"]],
            index=df.index,
        )
        df = pd.concat([df, generated], axis=1)
    elif len(present_features) != len(STANDARD_FEATURES):
        missing_features = [feature for feature in STANDARD_FEATURES if feature not in df.columns]
        raise ValueError(
            "输入文件只包含部分标准 staple 特征，无法确认特征定义；示例缺失："
            + ", ".join(missing_features[:10])
        )
    return df

# 从训练表中显式抽出标准 144 个 staple 特征作为委员会主输入。
# 额外列（如波长、亮度）不会进入五分类委员会输入，
# 但仍会保留在原始 DataFrame 中，供辅助回归分支和导出使用。
def select_standard_144_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    present = [c for c in STANDARD_FEATURES if c in df.columns]
    missing = [c for c in STANDARD_FEATURES if c not in df.columns]
    if missing:
        raise ValueError("输入文件缺少标准 144 个特征中的一部分，示例缺失：" + ", ".join(missing[:10]))
    ignored = [c for c in df.columns if c not in (["Sequence", "class_label"] + STANDARD_FEATURES)]
    return df[present].copy(), present, ignored

# 读取禁用序列文件。
# 支持 txt / csv / xlsx / xls。
# 这些序列会在候选生成后被统一排除，避免重复推荐或重复实验。
def load_banned_sequences(path_str: str) -> List[str]:
    path_str = str(path_str).strip()
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"禁用序列文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        seqs = [normalize_sequence(line) for line in path.read_text(encoding="utf-8").splitlines()]
    elif suffix == ".csv":
        tmp = pd.read_csv(path)
        col = "Sequence" if "Sequence" in tmp.columns else tmp.columns[0]
        seqs = tmp[col].astype(str).map(normalize_sequence).tolist()
    elif suffix in {".xlsx", ".xls"}:
        tmp = pd.read_excel(path)
        col = "Sequence" if "Sequence" in tmp.columns else tmp.columns[0]
        seqs = tmp[col].astype(str).map(normalize_sequence).tolist()
    else:
        raise ValueError(f"暂不支持的禁用序列文件格式：{suffix}")
    return sorted(set([s for s in seqs if is_valid_dna_10mer(s)]))


# ============================================================
# 4. 序列 -> 144 特征
# ============================================================
def sequence_to_144_feature_dict(seq: str) -> Dict[str, int]:
    seq = normalize_sequence(seq)
    out = {}
    for m in range(SEQ_LEN - 1):
        for x in DNA_BASES:
            for y in DNA_BASES:
                cnt = 0
                for i in range(SEQ_LEN - m - 1):
                    if seq[i] == x and seq[i + m + 1] == y:
                        cnt += 1
                out[f"{x}_{m}{y}"] = cnt
    return out

def encoded_batch_to_feature_matrix(encoded_batch: np.ndarray) -> np.ndarray:
    encoded_batch = np.asarray(encoded_batch, dtype=np.uint8)
    B = encoded_batch.shape[0]
    X = np.zeros((B, len(STANDARD_FEATURES)), dtype=np.float32)
    col_start = 0
    pair_code_values = np.arange(16, dtype=np.uint8)
    for m in range(SEQ_LEN - 1):
        left = encoded_batch[:, : SEQ_LEN - m - 1]
        right = encoded_batch[:, m + 1 :]
        pair_code = (left * 4 + right).astype(np.uint8)
        counts = (pair_code[:, :, None] == pair_code_values[None, None, :]).sum(axis=1)
        X[:, col_start:col_start + 16] = counts.astype(np.float32)
        col_start += 16
    return X


def sequence_list_to_candidate_df(seq_list: Sequence[str]) -> pd.DataFrame:
    """
    由于三路径候选池通常只有几千到几万条序列，而不是全空间 100 多万条，
    因此这里直接逐条计算 144 维特征仍然是可接受的。
    并且这里会在路径内部先按 Sequence 去重，避免同一路径重复生成的序列反复进入后续流程。
    """
    rows = []
    seen = set()
    for seq in seq_list:
        seq = normalize_sequence(seq)
        if not is_valid_dna_10mer(seq):
            continue
        if seq in seen:
            continue
        seen.add(seq)
        feat_dict = sequence_to_144_feature_dict(seq)
        rows.append({"Sequence": seq, **feat_dict})
    if not rows:
        return pd.DataFrame(columns=["Sequence"] + STANDARD_FEATURES)
    return pd.DataFrame(rows)


# ============================================================
# 5. motif prior：从 signed importance 文件中同时读取正向和负向 motif
# ============================================================
def parse_feature_theoretical_max(feature_name: str) -> int:
    middle = feature_name.split("_")[1]
    m = int(middle[:-1])
    return SEQ_LEN - m - 1



def load_signed_motif_prior_from_csv(csv_path: str, top_k: int) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    """
    读取 signed importance 文件，返回三样东西：

    1. promote_NIR 的 top-k 权重字典；
    2. suppress_NIR 的 top-k 权重字典；
    3. 完整排序表 full_df。

    注意：
    full_df 不只是给 prior 用，
    后面做 staple-space diversity、candidate pruning 和重要特征子空间分析时也会复用。
    """
    df = pd.read_csv(csv_path)
    required_cols = {"feature", "signed_importance", "abs_importance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"signed prior 文件缺少必要列：{missing}")

    df = df.copy()
    df["feature"] = df["feature"].map(normalize_feature_name)
    df = df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    pos_df = df[df["signed_importance"] > 0].sort_values("signed_importance", ascending=False).head(top_k)
    neg_df = df[df["signed_importance"] < 0].sort_values("signed_importance", ascending=True).head(top_k)

    pos_weights = {row["feature"]: float(abs(row["signed_importance"])) for _, row in pos_df.iterrows()}
    neg_weights = {row["feature"]: float(abs(row["signed_importance"])) for _, row in neg_df.iterrows()}
    return pos_weights, neg_weights, df



def build_signed_prior_weights(cfg: Config, n_samples: int) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame, str]:
    if not cfg.use_motif_prior:
        empty = pd.DataFrame(columns=["feature", "signed_importance", "abs_importance"])
        return {}, {}, empty, ""

    if cfg.motif_prior_csv:
        csv_path = cfg.motif_prior_csv
    else:
        csv_path = infer_signed_importance_csv_by_training_size(n_samples, cfg)

    pos_weights, neg_weights, full_df = load_signed_motif_prior_from_csv(csv_path, cfg.motif_prior_top_k)
    return pos_weights, neg_weights, full_df, csv_path



def motif_component_score(X_batch: np.ndarray, motif_weights: Dict[str, float]) -> np.ndarray:
    if not motif_weights:
        return np.zeros(X_batch.shape[0], dtype=np.float32)
    feature_to_index = {f: i for i, f in enumerate(STANDARD_FEATURES)}
    weighted_sum = np.zeros(X_batch.shape[0], dtype=np.float32)
    total_weight = 0.0
    for feat_name, weight in motif_weights.items():
        idx = feature_to_index[feat_name]
        theoretical_max = parse_feature_theoretical_max(feat_name)
        weighted_sum += float(weight) * (X_batch[:, idx] / float(theoretical_max))
        total_weight += float(weight)
    return np.zeros(X_batch.shape[0], dtype=np.float32) if total_weight <= 0 else np.clip(weighted_sum / total_weight, 0.0, 1.0).astype(np.float32)



def signed_motif_prior_score(X_batch: np.ndarray, pos_weights: Dict[str, float], neg_weights: Dict[str, float], cfg: Config) -> np.ndarray:
    """
    旧版逻辑只有“正向 motif 奖励”。
    新版改成 signed prior：
        promote_NIR - lambda * suppress_NIR
    这样当前的先验不再是单边鼓励，而是同时表达“哪些 pattern 值得靠近、哪些应该避开”。
    """
    pos_score = motif_component_score(X_batch, pos_weights)
    neg_score = motif_component_score(X_batch, neg_weights)
    signed_score = pos_score - cfg.motif_negative_scale * neg_score
    return np.clip(signed_score, -1.0, 1.0).astype(np.float32)


# ============================================================
# 6. 基础模型与委员会
# ============================================================












def _require_verified_backend(cfg: Config):
    if cfg.use_gpu_backend:
        raise ValueError(
            "The fixed ANOVA-16 + shrinkage LDA committee uses CPU/scikit-learn; "
            "set use_gpu_backend=False."
        )


def _fit_anova_indices(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= int(k) <= X.shape[1]:
        raise ValueError(f"classifier_anova_k 必须位于 [1, {X.shape[1]}]，当前为 {k}")
    scores = np.full(X.shape[1], -np.inf, dtype=float)
    nonconstant = np.var(X, axis=0) > 0
    if nonconstant.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores[nonconstant] = np.nan_to_num(
                f_classif(X[:, nonconstant], y)[0], nan=-np.inf
            )
    return np.argsort(scores)[::-1][: int(k)]


def build_base_estimator(cfg: Config):
    _require_verified_backend(cfg)
    if cfg.base_model_name.lower().strip() != "anova16_shrinkage_lda":
        raise ValueError("当前五分类委员会只支持 anova16_shrinkage_lda")
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage=float(cfg.classifier_lda_shrinkage),
                priors=None,
            ),
        ),
    ])



def _get_fitted_model_classes(model) -> List[str]:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return list(last.classes_)
    raise AttributeError("无法从模型中读取 classes_")


@dataclass
class CommitteeInfo:
    class_counts: Dict[str, int]
    committee_size: int
    base_model_name: str
    cv_folds: int
    inner_calibration_folds: int
    anova_k: int
    lda_shrinkage: float
    temperatures: List[float]


class CommitteeMemberWrapper:
    def __init__(
        self,
        model,
        selected_feature_indices: np.ndarray,
        target_class_order: Sequence[str],
        temperature: float,
    ):
        self.model = model
        self.selected_feature_indices = np.asarray(selected_feature_indices, dtype=int)
        self.target_class_order = list(target_class_order)
        self.model_classes = _get_fitted_model_classes(model)
        self.temperature = float(temperature)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_sub = X[:, self.selected_feature_indices]
        raw_scores = np.asarray(self.model.decision_function(X_sub), dtype=float)
        if raw_scores.ndim != 2:
            raise ValueError("五分类 LDA 应输出二维 decision scores")
        aligned_scores = np.empty(
            (raw_scores.shape[0], len(self.target_class_order)), dtype=float
        )
        for j, cls in enumerate(self.target_class_order):
            src = self.model_classes.index(cls)
            aligned_scores[:, j] = raw_scores[:, src]
        return softmax(aligned_scores / self.temperature, axis=1).astype(np.float32)


class CommitteeModel:
    def __init__(self, models: List[CommitteeMemberWrapper], info: CommitteeInfo):
        self.models = models
        self.info = info

    def predict_proba_mean_std(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        probs = [model.predict_proba(X).astype(np.float32) for model in self.models]
        probs = np.stack(probs, axis=0)
        mean_proba = probs.mean(axis=0)
        std_proba = probs.std(axis=0)
        nir_idx = CLASS_ORDER.index("NIR")
        nir_std = std_proba[:, nir_idx]
        member_nir_probabilities = probs[:, :, nir_idx].T
        return mean_proba, std_proba, nir_std, member_nir_probabilities


def _aligned_decision_scores(
    model, X: np.ndarray, target_class_order: Sequence[str]
) -> np.ndarray:
    raw_scores = np.asarray(model.decision_function(X), dtype=float)
    if raw_scores.ndim != 2:
        raise ValueError("五分类 LDA 应输出二维 decision scores")
    model_classes = _get_fitted_model_classes(model)
    aligned = np.empty((len(X), len(target_class_order)), dtype=float)
    for j, cls in enumerate(target_class_order):
        aligned[:, j] = raw_scores[:, model_classes.index(cls)]
    return aligned


def _fit_temperature(
    scores: np.ndarray,
    y: np.ndarray,
    class_order: Sequence[str],
    cfg: Config,
) -> float:
    lower = np.log(float(cfg.temperature_min))
    upper = np.log(float(cfg.temperature_max))

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = softmax(scores / temperature, axis=1)
        class_to_index = {label: index for index, label in enumerate(class_order)}
        true_indices = np.asarray([class_to_index[label] for label in y], dtype=int)
        true_probabilities = probabilities[np.arange(len(y)), true_indices]
        return float(-np.mean(np.log(np.maximum(true_probabilities, 1e-15))))

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"温度参数优化失败：{result.message}")
    return float(np.exp(result.x))


def _inner_oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    seed: int,
) -> np.ndarray:
    scores = np.empty((len(y), len(CLASS_ORDER)), dtype=float)
    splitter = StratifiedKFold(
        n_splits=cfg.committee_inner_folds,
        shuffle=True,
        random_state=seed,
    )
    for train_idx, validation_idx in splitter.split(X, y):
        selected_indices = _fit_anova_indices(
            X[train_idx], y[train_idx], cfg.classifier_anova_k
        )
        estimator = build_base_estimator(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            estimator.fit(X[train_idx][:, selected_indices], y[train_idx])
        scores[validation_idx] = _aligned_decision_scores(
            estimator,
            X[validation_idx][:, selected_indices],
            CLASS_ORDER,
        )
    return scores



def train_committee_model(X: pd.DataFrame, y: pd.Series, cfg: Config) -> CommitteeModel:
    _require_verified_backend(cfg)
    y = y.astype(str)
    class_counts = y.value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict()
    if min(class_counts.values()) <= 0:
        raise ValueError(f"至少有一个类别没有样本，无法训练五分类委员会：{class_counts}")
    if cfg.committee_size != 5:
        raise ValueError("固定协议要求 committee_size=5")
    if min(class_counts.values()) < cfg.committee_size:
        raise ValueError("每个类别至少需要5条样本才能建立分层五折委员会")

    X_np = X.to_numpy(dtype=np.float32)
    y_np = y.to_numpy()
    splitter = StratifiedKFold(
        n_splits=cfg.committee_size,
        shuffle=True,
        random_state=cfg.random_seed,
    )
    models: List[CommitteeMemberWrapper] = []
    temperatures: List[float] = []
    for fold, (train_idx, _) in enumerate(splitter.split(X_np, y_np)):
        inner_seed = cfg.random_seed + 100_003 + fold * 7919
        calibration_scores = _inner_oof_scores(
            X_np[train_idx], y_np[train_idx], cfg, inner_seed
        )
        temperature = _fit_temperature(
            calibration_scores, y_np[train_idx], CLASS_ORDER, cfg
        )
        selected_feature_indices = _fit_anova_indices(
            X_np[train_idx], y_np[train_idx], cfg.classifier_anova_k
        )
        estimator = build_base_estimator(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            estimator.fit(
                X_np[train_idx][:, selected_feature_indices], y_np[train_idx]
            )
        models.append(
            CommitteeMemberWrapper(
                estimator,
                selected_feature_indices,
                CLASS_ORDER,
                temperature,
            )
        )
        temperatures.append(temperature)

    info = CommitteeInfo(
        class_counts=class_counts,
        committee_size=cfg.committee_size,
        base_model_name=cfg.base_model_name,
        cv_folds=cfg.committee_size,
        inner_calibration_folds=cfg.committee_inner_folds,
        anova_k=cfg.classifier_anova_k,
        lda_shrinkage=float(cfg.classifier_lda_shrinkage),
        temperatures=temperatures,
    )
    return CommitteeModel(models=models, info=info)

def build_wavelength_regressor(cfg: Config, random_state: int):
    """
    构建波长回归分支的基学习器。

    第一版这里用 ElasticNet 回归，原因是：
    1. 与当前主分类模型的 elastic-net 风格一致；
    2. 在 144 维相关特征下通常比较稳；
    3. 代码修改成本小。
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", ElasticNet(
            alpha=cfg.wavelength_reg_alpha,
            l1_ratio=cfg.wavelength_reg_l1_ratio,
            max_iter=cfg.wavelength_reg_max_iter,
            random_state=random_state,
        )),
    ])

def train_wavelength_committee_model(X: pd.DataFrame, y_lambda: pd.Series, cfg: Config):
    """
    用当前训练集中的“非 Dark 且有真实波长”的样本训练一个小型回归委员会。

    这个分支不负责主分类，而是辅助回答：
    “这条序列预测离 NIR 边界 780 nm 还有多远”。

    训练策略：
    - 对可用样本做 bootstrap / 子采样；
    - 对特征继续做 bagging；
    - 最终输出一个成员列表，后续取均值和标准差。
    """
    _require_verified_backend(cfg)
    X_np = X.to_numpy(dtype=np.float32)
    y_np = pd.to_numeric(y_lambda, errors="coerce").to_numpy(dtype=np.float32)

    valid_mask = np.isfinite(y_np)
    X_np = X_np[valid_mask]
    y_np = y_np[valid_mask]

    if len(y_np) < 10:
        raise ValueError("可用于训练波长回归分支的样本过少。")

    total_feature_count = X_np.shape[1]
    feature_subsample_count = max(1, int(np.ceil(total_feature_count * cfg.feature_subsample_ratio)))
    all_feature_indices = np.arange(total_feature_count, dtype=int)

    rng_master = np.random.RandomState(cfg.random_seed + 999)
    member_seeds = [int(rng_master.randint(0, 1_000_000_000)) for _ in range(cfg.committee_size)]

    def fit_one_member(member_seed: int):
        rng = np.random.RandomState(member_seed)

        row_idx = rng.choice(np.arange(len(y_np)), size=len(y_np), replace=True)
        feat_idx = np.sort(rng.choice(all_feature_indices, size=feature_subsample_count, replace=False))

        estimator = build_wavelength_regressor(cfg, random_state=member_seed)
        estimator.fit(X_np[row_idx][:, feat_idx], y_np[row_idx])
        return estimator, feat_idx

    models = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
        delayed(fit_one_member)(seed) for seed in member_seeds
    )
    return models

def predict_wavelength_mean_std(models, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对候选序列预测波长，并返回委员会均值与标准差。
    """
    preds = []
    for model, feat_idx in models:
        pred = model.predict(X[:, feat_idx]).astype(np.float32)
        preds.append(pred)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def compute_wavelength_boundary_bonus(pred_lambda_nm: np.ndarray, cfg: Config) -> np.ndarray:
    """
    把预测主峰波长映射成 [0,1] 的边界加成项。

    设计思想：
    - 目标不是单纯越大越好，而是“越接近 / 越超过 780 nm 边界越值得关注”；
    - 用 sigmoid 做平滑映射，避免硬阈值造成不连续。

    当预测波长远低于 780 nm 时，加成接近 0；
    当预测波长接近 780 nm 时，加成快速上升；
    当预测波长超过 780 nm 较多时，加成接近 1。
    """
    x = (pred_lambda_nm - float(cfg.wavelength_boundary_nm)) / float(cfg.wavelength_sigmoid_tau)
    bonus = 1.0 / (1.0 + np.exp(-x))
    return np.clip(bonus, 0.0, 1.0).astype(np.float32)
def build_brightness_regressor(cfg: Config, random_state: int):
    """
    构建亮度回归分支的基学习器。

    这里和波长分支保持一致，仍然使用 ElasticNet：
    1. 与当前主分类模型和波长分支风格统一；
    2. 在 144 维相关特征下通常比较稳；
    3. 修改成本最低。
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", ElasticNet(
            alpha=cfg.brightness_reg_alpha,
            l1_ratio=cfg.brightness_reg_l1_ratio,
            max_iter=cfg.brightness_reg_max_iter,
            random_state=random_state,
        )),
    ])


def train_brightness_committee_model(X: pd.DataFrame, y_lli: pd.Series, cfg: Config):
    """
    用当前训练集中的“非 Dark 且有真实亮度”的样本训练一个小型亮度回归委员会。

    注意：
    - 训练目标不是原始 LLI，而是 log1p(LLI)；
    - 这样可以降低长尾亮度分布带来的不稳定；
    - 缺失值会自动被排除；
    - Dark 样本不进入亮度回归训练，因为 Dark 主要由五分类主模型负责。
    """
    _require_verified_backend(cfg)
    y_raw = pd.to_numeric(y_lli, errors="coerce").to_numpy(dtype=np.float32)
    y_log = np.log1p(np.clip(y_raw, a_min=0.0, a_max=None))

    valid_mask = np.isfinite(y_log)
    X_np = X.to_numpy(dtype=np.float32)[valid_mask]
    y_np = y_log[valid_mask]

    if len(y_np) < 10:
        raise ValueError("可用于训练亮度回归分支的样本过少。")

    total_feature_count = X_np.shape[1]
    feature_subsample_count = max(1, int(np.ceil(total_feature_count * cfg.feature_subsample_ratio)))
    all_feature_indices = np.arange(total_feature_count, dtype=int)

    rng_master = np.random.RandomState(cfg.random_seed + 1999)
    member_seeds = [int(rng_master.randint(0, 1_000_000_000)) for _ in range(cfg.committee_size)]

    def fit_one_member(member_seed: int):
        rng = np.random.RandomState(member_seed)

        row_idx = rng.choice(np.arange(len(y_np)), size=len(y_np), replace=True)
        feat_idx = np.sort(rng.choice(all_feature_indices, size=feature_subsample_count, replace=False))

        estimator = build_brightness_regressor(cfg, random_state=member_seed)
        estimator.fit(X_np[row_idx][:, feat_idx], y_np[row_idx])
        return estimator, feat_idx

    models = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
        delayed(fit_one_member)(seed) for seed in member_seeds
    )
    return models


def predict_brightness_mean_std(models, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对候选序列预测 log1p(LLI)，并返回委员会均值与标准差。
    """
    preds = []
    for model, feat_idx in models:
        pred = model.predict(X[:, feat_idx]).astype(np.float32)
        preds.append(pred)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def compute_brightness_bonus_and_dim_risk(
    pred_log_lli: np.ndarray,
    train_log_lli: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    亮度分支同时输出两样东西：

    1. brightness_bonus：
       亮度越高越接近 1，用作最终总分中的正向奖励。

    2. brightness_dim_risk：
       越暗越接近 1，用作最终 batch 里的硬过滤条件之一。

    做法：
    - 基于当前训练集亮度分布的 floor / good 分位数定义亮度等级；
    - 这样不依赖一个拍脑袋的绝对亮度阈值，更稳。
    """
    floor = float(np.quantile(train_log_lli, cfg.brightness_floor_quantile))
    good = float(np.quantile(train_log_lli, cfg.brightness_good_quantile))

    # 亮度奖励：低于 floor 接近 0，高于 good 接近 1
    bonus = np.clip((pred_log_lli - floor) / max(good - floor, 1e-12), 0.0, 1.0).astype(np.float32)

    # 偏暗风险：越低于 floor，风险越高
    x = (floor - pred_log_lli) / max(cfg.brightness_sigmoid_tau, 1e-12)
    dim_risk = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    return bonus, dim_risk

# ============================================================
# 7. 距离相关工具
# ============================================================
def compute_min_hamming_to_reference(encoded_batch: np.ndarray, reference_encoded: np.ndarray) -> np.ndarray:
    if reference_encoded.size == 0:
        return np.full(encoded_batch.shape[0], SEQ_LEN, dtype=np.uint8)
    matches = (encoded_batch[:, None, :] == reference_encoded[None, :, :]).sum(axis=2)
    dists = SEQ_LEN - matches
    return dists.min(axis=1).astype(np.uint8)

def normalize_uncertainty_from_std(std_values: np.ndarray) -> np.ndarray:
    return np.clip(std_values / 0.5, 0.0, 1.0).astype(np.float32)

def build_feature_theoretical_max_vector(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([parse_feature_theoretical_max(f) for f in feature_names], dtype=np.float32)

def normalized_l1_distance_to_reference(X_query: np.ndarray, X_ref: np.ndarray, max_vec: np.ndarray) -> np.ndarray:
    """
    在重要 staple 子空间里计算 normalized L1 distance。
    这里的设计思想是：
    Hamming 很远不一定代表机制模式很远；
    因此除了序列位点距离，我们还要看“staple 表示空间里是否真的不同”。
    """
    if X_ref.size == 0:
        return np.ones(X_query.shape[0], dtype=np.float32)
    Xq = X_query / max_vec[None, :]
    Xr = X_ref / max_vec[None, :]
    dists = np.abs(Xq[:, None, :] - Xr[None, :, :]).mean(axis=2)
    return dists.min(axis=1).astype(np.float32)


# ============================================================
# 8. 新目标项与新采集函数
# ============================================================
def compute_safe_nir_score(mean_probs: np.ndarray) -> np.ndarray:
    """
    旧版主目标：P_NIR_mean
    新版主目标：safe_nir_score

    它表达的是：
    “像 NIR，同时要和最容易混淆的 Far Red / Dark 拉开距离”。

    这里故意不用纯 margin，而是保留 P_NIR 本身作为主因子，
    再乘上两组 pairwise safety ratio，避免出现“P_NIR 很低但 ratio 看起来还行”的假高分。
    """
    p_farred = mean_probs[:, CLASS_ORDER.index("Far Red")]
    p_nir = mean_probs[:, CLASS_ORDER.index("NIR")]
    p_dark = mean_probs[:, CLASS_ORDER.index("Dark")]
    eps = 1e-8
    nir_vs_farred = p_nir / (p_nir + p_farred + eps)
    nir_vs_dark = p_nir / (p_nir + p_dark + eps)
    # p_green / p_red 没有显式放入安全项，是因为当前最关键的混淆边界主要是 Far Red 和 Dark。
    safe_score = p_nir * np.minimum(nir_vs_farred, nir_vs_dark)
    return np.clip(safe_score, 0.0, 1.0).astype(np.float32)

def compute_score(
    safe_nir_score: np.ndarray,
    uncertainty_norm: np.ndarray,
    diversity_norm: np.ndarray,
    motif_prior_signed: np.ndarray,
    wavelength_bonus: np.ndarray,
    brightness_bonus: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    return (
        cfg.w_target * safe_nir_score
        + cfg.w_uncertainty * uncertainty_norm
        + cfg.w_diversity * diversity_norm
        + cfg.w_motif * motif_prior_signed
        + cfg.w_wavelength * wavelength_bonus
        + cfg.w_brightness * brightness_bonus
    ).astype(np.float32)
def normalize_global_score(raw_score: np.ndarray, cfg: Config) -> np.ndarray:
    """
    把综合原始分数映射成“跨批次可比较”的全局归一化分数。

    这里绝对不能用“本批最大值=1”的方式，
    因为那样只能做批内比较，不能比较不同批次、不同轮次。

    当前总分定义为：
        score =
            w_target * safe_nir_score
          + w_uncertainty * uncertainty_norm
          + w_diversity * diversity_norm
          + w_motif * motif_prior_signed
          + w_wavelength * wavelength_bonus
          + w_brightness * brightness_bonus

    其中：
    - safe_nir_score ∈ [0, 1]
    - uncertainty_norm ∈ [0, 1]
    - diversity_norm ∈ [0, 1]
    - motif_prior_signed ∈ [-1, 1]
    - wavelength_bonus ∈ [0, 1]
    - brightness_bonus ∈ [0, 1]

    所以理论上下界是：
    - score_min = -w_motif
    - score_max = w_target + w_uncertainty + w_diversity + w_motif + w_wavelength + w_brightness

    用这个固定上下界做归一化后，
    得到的分数就可以跨不同批次直接比较。
    """
    score_min = -float(cfg.w_motif)
    score_max = (
            float(cfg.w_target)
            + float(cfg.w_uncertainty)
            + float(cfg.w_diversity)
            + float(cfg.w_motif)
            + float(cfg.w_wavelength)
            + float(cfg.w_brightness)
    )
    denom = max(score_max - score_min, 1e-12)
    normalized = (raw_score - score_min) / denom
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)

# ============================================================
# 9. 三路径候选池生成
# ============================================================
def mutate_sequence(seq: str, positions: Sequence[int], new_bases: Sequence[str]) -> str:
    arr = list(seq)
    for pos, new_b in zip(positions, new_bases):
        arr[pos] = new_b
    return "".join(arr)


def generate_local_mutation_candidates(
    seed_seq: str,
    rng: np.random.RandomState,
    max_three_mutations: int,
) -> List[str]:
    """
    路径 A 的核心：局部开发。
    对单个 seed 生成：
    - 全部 1 位突变
    - 全部 2 位突变
    - 少量 3 位突变（随机抽样，防止组合爆炸）
    """
    seed_seq = normalize_sequence(seed_seq)
    candidates = []

    # 1 位突变：10 * 3 = 30 条
    for pos in range(SEQ_LEN):
        orig = seed_seq[pos]
        for b in DNA_BASES:
            if b != orig:
                candidates.append(mutate_sequence(seed_seq, [pos], [b]))

    # 2 位突变：C(10,2) * 3 * 3 = 405 条
    for p1, p2 in combinations(range(SEQ_LEN), 2):
        b1_choices = [b for b in DNA_BASES if b != seed_seq[p1]]
        b2_choices = [b for b in DNA_BASES if b != seed_seq[p2]]
        for b1, b2 in product(b1_choices, b2_choices):
            candidates.append(mutate_sequence(seed_seq, [p1, p2], [b1, b2]))

    # 3 位突变：只随机抽一小部分，作为“比 2 位再远一点，但又不是完全乱跳”的补充。
    all_triplets = list(combinations(range(SEQ_LEN), 3))
    rng.shuffle(all_triplets)
    for triplet in all_triplets[:max_three_mutations]:
        new_bases = []
        for pos in triplet:
            choices = [b for b in DNA_BASES if b != seed_seq[pos]]
            new_bases.append(rng.choice(choices))
        candidates.append(mutate_sequence(seed_seq, triplet, new_bases))

    return candidates

def generate_path_A_local_candidates(
    train_df: pd.DataFrame,
    cfg: Config,
    pos_weights: Dict[str, float],
    neg_weights: Dict[str, float],
    banned_sequences: Sequence[str],
) -> pd.DataFrame:
    """
    路径 A：局部开发池。

    设计思想：
    - 不是从整个 4^10 空间乱抓，而是从“当前最有可能提供有效局部改进”的样本附近展开。
    - 这一步相当于把 2014/2020 那类“先生成 candidate，再筛 candidate”的思路，
      放到你现在的 AL 里，但尽量用最小代码代价落地。

    seed 的选择：
    - 必选当前训练集中的 NIR
    - 可选 Far Red，因为它是最关键的混淆边界之一
    """
    rng = np.random.RandomState(cfg.random_seed + 123)
    seed_mask = train_df["class_label"].eq("NIR")
    if cfg.path_a_include_farred_seeds:
        seed_mask = seed_mask | train_df["class_label"].eq("Far Red")

    seed_sequences = train_df.loc[seed_mask, "Sequence"].drop_duplicates().tolist()
    if len(seed_sequences) == 0:
        # 如果训练集里还没有 NIR/Far Red，退而求其次，拿所有非 Dark 做 seed。
        seed_sequences = train_df.loc[train_df["class_label"] != "Dark", "Sequence"].drop_duplicates().tolist()

    raw_candidates: List[str] = []
    for seed in seed_sequences:
        raw_candidates.extend(
            generate_local_mutation_candidates(
                seed_seq=seed,
                rng=rng,
                max_three_mutations=cfg.path_a_max_three_mutations_per_seed,
            )
        )

    raw_candidates = [normalize_sequence(s) for s in raw_candidates if is_valid_dna_10mer(s)]
    # 排除训练集中已有、以及人工禁用的序列
    forbidden = set(train_df["Sequence"].tolist()) | set(map(normalize_sequence, banned_sequences))
    raw_candidates = [s for s in raw_candidates if s not in forbidden]

    candidate_df = sequence_list_to_candidate_df(raw_candidates)
    if candidate_df.empty:
        return candidate_df

    X = candidate_df[STANDARD_FEATURES].to_numpy(dtype=np.float32)
    signed_prior = signed_motif_prior_score(X, pos_weights, neg_weights, cfg)
    candidate_df["signed_prior_seed_score"] = signed_prior

    # 路径 A 的目的本来就是 exploitation，因此先用 signed prior 做一轮便宜预筛。
    candidate_df = candidate_df.sort_values("signed_prior_seed_score", ascending=False).head(cfg.path_a_pool_size).reset_index(drop=True)
    candidate_df["candidate_path"] = "A_local"
    return candidate_df



def generate_single_hillclimb_sequence(rng: np.random.RandomState, cfg: Config, pos_weights: Dict[str, float], neg_weights: Dict[str, float]) -> str:
    """
    路径 B：用一个很轻量的 prior-guided hill climbing 代替完整生成模型。

    这样做的原因是：
    - 你现在还没全面切到 VAE/PrIVAE；
    - 但又不希望路径 B 只是“纯随机”。
    - 因此这里用 signed prior 做一个极低成本的 de novo 引导。
    """
    while True:
        seq = "".join(rng.choice(list(DNA_BASES), size=SEQ_LEN))
        if sum(ch in {"C", "G"} for ch in seq) >= cfg.path_b_min_cg_count:
            break

    current = seq
    current_df = sequence_list_to_candidate_df([current])
    current_X = current_df[STANDARD_FEATURES].to_numpy(dtype=np.float32)
    current_score = float(signed_motif_prior_score(current_X, pos_weights, neg_weights, cfg)[0])

    for _ in range(cfg.path_b_hillclimb_steps):
        pos = int(rng.randint(0, SEQ_LEN))
        choices = [b for b in DNA_BASES if b != current[pos]]
        new_b = str(rng.choice(choices))
        proposal = mutate_sequence(current, [pos], [new_b])
        # 路径 B 的 chemistry hard constraint：
        # 这条约束不是只限制随机起点，而是要求 hill-climbing 过程中
        # 的每一个被考虑的 proposal 都必须满足最低 C+G 数。
        if sum(ch in {"C", "G"} for ch in proposal) < cfg.path_b_min_cg_count:
            continue
        proposal_df = sequence_list_to_candidate_df([proposal])
        proposal_X = proposal_df[STANDARD_FEATURES].to_numpy(dtype=np.float32)
        proposal_score = float(signed_motif_prior_score(proposal_X, pos_weights, neg_weights, cfg)[0])

        # 非常简单的接受规则：
        # - 若 score 更高，直接接受
        # - 若 score 更低，小概率接受，避免完全卡死在局部最优
        if proposal_score >= current_score or rng.rand() < 0.05:
            current = proposal
            current_score = proposal_score
    return current


def generate_path_B_de_novo_candidates(
    train_df: pd.DataFrame,
    cfg: Config,
    pos_weights: Dict[str, float],
    neg_weights: Dict[str, float],
    banned_sequences: Sequence[str],
) -> pd.DataFrame:
    """
    路径 B：de novo 探索池。

    做法是：
    1. 从满足最低 C/G 数约束的随机起点出发；
    2. 用 signed prior 做轻量 hill climbing；
    3. 过滤训练集已有序列与禁用序列；
    4. 再按 signed prior_seed_score 做一轮便宜预筛。

    这一路径的目标不是局部 exploitation，而是补充更远、更 novel 的探索候选。
    """
    rng = np.random.RandomState(cfg.random_seed + 456)
    generated = [
        generate_single_hillclimb_sequence(rng, cfg, pos_weights, neg_weights)
        for _ in range(cfg.path_b_random_starts)
    ]

    forbidden = set(train_df["Sequence"].tolist()) | set(map(normalize_sequence, banned_sequences))
    generated = [
        s for s in generated
        if (s not in forbidden)
           and (sum(ch in {"C", "G"} for ch in s) >= cfg.path_b_min_cg_count)
    ]
    candidate_df = sequence_list_to_candidate_df(generated)
    if candidate_df.empty:
        return candidate_df

    X = candidate_df[STANDARD_FEATURES].to_numpy(dtype=np.float32)
    signed_prior = signed_motif_prior_score(X, pos_weights, neg_weights, cfg)
    candidate_df["signed_prior_seed_score"] = signed_prior
    candidate_df = candidate_df.sort_values("signed_prior_seed_score", ascending=False).head(cfg.path_b_pool_size).reset_index(drop=True)
    candidate_df["candidate_path"] = "B_denovo"
    return candidate_df

def generate_path_C_random_candidates(
    train_df: pd.DataFrame,
    cfg: Config,
    banned_sequences: Sequence[str],
) -> pd.DataFrame:
    """
    路径 C：随机覆盖池。

    这一路径不使用局部突变，也不使用 signed prior hill climbing，
    作用是给整个候选池补充“更无偏”的随机覆盖，
    避免路径 A 与路径 B 过度集中在已有高分局部区域和经验先验附近。

    实现策略：
    - 从 4^10 空间中随机生成 10-mer；
    - 排除训练集中已有序列；
    - 排除 banned list；
    - 最终保留固定数量 path_c_pool_size。
    """
    rng = np.random.RandomState(cfg.random_seed + 789)
    forbidden = set(train_df["Sequence"].tolist()) | set(map(normalize_sequence, banned_sequences))

    generated = set()
    max_trials = max(cfg.path_c_pool_size * 20, 10000)

    # 这里设置最大尝试次数，是为了避免在 forbidden 较多时随机采样陷入长时间空转。
    # 因此路径 C 的最终 unique 候选数不保证一定等于 path_c_pool_size，
    # 只保证“尽量在有限尝试次数内接近这个上限”。

    while len(generated) < cfg.path_c_pool_size and max_trials > 0:
        seq = "".join(rng.choice(list(DNA_BASES), size=SEQ_LEN))
        seq = normalize_sequence(seq)
        if seq not in forbidden:
            generated.add(seq)
        max_trials -= 1

    candidate_df = sequence_list_to_candidate_df(sorted(generated))
    if candidate_df.empty:
        return candidate_df

    candidate_df["candidate_path"] = "C_random"
    candidate_df["signed_prior_seed_score"] = 0.0
    return candidate_df

def prune_candidate_pool_by_staple_distance(
    candidate_df: pd.DataFrame,
    prior_rank_col: str,
    important_features: Sequence[str],
    min_distance: float,
) -> pd.DataFrame:
    """
    这一步对应 GNoME 里“候选过滤后再做聚类/去 polymorph 重复”的思想。
    DNA 这里没有晶体 polymorph，但会有大量“几乎同一类 pattern”的序列。

    实现上不使用重型聚类算法，而是做一个 greedy pruning：
    - 先按便宜分数排好序
    - 再逐个保留与已保留集合在重要 staple 子空间里足够远的序列
   
    这样做代码最省，行为也很稳定。
    """
    if candidate_df.empty:
        return candidate_df
    work = candidate_df.sort_values(prior_rank_col, ascending=False).reset_index(drop=True)
    X = work[list(important_features)].to_numpy(dtype=np.float32)
    max_vec = build_feature_theoretical_max_vector(important_features)

    keep_idx = []
    kept_vectors = []
    for i in range(len(work)):
        xi = X[i : i + 1]
        if not kept_vectors:
            keep_idx.append(i)
            kept_vectors.append(X[i])
            continue
        ref = np.stack(kept_vectors, axis=0)
        dist = normalized_l1_distance_to_reference(xi, ref, max_vec)[0]
        if dist >= min_distance:
            keep_idx.append(i)
            kept_vectors.append(X[i])
    return work.iloc[keep_idx].reset_index(drop=True)


def generate_tripath_candidates(
    train_df: pd.DataFrame,
    cfg: Config,
    pos_weights: Dict[str, float],
    neg_weights: Dict[str, float],
    full_signed_prior_df: pd.DataFrame,
    banned_sequences: Sequence[str],
) -> pd.DataFrame:
    """
    生成三路径候选池的总入口。

    执行顺序是：
    1. 分别生成路径 A、路径 B、路径 C 的候选；
    2. 合并三条路径的候选并按 Sequence 去重；
    3. 若开启 candidate pruning，则在重要 staple 特征子空间中做去近邻压缩。

    返回值是本轮进入统一打分阶段的候选池 DataFrame。
    """
    parts = []

    if cfg.path_a_enabled:
        path_a_df = generate_path_A_local_candidates(
            train_df, cfg, pos_weights, neg_weights, banned_sequences
        )
        print(f"    路径A候选数：{len(path_a_df)}")
        if not path_a_df.empty:
            parts.append(path_a_df)

    if cfg.path_b_enabled:
        path_b_df = generate_path_B_de_novo_candidates(
            train_df, cfg, pos_weights, neg_weights, banned_sequences
        )
        print(f"    路径B候选数：{len(path_b_df)}")
        if not path_b_df.empty:
            parts.append(path_b_df)

    if cfg.path_c_enabled:
        path_c_df = generate_path_C_random_candidates(
            train_df=train_df,
            cfg=cfg,
            banned_sequences=banned_sequences,
        )
        print(f"    路径C候选数：{len(path_c_df)}")
        if not path_c_df.empty:
            parts.append(path_c_df)

    if not parts:
        raise RuntimeError("三路径候选池为空，请检查路径 A/B/C 设置")

    candidate_df = pd.concat(parts, axis=0, ignore_index=True)
    print(f"    三路径合并后去重前：{len(candidate_df)}")
    # 记录同一条序列在三条路径中的全部来源，避免 keep="first" 之后把来源信息丢掉。
    path_all_map = (
        candidate_df.groupby("Sequence")["candidate_path"]
        .apply(lambda s: "|".join(dict.fromkeys(s.astype(str))))
        .to_dict()
    )

    candidate_df = candidate_df.drop_duplicates(subset=["Sequence"], keep="first").reset_index(drop=True)
    candidate_df["candidate_path_all"] = candidate_df["Sequence"].map(path_all_map)
    print(f"    按Sequence去重后：{len(candidate_df)}")



    # candidate pruning 依赖“重要特征排序”来定义去近邻所用的 staple 子空间。
    # 因此只有在开启 pruning 且成功读到完整 signed importance 排序表时，
    # 才会执行这一步；否则保留合并后的原始候选池进入后续打分。

    before_prune_n = len(candidate_df)
    if cfg.enable_candidate_pruning and not full_signed_prior_df.empty:
        important_features = full_signed_prior_df["feature"].tolist()[: cfg.candidate_prune_top_k_features]
        if important_features:
                candidate_df = prune_candidate_pool_by_staple_distance(
                candidate_df=candidate_df,
                prior_rank_col="signed_prior_seed_score",
                important_features=important_features,
                min_distance=cfg.candidate_prune_min_staple_distance,
            )
    print(f"    candidate pruning 前：{before_prune_n}，后：{len(candidate_df)}")
    return candidate_df.reset_index(drop=True)


# ============================================================
# 10. 旧流程：全空间枚举（保留做兼容/对照）
# ============================================================
def generate_encoded_sequence_batches(batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    powers = (4 ** np.arange(SEQ_LEN - 1, -1, -1, dtype=np.uint32))
    for start in range(0, TOTAL_SEQUENCE_SPACE, batch_size):
        end = min(start + batch_size, TOTAL_SEQUENCE_SPACE)
        codes = np.arange(start, end, dtype=np.uint32)
        digits = ((codes[:, None] // powers[None, :]) % 4).astype(np.uint8)
        yield codes, digits


def build_full_space_candidate_df(measured_sequences: Sequence[str], banned_sequences: Sequence[str], cfg: Config) -> pd.DataFrame:
    forbidden_codes = {encode_sequence_to_int(s) for s in measured_sequences if is_valid_dna_10mer(s)}
    forbidden_codes.update({encode_sequence_to_int(s) for s in banned_sequences if is_valid_dna_10mer(s)})
    all_codes = []
    all_digits = []
    for codes, digits in generate_encoded_sequence_batches(cfg.batch_size):
        mask = ~np.isin(codes, np.fromiter(forbidden_codes, dtype=np.uint32, count=len(forbidden_codes)))
        if mask.any():
            all_codes.append(codes[mask])
            all_digits.append(digits[mask])
    if not all_codes:
        return pd.DataFrame(columns=["Sequence"] + STANDARD_FEATURES)
    codes = np.concatenate(all_codes)
    digits = np.concatenate(all_digits)
    seqs = decode_codes_to_sequences(codes)
    X = encoded_batch_to_feature_matrix(digits)
    df = pd.DataFrame(X.astype(int), columns=STANDARD_FEATURES)
    df.insert(0, "Sequence", seqs)
    df["candidate_path"] = "full_space"
    return df


# ============================================================
# 11. 新版筛分：对“任意候选池 DataFrame”统一打分
# ============================================================
# 对整张候选池做统一预打分。
# 这一阶段会把：
# 1. 五分类委员会输出
# 2. uncertainty
# 3. train-set diversity
# 4. signed motif prior
# 5. 波长边界 bonus
# 6. 亮度 bonus / dim risk
# 全部汇总到一个 screened 字典中，
# 供后续 greedy_select_batch 做最终选批。
def screen_candidate_space(
        committee: CommitteeModel,
        wavelength_models,
        brightness_models,
        brightness_train_log,
        train_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        cfg: Config,
        pos_weights: Dict[str, float],
        neg_weights: Dict[str, float],
        full_signed_prior_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    if candidate_df.empty:
        raise RuntimeError("candidate_df 为空，无法打分")

    X_candidate = candidate_df[STANDARD_FEATURES].to_numpy(dtype=np.float32)
    mean_probs, _, nir_std, member_nir_probabilities = committee.predict_proba_mean_std(
        X_candidate
    )
    uncertainty_norm = normalize_uncertainty_from_std(nir_std)
    safe_nir_score = compute_safe_nir_score(mean_probs)

    candidate_digits = np.asarray([sequence_to_digits(s) for s in candidate_df["Sequence"]], dtype=np.uint8)
    train_digits = np.asarray([sequence_to_digits(s) for s in train_df["Sequence"]], dtype=np.uint8)
    min_hamming_to_train = compute_min_hamming_to_reference(candidate_digits, train_digits)
    hamming_div_train = (min_hamming_to_train.astype(np.float32) / float(SEQ_LEN)).astype(np.float32)

    # 重要 staple 子空间 diversity
    if full_signed_prior_df.empty:
        important_features = STANDARD_FEATURES[: cfg.staple_diversity_top_k]
    else:
        important_features = full_signed_prior_df["feature"].tolist()[: cfg.staple_diversity_top_k]
    X_train_imp = train_df[important_features].to_numpy(dtype=np.float32)
    X_cand_imp = candidate_df[important_features].to_numpy(dtype=np.float32)
    max_vec = build_feature_theoretical_max_vector(important_features)
    staple_div_train = normalized_l1_distance_to_reference(X_cand_imp, X_train_imp, max_vec)

    mixed_div_train = (
        cfg.diversity_hamming_weight * hamming_div_train
        + cfg.diversity_staple_weight * staple_div_train
    ).astype(np.float32)

    if cfg.use_motif_prior:
        motif_prior_signed = signed_motif_prior_score(X_candidate, pos_weights, neg_weights, cfg)
    else:
        motif_prior_signed = np.zeros(len(candidate_df), dtype=np.float32)
    # ============================================================
    # 波长回归分支：
    # 预测主峰波长，并将其转成边界加成项。
    # 这个分支不替代五分类，而是辅助判断“离 780 nm 有多近”。
    # ============================================================
    if cfg.use_wavelength_branch and wavelength_models is not None:
        pred_lambda_nm, pred_lambda_std = predict_wavelength_mean_std(wavelength_models, X_candidate)
        wavelength_bonus = compute_wavelength_boundary_bonus(pred_lambda_nm, cfg)
    else:
        pred_lambda_nm = np.full(len(candidate_df), np.nan, dtype=np.float32)
        pred_lambda_std = np.full(len(candidate_df), np.nan, dtype=np.float32)
        wavelength_bonus = np.zeros(len(candidate_df), dtype=np.float32)
    # ============================================================
    # 亮度回归分支：
    # 预测主峰亮度（log1p(LLI)），并同时产出：
    # 1. brightness_bonus：越亮越加分
    # 2. brightness_dim_risk：越暗风险越高
    #
    # 注意：
    # - 这一分支允许训练数据中存在缺失亮度；
    # - 只有非 Dark 且亮度列可数值化的样本会进入训练；
    # - 最终亮度不是简单替代 Dark，而是和 P_Dark 一起构成“别太暗”的双层控制。
    # ============================================================
    if cfg.use_brightness_branch and (brightness_models is not None) and (brightness_train_log is not None):
        pred_log_lli, pred_log_lli_std = predict_brightness_mean_std(brightness_models, X_candidate)
        brightness_bonus, brightness_dim_risk = compute_brightness_bonus_and_dim_risk(
            pred_log_lli=pred_log_lli,
            train_log_lli=brightness_train_log,
            cfg=cfg,
        )
    else:
        pred_log_lli = np.full(len(candidate_df), np.nan, dtype=np.float32)
        pred_log_lli_std = np.full(len(candidate_df), np.nan, dtype=np.float32)
        brightness_bonus = np.zeros(len(candidate_df), dtype=np.float32)
        brightness_dim_risk = np.zeros(len(candidate_df), dtype=np.float32)

    pre_score = compute_score(
        safe_nir_score=safe_nir_score,
        uncertainty_norm=uncertainty_norm,
        diversity_norm=mixed_div_train,
        motif_prior_signed=motif_prior_signed,
        wavelength_bonus=wavelength_bonus,
        brightness_bonus=brightness_bonus,
        cfg=cfg,
    )
    return {
        "candidate_df": candidate_df.reset_index(drop=True),
        "digits": candidate_digits,
        "mean_probs": mean_probs.astype(np.float32),
        "member_nir_probabilities": member_nir_probabilities.astype(np.float32),
        "safe_nir_score": safe_nir_score,
        "p_nir_std": nir_std.astype(np.float32),
        "uncertainty_norm": uncertainty_norm,
        "hamming_div_train": hamming_div_train,
        "staple_div_train": staple_div_train,
        "mixed_div_train": mixed_div_train,
        "motif_prior_signed": motif_prior_signed,
        "pre_score": pre_score,
        "pred_lambda_nm": pred_lambda_nm,
        "pred_lambda_std": pred_lambda_std,
        "wavelength_bonus": wavelength_bonus,
        "important_features": list(important_features),
        "important_feature_max": max_vec,
        "pred_log_lli": pred_log_lli,
        "pred_log_lli_std": pred_log_lli_std,
        "brightness_bonus": brightness_bonus,
        "brightness_dim_risk": brightness_dim_risk,
        "X_cand_imp": X_cand_imp,
        "X_train_imp": X_train_imp,
        "stats": {
            "candidate_pool_size": int(len(candidate_df)),
            "candidate_mode": cfg.candidate_mode,
            "path_counts": candidate_df["candidate_path"].value_counts().to_dict() if "candidate_path" in candidate_df.columns else {},
        },
    }


# ============================================================
# 12. 新版 greedy batch selection：同时考虑 Hamming 和 staple-space 动态 diversity
# ============================================================
# 在已经预打分的候选池上做最终 batch 选择。
# 这里不是简单取前 k 名，而是每次选出一条后，
# 重新计算候选对“已选集合”的动态 diversity，
# 同时应用若干硬过滤条件（如 Dark 风险、Far Red 风险、亮度过暗风险），
# 从而得到更分散、更稳健的一批最终推荐序列。
def greedy_select_batch(screened: Dict[str, np.ndarray], train_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    candidate_df = screened["candidate_df"].copy().reset_index(drop=True)
    digits = screened["digits"]
    mean_probs = screened["mean_probs"]
    member_nir_probabilities = screened["member_nir_probabilities"]
    safe_nir_score = screened["safe_nir_score"]
    p_nir_std = screened["p_nir_std"]
    uncertainty_norm = screened["uncertainty_norm"]
    pred_lambda_nm = screened["pred_lambda_nm"]
    pred_lambda_std = screened["pred_lambda_std"]
    wavelength_bonus = screened["wavelength_bonus"]
    pred_log_lli = screened["pred_log_lli"]
    pred_log_lli_std = screened["pred_log_lli_std"]
    brightness_bonus = screened["brightness_bonus"]
    brightness_dim_risk = screened["brightness_dim_risk"]
    hamming_div_train = screened["hamming_div_train"]
    staple_div_train = screened["staple_div_train"]
    motif_prior_signed = screened["motif_prior_signed"]
    X_cand_imp = screened["X_cand_imp"]
    important_features = screened["important_features"]
    max_vec = screened["important_feature_max"]

    available = np.ones(len(candidate_df), dtype=bool)
    selected_indices = []
    selected_digits = []
    selected_imp_vectors = []
    selected_scores = []
    selected_mix_div = []
    selected_hamming_div = []
    selected_staple_div = []

    train_digits = np.asarray([sequence_to_digits(s) for s in train_df["Sequence"]], dtype=np.uint8)

    for _ in range(cfg.recommend_k):
        if selected_digits:
            selected_digits_arr = np.stack(selected_digits, axis=0)
            min_hamming_to_selected = compute_min_hamming_to_reference(digits, selected_digits_arr)
            hamming_div_selected = (min_hamming_to_selected.astype(np.float32) / float(SEQ_LEN)).astype(np.float32)

            selected_imp_arr = np.stack(selected_imp_vectors, axis=0).astype(np.float32)
            staple_div_selected = normalized_l1_distance_to_reference(X_cand_imp, selected_imp_arr, max_vec)
        else:
            hamming_div_selected = np.ones(len(candidate_df), dtype=np.float32)
            staple_div_selected = np.ones(len(candidate_df), dtype=np.float32)

        # 同时避免靠训练集太近，也避免 batch 内自己扎堆。
        hamming_div_dynamic = np.minimum(hamming_div_train, hamming_div_selected)
        staple_div_dynamic = np.minimum(staple_div_train, staple_div_selected)
        mixed_div_dynamic = (
            cfg.diversity_hamming_weight * hamming_div_dynamic
            + cfg.diversity_staple_weight * staple_div_dynamic
        ).astype(np.float32)

        current_score = compute_score(
            safe_nir_score=safe_nir_score,
            uncertainty_norm=uncertainty_norm,
            diversity_norm=mixed_div_dynamic,
            motif_prior_signed=motif_prior_signed,
            wavelength_bonus=wavelength_bonus,
            brightness_bonus=brightness_bonus,
            cfg=cfg,
        )
        # 额外加一个很便宜但很有用的硬过滤：
        # 明显像 Dark 的直接淘汰；
        # 明显更像 Far Red 而不是 NIR 的也淘汰。
        p_farred = mean_probs[:, CLASS_ORDER.index("Far Red")]
        p_dark = mean_probs[:, CLASS_ORDER.index("Dark")]
        p_nir = mean_probs[:, CLASS_ORDER.index("NIR")]

        # 这里的 p_dark > 0.35 是经验型硬阈值：
        # 只要候选被委员会较明显地判断为 Dark，就直接从最终 batch 中剔除。
        invalid_mask = (
            (p_dark > 0.35)
            | (p_farred > p_nir)
            | (brightness_dim_risk > cfg.brightness_hard_risk_threshold)
        )
        current_score[~available] = -np.inf
        current_score[invalid_mask] = -np.inf

        if cfg.min_hamming_distance > 0 and selected_digits:
            too_close_to_selected = (min_hamming_to_selected < cfg.min_hamming_distance)
            current_score[too_close_to_selected] = -np.inf

        best_idx = int(np.argmax(current_score))
        if not np.isfinite(current_score[best_idx]):
            raise RuntimeError("无法再选出满足条件的候选，请放宽过滤条件或增大候选池")

        available[best_idx] = False
        selected_indices.append(best_idx)
        selected_digits.append(digits[best_idx])
        selected_imp_vectors.append(X_cand_imp[best_idx])
        selected_scores.append(float(current_score[best_idx]))
        selected_mix_div.append(float(mixed_div_dynamic[best_idx]))
        selected_hamming_div.append(float(hamming_div_dynamic[best_idx]))
        selected_staple_div.append(float(staple_div_dynamic[best_idx]))

    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_df = candidate_df.iloc[selected_indices].copy().reset_index(drop=True)
    selected_df.insert(0, "Rank", np.arange(1, len(selected_df) + 1))
    raw_score_arr = np.asarray(selected_scores, dtype=np.float32)
    selected_df["Score"] = raw_score_arr
    selected_df["GlobalScore_norm"] = normalize_global_score(raw_score_arr, cfg)
    selected_df["GlobalScore_100"] = (selected_df["GlobalScore_norm"] * 100.0).astype(np.float32)
    selected_df["Safe_NIR_score"] = safe_nir_score[selected_indices]
    selected_df["Pred_lambda_nm"] = pred_lambda_nm[selected_indices]
    selected_df["Pred_lambda_std"] = pred_lambda_std[selected_indices]
    selected_df["WavelengthBoundary_bonus"] = wavelength_bonus[selected_indices]
    selected_df["Pred_logLLI"] = pred_log_lli[selected_indices]
    selected_df["Pred_logLLI_std"] = pred_log_lli_std[selected_indices]
    selected_df["Brightness_bonus"] = brightness_bonus[selected_indices]
    selected_df["Brightness_dim_risk"] = brightness_dim_risk[selected_indices]
    selected_df["P_NIR_mean"] = mean_probs[selected_indices, CLASS_ORDER.index("NIR")]
    selected_df["P_FarRed_mean"] = mean_probs[selected_indices, CLASS_ORDER.index("Far Red")]
    selected_df["P_Dark_mean"] = mean_probs[selected_indices, CLASS_ORDER.index("Dark")]
    selected_df["P_Green_mean"] = mean_probs[selected_indices, CLASS_ORDER.index("Green")]
    selected_df["P_Red_mean"] = mean_probs[selected_indices, CLASS_ORDER.index("Red")]
    selected_df["P_NIR_std"] = p_nir_std[selected_indices]
    for member_index in range(member_nir_probabilities.shape[1]):
        selected_df[f"P_NIR_member_{member_index + 1}"] = (
            member_nir_probabilities[selected_indices, member_index]
        )
    selected_df["Uncertainty_score"] = uncertainty_norm[selected_indices]
    selected_df["Diversity_score"] = np.asarray(selected_mix_div, dtype=np.float32)
    selected_df["Hamming_diversity_score"] = np.asarray(selected_hamming_div, dtype=np.float32)
    selected_df["Staple_diversity_score"] = np.asarray(selected_staple_div, dtype=np.float32)
    selected_df["Motif_prior_signed"] = motif_prior_signed[selected_indices]
    selected_df["Min_Hamming_to_Train"] = (hamming_div_train[selected_indices] * SEQ_LEN).astype(int)

    # 额外把重要 staple 子空间的列也导出，方便你后面人工复盘“为什么选了这些序列”。
    for feat in important_features:
        selected_df[feat] = candidate_df.iloc[selected_indices][feat].to_numpy()

    # ============================================================
    # 重新排列导出列顺序：
    # 1. 第一列放 10mer 序列
    # 2. 前面优先放路径来源、综合分数、概率、不确定性、多样性等核心信息
    # 3. 144 个结构描述符统一放到最后
    # ============================================================
    front_cols = [
        "Sequence",
        "Score",
        "Rank",
        "candidate_path",
        "GlobalScore_norm",
        "GlobalScore_100",
        "Safe_NIR_score",
        "P_NIR_mean",
        "P_FarRed_mean",
        "P_Dark_mean",
        "P_Green_mean",
        "P_Red_mean",
        "P_NIR_std",
        *[f"P_NIR_member_{index + 1}" for index in range(cfg.committee_size)],
        "Uncertainty_score",
        "Diversity_score",
        "Hamming_diversity_score",
        "Staple_diversity_score",
        "Pred_lambda_nm",
        "Pred_lambda_std",
        "WavelengthBoundary_bonus",
        "Pred_logLLI",
        "Pred_logLLI_std",
        "Brightness_bonus",
        "Brightness_dim_risk",
        "Motif_prior_signed",
        "Min_Hamming_to_Train",
    ]

    descriptor_cols = [f for f in STANDARD_FEATURES if f in selected_df.columns]

    middle_cols = [
        c for c in selected_df.columns
        if c not in front_cols and c not in descriptor_cols
    ]

    selected_df = selected_df[front_cols + middle_cols + descriptor_cols]

    return selected_df

# ============================================================
# 13. 导出
# ============================================================
# 从统一预打分结果中导出 top-N 高分候选池。
# 这里导出的不是最终 greedy batch，而是“进入最终选批前”的高分候选，
# 主要用于人工复盘候选来源、分数构成和路径分布。
def build_top_pool_dataframe(screened: Dict[str, np.ndarray], cfg: Config) -> pd.DataFrame:
    candidate_df = screened["candidate_df"]
    n = min(cfg.export_top_n_pool, len(candidate_df))
    order = np.argsort(screened["pre_score"])[::-1][:n]
    df = candidate_df.iloc[order].copy().reset_index(drop=True)
    mean_probs = screened["mean_probs"][order]
    member_nir_probabilities = screened["member_nir_probabilities"][order]
    df["PreScore"] = screened["pre_score"][order]
    df["GlobalPreScore_norm"] = normalize_global_score(df["PreScore"].to_numpy(dtype=np.float32), cfg)
    df["GlobalPreScore_100"] = (df["GlobalPreScore_norm"] * 100.0).astype(np.float32)
    df["Safe_NIR_score"] = screened["safe_nir_score"][order]
    df["Pred_lambda_nm"] = screened["pred_lambda_nm"][order]
    df["Pred_lambda_std"] = screened["pred_lambda_std"][order]
    df["WavelengthBoundary_bonus"] = screened["wavelength_bonus"][order]
    df["Pred_logLLI"] = screened["pred_log_lli"][order]
    df["Pred_logLLI_std"] = screened["pred_log_lli_std"][order]
    df["Brightness_bonus"] = screened["brightness_bonus"][order]
    df["Brightness_dim_risk"] = screened["brightness_dim_risk"][order]
    df["P_NIR_mean"] = mean_probs[:, CLASS_ORDER.index("NIR")]
    df["P_FarRed_mean"] = mean_probs[:, CLASS_ORDER.index("Far Red")]
    df["P_Dark_mean"] = mean_probs[:, CLASS_ORDER.index("Dark")]
    df["P_Green_mean"] = mean_probs[:, CLASS_ORDER.index("Green")]
    df["P_Red_mean"] = mean_probs[:, CLASS_ORDER.index("Red")]
    df["P_NIR_std"] = screened["p_nir_std"][order]
    for member_index in range(member_nir_probabilities.shape[1]):
        df[f"P_NIR_member_{member_index + 1}"] = member_nir_probabilities[:, member_index]
    df["Uncertainty_score"] = screened["uncertainty_norm"][order]
    df["Hamming_div_train"] = screened["hamming_div_train"][order]
    df["Staple_div_train"] = screened["staple_div_train"][order]
    df["Mixed_div_train"] = screened["mixed_div_train"][order]
    df["Motif_prior_signed"] = screened["motif_prior_signed"][order]
    return df


# 导出本轮真实推荐的全部结果文件。
# 包括：
# - recommended_sequences_{n}.csv：最终推荐序列，n 为当前训练集大小
# - top_candidate_pool.csv：高分候选池
# - run_summary.json：运行配置与关键信息
# - README_results.txt：文字说明
def export_results(
    out_dir: Path,
    cfg: Config,
    cleaned_df: pd.DataFrame,
    ignored_extra_cols: List[str],
    committee: CommitteeModel,
    screened: Dict[str, np.ndarray],
    selected_df: pd.DataFrame,
    runtime_seconds: float,
    prior_csv_path: str,
):
    n_training = len(cleaned_df)
    selected_df.to_csv(out_dir / f"recommended_sequences_{n_training}.csv", index=False, encoding="utf-8-sig")
    top_pool_df = build_top_pool_dataframe(screened, cfg)
    top_pool_df.to_csv(out_dir / "top_candidate_pool.csv", index=False, encoding="utf-8-sig")

    summary = {
        "config": asdict(cfg),
        "n_training_samples": int(len(cleaned_df)),
        "class_distribution": cleaned_df["class_label"].value_counts().to_dict(),
        "ignored_extra_columns": ignored_extra_cols,
        "committee_info": asdict(committee.info),
        "screen_stats": screened["stats"],
        "motif_prior_runtime": {
            "effective_use_motif_prior": cfg.use_motif_prior,
            "motif_prior_csv_used": prior_csv_path,
            "motif_prior_top_k": cfg.motif_prior_top_k,
            "motif_negative_scale": cfg.motif_negative_scale,
            "prior_gate_json": cfg.prior_gate_json,
        },
        "runtime_seconds": runtime_seconds,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_text = f"""本轮三路径主动学习结果说明
==========================

1. recommended_sequences_{len(cleaned_df)}.csv
   这是最终建议优先实验的 {cfg.recommend_k} 条序列。
   新版分数由以下六部分组成：
   - Safe_NIR_score：核心目标项，不再只是 P_NIR，而是“像 NIR 且不像 Far Red / Dark”
   - WavelengthBoundary_bonus：波长边界加成项，用预测主峰波长衡量“离 780 nm 是否接近”
   - Brightness_bonus：亮度加成项，用预测主峰亮度衡量“是否更值得优先实验”
   - Uncertainty_score：委员会分歧
   - Diversity_score：Hamming 与重要 staple 子空间的混合多样性
   - Motif_prior_signed：signed prior，不再只有正向奖励，也会惩罚 suppress_NIR 模式

2. top_candidate_pool.csv
   这是预评分最高的候选池，便于人工复核。
   它能帮助你看清楚：
  - 哪些是路径 A（局部开发）给出来的
  - 哪些是路径 B（de novo 探索）给出来的
  - 哪些是路径 C（随机覆盖）给出来的
  - 为什么它们预分高

3. run_summary.json
   保存了：
   - 当前训练集大小与类别分布
   - 委员会模型参数
   - 候选池三条路径的大小（A_local / B_denovo / C_random）
   - 实际使用的 signed prior 文件

当前训练集大小：{len(cleaned_df)}
当前轮次文件夹：{out_dir.name}
当前候选模式：{cfg.candidate_mode}
路径 A 是否开启：{cfg.path_a_enabled}
路径 B 是否开启：{cfg.path_b_enabled}
本轮是否启用 motif prior：{cfg.use_motif_prior}
本轮 signed prior 文件：{prior_csv_path}
运行总时间：{runtime_seconds:.2f} 秒
"""
    (out_dir / "README_results.txt").write_text(readme_text, encoding="utf-8")


# ============================================================
# 14. 主流程
# ============================================================
def main(cfg: Config):
    start_time = time.time()
    print("=" * 70)
    print("DNA-AgNC 10-mer 三路径主动学习脚本开始运行")
    print("=" * 70)

    # 1) 读数据
    print("[1/8] 读取并清洗训练集...")
    df = load_and_clean_data(cfg.input_csv)
    print(f"    当前训练集样本数：{len(df)}")
    print(f"    当前标签分布：{df['class_label'].value_counts().to_dict()}")

    # 2) 标准 144 特征
    print("[2/8] 检查 144 个标准 staple features...")
    X_df, feature_cols, ignored_extra_cols = select_standard_144_features(df)
    print(f"    输入特征数：{len(feature_cols)}")
    if ignored_extra_cols:
        print(f"    [提示] 额外列会被保留在 DataFrame 中，但不会进入委员会输入：{ignored_extra_cols[:10]}")

    # 3) 训练委员会
    print("[3/8] 训练五分类委员会...")
    committee = train_committee_model(X_df, df["class_label"], cfg)
    print(f"    委员会规模：{committee.info.committee_size}")
    print(f"    折内 ANOVA 特征数：{committee.info.anova_k}")
    print(f"    LDA shrinkage：{committee.info.lda_shrinkage}")
    print(f"    各成员温度参数：{committee.info.temperatures}")

    wavelength_models = None
    brightness_models = None
    brightness_train_log = None

    # ============================================================
    # 3.4) 训练亮度回归分支
    # ------------------------------------------------------------
    # 只使用“非 Dark 且有真实亮度”的样本。
    # 这一分支应该独立于波长分支存在，不能套在 use_wavelength_branch 里面。
    # ============================================================
    if cfg.use_brightness_branch:
        if cfg.brightness_col in df.columns:
            bright_mask = (
                    df["class_label"].ne("Dark")
                    & pd.to_numeric(df[cfg.brightness_col], errors="coerce").notna()
            )
            n_bright = int(bright_mask.sum())
            print(f"[3.4/8] 训练亮度回归分支，可用样本数：{n_bright}")

            if n_bright >= 10:
                brightness_train_raw = pd.to_numeric(
                    df.loc[bright_mask, cfg.brightness_col],
                    errors="coerce"
                ).to_numpy(dtype=np.float32)
                brightness_train_log = np.log1p(
                    np.clip(brightness_train_raw, a_min=0.0, a_max=None)
                )

                brightness_models = train_brightness_committee_model(
                    X=X_df.loc[bright_mask],
                    y_lli=df.loc[bright_mask, cfg.brightness_col],
                    cfg=cfg,
                )
            else:
                print("[提示] 可用于亮度回归的样本过少，本轮跳过亮度分支。")
        else:
            print(f"[提示] 当前训练集不含亮度列 {cfg.brightness_col}，本轮跳过亮度分支。")
    # ============================================================
    # 3.5) 训练波长回归分支
    # ------------------------------------------------------------
    # 只使用“非 Dark 且有真实波长”的样本。
    # 这个分支与亮度分支并列，不应控制亮度分支是否运行。
    # ============================================================
    if cfg.use_wavelength_branch:
        if cfg.wavelength_col in df.columns:
            lambda_mask = (
                    df["class_label"].ne("Dark")
                    & pd.to_numeric(df[cfg.wavelength_col], errors="coerce").notna()
            )
            n_lambda = int(lambda_mask.sum())
            print(f"[3.5/8] 训练波长回归分支，可用样本数：{n_lambda}")

            if n_lambda >= 10:
                wavelength_models = train_wavelength_committee_model(
                    X=X_df.loc[lambda_mask],
                    y_lambda=df.loc[lambda_mask, cfg.wavelength_col],
                    cfg=cfg,
                )
            else:
                print("[提示] 可用于波长回归的样本过少，本轮跳过波长分支。")
        else:
            print(f"[提示] 当前训练集不含波长列 {cfg.wavelength_col}，本轮跳过波长分支。")

    # 4) 读取禁用序列与 gating
    print("[4/8] 读取禁用序列和 prior gating 结果...")
    banned_sequences = load_banned_sequences(cfg.banned_sequences_path)
    cfg.use_motif_prior = resolve_effective_use_motif_prior(cfg)
    pos_weights, neg_weights, full_signed_prior_df, prior_csv_path = build_signed_prior_weights(cfg, len(df))
    print(f"    禁用序列数：{len(banned_sequences)}")
    print(f"    effective_use_motif_prior：{cfg.use_motif_prior}")
    if cfg.use_motif_prior:
        print(f"    promote_NIR motifs：{len(pos_weights)} 条")
        print(f"    suppress_NIR motifs：{len(neg_weights)} 条")
        print(f"    signed prior 文件：{prior_csv_path}")

    # 5) 生成候选池
    print("[5/8] 生成候选池...")
    if cfg.candidate_mode == "tripath":
        candidate_df = generate_tripath_candidates(
            train_df=df,
            cfg=cfg,
            pos_weights=pos_weights,
            neg_weights=neg_weights,
            full_signed_prior_df=full_signed_prior_df,
            banned_sequences=banned_sequences,
        )
    elif cfg.candidate_mode == "full_space":
        candidate_df = build_full_space_candidate_df(df["Sequence"].tolist(), banned_sequences, cfg)
    else:
        raise ValueError(f"未知 candidate_mode：{cfg.candidate_mode}")
    print(f"    候选池大小：{len(candidate_df)}")
    if "candidate_path" in candidate_df.columns:
        print(f"    路径来源分布：{candidate_df['candidate_path'].value_counts().to_dict()}")

    # 6) 统一打分
    print("[6/8] 对候选池做 committee 过滤与预打分...")
    screened = screen_candidate_space(
        committee=committee,
        wavelength_models=wavelength_models,
        brightness_models=brightness_models,
        brightness_train_log=brightness_train_log,
        train_df=df,
        candidate_df=candidate_df,
        cfg=cfg,
        pos_weights=pos_weights,
        neg_weights=neg_weights,
        full_signed_prior_df=full_signed_prior_df,
    )

    # 7) greedy batch selection
    print("[7/8] 进行最终 batch 推荐...")
    selected_df = greedy_select_batch(screened, df, cfg)
    print(selected_df[[
        "Sequence", "Rank", "candidate_path", "Score", "GlobalScore_norm",
        "Safe_NIR_score", "Pred_lambda_nm", "WavelengthBoundary_bonus",
        "Pred_logLLI", "Brightness_bonus", "Brightness_dim_risk",
        "P_NIR_mean", "P_FarRed_mean", "P_Dark_mean",
        "Diversity_score", "Motif_prior_signed"
    ]])

    # 8) 导出
    print("[8/8] 导出结果...")
    out_dir = make_output_dir(cfg, n_samples=len(df))
    runtime_seconds = time.time() - start_time
    export_results(
        out_dir=out_dir,
        cfg=cfg,
        cleaned_df=df,
        ignored_extra_cols=ignored_extra_cols,
        committee=committee,
        screened=screened,
        selected_df=selected_df,
        runtime_seconds=runtime_seconds,
        prior_csv_path=prior_csv_path,
    )
    print("-" * 70)
    print(f"运行完成，结果目录：{out_dir}")
    print(f"总耗时：{runtime_seconds:.2f} 秒")
    print("-" * 70)

# 下面这组参数是“直接运行本脚本”时使用的默认真实推荐配置。
# 平时最常改的通常是：
# input_csv、recommend_k、A/B/C 路径大小、candidate pruning 强度、
# 以及是否启用波长/亮度辅助分支。
if __name__ == "__main__":
    cfg = Config(
        # 如果你决定“从当前这一版新流程重新开始”，
        # 这里应该指向当前定义的初始训练集文件（例如 Initial120.csv），
        # 并把 initial_training_size 设回对应的初始样本数（例如 120）。
        # 一旦进入后续轮次，就只改 input_csv 为最新 verified 训练集，
        # 不要再改 initial_training_size。
        input_csv="Iteration3_156.csv",
        output_root="active_learning_outputs_tripath",
        banned_sequences_path="merged_unique_sequences.csv",
        initial_training_size=120,

        recommend_k=14,
        min_hamming_distance=0,

        candidate_mode="tripath",
        path_a_enabled=True,
        path_a_include_farred_seeds=True,
        path_a_max_three_mutations_per_seed=8,
        path_a_pool_size=20000,
        path_b_enabled=True,
        path_b_random_starts=10000,
        path_b_hillclimb_steps=12,
        path_b_min_cg_count=4,
        path_b_pool_size=20000,
        path_c_enabled=True,
        path_c_pool_size=5000,
        enable_candidate_pruning=True,
        candidate_prune_top_k_features=40,
        candidate_prune_min_staple_distance=0.05,

        base_model_name="anova16_shrinkage_lda",
        committee_size=5,
        committee_inner_folds=4,
        classifier_anova_k=16,
        classifier_lda_shrinkage=0.50,
        temperature_min=0.05,
        temperature_max=20.0,
        n_jobs=-1,
        use_gpu_backend=False,
        gpu_device="cuda",
        torch_logreg_max_iter=120,
        torch_reg_max_iter=120,
        random_seed=42,
        per_class_sample_size=None,
        feature_subsample_ratio=0.80,
        logreg_C=0.30,
        logreg_l1_ratio=0.30,
        logreg_max_iter=6000,
        batch_size=50000,

        w_target=0.375,
        w_uncertainty=0.20,
        w_diversity=0.30,
        w_motif=0.125,
        diversity_hamming_weight=0.50,
        diversity_staple_weight=0.50,
        staple_diversity_top_k=30,

        use_motif_prior=True,
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
        motif_prior_csv=None,
        motif_prior_top_k=15,
        motif_negative_scale=0.70,
        prior_gate_json="offline_replay_outputs_tripath_iter4_156/prior_gate_decision.json",
        fallback_use_motif_prior_when_gate_missing=True,

    )
    main(cfg)
