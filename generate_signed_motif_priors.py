# -*- coding: utf-8 -*-
"""
对当前训练集中的 144 个标准 staple features 做“针对 NIR 的 signed importance 分析”。

这份脚本是新版三路径主动学习流程的配套脚本，作用有三层：
1. 给运行主脚本提供 signed prior 文件；
2. 明确区分 promote_NIR 与 suppress_NIR 特征；
3. 为后面的 staple-space diversity / candidate pruning 提供 abs_importance 排序依据。

这里：
- 不再把“除了 Sequence / class_label 之外的所有列”都当成特征；
- 而是显式只读取 144 个标准 staple features。

这样以后即使训练表里增加了 brightness、备注列、实验批次号等字段，
也不会污染 importance 计算。
"""

from pathlib import Path
import importlib.util
import sys
import pandas as pd
import time
from scipy.stats import pointbiserialr

THIS_DIR = Path(__file__).resolve().parent
# 动态导入主动学习脚本，而不是在本文件里手写 STANDARD_FEATURES，
# 是为了保证 importance 分析脚本与主流程始终共用同一套 144 维特征定义。
RUN_SCRIPT = THIS_DIR / "run_dna_agn_active_learning_tripath.py"

spec = importlib.util.spec_from_file_location("al_tripath", RUN_SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# 从主主动学习脚本中直接复用标准 144 个 staple feature 名称，
# 这样可以保证：
# 1. importance 分析脚本与主流程使用完全同一套特征定义；
# 2. 后续即使主脚本里标准特征顺序有调整，这里也不会手工写错。
STANDARD_FEATURES = module.STANDARD_FEATURES

# ============================================================
# 1. 你通常只需要改这里
# ============================================================
INPUT_CSV = Path("Iteration3_156.csv")
OUTPUT_DIR = Path("全集重要性分析")
LABEL_COL = "class_label"
SEQ_COL = "Sequence"
TARGET_CLASS = "NIR"
# INITIAL_TRAIN_SIZE 用来判断当前输入文件是否是“初始训练集”。
# 若样本数正好等于这个值，则输出文件名使用 small；
# 否则输出文件名使用 train{样本数}，便于和后续各轮训练集区分。
INITIAL_TRAIN_SIZE = 120
WAVELENGTH_COL = "main_peak_lambda_nm"
BRIGHTNESS_COL = "main_peak_lli_raw"
# ============================================================
# 2. 读取数据，并校验后续分析只使用 144 个标准特征
# ============================================================
start_time = time.time()
print("开始执行 signed importance 分析 ...")


df = module.load_and_clean_data(str(INPUT_CSV))
required = {SEQ_COL, LABEL_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"输入文件缺少必要列：{missing}")
if WAVELENGTH_COL in df.columns:
    print(f"[提示] 当前训练集包含波长列：{WAVELENGTH_COL}，但本脚本不会把它当作输入特征。")
else:
    print(f"[提示] 当前训练集不包含波长列：{WAVELENGTH_COL}，这不会影响 signed importance 计算。")
if BRIGHTNESS_COL in df.columns:
    print(f"[提示] 当前训练集包含亮度列：{BRIGHTNESS_COL}，但本脚本不会把它当作输入特征。")
else:
    print(f"[提示] 当前训练集不包含亮度列：{BRIGHTNESS_COL}，这不会影响 signed importance 计算。")

# 这里不直接把 df 裁剪成只剩 144 列，
# 而是仅显式指定“后续 importance 计算要遍历哪些标准特征”。
# 这样额外列仍可保留用于打印提示或人工核对，
# 但不会进入 signed importance 的统计过程。
feature_cols = [c for c in STANDARD_FEATURES if c in df.columns]
missing_feat = [c for c in STANDARD_FEATURES if c not in df.columns]
if missing_feat:
    raise ValueError(f"输入文件缺少标准 144 特征中的一部分，示例：{missing_feat[:10]}")

# 根据当前训练集大小自动决定输出标签
if len(df) == INITIAL_TRAIN_SIZE:
    output_tag = "small"
else:
    output_tag = f"train{len(df)}"

# 一共导出三份结果：
# 1. 完整 signed importance 排序表；
# 2. promote_NIR 最强的前 15 个特征；
# 3. suppress_NIR 最强的前 15 个特征。
# 主流程通常读取完整表；前 15 表主要用于人工检查和快速复盘。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / f"nir_feature_importance_signed_{output_tag}.csv"
OUTPUT_POS15_CSV = OUTPUT_DIR / f"nir_feature_importance_positive15_{output_tag}.csv"
OUTPUT_NEG15_CSV = OUTPUT_DIR / f"nir_feature_importance_negative15_{output_tag}.csv"

# ============================================================
# 3. NIR vs non-NIR 的二元标签
# ============================================================
# 这里把问题转成“当前特征是否更偏向 NIR”这一二元任务：
# NIR 记为 1，其它所有颜色类别统一记为 0。
# 这样 point-biserial correlation 的符号就可以直接解释为：
# 正值更偏向 promote_NIR，负值更偏向 suppress_NIR。
y = (df[LABEL_COL].astype(str).str.strip() == TARGET_CLASS).astype(int)
if int(y.sum()) == 0:
    raise ValueError("当前训练集中没有 NIR 样本，无法计算针对 NIR 的 signed importance。")

# ============================================================
# 4. 点双列相关：得到天然带方向的 importance
# ============================================================
results = []
for col in feature_cols:
    x = pd.to_numeric(df[col], errors="coerce")
    valid_mask = x.notna()
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # 若某个特征在当前训练集里几乎是常数，
    # 则它无法提供有效相关性信息，此时直接记为 0。
    if x_valid.nunique() <= 1:
        r = 0.0
        p_value = 1.0
    else:
        r, p_value = pointbiserialr(y_valid, x_valid)
        if pd.isna(r):
            r = 0.0
        if pd.isna(p_value):
            p_value = 1.0

    mean_nir = x[y == 1].mean()
    mean_non_nir = x[y == 0].mean()
    results.append({
        "feature": col,
        "signed_importance": float(r),
        "abs_importance": float(abs(r)),
        "direction": "promote_NIR" if r > 0 else ("suppress_NIR" if r < 0 else "neutral"),
        "mean_in_NIR": float(mean_nir),
        "mean_in_nonNIR": float(mean_non_nir),
        "difference_mean": float(mean_nir - mean_non_nir),
        "p_value": float(p_value),
    })

# 完整表按 signed_importance 从大到小排序：
# - 越靠前，越偏向 promote_NIR；
# - 越靠后，越偏向 suppress_NIR。
# 同时另外单独导出正向前 15 和负向前 15，
# 供主流程读取 signed prior，或供人工检查当前训练集的 motif 倾向。
result_df = pd.DataFrame(results).sort_values(by="signed_importance", ascending=False).reset_index(drop=True)
positive15_df = (
    result_df[result_df["signed_importance"] > 0]
    .sort_values(by="signed_importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)
negative15_df = (
    result_df[result_df["signed_importance"] < 0]
    .sort_values(by="signed_importance", ascending=True)
    .head(15)
    .reset_index(drop=True)
)

result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
positive15_df.to_csv(OUTPUT_POS15_CSV, index=False, encoding="utf-8-sig")
negative15_df.to_csv(OUTPUT_NEG15_CSV, index=False, encoding="utf-8-sig")

print("\n=== 训练集概况 ===")
print(f"输入文件：{INPUT_CSV}")
print(f"样本数：{len(df)}")
print(f"NIR 样本数：{int(y.sum())}")
print(f"非 NIR 样本数：{int((1 - y).sum())}")
print(f"标准特征数：{len(feature_cols)}")
print(f"输出标签：{output_tag}")

print("\n=== promote_NIR 最强的前 15 个特征 ===")
print(positive15_df.to_string(index=False) if not positive15_df.empty else "无")

print("\n=== suppress_NIR 最强的前 15 个特征 ===")
print(negative15_df.to_string(index=False) if not negative15_df.empty else "无")

print(f"\n完整结果已保存到：{OUTPUT_CSV}")
print(f"正向前15已保存到：{OUTPUT_POS15_CSV}")
print(f"负向前15已保存到：{OUTPUT_NEG15_CSV}")
elapsed = time.time() - start_time
print(f"运行完成，总耗时：{elapsed:.2f} 秒")
