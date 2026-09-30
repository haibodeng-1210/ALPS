import pandas as pd


# =========================
# 1. 修改这里的文件路径
# =========================
INPUT_CSV = "120-144 staple/iter 3 on/iter3.csv"
OUTPUT_CSV = "120-144 staple/iter 3 on/iter3_with_144_staple_features.csv"

SEQUENCE_COL = "Sequence"
REQUIRE_10MER = True


# =========================
# 2. 生成 144 个特征名
# 顺序严格为：
# A_0A A_0C ... T_0T
# A_1A A_1C ... T_1T
# ...
# A_8A A_8C ... T_8T
# =========================
BASES = ["A", "C", "G", "T"]


def get_144_staple_feature_names():
    feature_names = []

    for gap in range(9):
        for left_base in BASES:
            for right_base in BASES:
                feature_names.append(f"{left_base}_{gap}{right_base}")

    return feature_names


# =========================
# 3. 对单条序列提取 144 个特征
# =========================
def extract_144_staple_features(sequence):
    seq = str(sequence).strip().upper()

    if REQUIRE_10MER and len(seq) != 10:
        raise ValueError(f"序列 {seq} 的长度为 {len(seq)}，不是 10-mer。")

    invalid_bases = set(seq) - set(BASES)
    if invalid_bases:
        raise ValueError(f"序列 {seq} 中包含非法碱基: {invalid_bases}。只允许 A, C, G, T。")

    feature_names = get_144_staple_feature_names()
    features = {name: 0 for name in feature_names}

    sequence_length = len(seq)

    for gap in range(9):
        distance = gap + 1

        for i in range(sequence_length - distance):
            left_base = seq[i]
            right_base = seq[i + distance]

            feature_name = f"{left_base}_{gap}{right_base}"
            features[feature_name] += 1

    return features


# =========================
# 4. 读取 CSV 并把 144 个特征接到后面
# =========================
def main():
    df = pd.read_csv(INPUT_CSV, dtype={SEQUENCE_COL: str})

    if SEQUENCE_COL not in df.columns:
        raise ValueError(f"输入文件中没有找到列名 {SEQUENCE_COL}。请检查第一列列名。")

    feature_names = get_144_staple_feature_names()

    feature_rows = []

    for seq in df[SEQUENCE_COL]:
        feature_dict = extract_144_staple_features(seq)
        feature_rows.append(feature_dict)

    feature_df = pd.DataFrame(feature_rows, columns=feature_names)

    output_df = pd.concat([df, feature_df], axis=1)

    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"处理完成：{OUTPUT_CSV}")
    print(f"原始列数：{df.shape[1]}")
    print(f"新增 staple 特征数：{feature_df.shape[1]}")
    print(f"最终列数：{output_df.shape[1]}")


if __name__ == "__main__":
    main()