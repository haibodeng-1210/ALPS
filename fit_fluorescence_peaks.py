import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter


# ============================================================
# 0. 你最常改的参数
# ============================================================
MODE = "batch"  # "single" 或 "batch"

# 单文件模式参数
SAMPLE_FILE = "120-144 staple/iter 3 on/iter 3(1)/3-1.csv"

# 批量模式参数
DATA_FOLDER = "120-144 staple/iter 3 on/iter 3(2)"
FILE_PATTERN = "*.csv"

# 拟合参数
FIT_MIN_NM = 450
FIT_MAX_NM = 850
FORCE_N_PEAKS = None  # 全局默认峰数：None / 1 / 2 / 3
SAVE_PNG = True
PNG_FOLDER_NAME = "fit_png"

# 输出文件
SUMMARY_CSV = "all_fit_summary iter 3.csv"
FAILED_CSV = "fit_failed.csv"

# ----------------------------
# 按文件名手动指定峰数
# 只对这里列出的文件生效
# 其余文件继续自动拟合
# ----------------------------
MANUAL_PEAK_COUNT = {
    "3-3.csv": 2,
    "3-11.csv": 2
    # "xxx.csv": 1,
}


# ============================================================
# 1. 自然排序
# ============================================================
def natural_sort_key(path_obj):
    """
    自然排序键。
    例如：
    1-1.csv, 1-2.csv, ..., 1-10.csv, 1-11.csv
    而不是字符串排序的
    1-1.csv, 1-10.csv, 1-11.csv, 1-2.csv
    """
    name = Path(path_obj).name if not isinstance(path_obj, str) else Path(path_obj).name
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


# ============================================================
# 2. 标准 Gaussian（wavelength domain）
#    I(lambda) = a0 + sum A_i * exp(-(lambda-lambda_i)^2 / (2*sigma_i^2))
# ============================================================
def gaussian_sum_wavelength(x_nm, a0, *params):
    """
    params = [A1, lambda1_nm, sigma1_nm, A2, lambda2_nm, sigma2_nm, ...]
    """
    y = np.full_like(x_nm, a0, dtype=float)
    for i in range(0, len(params), 3):
        A = params[i]
        lambda0 = params[i + 1]
        sigma = params[i + 2]
        y += A * np.exp(-((x_nm - lambda0) ** 2) / (2.0 * sigma**2))
    return y


# ============================================================
# 3. 读取 Tecan 风格 CSV
#    前面可能有 metadata，后面才是数值型光谱
#    做法：把前两列转数值，非数值行自动丢弃
# ============================================================
def load_tecan_export_csv(csv_path):
    raw = pd.read_csv(csv_path, sep=None, engine="python")

    if raw.shape[1] < 2:
        raise ValueError(f"{csv_path} 读取后少于两列，无法处理。")

    col_x = raw.columns[0]
    col_y = raw.columns[1]

    x_nm = pd.to_numeric(raw[col_x], errors="coerce")
    y = pd.to_numeric(raw[col_y], errors="coerce")

    mask = x_nm.notna() & y.notna()
    x_nm = x_nm[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=float)

    if len(x_nm) == 0:
        raise ValueError(f"{csv_path} 没有读到有效数值型光谱数据。")

    order = np.argsort(x_nm)
    x_nm = x_nm[order]
    y = y[order]

    return x_nm, y


# ============================================================
# 4. 预处理
#    - 截取指定波长范围
#    - 保持 wavelength 升序
# ============================================================
def preprocess_wavelength_data(csv_path, fit_min_nm=400, fit_max_nm=850):
    x_nm, y = load_tecan_export_csv(csv_path)

    mask = (x_nm >= fit_min_nm) & (x_nm <= fit_max_nm)
    x_nm = x_nm[mask]
    y = y[mask]

    if len(x_nm) < 10:
        raise ValueError(f"{csv_path} 在 {fit_min_nm}-{fit_max_nm} nm 内有效点太少。")

    return x_nm, y


# ============================================================
# 5. 自动找峰
#    直接在 wavelength domain 上找峰位初值
# ============================================================
def auto_detect_peaks(x_nm, y, max_peaks=3):
    n = len(y)
    if n >= 11:
        win = min(21, n if n % 2 == 1 else n - 1)
        win = max(win, 7)
        if win % 2 == 0:
            win -= 1
        y_smooth = savgol_filter(y, window_length=win, polyorder=3)
    else:
        y_smooth = y.copy()

    dynamic = max(np.max(y_smooth) - np.min(y_smooth), 1e-12)

    prominence_main = 0.03 * dynamic
    distance_main = max(3, len(y_smooth) // 35)

    peaks_main, props_main = find_peaks(
        y_smooth,
        prominence=prominence_main,
        distance=distance_main,
    )

    if len(peaks_main) == 0:
        peaks_ranked = np.array([int(np.argmax(y_smooth))], dtype=int)
    else:
        prominences = props_main["prominences"]
        order = np.argsort(prominences)[::-1]
        peaks_ranked = peaks_main[order][:max_peaks]

    # 若只找到一个主峰，则额外尝试找 shoulder
    if len(peaks_ranked) == 1 and max_peaks >= 2:
        main_idx = int(peaks_ranked[0])

        prominence_shoulder = 0.015 * dynamic
        distance_shoulder = max(2, distance_main // 2)

        peaks_shoulder, _ = find_peaks(
            y_smooth,
            prominence=prominence_shoulder,
            distance=distance_shoulder,
        )

        shoulder_candidates = []
        for p in peaks_shoulder:
            # 至少相隔 20 nm，避免重复选到主峰附近
            if abs(x_nm[p] - x_nm[main_idx]) >= 20:
                shoulder_candidates.append(int(p))

        if len(shoulder_candidates) > 0:
            second_idx = max(shoulder_candidates, key=lambda p: y_smooth[p])
            peaks_ranked = np.array([main_idx, second_idx], dtype=int)

    peak_lambda_guesses = x_nm[peaks_ranked]

    return np.array(peak_lambda_guesses, dtype=float), y_smooth


# ============================================================
# 6. 构造拟合初值和边界
# ============================================================
def build_initial_guess(x_nm, y, lambda_guesses):
    if lambda_guesses is None or len(lambda_guesses) == 0:
        lambda_guesses = np.array([x_nm[np.argmax(y)]], dtype=float)

    y_dynamic = max(np.max(y) - np.min(y), 1.0)
    a0_guess = np.percentile(y, 5)

    p0 = [a0_guess]
    lower = [np.min(y) - y_dynamic]
    upper = [np.max(y)]

    for lambda_guess in np.sort(np.asarray(lambda_guesses, dtype=float)):
        idx = int(np.argmin(np.abs(x_nm - lambda_guess)))
        A_guess = max(y[idx] - a0_guess, 1.0)
        sigma_guess = 20.0

        p0.extend([A_guess, lambda_guess, sigma_guess])

        lower.extend([0.0, np.min(x_nm), 1.0])
        upper.extend([10 * y_dynamic, np.max(x_nm), 200.0])

    return np.array(p0, dtype=float), (np.array(lower, dtype=float), np.array(upper, dtype=float))


# ============================================================
# 7. 执行拟合
# ============================================================
def fit_model(x_nm, y, lambda_guesses):
    p0, bounds = build_initial_guess(x_nm, y, lambda_guesses)

    popt, pcov = curve_fit(
        lambda x, *params: gaussian_sum_wavelength(x, *params),
        x_nm,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=100000,
    )

    y_fit = gaussian_sum_wavelength(x_nm, *popt)
    return popt, pcov, y_fit


# ============================================================
# 8. 从拟合参数中提取各峰信息
#    不再计算 LII 面积，只保留峰位、峰高和 sigma
# ============================================================
def extract_peak_table(popt):
    a0 = popt[0]
    peaks = []

    n_peaks = (len(popt) - 1) // 3
    for i in range(n_peaks):
        A = popt[1 + 3 * i]
        lambda_nm = popt[1 + 3 * i + 1]
        sigma_nm = popt[1 + 3 * i + 2]

        peaks.append({
            "peak_index": i + 1,
            "A": float(A),
            "lambda_nm": float(lambda_nm),
            "sigma_nm": float(sigma_nm),
        })

    peaks = sorted(peaks, key=lambda d: d["lambda_nm"])
    for i, peak in enumerate(peaks, start=1):
        peak["peak_index"] = i
    return a0, peaks


# ============================================================
# 9. 选择主峰
#    现在按峰高 A 最大选，不再按面积选
# ============================================================
def choose_main_peak(peaks):
    return max(peaks, key=lambda d: d["A"])


# ============================================================
# 10. 画图
#    只画 wavelength domain
# ============================================================
def plot_fit(x_nm, y, popt, y_fit, save_png=None, title=None):
    a0 = popt[0]
    n_peaks = (len(popt) - 1) // 3

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))

    ax.plot(x_nm, y, lw=1.8, label="Raw spectrum")
    ax.plot(x_nm, y_fit, lw=2.2, label="Total fit")
    ax.axhline(a0, color="0.5", ls=":", lw=1.0, label="Baseline")

    for i in range(n_peaks):
        A = popt[1 + 3 * i]
        lambda0 = popt[1 + 3 * i + 1]
        sigma = popt[1 + 3 * i + 2]
        y_comp = a0 + A * np.exp(-((x_nm - lambda0) ** 2) / (2.0 * sigma**2))
        ax.plot(x_nm, y_comp, "--", lw=1.2, label=f"Peak {i + 1}")

    ax.set_xlabel("Wavelength / nm")
    ax.set_ylabel("Counts")
    ax.set_title("Wavelength domain")
    ax.legend(fontsize=8)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if save_png is not None:
        plt.savefig(save_png, dpi=300, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# 11. helper：从自动找峰结果中选择指定数量的初值
# ============================================================
def select_lambda_guesses(auto_lambda_guesses, x_nm, y, n_needed):
    """
    从 auto_detect_peaks 给出的 lambda 初值中取前 n_needed 个。
    如果不够，就从原始 y 的高点里补。
    """
    lambda_guesses = list(np.asarray(auto_lambda_guesses, dtype=float)[:n_needed])

    if len(lambda_guesses) < n_needed:
        extra_idx = np.argsort(y)[::-1]
        for idx in extra_idx:
            lambda_extra = float(x_nm[idx])

            # 避免重复补到已经有的峰附近
            if all(abs(lambda_extra - lam) > 20.0 for lam in lambda_guesses):
                lambda_guesses.append(lambda_extra)

            if len(lambda_guesses) == n_needed:
                break

    return np.array(lambda_guesses, dtype=float)


# ============================================================
# 12. helper：按文件名决定实际使用的峰数
#    优先级：
#    1. 按文件名单独指定
#    2. 单文件/全局 FORCE_N_PEAKS
#    3. 自动判定
# ============================================================
def resolve_effective_peak_count(csv_path, force_n_peaks=None, manual_peak_count=None):
    filename = Path(csv_path).name

    if manual_peak_count is None:
        manual_peak_count = {}

    if filename in manual_peak_count:
        peak_count = manual_peak_count[filename]
        if peak_count not in [1, 2, 3]:
            raise ValueError(f"MANUAL_PEAK_COUNT 里 {filename} 的峰数只能是 1/2/3。")
        return peak_count, "manual"

    if force_n_peaks is not None:
        if force_n_peaks not in [1, 2, 3]:
            raise ValueError("force_n_peaks 只能取 None、1、2、3。")
        return force_n_peaks, "global"

    return None, "auto"


# ============================================================
# 13. helper：拟合一个候选模型并返回 BIC
# ============================================================
def fit_model_candidate(x_nm, y, lambda_guesses):
    popt, pcov, y_fit = fit_model(x_nm, y, lambda_guesses)

    rss = np.sum((y - y_fit) ** 2)
    n = len(y)
    k = len(popt)
    bic = n * np.log(rss / n + 1e-12) + k * np.log(n)

    return popt, pcov, y_fit, rss, bic


# ============================================================
# 14. 拟合单个 CSV
#    - wavelength domain 拟合
#    - 自动做候选峰数竞争
#    - 支持按文件名单独手动指定峰数
# ============================================================
def fit_one_csv(
    csv_path,
    force_n_peaks=None,
    fit_min_nm=400,
    fit_max_nm=850,
    save_png=True,
    png_dir=None,
    manual_peak_count=None,
):
    x_nm, y = preprocess_wavelength_data(
        csv_path=csv_path,
        fit_min_nm=fit_min_nm,
        fit_max_nm=fit_max_nm,
    )

    auto_lambda_guesses, y_smooth = auto_detect_peaks(
        x_nm=x_nm,
        y=y,
        max_peaks=3,
    )

    effective_force_n_peaks, peak_count_source = resolve_effective_peak_count(
        csv_path=csv_path,
        force_n_peaks=force_n_peaks,
        manual_peak_count=manual_peak_count,
    )

    if effective_force_n_peaks is None:
        max_try = min(len(auto_lambda_guesses), 3)
        if max_try == 0:
            candidate_ns = [1]
        else:
            candidate_ns = list(range(1, max_try + 1))
    else:
        candidate_ns = [effective_force_n_peaks]

    best = None
    candidate_errors = []

    for n_try in candidate_ns:
        try:
            lambda_guesses = select_lambda_guesses(auto_lambda_guesses, x_nm, y, n_try)
            popt, pcov, y_fit, rss, bic = fit_model_candidate(x_nm, y, lambda_guesses)
            a0, peaks = extract_peak_table(popt)
            main_peak = choose_main_peak(peaks)

            result = {
                "n_try": n_try,
                "lambda_guesses": lambda_guesses,
                "popt": popt,
                "pcov": pcov,
                "y_fit": y_fit,
                "bic": bic,
                "a0": a0,
                "peaks": peaks,
                "main_peak": main_peak,
            }

            if best is None or bic < best["bic"]:
                best = result

        except Exception as e:
            candidate_errors.append(f"{n_try} peaks: {e}")

    if best is None:
        error_text = "; ".join(candidate_errors) if candidate_errors else "unknown error"
        raise RuntimeError(f"{csv_path} 无法完成有效拟合。{error_text}")

    popt = best["popt"]
    y_fit = best["y_fit"]
    a0 = best["a0"]
    peaks = best["peaks"]
    main_peak = best["main_peak"]

    png_path = None
    if save_png:
        if png_dir is None:
            png_dir = Path(csv_path).parent / PNG_FOLDER_NAME

        png_dir = Path(png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

        png_path = png_dir / f"{Path(csv_path).stem}_fit.png"

        plot_fit(
            x_nm=x_nm,
            y=y,
            popt=popt,
            y_fit=y_fit,
            save_png=str(png_path),
            title=Path(csv_path).name,
        )

    row = {
        "file": Path(csv_path).name,
        "n_peaks_used": len(peaks),
        "selected_peak_index": main_peak["peak_index"],
    }

    for i, p in enumerate(peaks, start=1):
        row[f"peak{i}_lambda_nm"] = p["lambda_nm"]
        row[f"peak{i}_value"] = p["A"]

    return row, peaks


# ============================================================
# 15. 单文件模式
# ============================================================
def process_single_sample(
    sample_file,
    force_n_peaks=None,
    fit_min_nm=400,
    fit_max_nm=850,
    save_png=True,
    manual_peak_count=None,
):
    sample_row, sample_peaks = fit_one_csv(
        csv_path=sample_file,
        force_n_peaks=force_n_peaks,
        fit_min_nm=fit_min_nm,
        fit_max_nm=fit_max_nm,
        save_png=save_png,
        manual_peak_count=manual_peak_count,
    )

    return {
        "sample_row": sample_row,
        "sample_peaks": sample_peaks,
    }


# ============================================================
# 16. 批量模式
# ============================================================
def should_skip_batch_file(path_obj, summary_csv=SUMMARY_CSV, failed_csv=FAILED_CSV):
    filename = Path(path_obj).name
    if filename in {summary_csv, failed_csv}:
        return True
    if filename.startswith("all_fit_summary"):
        return True
    if filename.startswith("fit_failed"):
        return True
    return False


def format_peak_summary(peaks):
    return "; ".join(
        f"{p['lambda_nm']:.2f} nm / A={p['A']:.2f}"
        for p in peaks
    )


def batch_process(
    folder,
    pattern="*.csv",
    force_n_peaks=None,
    fit_min_nm=400,
    fit_max_nm=850,
    save_png=True,
    summary_csv="all_fit_summary.csv",
    failed_csv="fit_failed.csv",
    manual_peak_count=None,
):
    folder = Path(folder)
    files = sorted(folder.glob(pattern), key=natural_sort_key)
    files = [f for f in files if not should_skip_batch_file(f, summary_csv, failed_csv)]

    if len(files) == 0:
        raise FileNotFoundError(f"{folder} 下没有找到匹配 {pattern} 的样品 CSV 文件。")

    png_dir = folder / PNG_FOLDER_NAME

    rows = []
    failed = []

    for f in files:
        try:
            row, peaks = fit_one_csv(
                csv_path=f,
                force_n_peaks=force_n_peaks,
                fit_min_nm=fit_min_nm,
                fit_max_nm=fit_max_nm,
                save_png=save_png,
                png_dir=png_dir,
                manual_peak_count=manual_peak_count,
            )

            print(
                f"[OK] {f.name} -> "
                f"n_peaks={row['n_peaks_used']}, "
                f"selected_peak=peak{row['selected_peak_index']}, "
                f"peaks={format_peak_summary(peaks)}"
            )

            rows.append(row)

        except Exception as e:
            failed.append({
                "file": f.name,
                "error": str(e),
            })
            print(f"[FAILED] {f.name} -> {e}")

    rows = sorted(rows, key=lambda r: natural_sort_key(r["file"]))
    df = pd.DataFrame(rows)
    summary_path = folder / summary_csv
    df.to_csv(summary_path, index=False)

    failed_path = folder / failed_csv
    if failed:
        pd.DataFrame(failed).to_csv(failed_path, index=False)
    elif failed_path.exists():
        failed_path.unlink()

    print("\n========================================")
    print(f"样品汇总表已保存: {summary_path}")
    if failed:
        print(f"失败文件记录已保存: {failed_path}")
    print("========================================")

    return df, failed


# ============================================================
# 17. 主程序入口
# ============================================================
if __name__ == "__main__":
    if MODE == "single":
        result = process_single_sample(
            sample_file=SAMPLE_FILE,
            force_n_peaks=FORCE_N_PEAKS,
            fit_min_nm=FIT_MIN_NM,
            fit_max_nm=FIT_MAX_NM,
            save_png=SAVE_PNG,
            manual_peak_count=MANUAL_PEAK_COUNT,
        )

        sample_row = result["sample_row"]
        sample_peaks = result["sample_peaks"]

        print("\n================ 样品结果 ================")
        print("file =", sample_row["file"])
        print("n_peaks_used =", sample_row["n_peaks_used"])
        print("selected_peak_index =", sample_row["selected_peak_index"])

        print("\n================ 样品所有分峰 ================")
        for p in sample_peaks:
            print(
                f"Peak {p['peak_index']}: "
                f"lambda = {p['lambda_nm']:.2f} nm, "
                f"A = {p['A']:.6f}"
            )

    elif MODE == "batch":
        df, failed = batch_process(
            folder=DATA_FOLDER,
            pattern=FILE_PATTERN,
            force_n_peaks=FORCE_N_PEAKS,
            fit_min_nm=FIT_MIN_NM,
            fit_max_nm=FIT_MAX_NM,
            save_png=SAVE_PNG,
            summary_csv=SUMMARY_CSV,
            failed_csv=FAILED_CSV,
            manual_peak_count=MANUAL_PEAK_COUNT,
        )

    else:
        raise ValueError("MODE 只能是 'single' 或 'batch'。")
