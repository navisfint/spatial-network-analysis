# -*- coding: utf-8 -*-
"""
common.py — 公共工具：数据读取 / 标准化 / 赋权 / 中文绘图 / 地理距离

设计原则：只保留两个项目真正共用的部分，不做过度封装。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 中文字体（Windows 优先 SimHei / 微软雅黑）
# ---------------------------------------------------------------------------
def setup_cn_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]:
        try:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# ---------------------------------------------------------------------------
# 数据读取：支持长表 / 宽表自动识别
# ---------------------------------------------------------------------------
def read_panel(path: str, sheet=0) -> pd.DataFrame:
    """读取面板数据，统一为长表 [地区, 年份, 指标, 值]。

    宽表：地区, 年份, 指标1, 指标2, ...（列名即指标名）
    长表：地区, 年份, 指标, 值
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    df.columns = [str(c).strip() for c in df.columns]
    cols = set(df.columns)

    if {"地区", "年份", "指标", "值"} <= cols:
        out = df[["地区", "年份", "指标", "值"]].copy()
    elif {"地区", "年份"} <= cols:
        id_cols = ["地区", "年份"]
        val_cols = [c for c in df.columns if c not in id_cols]
        out = df.melt(id_vars=id_cols, value_vars=val_cols,
                      var_name="指标", value_name="值")
    else:
        raise ValueError(
            f"无法识别的表结构。需要列 [地区,年份,指标,值] 或 [地区,年份,指标1,...]。"
            f"当前列：{list(df.columns)}"
        )

    out["地区"] = out["地区"].astype(str).str.strip()
    out["指标"] = out["指标"].astype(str).str.strip()
    out["年份"] = pd.to_numeric(out["年份"], errors="coerce").astype("Int64")
    out["值"] = pd.to_numeric(out["值"], errors="coerce")
    out = out.dropna(subset=["年份", "值"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 缺省值填补：按地区用年平均增长率外推/内插，兜底用指标均值
# ---------------------------------------------------------------------------
def fill_missing(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["地区", "指标", "年份"]).copy()
    df["值"] = pd.to_numeric(df["值"], errors="coerce").astype(float)
    grp = df.groupby(["地区", "指标"], sort=False)["值"]
    # 线性内插（等价于按年均增长率补齐的简化实现），再前后向填充
    df["值"] = grp.transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    df["值"] = df.groupby(["地区", "指标"], sort=False)["值"].transform(
        lambda s: s.ffill().bfill())
    # 跨地区兜底
    df["值"] = df["值"].fillna(df.groupby(["年份", "指标"])["值"].transform("mean"))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 标准化
# ---------------------------------------------------------------------------
def standardize(panel: pd.DataFrame, direction: dict | None = None,
                method: str = "minmax", by_year: bool = False) -> pd.DataFrame:
    """把原始值转成可比的正向标准化值 X。

    method:
      minmax : (x-min)/(max-min)          → X ∈ [0,1]，零值会成为 0
      max    : x/max                      → X ∈ (0,1]，无零值
      mean   : x/mean                     → X > 0，均值化，常用于"相对标准化"
      zscore : 仅作对照，会破坏两模型的非负前提（本函数会直接报错）

    direction: {指标: '+'/'-'}，'-' 表示逆向指标，标准化后取 1-X 翻转。
    by_year : True 表示逐年标准化（跨地区比较）；False 表示全样本统一标准化。
    """
    df = panel.copy()
    df = fill_missing(df)
    direction = direction or {}

    if method == "zscore":
        raise ValueError(
            "zscore 标准化会产生负值，传统障碍度模型的 (1-X) 和分母会失去意义，"
            "改进模型同样会出现负的驱动度。两个模型都必须建立在非负标准化之上，"
            "请改用 minmax / max / mean。"
        )

    keys = ["年份", "指标"] if by_year else ["指标"]

    def _norm(s: pd.Series) -> pd.Series:
        x = s.astype(float)
        if method == "minmax":
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=x.index)
        if method == "max":
            mx = x.max()
            return x / mx if mx != 0 else pd.Series(0.0, index=x.index)
        if method == "mean":
            mu = x.mean()
            return x / mu if mu != 0 else pd.Series(0.0, index=x.index)
        raise ValueError(f"未知标准化方法: {method}")

    df["X"] = df.groupby(keys, sort=False)["值"].transform(_norm).astype(float)

    # 逆向指标翻转（组内最大值 - X）
    neg = {k for k, v in direction.items() if str(v).strip() == "-"}
    if neg:
        mask = df["指标"].isin(neg)
        if mask.any():
            mx = df.loc[mask].groupby(keys, sort=False)["X"].transform("max")
            df.loc[mask, "X"] = mx - df.loc[mask, "X"]

    # 收尾：重新压到 [0,1]，保证非负且组内有区分度
    df["X"] = df.groupby(keys, sort=False)["X"].transform(_norm).astype(float)
    return df


# ---------------------------------------------------------------------------
# 赋权
# ---------------------------------------------------------------------------
def compute_weights(panel: pd.DataFrame, method: str = "cv",
                    given: dict | None = None, base: str | None = None) -> pd.DataFrame:
    """返回 DataFrame [指标, 权重]。

    method:
      cv          : 变异系数法（文献 [A] 主用）
      entropy     : 熵权法
      critic      : CRITIC 法
      independent : 独立性权重法（复相关系数倒数）
      given       : 直接采用外部给定权重

    base —— 权重算在哪个量上，这一步很关键：
      "raw" : 原始值。CV 本身无量纲（σ/μ），在原始值上算才反映指标真实的相对离散度，
              这是文献口径，也是唯一能复现论文权重结构的做法。
      "std" : 标准化后的 X。注意 min-max 会把每个指标的极差强行拉成 1，
              各指标的 CV 因此被拉平，权重会退化成近似等权 —— 除非你明确知道要这个效果，
              否则不要用。
    base=None 时按方法取默认：cv/entropy → raw，critic/independent → std。
    """
    if method == "given":
        if given is None:
            raise ValueError("method='given' 时必须提供 given 权重字典")
        w = pd.DataFrame({"指标": list(given.keys()),
                          "权重": [float(v) for v in given.values()]})
        return _normalize_w(w)

    if base is None:
        base = "raw" if method in ("cv", "entropy") else "std"
    if base == "raw":
        if "值" not in panel.columns:
            raise ValueError("base='raw' 需要 panel 含原始值列『值』")
        col = "值"
        piv = panel.pivot_table(index=["地区", "年份"], columns="指标",
                                values=col, aggfunc="mean")
        if method == "entropy":
            # 熵权法要求非负：原始值若全正可直接用，否则退化为标准化值
            if (piv < 0).any().any():
                base = "std"
            else:
                piv = piv.clip(lower=0)
    if base == "std":
        if "X" not in panel.columns:
            raise ValueError("base='std' 需要 panel 已标准化（含 X 列）")
        piv = panel.pivot_table(index=["地区", "年份"], columns="指标",
                                values="X", aggfunc="mean")

    n = piv.shape[0]

    if method == "cv":
        # 变异系数 = 标准差 / 均值（无量纲，尺度不变）
        mu = piv.mean()
        sd = piv.std(ddof=1)
        raw = (sd / mu.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        raw = raw.fillna(0.0).clip(lower=0)
    elif method == "entropy":
        p = piv.clip(lower=0)
        p = p / p.sum(axis=0).replace(0, np.nan)
        p = p.fillna(0.0)
        k = 1.0 / math.log(n) if n > 1 else 0.0
        e = -(p * np.log(p.replace(0, np.nan))).sum(axis=0).fillna(0.0) * k
        raw = (1 - e).clip(lower=0)
    elif method == "critic":
        sd = piv.std(ddof=1).fillna(0.0)
        corr = piv.corr().abs().fillna(0.0)
        conflict = (1 - corr).sum(axis=0)          # Σ(1-|r_ij|)
        raw = sd * conflict
    elif method == "independent":
        corr = piv.corr().fillna(0.0)
        try:
            inv = np.linalg.pinv(corr.values + np.eye(corr.shape[0]) * 1e-8)
            r2 = np.clip(1 - 1.0 / np.diag(inv), 0, 0.999)
            raw = pd.Series(1.0 / (1.0 - r2), index=corr.columns)
        except Exception:
            raw = pd.Series(1.0, index=corr.columns)
    else:
        raise ValueError(f"未知赋权方法: {method}")

    w = pd.DataFrame({"指标": raw.index, "权重": raw.values})
    return _normalize_w(w)


def _normalize_w(w: pd.DataFrame) -> pd.DataFrame:
    s = w["权重"].astype(float).clip(lower=0)
    if s.sum() <= 0:
        s = pd.Series(np.ones(len(s)) / len(s), index=s.index)
    w = w.copy()
    w["权重"] = s / s.sum()
    return w.sort_values("权重", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 地理距离
# ---------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def distance_matrix(provinces: list[str], lonlat: dict) -> pd.DataFrame:
    """按代表点经纬度生成 n×n 地理距离矩阵（公里）。"""
    missing = [p for p in provinces if p not in lonlat]
    if missing:
        raise KeyError(f"缺经纬度：{missing}。请在 indicator_ref.PROVINCE_LONLAT 中补充。")
    n = len(provinces)
    M = np.zeros((n, n))
    for i, a in enumerate(provinces):
        for j, b in enumerate(provinces):
            if i == j:
                M[i, j] = 0.0
            elif j > i:
                lo1, la1 = lonlat[a]
                lo2, la2 = lonlat[b]
                M[i, j] = M[j, i] = haversine_km(lo1, la1, lo2, la2)
    return pd.DataFrame(M, index=provinces, columns=provinces)


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------
def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def write_excel(sheets: dict, path: str):
    """sheets: {sheet名: DataFrame}"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            safe = str(name)[:31]
            df.to_excel(w, sheet_name=safe, index=False)


def say(*a):
    print(*a)
    sys.stdout.flush()
