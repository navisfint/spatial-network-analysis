# -*- coding: utf-8 -*-
"""
network_analysis.py — 修正引力模型 + 社会网络分析

覆盖文献：
  [B] 魏巍(2025). 我国共同富裕水平测度、演进态势及空间关联.
      → 引力模型：P_ij = g_ij · (G_i·G_j) / D_ij² ,  g_ij = G_i/(G_i+G_j)
      → 二值化：与所在行均值比较（行均值法）
      → 整体网络（密度/关联度）、个体网络（三种中心度）、CONCOR 板块
  [C] 王梦宇等(2024). 中国共同富裕水平的空间关联网络结构及演进特征.
      → 引力模型：R_ij = k_ij · (V_i·V_j) / [ D_ij / |pgdp_i - pgdp_j| ]^λ ,
                   k_ij = V_i/(V_i+V_j), λ = 2   （经济地理距离）
      → 二值化：固定值阈值
      → 整体网络（密度/关联度/等级度/效率/小世界）、个体网络、核心-边缘结构

用法（零参数可直接跑通）：
    python network_analysis.py
    python network_analysis.py --index ../data/我的综合指数.csv --year 2023
    python network_analysis.py --raw ../data/demo_共同富裕面板.csv --pgdp ../data/demo_人均GDP.csv

输出（写到 ../output/）：network_result.xlsx + network_<年份>.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from indicator_ref import CP_SYSTEM, PROVINCE_LONLAT, FOUR_REGIONS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")

CP_DIRECTION = {t[0]: t[2] for t in CP_SYSTEM}


# ===========================================================================
# 1. 修正引力模型
# ===========================================================================
def gravity_matrix(level: pd.Series, D: pd.DataFrame,
                   pgdp: pd.Series | None = None, kind: str = "geo",
                   lam: float = 2.0, eps_rel: float = 1e-3) -> pd.DataFrame:
    """计算引力矩阵。

    kind="geo"  : P_ij = g_ij · G_i G_j / D_ij²          （文献 [B] 魏巍 2025）
    kind="econ" : R_ij = k_ij · V_i V_j / (D_ij/|Δpgdp|)^λ（文献 [C] 王梦宇 2024）

    两者修正系数都是 G_i/(G_i+G_j)，差别只在"距离"用地理距离还是经济地理距离。

    ⚠ 经济地理距离的坑：两省人均 GDP 接近时 D/|Δpgdp| → ∞，引力爆掉；
      完全相同则直接除零。这里用 eps_rel 做一个相对下限保护：
      |Δpgdp| 的下限取「全省对人均GDP差距的中位数的 eps_rel 倍」。
      论文原文没写这个保护，用真实数据时必须加，否则会出现 inf。
    """
    provs = list(D.index)
    G = level.reindex(provs).astype(float)

    num = np.outer(G.values, G.values)
    gij = G.values[:, None] / (G.values[:, None] + G.values[None, :])   # 贡献度/修正系数

    if kind == "geo":
        denom = D.values ** 2
    elif kind == "econ":
        if pgdp is None:
            raise ValueError("kind='econ' 需要 --pgdp（人均GDP）")
        p = pgdp.reindex(provs).astype(float).values
        gap = np.abs(p[:, None] - p[None, :])
        off = ~np.eye(len(provs), dtype=bool)
        scale = np.median(gap[off]) if gap[off].size else 1.0
        gap = np.maximum(gap, max(scale * eps_rel, 1e-9))
        denom = (D.values / gap) ** lam
    else:
        raise ValueError(f"未知引力模型: {kind}")

    with np.errstate(divide="ignore", invalid="ignore"):
        M = gij * num / denom
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(M, 0.0)
    return pd.DataFrame(M, index=provs, columns=provs)


def binarize(P: pd.DataFrame, method: str = "rowmean",
             fixed=None, density: float | None = None,
             topk: int | None = None) -> pd.DataFrame:
    """引力矩阵 → 有向二值邻接矩阵。

    method:
      rowmean : 与所在行均值比较，大于则记 1（文献 [B] 魏巍）
      fixed   : 大于固定阈值记 1（文献 [C] 王梦宇）
      topk    : 每行取引力最大的 k 个（密度可控）

    fixed 参数（重要）：
      * 传数字 → 直接当阈值用，**这是做时间演化分析时唯一正确的用法**
      * 传 "mean" → 用「全样本所有年份、所有省对」引力的均值作统一阈值（推荐，见下）
      * 传 "mean_ratio:r" → 用「全样本引力均值 × r」作统一阈值（r 省略时默认 0.58）
        ★ 这是复现文献 [C] 王梦宇(2024) 的推荐口径：以该文表 2 三年锚点
          （2013/2014/2015 = 155/175/204 条边）反推，其阈值 ≈ 全样本引力均值的 0.58 倍，
          本文实测可复现为 159/177/191 条，密度 0.183/0.203/0.220
          （文献 0.1782/0.2011/0.2345）。详见说明文档第三、七节。
      * 传 "median_ratio:r" → 用「全样本引力中位数 × r」作统一阈值（r 省略时默认 1.0）
      * 传 None → 用当前年引力矩阵的中位数

    ⚠️ **为什么 fixed=None 不能用来做时间趋势分析**
      对任意一组连续值取中位数作为「大于则记 1」的阈值，都**数学上保证 50% 的密度**。
      所以 fixed=None 时，无论数据怎么变，网络密度恒为 0.5、关系数恒为 n(n-1)/2 ——
      「网络密度逐年怎么变」这个维度被永久钉死，画出来的趋势图必然是水平直线。
      这跟 `rowmean` 的毛病本质相同，只是把「随行漂移」换成了「随年漂移」。

      → 要做演化分析，必须给一个**不随数据平移的固定参考水平**：
        · `--fixed mean`：用全样本引力均值。引力分布为右偏长尾，均值远高于中位数，
          密度会落到 20~40% 的合理区间，且该按年变时天然会变。
        · 或者你自己定一个可解释的数（如「大于全样本引力均值的省份对视为显著溢出」）。
      结论：**阈值必须来自研究者的口径设定，不能交给数据自适应**，否则阈值跟着数据走，
      趋势就被吃掉了。

    topk 与 fixed 是两个量纲（条数 vs 阈值），绝不互相借用。
    """
    provs = list(P.index)
    n = len(provs)
    A = np.zeros((n, n), dtype=int)
    off_mask = ~np.eye(n, dtype=bool)

    if method == "rowmean":
        row_mean = P.mean(axis=1).values
        A = (P.values > row_mean[:, None]).astype(int)
    elif method == "fixed":
        ref = _resolve_threshold(fixed, P, provs, off_mask)
        A = (P.values > ref).astype(int)
    elif method == "topk":
        k = topk if topk else max(1, int(round((density or 0.1) * (n - 1))))
        k = int(k)
        for i in range(n):
            row = P.values[i].copy()
            idx = np.argsort(-row)[:k]
            A[i, idx] = (row[idx] > 0).astype(int)
    else:
        raise ValueError(f"未知二值化方法: {method}")

    np.fill_diagonal(A, 0)
    return pd.DataFrame(A, index=provs, columns=provs)


# 全样本引力参考值缓存（用于 --fixed mean）
_GLOBAL_REF: dict = {}


def register_global_ref(name: str, value: float):
    """登记全样本参考阈值，供 --fixed mean / median_all 使用。"""
    _GLOBAL_REF[name] = float(value)


def global_ref(name: str) -> float | None:
    return _GLOBAL_REF.get(name)


def _resolve_threshold(fixed, P: pd.DataFrame, provs: list, off_mask) -> float:
    """把 --fixed 的取值解析成真正的阈值数字。

    字符串口径：
      mean / nanmean          → 全样本引力均值（需先 register_global_ref）
      median_all / median     → 全样本引力中位数
      mean_ratio:r            → 全样本均值 × r      （r 省略 = 0.58）
      median_ratio:r          → 全样本中位数 × r    （r 省略 = 1.0）
      "0.5" 之类的纯数字字符串 → 直接当阈值
    """
    if fixed is None:
        # 当前年矩阵的中位数（仅作可运行兜底，不做演化分析用）
        return float(np.median(P.values[off_mask]))

    if isinstance(fixed, str):
        key = fixed.strip().lower()

        # ---- 比例型口径：base × r（优先用已登记的全样本参考值）----
        # 支持两种写法：mean_ratio:0.58（显式比例）、mean_ratio（用默认 0.58）
        for prefix, bare, base_name, default_r in (
            ("mean_ratio:", "mean_ratio", "mean", 0.58),
            ("median_ratio:", "median_ratio", "median_all", 1.0),
        ):
            if key.startswith(prefix) or key == bare:
                raw_r = key[len(prefix):].strip() if key.startswith(prefix) else ""
                try:
                    r = float(raw_r) if raw_r else default_r
                except ValueError:
                    raise ValueError(
                        f"--fixed {prefix}<比例> 的比例部分必须是数字。收到：{fixed!r}")
                base = global_ref(base_name)
                if base is None:
                    if P is None:
                        raise ValueError(
                            f"--fixed {key} 需要先跨全部年份登记全样本参考值，"
                            "但当前未登记且未提供引力矩阵。")
                    # ⚠️ 退化路径：base 用了「当年」矩阵的均值/中位数，
                    #    这会重新引入 50% 密度的病根。必须显式报警，不能静默。
                    off_ = P.values[off_mask]
                    base = (float(np.nanmean(off_)) if base_name == "mean"
                            else float(np.median(off_)))
                    import sys as _sys
                    print(f"[警告] --fixed {key} 未登记全样本参考值，已退化为"
                          f"「当年{base_name}×{r}」——这会让网络密度被钉死，"
                          f"结果不可用于时间演化分析！请检查 is_global_ref_fixed() 是否漏判。",
                          file=_sys.stderr)
                return float(base) * r

        # ---- 直接口径 ----
        if key == "mean" and global_ref("mean") is not None:
            return global_ref("mean")
        if key == "median_all" and global_ref("median_all") is not None:
            return global_ref("median_all")
        if P is not None:
            off = P.values[off_mask]
            if key in ("mean", "nanmean"):
                return float(np.nanmean(off))
            if key in ("median", "median_all"):
                return float(np.median(off))
        # 兼容命令行把数字读成字符串的情况（如 --fixed 0.5 被传成 '0.5'）
        try:
            return float(key)
        except ValueError:
            raise ValueError(
                "--fixed 只接受数字，或字符串 mean / median_all / "
                f"mean_ratio[:r] / median_ratio[:r]。收到：{fixed!r}")
    return float(fixed)


def is_global_ref_fixed(fixed) -> bool:
    """该 --fixed 取值是否是「全样本口径」（需要先跨全部年份登记参考值）。

    ⚠️ 必须同时认「带冒号」和「裸写」两种形式：
       mean_ratio / mean_ratio:0.58 都要走全样本分支。
       否则会静默退化成「当年均值×r」——那是 50% 病的新变种，趋势照样被吃掉。
    """
    if not isinstance(fixed, str):
        return False
    key = fixed.strip().lower()
    if key in ("mean", "median_all"):
        return True
    for prefix, bare in (("mean_ratio:", "mean_ratio"),
                         ("median_ratio:", "median_ratio")):
        if key.startswith(prefix) or key == bare:
            return True
    return False

    return pd.DataFrame(A, index=provs, columns=provs)


# ===========================================================================
# 2. 整体网络特征
# ===========================================================================
def overall_features(A: pd.DataFrame, n_years: int = 1) -> dict:
    """整体网络特征。各指标的定义在文档里写死，避免"同名不同义"。

    关系数 / 网络密度的「趋势诊断」放在 run() 里跨年份统一做，这里只出单年值。
    """
    provs = list(A.index)
    n = len(provs)
    M = A.values.astype(int)
    m = int(M.sum())
    dg = nx.from_pandas_adjacency(A, create_using=nx.DiGraph)

    # 网络密度
    density = m / (n * (n - 1)) if n > 1 else 0.0

    # 网络关联度：1 - 不可达点对数 / 最大可能点对数（点对按无向计）
    und = dg.to_undirected()
    comps = list(nx.connected_components(und))
    unreachable = 0
    for c in comps:
        k = len(c)
        unreachable += k * (n - k)
    unreachable = unreachable // 2 if comps else 0
    max_pairs = n * (n - 1) // 2
    reach = 1 - unreachable / max_pairs if max_pairs else 0.0

    # 网络等级度：1 - 互惠（双向）点对数 / 最大可能
    mutual = sum(1 for u, v in dg.edges() if dg.has_edge(v, u)) // 2
    hierarchy = 1 - mutual / max_pairs if max_pairs else 0.0

    # 网络效率：1 - 冗余连线数 / 最大可能冗余连线数
    redundant = m - (n - 1) if m >= (n - 1) else 0
    max_redundant = (n - 1) ** 2 if n > 1 else 1
    efficiency = 1 - redundant / max_redundant if max_redundant else 0.0

    # 小世界：聚类系数、平均路径长度
    if len(comps) and max(len(c) for c in comps) > 1:
        giant = und.subgraph(max(comps, key=len)).copy()
        avg_path = nx.average_shortest_path_length(giant)
    else:
        avg_path = np.nan
    clustering = nx.average_clustering(und)

    # 互惠比例
    recip = nx.reciprocity(dg) if dg.number_of_edges() else 0.0

    return {
        "省份数": n, "关系数": m, "网络密度": round(density, 4),
        "网络关联度": round(reach, 4), "网络等级度": round(hierarchy, 4),
        "网络效率": round(efficiency, 4), "互惠性": round(float(recip), 4),
        "网络集聚系数": round(float(clustering), 4),
        "平均路径长度": round(float(avg_path), 4) if pd.notna(avg_path) else np.nan,
        "连通片数": len(comps),
    }


def _trend_diagnosis(overall_df: pd.DataFrame, method: str, fixed, n_years: int) -> str:
    """检查关系数序列是否"纹丝不动"——这是阈值自适应把趋势吃掉的典型症状。"""
    if n_years < 3 or "关系数" not in overall_df.columns:
        return ""
    m = overall_df["关系数"].astype(int).tolist()
    uniq = len(set(m))
    lines = ["", "-" * 78, "二值化趋势诊断", "-" * 78]
    lines.append(f"  关系数序列: {m}")
    lines.append(f"  不同取值个数: {uniq} / {n_years}")

    if uniq == 1:
        lines.append("  [X] 关系数年年相同 -> 阈值已把时间维度钉死，网络密度无演化可言。")
        if method == "fixed" and fixed is None:
            lines.append("      原因: `--binarize fixed` 未指定 `--fixed`，阈值取当年中位数，")
            lines.append("            而「大于中位数」数学上保证 50% 密度。")
            lines.append("      修法: 改用 `--fixed mean_ratio`（复现文献[C]，推荐）")
            lines.append("            或 `--fixed mean` / `--fixed median_all`。")
        elif method == "rowmean":
            lines.append("      原因: 行均值法每行阈值随行平移，密度天然锁在 0.5 附近。")
            lines.append("      修法: 改用 `--binarize fixed --fixed mean_ratio`。")
        else:
            lines.append("      原因: 你给的阈值可能落在引力分布的极值区外，请核对量纲。")
    elif uniq <= max(2, n_years // 4):
        lines.append("  [!] 关系数变化很小，趋势可能不显著——建议换个固定参考阈值对比。")
    else:
        lines.append("  [OK] 关系数逐年有变化，可以用于时间演化分析。")
    lines.append("-" * 78)
    return "\n".join(lines)


# ===========================================================================
# 3. 个体网络特征（中心度）
# ===========================================================================
def centrality_table(A: pd.DataFrame) -> pd.DataFrame:
    provs = list(A.index)
    n = len(provs)
    dg = nx.from_pandas_adjacency(A, create_using=nx.DiGraph)
    und = dg.to_undirected()

    outd = np.array([dg.out_degree(p) for p in provs], dtype=float)
    ind = np.array([dg.in_degree(p) for p in provs], dtype=float)
    # 点度中心度 = (点出度+点入度) / (2(n-1)) × 100
    # 该口径已用文献[C]表3数据核对通过：北京 (28+28)/58=96.552 ✓
    degree = (outd + ind) / (2 * (n - 1)) * 100 if n > 1 else np.zeros(n)

    # 接近中心度 = (n-1)/Σd × 100，d 为有向可达距离；不可达按 n-1 保守替换。
    # 注意：网络越稀疏，"可达省数"越小，接近中心度就越低——这不是算错，
    # 而是接近中心度在稀疏有向网里本来就不稳定。判断时务必连"可达省数"一起看。
    close, reach_n = [], []
    for p in provs:
        ln = nx.single_source_shortest_path_length(dg, p)
        reach_n.append(len(ln) - 1)
        d = [ln.get(q, n - 1) for q in provs if q != p]
        s = sum(d)
        close.append((n - 1) / s * 100 if s > 0 else 0.0)
    close = np.array(close)

    # 中介中心度（UCINET 口径：未归一化的点对计数占比 × 100）
    bw = nx.betweenness_centrality(dg, normalized=False, endpoints=False)
    betw = np.array([bw[p] for p in provs], dtype=float)
    betw = betw / max(betw.max(), 1e-12) * 100 if betw.max() > 0 else betw

    df = pd.DataFrame({
        "省份": provs, "点出度": outd.astype(int), "点入度": ind.astype(int),
        "可达省数": np.array(reach_n, dtype=int),
        "点度中心度": np.round(degree, 4),
        "接近中心度": np.round(close, 4),
        "中介中心度": np.round(betw, 4),
    })
    for col in ["点度中心度", "接近中心度", "中介中心度"]:
        df[col + "_排名"] = df[col].rank(ascending=False, method="min").astype(int)
    return df.sort_values("点度中心度_排名").reset_index(drop=True)


# ===========================================================================
# 4. CONCOR 板块划分
# ===========================================================================
def concor(A: pd.DataFrame, depth: int = 2, max_iter: int = 25) -> tuple[np.ndarray, list]:
    """CONCOR：反复对（子）矩阵求相关系数矩阵直到元素收敛到 ±1，
    再用第一特征向量的符号二分。depth=2 得到 4 个板块。"""
    adj = A.values.astype(float)
    n = adj.shape[0]
    blocks = [list(range(n))]
    for _ in range(depth):
        newb = []
        for members in blocks:
            if len(members) < 2:
                newb.append(members)
                continue
            sub = adj[np.ix_(members, members)]
            M = sub.copy()
            for _ in range(max_iter):
                with np.errstate(invalid="ignore"):
                    M2 = np.corrcoef(M)
                M2 = np.nan_to_num(M2, nan=0.0)
                if M2.shape != M.shape:
                    break
                if np.allclose(np.abs(M2), 1.0, atol=1e-3):
                    M = M2
                    break
                M = M2
            w, v = np.linalg.eigh(M)
            vec = v[:, -1]
            pos = [members[i] for i in range(len(members)) if vec[i] >= 0]
            neg = [members[i] for i in range(len(members)) if vec[i] < 0]
            if not pos or not neg:
                newb.append(members)
            else:
                newb.extend([pos, neg])
        blocks = newb

    labels = np.zeros(n, dtype=int)
    for bi, mem in enumerate(blocks, start=1):
        for i in mem:
            labels[i] = bi
    return labels, blocks


def block_density_matrix(A: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    provs = list(A.index)
    M = A.values.astype(float)
    ks = sorted(set(labels))
    out = pd.DataFrame(index=[f"板块{k}" for k in ks],
                       columns=[f"板块{k}" for k in ks], dtype=float)
    for ki in ks:
        for kj in ks:
            I = np.where(labels == ki)[0]
            J = np.where(labels == kj)[0]
            sub = M[np.ix_(I, J)]
            if ki == kj:
                k = len(I)
                denom = k * (k - 1)
                val = sub.sum() / denom if denom > 0 else 0.0
            else:
                val = sub.mean() if sub.size else 0.0
            out.loc[f"板块{ki}", f"板块{kj}"] = round(float(val), 4)
    return out, provs


# ===========================================================================
# 5. 核心-边缘结构
# ===========================================================================
def core_periphery(A: pd.DataFrame, n_core: int | None = None,
                   core_ratio: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """连续核心度（对称邻接矩阵的第一特征向量）→ 确定核心区/边缘区 → 输出四格密度矩阵。

    理想模式矩阵（有向）：P_ij = 0 当且仅当 i 与 j 同属边缘区，其余为 1。
    切点用观测矩阵与理想模式的相关系数来挑，避免"核心区越小拟合度越高"的尺寸偏误。

    n_core / core_ratio（可选，用于对齐文献的核心区规模）：
      * n_core      → 直接指定核心区省份个数（如文献写"核心区 6 个省"）
      * core_ratio  → 指定核心区占全部省份的比例（如 0.2 → 30 省取 6）
      两者都不传 → 走 Borgatti & Everett (1999) 原版的自动切点（默认）。
      ⚠️ 手动指定会覆盖最优拟合度切点，仅在"必须对齐某文献核心区个数"时使用，
         并在报告里注明核心区规模是按文献口径设定而非自动求得。
    """
    provs = list(A.index)
    n = len(provs)
    S = ((A.values + A.values.T) / 2.0).astype(float)
    w, v = np.linalg.eigh(S)
    coreness = np.abs(v[:, -1])
    order = np.argsort(-coreness)

    off = ~np.eye(n, dtype=bool)
    obs = A.values.astype(float)[off]

    best_k, best_f = 1, -np.inf
    for k in range(1, n):
        core = set(order[:k])
        ideal = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i in core or j in core:
                    ideal[i, j] = 1.0
                else:
                    ideal[i, j] = 0.0
        i_vec = ideal[off]
        if obs.std() < 1e-12 or i_vec.std() < 1e-12:
            continue
        fk = float(np.corrcoef(obs, i_vec)[0, 1])
        if fk > best_f:
            best_f, best_k = fk, k

    # ---- 手动指定核心区规模（覆盖自动切点）----
    if n_core is not None:
        best_k = int(max(1, min(n - 1, n_core)))
    elif core_ratio is not None:
        best_k = int(max(1, min(n - 1, round(core_ratio * n))))

    core = set(order[:best_k])

    labels = np.array([0 if i in core else 1 for i in range(n)])
    dens, _ = block_density_matrix(A, labels)
    dens.index = ["核心区", "边缘区"]
    dens.columns = ["核心区", "边缘区"]

    tab = pd.DataFrame({
        "省份": provs,
        "核心度": np.round(coreness, 4),
        "类型": ["核心区" if i in core else "边缘区" for i in range(n)],
    }).sort_values("核心度", ascending=False).reset_index(drop=True)
    return tab, dens


# ===========================================================================
# 6. 绘图
# ===========================================================================
def plot_network(A: pd.DataFrame, year, path: str, region_map=None):
    try:
        plt = C.setup_cn_font()
        dg = nx.from_pandas_adjacency(A, create_using=nx.DiGraph)
        dg.remove_nodes_from([n for n in list(dg) if dg.degree(n) == 0] or [])
        if dg.number_of_nodes() == 0:
            return None

        cent = nx.degree_centrality(dg)
        pos = nx.circular_layout(dg)
        sizes = [900 + 4200 * cent[n] for n in dg]

        fig, ax = plt.subplots(figsize=(11.5, 10))
        nx.draw_networkx_edges(dg, pos, ax=ax, alpha=0.22, arrows=True,
                               arrowsize=7, edge_color="#7f8c8d", width=0.7,
                               connectionstyle="arc3,rad=0.08")
        nx.draw_networkx_nodes(dg, pos, ax=ax, node_size=sizes,
                               node_color="#e74c3c", edgecolors="white",
                               linewidths=1.2, alpha=0.92)
        nx.draw_networkx_labels(dg, pos, ax=ax, font_size=9.5,
                                font_color="#1a1a1a")
        ax.set_title(f"{year} 年空间关联网络（节点大小 = 点度中心度，"
                     f"节点数 {dg.number_of_nodes()}，连线数 {dg.number_of_edges()}）",
                     fontsize=12.5)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=190, facecolor="white")
        plt.close(fig)
        return path
    except Exception as e:
        C.say(f"[警告] 绘图失败：{e}")
        return None


def plot_trend(overall_df: pd.DataFrame, path: str, title_suffix: str = ""):
    """关系数与网络密度的逐年趋势（双纵轴）。文献里"关系数和密度都不一样"就是这个图。"""
    try:
        plt = C.setup_cn_font()
        df = overall_df.sort_values("年份")
        yrs = df["年份"].astype(int).tolist()
        m = df["关系数"].astype(float).tolist()
        d = df["网络密度"].astype(float).tolist()

        fig, ax1 = plt.subplots(figsize=(10.5, 5.2))
        c1, c2 = "#c0392b", "#1f6fb4"
        ax1.plot(yrs, m, "-o", color=c1, lw=2.1, ms=5.5, label="关系数")
        ax1.set_xlabel("年份")
        ax1.set_ylabel("关系数", color=c1)
        ax1.tick_params(axis="y", labelcolor=c1)
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(yrs, d, "-s", color=c2, lw=2.1, ms=5.0, label="网络密度")
        ax2.set_ylabel("网络密度", color=c2)
        ax2.tick_params(axis="y", labelcolor=c2)

        flat = len(set(m)) == 1
        ax1.set_title(
            f"关系数与网络密度的逐年演变{title_suffix}"
            + ("\n⚠ 序列完全水平 —— 阈值把趋势吃掉了" if flat else ""),
            fontsize=12.5, color="#b03030" if flat else "#1a1a1a")
        if flat:
            ax1.axhspan(min(m) * 0.97, max(m) * 1.03, color="#e74c3c", alpha=0.10)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9.5)

        fig.tight_layout()
        fig.savefig(path, dpi=185, facecolor="white")
        plt.close(fig)
        return path
    except Exception as e:
        C.say(f"[警告] 趋势图绘制失败：{e}")
        return None


# ===========================================================================
# 7. 主流程
# ===========================================================================
def build_index(panel_path: str, std="max", weight="entropy", by_year=False):
    panel = C.read_panel(panel_path)
    X = C.standardize(panel, direction=CP_DIRECTION, method=std, by_year=by_year)
    w = C.compute_weights(X, method=weight)
    d = X.merge(w, on="指标", how="left")
    idx = (d.assign(c=d["X"] * d["权重"])
             .groupby(["地区", "年份"], as_index=False)["c"].sum()
             .rename(columns={"c": "指数"}))
    return idx


def run(index_path, raw_path, dist_path, pgdp_path, year, kind, bin_method,
        fixed, topk, std, weight, by_year, depth,
        n_core=None, core_ratio=None):
    if index_path:
        if index_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(index_path)
        else:
            df = pd.read_csv(index_path, encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]
        if "指数" not in df.columns:
            cand = [c for c in df.columns if c not in ("地区", "年份")]
            df = df.rename(columns={cand[0]: "指数"})
        idx = df[["地区", "年份", "指数"]].copy()
        idx["地区"] = idx["地区"].astype(str).str.strip()
        C.say(f"[读取] 综合指数 {index_path}  行={len(idx)}")
    else:
        src = raw_path or os.path.join(DATA, "demo_共同富裕面板.csv")
        C.say(f"[数据] 从原始指标面板测算综合指数：{src}")
        idx = build_index(src, std=std, weight=weight, by_year=by_year)
        C.say(f"[指数] {idx['地区'].nunique()} 省 × {idx['年份'].nunique()} 年，"
              f"{int(idx['年份'].min())}-{int(idx['年份'].max())}")

    provs = sorted(idx["地区"].unique())
    # 距离矩阵
    if dist_path:
        D = pd.read_csv(dist_path, encoding="utf-8-sig", index_col=0)
        D.index = D.index.astype(str)
        D.columns = D.columns.astype(str)
        C.say(f"[距离] 读取 {dist_path}")
    else:
        D = C.distance_matrix(provs, PROVINCE_LONLAT)
        C.say(f"[距离] 按省级代表点经纬度计算（Haversine），{len(provs)}×{len(provs)}")

    # 人均GDP
    pgdp_by_year = None
    if kind == "econ":
        gp = pgdp_path or os.path.join(DATA, "demo_人均GDP.csv")
        p = pd.read_csv(gp, encoding="utf-8-sig")
        p.columns = [str(c).strip() for c in p.columns]
        pgdp_by_year = p

    years = [int(y) for y in sorted(idx["年份"].unique())]
    if year not in years:
        year = years[-1]

    # ---- 全样本参考阈值：--fixed mean / median_all / *_ratio 需先跨全部年份登记 ----
    if is_global_ref_fixed(fixed):
        all_off = []
        for y in years:
            lv = idx[idx["年份"] == y].set_index("地区")["指数"]
            pg = None
            if pgdp_by_year is not None:
                pg = pgdp_by_year[pgdp_by_year["年份"] == y].set_index("地区")["人均GDP"]
            Pp = gravity_matrix(lv, D, pg, kind=kind)
            all_off.append(Pp.values[~np.eye(len(provs), dtype=bool)])
        cat = np.concatenate(all_off)
        register_global_ref("mean", float(np.nanmean(cat)))
        register_global_ref("median_all", float(np.median(cat)))
        key = str(fixed).strip().lower()
        ref = _resolve_threshold(fixed, None, provs, None)
        C.say(f"[阈值] --fixed {key} → 全样本参考阈值 = {ref:.6g} "
              f"（跨 {len(years)} 年全部省对计算：均值={global_ref('mean'):.6g}、"
              f"中位数={global_ref('median_all'):.6g}，逐年套用同一值）")

    # ---- 逐年整体特征 + 二值矩阵 ----
    overall, binmats = [], {}
    for y in years:
        lv = idx[idx["年份"] == y].set_index("地区")["指数"]
        pg = None
        if pgdp_by_year is not None:
            sub = pgdp_by_year[pgdp_by_year["年份"] == y]
            pg = sub.set_index("地区")["人均GDP"]
        P = gravity_matrix(lv, D, pg, kind=kind)
        A = binarize(P, method=bin_method, fixed=fixed, topk=topk)
        feat = overall_features(A)
        feat["年份"] = y
        overall.append(feat)
        binmats[y] = (P, A)

    overall_df = pd.DataFrame(overall)
    cols = ["年份"] + [c for c in overall_df.columns if c != "年份"]
    overall_df = overall_df[cols]

    # ---- 趋势诊断：一眼看出阈值有没有把"网络密度的时间演变"吃掉 ----
    trend = _trend_diagnosis(overall_df, bin_method, fixed, len(years))
    if trend:
        C.say(trend)

    P_sel, A_sel = binmats[year]
    cent = centrality_table(A_sel)
    labels, blocks = concor(A_sel, depth=depth)
    dens, provs_ = block_density_matrix(A_sel, labels)
    blocks_df = pd.DataFrame({
        "省份": provs_,
        "板块": [f"第{labels[i]}板块" for i in range(len(provs_))],
    })
    # 板块成员
    mem = blocks_df.groupby("板块")["省份"].apply(lambda s: "、".join(s)).reset_index()
    mem.columns = ["板块", "成员省份"]
    mem["省份数"] = blocks_df.groupby("板块").size().values
    cp_tab, cp_dens = core_periphery(A_sel, n_core=n_core, core_ratio=core_ratio)

    # 四大板块（行政区划）汇总，便于和文献口径对照
    four = []
    for rn, ps in FOUR_REGIONS.items():
        sub = idx[(idx["地区"].isin(ps))]
        for y in years:
            v = sub[sub["年份"] == y]["指数"]
            four.append({"板块": rn, "年份": y, "均值": round(v.mean(), 4)})
    four_df = pd.DataFrame(four)

    sheets = {
        "0_方法与公式": pd.DataFrame([
            {"项目": "修正引力模型-A 地理距离", "公式": "P_ij = G_i/(G_i+G_j) · G_i·G_j / D_ij²", "出处": "[B]魏巍 2025"},
            {"项目": "修正引力模型-B 经济地理距离", "公式": "R_ij = V_i/(V_i+V_j) · V_i·V_j / (D_ij/|pgdp_i-pgdp_j|)^λ, λ=2", "出处": "[C]王梦宇 2024"},
            {"项目": "二值化-行均值法", "公式": "P_ij > rowmean(P_i) → 1，否则 0", "出处": "[B]魏巍 2025"},
            {"项目": "二值化-固定阈值", "公式": "P_ij > 阈值 → 1", "出处": "[C]王梦宇 2024（论文只写『以固定值为阈值』，未给数值；本文反推 ≈ 全样本引力均值×0.58）"},
            {"项目": "二值化-全样本均值比例法", "公式": "阈值 = 全样本引力均值 × r（默认 r=0.58）", "出处": "反推 [C]王梦宇 2024 表2（2013/14/15=155/175/204条）所得，实测复现 159/177/191 条" if str(fixed).startswith("mean_ratio") else "本工具扩展口径"},
            {"项目": "二值化-全样本均值法", "公式": "阈值 = 全样本所有年份、所有省对引力均值", "出处": "本工具扩展口径"},
            {"项目": "二值化-全样本中位数法", "公式": "阈值 = 全样本引力中位数", "出处": "本工具扩展口径"},
            {"项目": "网络密度", "公式": "m / [n(n-1)]", "出处": "标准定义（已用文献[C]表2核对：155/870=0.1782 ✓）"},
            {"项目": "网络关联度", "公式": "1 − 不可达点对数 / [n(n-1)/2]", "出处": "刘军《整体网分析》"},
            {"项目": "网络等级度", "公式": "1 − 互惠点对数 / [n(n-1)/2]", "出处": "刘军《整体网分析》（各文献口径不一，务必核对）"},
            {"项目": "网络效率", "公式": "1 − [m − (n−1)] / (n−1)²", "出处": "刘华军等 2015（各文献口径不一，务必核对）"},
            {"项目": "点度中心度", "公式": "(点出度+点入度) / [2(n−1)] × 100", "出处": "已用文献[C]表3核对：北京 (28+28)/58=96.552 ✓"},
            {"项目": "接近中心度", "公式": "(n−1) / Σd × 100，d 为有向最短路；不可达按 n−1 处理", "出处": "已用文献[C]表3核对：北京 29/30=96.667 ✓"},
            {"项目": "中介中心度", "公式": "中介点对计数（未归一化），再线性缩放到 0-100", "出处": "UCINET 口径"},
            {"项目": "CONCOR 板块", "公式": "反复求相关系数矩阵至收敛，按第一特征向量符号二分，depth=2 得 4 板块", "出处": "[B]魏巍 2025"},
            {"项目": "核心-边缘", "公式": ("对称邻接矩阵第一特征向量作核心度，按分类拟合度最优切点划分"
                                        + (f"；核心区固定为 {int(n_core)} 个" if n_core else
                                           (f"；核心区比例固定为 {core_ratio}" if core_ratio else ""))),
             "出处": "Borgatti & Everett (1999)"},
        ]),
        "1_整体网络特征(逐年)": overall_df,
        f"2_引力矩阵_{year}": P_sel.round(6).reset_index().rename(columns={"index": "省份"}),
        f"3_二值矩阵_{year}": A_sel.reset_index().rename(columns={"index": "省份"}),
        f"4_中心度_{year}": cent,
        f"5_CONCOR板块_{year}": blocks_df,
        f"5b_板块成员_{year}": mem,
        f"5c_板块密度矩阵_{year}": dens.reset_index().rename(columns={"index": "板块"}),
        f"6_核心边缘_{year}": cp_tab,
        f"6b_核心边缘密度_{year}": cp_dens.reset_index().rename(columns={"index": "区域"}),
        "7_四大地区指数均值": four_df,
    }

    out_xlsx = os.path.join(OUT_DIR, "network_result.xlsx")
    C.write_excel(sheets, out_xlsx)
    C.say(f"[写出] {out_xlsx}")

    png = os.path.join(OUT_DIR, f"network_{year}.png")
    got = plot_network(A_sel, year, png)
    if got:
        C.say(f"[写出] {got}")

    tag = f"（阈值口径：{bin_method}" + (f" / fixed={fixed}" if bin_method == "fixed" else "") + "）"
    png2 = os.path.join(OUT_DIR, "trend_关系数与网络密度.png")
    got2 = plot_trend(overall_df, png2, tag)
    if got2:
        C.say(f"[写出] {got2}")

    # ---- 控制台摘要 ----
    C.say("\n" + "=" * 78)
    C.say(f"整体网络特征（{years[0]}-{years[-1]}）")
    C.say("=" * 78)
    C.say(overall_df.to_string(index=False))
    C.say(f"\n{year} 年中心度前 8（点度中心度）：")
    for _, r in cent.head(8).iterrows():
        C.say(f"  {r['省份']:<4} 出度={r['点出度']:<3} 入度={r['点入度']:<3} "
              f"点度={r['点度中心度']:>6} 接近={r['接近中心度']:>7} 中介={r['中介中心度']:>7}")
    C.say(f"\n{year} 年 CONCOR 板块（depth={depth}）：")
    for _, r in mem.iterrows():
        C.say(f"  {r['板块']}（{r['省份数']}省）：{r['成员省份']}")
    C.say(f"\n{year} 年核心-边缘：核心区 {int((cp_tab['类型'] == '核心区').sum())} 省，"
          f"边缘区 {int((cp_tab['类型'] == '边缘区').sum())} 省")
    C.say(f"  核心区：{'、'.join(cp_tab[cp_tab['类型'] == '核心区']['省份'])}")
    return sheets


def main():
    ap = argparse.ArgumentParser(description="修正引力模型 + 社会网络分析")
    ap.add_argument("--index", default=None, help="综合指数文件 [地区,年份,指数]；不传则用原始面板自算")
    ap.add_argument("--raw", default=None, help="原始指标面板（长表/宽表），用于自算综合指数")
    ap.add_argument("--dist", default=None, help="地理距离矩阵 CSV（省×省）；不传则按经纬度算")
    ap.add_argument("--pgdp", default=None, help="人均GDP [地区,年份,人均GDP]，经济地理距离需要")
    ap.add_argument("--year", type=int, default=2023, help="重点分析年份（默认取最后一年）")
    ap.add_argument("--gravity", default="geo", choices=["geo", "econ"], help="引力模型口径")
    ap.add_argument("--binarize", default="rowmean", choices=["rowmean", "fixed", "topk"])
    ap.add_argument("--fixed", default=None,
                    help="fixed 法阈值。可传数字；或传 mean=全样本引力均值、"
                         "median_all=全样本中位数、"
                         "mean_ratio:r=全样本均值×r（★默认 r=0.58，复现文献[C]的推荐口径）、"
                         "median_ratio:r=全样本中位数×r。"
                         "⚠️ 不传则用当前年中位数，密度会恒为 0.5、趋势被钉死，"
                         "不要用它做时间趋势分析")
    ap.add_argument("--core-n", type=int, default=None,
                    help="核心-边缘：直接指定核心区省份个数（对齐文献核心区规模时用）")
    ap.add_argument("--core-ratio", type=float, default=None,
                    help="核心-边缘：指定核心区占全部省份的比例（如 0.2 → 30 省取 6 个）")
    ap.add_argument("--topk", type=int, default=12, help="topk 法每行保留的连边数")
    ap.add_argument("--std", default="max", choices=["max", "minmax", "mean"])
    ap.add_argument("--weight", default="entropy", choices=["cv", "entropy", "critic", "independent"])
    ap.add_argument("--by-year", action="store_true")
    ap.add_argument("--depth", type=int, default=2, help="CONCOR 层数，2 → 4 板块")
    a = ap.parse_args()
    C.ensure_dirs(OUT_DIR)
    run(a.index, a.raw, a.dist, a.pgdp, a.year, a.gravity, a.binarize,
        a.fixed, a.topk, a.std, a.weight, a.by_year, a.depth,
        n_core=a.core_n, core_ratio=a.core_ratio)


if __name__ == "__main__":
    main()
