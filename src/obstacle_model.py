# -*- coding: utf-8 -*-
"""
obstacle_model.py — 传统障碍度模型 vs 改进障碍度（驱动度）模型的并排测算与对比

核心：
    传统障碍度   Z_ij = (1 - X_ij) * f_ij / Σ_j (1 - X_ij) * f_ij
    改进驱动度   Q_ij =      X_ij  * f_ij / Σ_j      X_ij  * f_ij

两者是同一个算术结构，唯一区别是分子里的 X 换成了 1-X。
因此：Z 分解的是"缺口总量"，Q 分解的是"得分总量"，两者的排名结构性地接近相反。

用法：
    python obstacle_model.py                                   # 用内置演示数据跑通
    python obstacle_model.py --data ../data/你的数据.xlsx

输出（默认写到 ../output/）：
    obstacle_result.xlsx   含 8 个 sheet，其中"排名对比"是关键
    rank_scatter.png       Q 排名 vs Z 排名 散点图（直观看到反转）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from indicator_ref import (NQP_SYSTEM, SEVEN_REGIONS, nqp_long_names,
                           nqp_meta)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "output")


# ===========================================================================
# 1. 两套模型
# ===========================================================================
def traditional_obstacle(df: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """传统障碍度：(1-X)*f 归一化。返回 df + [缺口, 障碍度]"""
    d = df.merge(w, on="指标", how="left")
    d["权重"] = d["权重"].fillna(0.0)
    d["缺口"] = (1.0 - d["X"]) * d["权重"]
    tot = d.groupby(["地区", "年份"])["缺口"].transform("sum")
    d["障碍度"] = np.where(tot > 0, d["缺口"] / tot, 0.0)
    return d


def improved_obstacle(df: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """改进障碍度（驱动度）：X*f 归一化。返回 df + [贡献, 驱动度]"""
    d = df.merge(w, on="指标", how="left")
    d["权重"] = d["权重"].fillna(0.0)
    d["贡献"] = d["X"] * d["权重"]
    tot = d.groupby(["地区", "年份"])["贡献"].transform("sum")
    d["驱动度"] = np.where(tot > 0, d["贡献"] / tot, 0.0)
    return d


# ===========================================================================
# 2. 区域汇总 + 排名
# ===========================================================================
def region_table(d: pd.DataFrame, region_map: dict | None,
                 value_col: str, meta: dict) -> pd.DataFrame:
    """把逐省逐年的值按区域求平均，并给出排名。region_map=None → 全国整体。"""
    out = []
    scopes = {"全国": None}
    if region_map:
        scopes.update(region_map)

    for scope, provs in scopes.items():
        sub = d if provs is None else d[d["地区"].isin(provs)]
        if sub.empty:
            continue
        g = sub.groupby("指标", as_index=False).agg(**{value_col: (value_col, "mean")})
        # 用区域均值重新归一化（保证区域内合计 100%）
        s = g[value_col].sum()
        g[value_col] = g[value_col] / s if s > 0 else 0.0
        g["区域"] = scope
        out.append(g)

    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["排名"] = res.groupby("区域")[value_col].rank(ascending=False,
                                                      method="min").astype(int)
    return res[["区域", "指标", value_col, "排名"]]


# ===========================================================================
# 3. 反转检验（这是回答"改进改了什么"的关键证据）
# ===========================================================================
def reversal_test(q: pd.DataFrame, z: pd.DataFrame,
                  w: pd.DataFrame | None = None,
                  top_n: int = 10) -> pd.DataFrame:
    """两模型的排名结构检验。

    理论：设 a_j = X_j f_j（改进分子），b_j = (1-X_j) f_j（传统分子）。
    因为 b_j = f_j - a_j，当所有 f_j 相等时，按 b 排序等价于按 a 倒序 —— 排名严格反转。
    权重越离散，(1-X) 那一项就越压不住 f 的差异，"高权重指标"会同时挤进两个榜单前列。
    """
    fmap = dict(zip(w["指标"], w["权重"])) if w is not None else {}
    rows = []
    for scope in sorted(set(q["区域"]) & set(z["区域"])):
        a = q[q["区域"] == scope][["指标", "驱动度", "排名"]]
        b = z[z["区域"] == scope][["指标", "障碍度", "排名"]]
        m = a.merge(b, on="指标", suffixes=("_改进", "_传统"))
        m["权重"] = m["指标"].map(fmap).fillna(0.0)

        rho = m["驱动度"].corr(m["障碍度"], method="spearman")
        tau = m["驱动度"].corr(m["障碍度"], method="kendall")
        n = len(m)
        k = max(1, int(round(n * top_n / 100.0))) if top_n <= 20 else top_n
        top_q = set(m.nsmallest(k, "排名_改进")["指标"])
        top_z = set(m.nsmallest(k, "排名_传统")["指标"])
        bot_q = set(m.nlargest(k, "排名_改进")["指标"])
        bot_z = set(m.nlargest(k, "排名_传统")["指标"])
        inter = top_q & top_z

        # 权重离散度：用变异系数衡量
        f = m["权重"].values
        cv = float(f.std(ddof=1) / f.mean()) if f.mean() > 0 else np.nan

        rows.append({
            "区域": scope,
            "指标数": n,
            f"前{k}比较": f"改进Top{k} 与 传统Top{k}",
            "Spearman(驱动度,障碍度)": round(rho, 4) if pd.notna(rho) else np.nan,
            "Kendall(驱动度,障碍度)": round(tau, 4) if pd.notna(tau) else np.nan,
            "两榜交集数": len(inter),
            "交集指标（若有）": "、".join(sorted(inter)) if inter else "（无交集）",
            "改进Top同时是传统Bottom数": len(top_q & bot_z),
            "传统Top同时是改进Bottom数": len(top_z & bot_q),
            "权重变异系数CV": round(cv, 3) if pd.notna(cv) else np.nan,
            "解读": _interpret(rho, len(inter), k, cv),
        })
    return pd.DataFrame(rows)


def _interpret(rho, inter, k, cv) -> str:
    if pd.isna(rho):
        return "无法计算"
    if rho <= -0.85:
        base = "严格反转：两榜互为倒序，改进=传统的前途镜像"
    elif rho <= -0.5:
        base = "显著反转：两榜大面积为倒序"
    elif rho <= -0.2:
        base = "弱反转：反向趋势存在，但被权重差异抵消了一部分"
    elif rho < 0.2:
        base = "几乎不相关：水平差异与权重差异共同作用，两榜互不预示"
    else:
        base = "同向：本区域两榜趋同，说明排序主要由权重驱动、指标水平差异小"
    if inter > 0:
        base += f"；Top{k}有{inter}个交集，且权重CV={cv:.2f}，"
        base += "高权重指标会同时挤进两榜前列（X≈0.5 时 X·f 与 (1-X)·f 都大），属正常"
    else:
        base += f"；Top{k}零交集"
    return base


def theoretical_note():
    """五组对照，把"两榜差异从哪来"讲死。

    恒等式：传统分子 b_j = f_j - a_j（a_j 为改进分子 X_j·f_j）。
    于是两榜分别是"权重×水平"与"权重×(1-水平)"的排序，谁主导取决于
    权重离散度 Var(f) 与水平离散度 Var(X) 谁更大。
    """
    names = ["A", "B", "C", "D", "E", "F"]
    scenarios = [
        ("①权重完全相等、水平分化",
         [0.90, 0.80, 0.50, 0.40, 0.20, 0.10],
         [1, 1, 1, 1, 1, 1],
         "权重相同 → 两榜严格互为倒序（教科书情形）"),
        ("②权重轻度分化、水平分化",
         [0.90, 0.80, 0.50, 0.40, 0.20, 0.10],
         [0.20, 0.15, 0.20, 0.15, 0.18, 0.12],
         "权重离散被引入，反转被削弱但仍是主导趋势"),
        ("③权重高度分化、水平全部相同",
         [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
         [0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
         "关键反例：水平一样时，两榜完全相同（都跟着权重走）"),
        ("④权重与水平都分化（最常见）",
         [0.90, 0.80, 0.50, 0.50, 0.20, 0.10],
         [0.30, 0.05, 0.25, 0.20, 0.15, 0.05],
         "两股力量对冲，相关趋近 0：两榜互不预示"),
        ("⑤存在一个超高权重且水平中等的指标",
         [0.55, 0.95, 0.90, 0.15, 0.10, 0.05],
         [0.50, 0.10, 0.10, 0.10, 0.10, 0.10],
         "该指标同时进两榜首位（X≈0.5 时 X·f 与 (1-X)·f 都大）"),
    ]
    out, detail = [], []
    for tag, x, f, note in scenarios:
        x = np.array(x, float)
        f = np.array(f, float)
        f = f / f.sum()
        a = x * f
        b = (1 - x) * f
        q = a / a.sum()
        z = b / b.sum()
        rq = pd.Series(q).rank(ascending=False, method="min").astype(int).tolist()
        rz = pd.Series(z).rank(ascending=False, method="min").astype(int).tolist()
        rho = float(pd.Series(q).corr(pd.Series(z), method="spearman"))
        out.append({
            "情景": tag,
            "指标(A~F)": " / ".join(names),
            "标准化值X": " / ".join(f"{v:.2f}" for v in x),
            "权重f": " / ".join(f"{v:.2f}" for v in f),
            "改进驱动度": " / ".join(f"{v:.3f}" for v in q),
            "改进排名": " / ".join(str(v) for v in rq),
            "传统障碍度": " / ".join(f"{v:.3f}" for v in z),
            "传统排名": " / ".join(str(v) for v in rz),
            "Spearman": round(rho, 4),
            "机制说明": note,
        })
        for i, nm in enumerate(names):
            detail.append({
                "情景": tag, "指标": nm, "X": round(float(x[i]), 3),
                "f": round(float(f[i]), 3),
                "改进分子 X·f": round(float(a[i]), 4),
                "传统分子 (1-X)·f": round(float(b[i]), 4),
                "改进驱动度": round(float(q[i]), 4), "改进排名": rq[i],
                "传统障碍度": round(float(z[i]), 4), "传统排名": rz[i],
            })
    return pd.DataFrame(out), pd.DataFrame(detail)


# ===========================================================================
# 4. 主流程
# ===========================================================================
def run(data_path: str | None, weight_method: str, std_method: str,
        top_n: int, by_year: bool, meta_kind: str):
    meta = nqp_meta()
    long_names = nqp_long_names()
    dir_map = {}   # 新质生产力均为正指标

    if data_path:
        panel = C.read_panel(data_path)
        C.say(f"[读取] {data_path}  行数={len(panel)}  "
              f"地区={panel['地区'].nunique()} 年份={panel['年份'].nunique()} "
              f"指标={panel['指标'].nunique()}")
    else:
        C.say("[数据] 未提供 --data，使用内置演示数据（与论文同量级的模拟面板）")
        panel = demo_panel()

    # 只保留参照表里认识的指标
    known = set(long_names)
    unk = sorted(set(panel["指标"]) - known)
    if unk:
        C.say(f"[提示] 以下 {len(unk)} 个指标不在新质生产力参照表中，将按原样保留："
              f"{unk[:5]}{'...' if len(unk) > 5 else ''}")
    used = [i for i in panel["指标"].unique() if i in known] or list(panel["指标"].unique())

    X = C.standardize(panel, direction=dir_map, method=std_method, by_year=by_year)
    X = X[X["指标"].isin(used)]

    if weight_method == "paper":
        given = {k: v["权重"] for k, v in meta.items()}
        w = C.compute_weights(X, method="given", given=given)
        w = w[w["指标"].isin(used)]
        C.say(f"[赋权] 采用文献[柴士改2026]表1公布的变异系数权重（{len(w)} 项，已归一化）")
    else:
        w = C.compute_weights(X, method=weight_method)
        w = w[w["指标"].isin(used)]
        C.say(f"[赋权] {weight_method} 法，{len(w)} 项")

    dq = improved_obstacle(X, w)
    dz = traditional_obstacle(X, w)

    rt_q = region_table(dq, SEVEN_REGIONS, "驱动度", meta)
    rt_z = region_table(dz, SEVEN_REGIONS, "障碍度", meta)

    # ---- 明细表：把指标的全部信息拼在一起（含含义、二级、一级）----
    info = pd.DataFrame([
        {"指标": k, "三级指标": v["三级"], "指标含义": v["含义"],
         "二级指标": v["二级"], "一级指标": v["一级"]}
        for k, v in meta.items() if k in used
    ])

    detail = (dq.groupby("指标", as_index=False)
                .agg(X均值=("X", "mean"), 驱动度=("驱动度", "mean"))
                .merge(dz.groupby("指标", as_index=False).agg(障碍度=("障碍度", "mean")),
                       on="指标", how="outer")
                .merge(w, on="指标", how="left")
                .merge(info, on="指标", how="left"))
    detail["改进排名"] = detail["驱动度"].rank(ascending=False, method="min").astype(int)
    detail["传统排名"] = detail["障碍度"].rank(ascending=False, method="min").astype(int)
    detail["排名差(传统-改进)"] = detail["传统排名"] - detail["改进排名"]
    detail = detail.sort_values("改进排名").reset_index(drop=True)

    rev = reversal_test(rt_q, rt_z, w, top_n)
    theo_summary, theo_detail = theoretical_note()

    # ---- 二级/一级汇总对比 ----
    lvl2 = _aggregate_level(detail, "二级指标")
    lvl1 = _aggregate_level(detail, "一级指标")

    # ---- 全国 Top-N 对照表（用户截图里的那种表，但带上完整含义）----
    tops = _top_table(rt_q, rt_z, info, top_n)

    sheets = {
        "0_模型公式与说明": pd.DataFrame([
            {"项": "传统障碍度 Z", "公式": "(1-X)*f / Σ(1-X)*f", "分解对象": "缺口（离理想值的距离）", "高值含义": "短板、制约因素", "政策方向": "提升该指标"},
            {"项": "改进驱动度 Q", "公式": "X*f / Σ X*f", "分解对象": "得分（当前水平）", "高值含义": "长板、贡献来源", "政策方向": "继续强化该指标"},
            {"项": "关系", "公式": "Q_j + Z_j ≠ 1（分母不同）", "分解对象": "同一套 X 与 f", "高值含义": "同一指标在两个模型里排名天然相反", "政策方向": "两者回答的是两个不同的问题"},
            {"项": "理论断言", "公式": "权重相等时，两模型排名严格互为倒序", "分解对象": "见 sheet 理论演示", "高值含义": "-", "政策方向": "-"},
            {"项": "综合指数", "公式": "S_i = Σ_j f_j X_ij", "分解对象": "Q_ij = f_j X_ij / S_i", "高值含义": "Q 就是综合指数的贡献度分解", "政策方向": "-"},
        ]),
        "1_指标明细与两模型排名": detail,
        "2_全国Top对照": tops,
        "3_七大地区_改进驱动度": rt_q.sort_values(["区域", "排名"]),
        "4_七大地区_传统障碍度": rt_z.sort_values(["区域", "排名"]),
        "5_反转检验": rev,
        "6_理论演示_权重情景": theo_summary,
        "6b_理论演示_逐行明细": theo_detail,
        "7_二级指标汇总": lvl2,
        "8_一级指标汇总": lvl1,
    }

    out_xlsx = os.path.join(OUT_DIR, "obstacle_result.xlsx")
    C.write_excel(sheets, out_xlsx)
    C.say(f"[写出] {out_xlsx}")

    # ---- 图：排名散点 ----
    png = _plot_rank_scatter(detail, os.path.join(OUT_DIR, "rank_scatter.png"))
    if png:
        C.say(f"[写出] {png}")

    # ---- 控制台摘要 ----
    C.say("\n" + "=" * 78)
    C.say("全国层面：改进驱动度 Top%d  vs  传统障碍度 Top%d" % (top_n, top_n))
    C.say("=" * 78)
    disp = tops[tops["区域"] == "全国"]
    for _, r in disp.iterrows():
        C.say(f"  [{r['模型']}] #{r['排名']:<2} {r['指标含义'][:34]:<34} "
              f"占比={r['平均占比']*100:5.2f}%")
    C.say("\n反转检验（判断两模型差异是否符合预期的自查表）：")
    for _, r in rev.iterrows():
        C.say(f"  {str(r['区域']):<4} Spearman={r['Spearman(驱动度,障碍度)']:>7}  "
              f"Top交集={r['两榜交集数']}  权重CV={r['权重变异系数CV']}")
    C.say("\n解读：Spearman 明显为负 = 两者确实在问相反的问题，不是算错；")
    C.say("      交集不为 0 且权重CV 大 = 高权重指标天然会在两榜同时靠前，属正常。")
    return sheets


def _aggregate_level(detail: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in detail.columns:
        return pd.DataFrame()
    g = (detail.groupby(col, as_index=False)
               .agg(权重合计=("权重", "sum"),
                    X均值=("X均值", "mean"),
                    改进驱动度合计=("驱动度", "sum"),
                    传统障碍度合计=("障碍度", "sum"),
                    指标数=("指标", "count")))
    g = g.sort_values("改进驱动度合计", ascending=False).reset_index(drop=True)
    g["改进排名"] = g.index + 1
    g["传统排名"] = g["传统障碍度合计"].rank(ascending=False, method="min").astype(int)
    return g


def _top_table(rt_q, rt_z, info, top_n: int) -> pd.DataFrame:
    q = rt_q.merge(info, on="指标", how="left")
    z = rt_z.merge(info, on="指标", how="left")
    rows = []
    for scope in sorted(set(q["区域"])):
        for model, t, col in [("改进驱动度", q, "驱动度"), ("传统障碍度", z, "障碍度")]:
            sub = t[t["区域"] == scope].nsmallest(top_n, "排名")
            for _, r in sub.iterrows():
                rows.append({
                    "区域": scope, "模型": model, "指标": r["指标"],
                    "三级指标": r.get("三级指标"), "指标含义": r.get("指标含义"),
                    "二级指标": r.get("二级指标"), "一级指标": r.get("一级指标"),
                    "平均占比": round(float(r[col]), 4), "排名": int(r["排名"]),
                })
    return pd.DataFrame(rows)


def _plot_rank_scatter(detail: pd.DataFrame, path: str):
    try:
        plt = C.setup_cn_font()
        fig = plt.figure(figsize=(15.5, 7.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.32)
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(detail["改进排名"], detail["传统排名"], s=42,
                   c="#c0392b", alpha=0.75, edgecolors="white", linewidths=0.6)
        n = len(detail)
        ax.plot([0.5, n + 0.5], [0.5, n + 0.5], "--", color="#95a5a6",
                lw=1, label="若两模型一致（对角线）")
        ax.plot([0.5, n + 0.5], [n + 0.5, 0.5], "-", color="#27ae60",
                lw=1.4, label="完全反转线 (y = n+1-x)")
        for _, r in detail.head(3).iterrows():
            ax.annotate(str(r["指标"])[:16], (r["改进排名"], r["传统排名"]),
                        fontsize=7.5, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("改进驱动度 排名（1 = 贡献最大）")
        ax.set_ylabel("传统障碍度 排名（1 = 障碍最大）")
        ax.set_title("两模型排名散点：点越贴绿线，反转越彻底", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(alpha=0.25)

        ax = fig.add_subplot(gs[0, 1])
        d = detail.sort_values("改进排名")
        y = np.arange(len(d))
        ax.barh(y - 0.2, d["驱动度"], height=0.38, color="#c0392b", label="改进驱动度")
        ax.barh(y + 0.2, d["障碍度"], height=0.38, color="#2e86c1", label="传统障碍度")
        ax.set_yticks(y)
        ax.set_yticklabels([str(s)[:18] for s in d["指标"]], fontsize=6.4)
        ax.invert_yaxis()
        ax.set_xlabel("占比")
        ax.set_title("同一批指标，两条分布几乎镜像", fontsize=11)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, axis="x")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        C.say(f"[警告] 绘图失败：{e}")
        return None


# ===========================================================================
# 演示数据（未提供 --data 时使用）
# ===========================================================================
def demo_panel(seed: int = 20260925) -> pd.DataFrame:
    """构造与文献[柴士改2026]量级相当的模拟面板。

    关键：让各指标的相对离散度(CV)与论文公布的权重同比例，
    这样用 cv 法赋权能大致复现论文的权重结构（可用来交叉验证实现是否正确）。
    """
    rng = np.random.default_rng(seed)
    from indicator_ref import PROVINCE_LONLAT
    provs = list(PROVINCE_LONLAT.keys())
    years = list(range(2015, 2024))
    east = {"北京", "天津", "上海", "江苏", "浙江", "福建", "山东", "广东"}
    mid = {"河北", "山西", "安徽", "江西", "河南", "湖北", "湖南", "辽宁", "吉林",
           "黑龙江", "重庆", "四川", "陕西", "海南"}
    base = {p: (0.72 if p in east else 0.50 if p in mid else 0.33) for p in provs}

    meta = nqp_meta()
    ws = np.array([v["权重"] for v in meta.values()])
    wmin, wmax = ws.min(), ws.max()
    # 目标 CV：论文权重最小的指标 CV≈0.06，最大的 CV≈1.10，中间等比插值
    target_cv = {}
    for k, v in meta.items():
        rel = (v["权重"] - wmin) / (wmax - wmin)
        target_cv[k] = 0.06 + rel * 1.04

    rows = []
    for k, v in meta.items():
        sigma = target_cv[k]                 # 对数正态离散度 ≈ CV
        growth = 0.02 + 0.06 * (v["权重"] / wmax)
        # 每个指标自身的"成熟度"不同（现实里数字基建已成熟、政府关注度还很低），
        # 这一项让各指标的平均水平 X̄_j 拉开差距，两模型的排名差异才会显现
        achieve = float(rng.uniform(0.25, 0.95))
        for p in provs:
            lvl = base[p] * achieve * float(np.exp(rng.normal(0, sigma * 0.8)))
            for t, y in enumerate(years):
                val = lvl * (1 + growth) ** t * float(np.exp(rng.normal(0, sigma * 0.5)))
                rows.append({"地区": p, "年份": y, "指标": k,
                             "值": float(max(val, 1e-8))})
    return pd.DataFrame(rows)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="传统障碍度 vs 改进驱动度 对比测算")
    ap.add_argument("--data", default=None,
                    help="面板数据文件（.xlsx/.csv）。不传则用内置演示数据。"
                         "列结构：[地区,年份,指标,值] 或 [地区,年份,指标1,指标2,...]")
    ap.add_argument("--weight", default="cv",
                    choices=["cv", "entropy", "critic", "independent", "paper"],
                    help="赋权法；paper = 用文献[柴士改2026]表1公布的权重（仅新质生产力体系）")
    ap.add_argument("--std", default="max", choices=["max", "minmax", "mean"],
                    help="标准化方法，默认 max（即 x/max，文献口径，落在 (0,1] 内）。"
                         "minmax 会让最小观测取到 X=0（不满足传统模型的 (0,1) 前提），"
                         "mean（x/均值）会超过 1；两者都会改变两榜的差异结构。"
                         "zscore 被禁用（会产生负值，两个模型的非负前提都失效）")
    ap.add_argument("--by-year", action="store_true", help="逐年标准化（跨地区可比）")
    ap.add_argument("--top", type=int, default=10, help="输出前 N 名，默认 10")
    a = ap.parse_args()
    C.ensure_dirs(OUT_DIR)
    run(a.data, a.weight, a.std, a.top, a.by_year, "nqp")


if __name__ == "__main__":
    main()
