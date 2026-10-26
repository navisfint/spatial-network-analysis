# -*- coding: utf-8 -*-
"""
make_demo_data.py — 生成两份演示数据 + 数据模板

产出（写到 ../data/）：
  demo_新质生产力面板.csv    30省 × 2015-2023 × 30指标，长表 [地区,年份,指标,值]
                             → 给 obstacle_model.py 用
  demo_共同富裕面板.csv      30省 × 2011-2023 × 49指标，长表
                             → 给 network_analysis.py 的 --raw 用
  demo_人均GDP.csv           30省 × 2011-2023，[地区,年份,人均GDP]
                             → 给 network_analysis.py 的 --pgdp 用
  模板_长表.csv / 模板_宽表.csv  空模板，照着填自己的数据即可

数据是模拟的，只用来跑通流程 / 做对照，不是真实统计值。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicator_ref import CP_SYSTEM, PROVINCE_LONLAT, nqp_meta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

EAST = {"北京", "天津", "上海", "江苏", "浙江", "福建", "山东", "广东"}
MID = {"河北", "山西", "安徽", "江西", "河南", "湖北", "湖南", "辽宁", "吉林",
       "黑龙江", "重庆", "四川", "陕西", "海南"}


def _base(prov: str) -> float:
    return 0.74 if prov in EAST else 0.50 if prov in MID else 0.32


def gen_nqp(seed=20260925) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meta = nqp_meta()
    ws = np.array([v["权重"] for v in meta.values()])
    wmin, wmax = ws.min(), ws.max()
    rows = []
    for k, v in meta.items():
        rel = (v["权重"] - wmin) / (wmax - wmin)
        sigma = 0.06 + rel * 1.04
        growth = 0.02 + 0.06 * (v["权重"] / wmax)
        achieve = float(rng.uniform(0.25, 0.95))
        for p in PROVINCE_LONLAT:
            lvl = _base(p) * achieve * float(np.exp(rng.normal(0, sigma * 0.8)))
            for t, y in enumerate(range(2015, 2024)):
                val = lvl * (1 + growth) ** t * float(np.exp(rng.normal(0, sigma * 0.5)))
                rows.append({"地区": p, "年份": y, "指标": k, "值": round(max(val, 1e-8), 6)})
    return pd.DataFrame(rows)


def gen_cp(seed=20260926) -> pd.DataFrame:
    """共同富裕 49 指标面板。逆向指标方向按参照表处理，这里生成的都是"原始值"。"""
    rng = np.random.default_rng(seed)
    rows = []
    for name, unit, direction, lvl2, lvl1 in CP_SYSTEM:
        sigma = float(rng.uniform(0.10, 0.55))
        growth = float(rng.uniform(0.015, 0.075))
        achieve = float(rng.uniform(0.30, 0.92))
        for p in PROVINCE_LONLAT:
            lvl = _base(p) * achieve * float(np.exp(rng.normal(0, sigma * 0.8)))
            for t, y in enumerate(range(2011, 2024)):
                v = lvl * (1 + growth) ** t * float(np.exp(rng.normal(0, sigma * 0.45)))
                # 逆向指标（失业率/恩格尔系数/房价…）真实含义是"越低越好"，
                # 生成时让它随地区禀赋反向变动
                if direction == "-":
                    v = (2.2 - 1.6 * _base(p)) * (1 - growth / 2) ** t * (0.85 + 0.3 * rng.random())
                rows.append({"地区": p, "年份": y, "指标": name, "值": round(max(v, 1e-6), 6)})
    return pd.DataFrame(rows)


def gen_pgdp(seed=20260927) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in PROVINCE_LONLAT:
        lvl = 35000 + 65000 * _base(p)
        g = float(rng.uniform(0.045, 0.085))
        for t, y in enumerate(range(2011, 2024)):
            rows.append({"地区": p, "年份": y,
                         "人均GDP": round(lvl * (1 + g) ** t * (1 + rng.normal(0, 0.02)), 1)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(DATA, exist_ok=True)
    nqp = gen_nqp()
    nqp.to_csv(os.path.join(DATA, "demo_新质生产力面板.csv"), index=False, encoding="utf-8-sig")
    print("demo_新质生产力面板.csv", nqp.shape)

    cp = gen_cp()
    cp.to_csv(os.path.join(DATA, "demo_共同富裕面板.csv"), index=False, encoding="utf-8-sig")
    print("demo_共同富裕面板.csv ", cp.shape)

    pg = gen_pgdp()
    pg.to_csv(os.path.join(DATA, "demo_人均GDP.csv"), index=False, encoding="utf-8-sig")
    print("demo_人均GDP.csv       ", pg.shape)

    # 模板
    tpl_long = pd.DataFrame({
        "地区": ["北京", "北京", "天津", "天津"],
        "年份": [2020, 2020, 2020, 2020],
        "指标": ["人均可支配收入", "城镇化率", "人均可支配收入", "城镇化率"],
        "值": [69434, 87.4, 43854, 84.7],
    })
    tpl_long.to_csv(os.path.join(DATA, "模板_长表.csv"), index=False, encoding="utf-8-sig")

    tpl_wide = pd.DataFrame({
        "地区": ["北京", "天津"],
        "年份": [2020, 2020],
        "人均可支配收入": [69434, 43854],
        "城镇化率": [87.4, 84.7],
    })
    tpl_wide.to_csv(os.path.join(DATA, "模板_宽表.csv"), index=False, encoding="utf-8-sig")
    print("模板_长表.csv / 模板_宽表.csv")


if __name__ == "__main__":
    main()
