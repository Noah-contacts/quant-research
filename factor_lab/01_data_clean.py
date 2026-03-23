# ============================================================
#  01_data_clean.py — 数据获取与清洗
#
#  这个文件只做一件事：
#  从Tushare拉取原始数据 → 清洗 → 保存到 data/ 文件夹
#
#  运行完后 data/ 里会出现每只股票的CSV文件
#  后续所有模块都从这里读数据，不再重复请求Tushare
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts
import pandas as pd
import numpy as np
from config import TUSHARE_TOKEN, DATA_DIR, START_DATE, END_DATE, STOCK_POOL

# ── 初始化Tushare ─────────────────────────────────────────
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

def clean_single_stock(df):
    """
    清洗单只股票数据
    输入：从Tushare拉下来的原始DataFrame
    输出：干净的DataFrame，可以直接用于因子计算
    """

    # 第一步：按日期排序（Tushare默认倒序，改成正序）
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 第二步：删除成交量为0的行（停牌日）
    # 停牌时价格不变，会让动量因子误以为股票很稳定
    df = df[df["vol"] > 0]

    # 第三步：删除涨跌幅异常的行（数据错误）
    # A股涨跌停是±10%（ST股±5%），超过±11%基本是数据错误
    df = df[df["pct_chg"].abs() <= 11]

    # 第四步：极端值处理（Winsorize缩尾）
    # 把换手率、成交量等指标中的离群值截断到合理范围
    for col in ["vol", "amount", "pct_chg"]:
        if col in df.columns:
            mean = df[col].mean()
            std  = df[col].std()
            df[col] = df[col].clip(
                lower = mean - 3 * std,
                upper = mean + 3 * std
            )

    # 第五步：重置索引
    df = df.reset_index(drop=True)

    return df


def fetch_and_save(ts_code):
    """
    拉取单只股票数据，清洗后保存为CSV
    文件名格式：data/000001.SZ.csv
    """
    print(f"  正在处理 {ts_code} ...", end=" ")

    try:
        # 拉取日K数据
        df = pro.daily(
            ts_code    = ts_code,
            start_date = START_DATE,
            end_date   = END_DATE,
            fields     = "ts_code,trade_date,open,high,low,close,vol,amount,pct_chg"
        )

        if df is None or df.empty:
            print("❌ 无数据")
            return False

        # 拉取基本面数据（PE、PB、换手率）
        basic = pro.daily_basic(
            ts_code    = ts_code,
            start_date = START_DATE,
            end_date   = END_DATE,
            fields = "trade_date,pe_ttm,pb,turnover_rate,volume_ratio,total_mv,circ_mv"
        )

        # 合并两张表
        if basic is not None and not basic.empty:
            df = df.merge(basic, on="trade_date", how="left")

        # 清洗数据
        df = clean_single_stock(df)

        # 保存到 data/ 文件夹
        os.makedirs(DATA_DIR, exist_ok=True)
        save_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"✅ {len(df)} 条记录已保存")
        return True

    except Exception as e:
        print(f"❌ 失败：{e}")
        return False


# ── 主程序 ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  数据获取与清洗 启动")
    print(f"  区间：{START_DATE} → {END_DATE}")
    print(f"  股票数量：{len(STOCK_POOL)} 只")
    print("=" * 50)

    success = 0
    for code in STOCK_POOL:
        if fetch_and_save(code):
            success += 1

    print("=" * 50)
    print(f"  完成！成功 {success}/{len(STOCK_POOL)} 只")
    print(f"  数据已保存至：{DATA_DIR}")
    print("=" * 50)