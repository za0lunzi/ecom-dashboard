import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# =========================
# Page
# =========================
st.set_page_config(page_title="Ecom Dashboard", layout="wide")
st.title("Ecom Dashboard 🚀")

# =========================
# Visual preferences
# =========================
HOVER_FONT_SIZE = 18
AXIS_TICK_FORMAT = ",.0f"   # 千分位 + 0小数（不要k、不要小数）
PCT_FORMAT = ".1%"          # 百分比显示

# 成本列（按你的字段名）
COST_COLS = ["采购费", "海运费", "佣金", "配送费", "广告费", "仓储费"]

# =========================
# Robust paths (Cloud-safe)
# =========================
BASE_DIR = Path(__file__).resolve().parent

def pick_file(name: str) -> Path:
    """优先读 data/name，如果不存在就读根目录 name"""
    p1 = BASE_DIR / "data" / name
    p2 = BASE_DIR / name
    if p1.exists():
        return p1
    return p2

SALES_FILE = pick_file("sales.xlsx")
MAP_FILE   = pick_file("mapping.xlsx")

# =========================
# Helpers
# =========================
@st.cache_data
def load_excel(path: Path):
    return pd.read_excel(path)

def norm_platform(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def make_product_key(platform: str, sku: str) -> str:
    return f"{platform}|{sku}"

def safe_to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def wow_delta(s: pd.Series) -> pd.Series:
    return s - s.shift(1)

def wow_pct(s: pd.Series) -> pd.Series:
    prev = s.shift(1)
    return np.where(prev.notna() & (prev != 0), (s - prev) / np.abs(prev), np.nan)

def fmt_money(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:,.0f}"

def fmt_pct(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.1%}"

def ensure_exists_or_stop(p: Path, label: str):
    if not p.exists():
        st.error(f"❌ 找不到 {label}：{p}\n\n"
                 f"请确认文件已上传到仓库根目录（或 data/ 目录）且文件名一致。")
        st.stop()

# =========================
# Load
# =========================
ensure_exists_or_stop(SALES_FILE, "sales.xlsx")
ensure_exists_or_stop(MAP_FILE, "mapping.xlsx")

sales = load_excel(SALES_FILE)
mp = load_excel(MAP_FILE)

# =========================
# Clean sales
# =========================
sales = sales.loc[:, ~sales.columns.astype(str).str.startswith("Unnamed")].copy()

# 必需字段检查（不够就提示）
required_cols = ["平台", "ORDER_PLACED_DT", "year", "week", "VENDOR_SKU", "QUANTITY", "gmv（￥）", "利润", "利润率"]
missing = [c for c in required_cols if c not in sales.columns]
if missing:
    st.error(f"❌ sales.xlsx 缺少字段：{missing}\n请检查表头是否一致。")
    st.stop()

sales["平台"] = sales["平台"].apply(norm_platform)
sales["VENDOR_SKU"] = sales["VENDOR_SKU"].astype(str).str.strip()

sales["ORDER_PLACED_DT"] = pd.to_datetime(sales["ORDER_PLACED_DT"], errors="coerce")
sales["year"] = pd.to_numeric(sales["year"], errors="coerce")
sales["week"] = pd.to_numeric(sales["week"], errors="coerce")

sales["year_week"] = sales.apply(
    lambda r: f"{int(r['year'])}-W{int(r['week']):02d}"
    if pd.notna(r["year"]) and pd.notna(r["week"]) else None,
    axis=1
)

sales["product_key"] = sales.apply(lambda r: make_product_key(r["平台"], r["VENDOR_SKU"]), axis=1)

# 数值列
num_cols = ["gmv（￥）", "QUANTITY", "利润", "利润率"] + COST_COLS
sales = safe_to_numeric(sales, num_cols)

# =========================
# Clean mapping -> long (by platform)
# =========================
mp = mp.copy()
# 允许 mapping 表字段存在也可能不一致，做“尽力匹配”
# 你当前示例里有这些列名：
# 沃尔玛SKU / 亚马逊sku / 产品名 / 产品品类 / 月度目标-wfs / 目前库存-wfs / 在途库存-wfs / 月度目标-fba / 目前库存-fba / 在途库存-fba
for c in ["沃尔玛SKU", "亚马逊sku"]:
    if c in mp.columns:
        mp[c] = mp[c].astype(str).str.strip()

def build_mp_long():
    parts = []

    if all(c in mp.columns for c in ["沃尔玛SKU", "产品名", "产品品类", "月度目标-wfs", "目前库存-wfs", "在途库存-wfs"]):
        wm = mp[["沃尔玛SKU", "产品名", "产品品类", "月度目标-wfs", "目前库存-wfs", "在途库存-wfs"]].copy()
        wm = wm.rename(columns={
            "沃尔玛SKU": "VENDOR_SKU",
            "产品品类": "产品类别_map",
            "月度目标-wfs": "月度目标_QUANTITY",
            "目前库存-wfs": "当前库存",
            "在途库存-wfs": "在途库存",
        })
        wm["平台"] = "沃尔玛"
        wm["VENDOR_SKU"] = wm["VENDOR_SKU"].astype(str).str.strip()
        wm["product_key"] = wm.apply(lambda r: make_product_key(r["平台"], r["VENDOR_SKU"]), axis=1)
        parts.append(wm)

    if all(c in mp.columns for c in ["亚马逊sku", "产品名", "产品品类", "月度目标-fba", "目前库存-fba", "在途库存-fba"]):
        amz = mp[["亚马逊sku", "产品名", "产品品类", "月度目标-fba", "目前库存-fba", "在途库存-fba"]].copy()
        amz = amz.rename(columns={
            "亚马逊sku": "VENDOR_SKU",
            "产品品类": "产品类别_map",
            "月度目标-fba": "月度目标_QUANTITY",
            "目前库存-fba": "当前库存",
            "在途库存-fba": "在途库存",
        })
        amz["平台"] = "亚马逊"
        amz["VENDOR_SKU"] = amz["VENDOR_SKU"].astype(str).str.strip()
        amz["product_key"] = amz.apply(lambda r: make_product_key(r["平台"], r["VENDOR_SKU"]), axis=1)
        parts.append(amz)

    if not parts:
        return pd.DataFrame(columns=["product_key", "月度目标_QUANTITY", "当前库存", "在途库存", "产品类别_map"])

    out = pd.concat(parts, ignore_index=True)
    out = safe_to_numeric(out, ["月度目标_QUANTITY", "当前库存", "在途库存"])
    return out

mp_long = build_mp_long()

# =========================
# Merge (keep sales rows)
# =========================
df = sales.merge(
    mp_long[["product_key", "月度目标_QUANTITY", "当前库存", "在途库存", "产品类别_map"]],
    on="product_key",
    how="left"
)

# 品类：优先 sales 的“产品类别”，为空用 mapping 补
if "产品类别" in df.columns:
    df["产品类别"] = df["产品类别"].fillna(df["产品类别_map"])
else:
    df["产品类别"] = df["产品类别_map"]

df = df.drop(columns=["产品类别_map"])

# =========================
# Sidebar filters
# =========================
st.sidebar.header("筛选")
hide_no_target = st.sidebar.checkbox("隐藏无目标SKU（推荐）", value=True)

platform_options = ["全部"] + sorted([x for x in df["平台"].dropna().unique().tolist() if x != ""])
platform_sel = st.sidebar.selectbox("平台", platform_options, index=0)

cat_options = ["全部"] + sorted(df["产品类别"].dropna().unique().tolist())
cat_sel = st.sidebar.selectbox("产品类别", cat_options, index=0)

sku_options = ["全部"] + sorted(df["VENDOR_SKU"].dropna().unique().tolist())
sku_sel = st.sidebar.selectbox("VENDOR_SKU", sku_options, index=0)

df_f = df.copy()
if hide_no_target:
    df_f = df_f[df_f["月度目标_QUANTITY"].notna()].copy()
if platform_sel != "全部":
    df_f = df_f[df_f["平台"] == platform_sel].copy()
if cat_sel != "全部":
    df_f = df_f[df_f["产品类别"] == cat_sel].copy()
if sku_sel != "全部":
    df_f = df_f[df_f["VENDOR_SKU"] == sku_sel].copy()

st.success("✅ 数据准备完成")

c1, c2, c3, c4 = st.columns(4)
c1.metric("销售明细行数", f"{len(df):,}")
c2.metric("筛选后明细行数", f"{len(df_f):,}")
c3.metric("有目标明细行数", f"{df['月度目标_QUANTITY'].notna().sum():,}")
c4.metric("无目标明细行数", f"{df['月度目标_QUANTITY'].isna().sum():,}")

# =========================
# Weekly aggregate (one place for all modules)
# =========================
agg_dict = {
    "gmv（￥）": "sum",
    "QUANTITY": "sum",
    "利润": "sum",
    "采购费": "sum",
    "海运费": "sum",
    "佣金": "sum",
    "配送费": "sum",
    "广告费": "sum",
    "仓储费": "sum",
}
weekly = df_f.groupby("year_week", as_index=False).agg(agg_dict)
weekly["利润率"] = np.where(weekly["gmv（￥）"] != 0, weekly["利润"] / weekly["gmv（￥）"], np.nan)

# sort by year/week
weekly["year_num"] = weekly["year_week"].str.extract(r"(\d{4})").astype(float)
weekly["week_num"] = weekly["year_week"].str.extract(r"W(\d{2})").astype(float)
weekly = weekly.sort_values(["year_num", "week_num"]).drop(columns=["year_num", "week_num"])

# =========================
# Trend module (single / dual)
# =========================
st.subheader("周趋势")

metric_map = {
    "GMV（￥）": "gmv（￥）",
    "销量（QUANTITY）": "QUANTITY",
    "利润": "利润",
    "广告费": "广告费",
    "利润率": "利润率",
}

st.sidebar.subheader("趋势图")
chart_mode = st.sidebar.selectbox("趋势模式", ["单指标", "双指标对比（双轴）"], index=0)

def make_single_line(metric_label, metric_col):
    fig = px.line(weekly, x="year_week", y=metric_col, markers=True, title=f"{metric_label}（按周）")
    if metric_col == "利润率":
        fig.update_yaxes(tickformat=PCT_FORMAT)
        fig.update_traces(hovertemplate="%{x}<br>" + f"{metric_label}: %{{y:{PCT_FORMAT}}}<extra></extra>")
    else:
        fig.update_yaxes(tickformat=AXIS_TICK_FORMAT)
        fig.update_traces(hovertemplate="%{x}<br>" + f"{metric_label}: %{{y:{AXIS_TICK_FORMAT}}}<extra></extra>")
    fig.update_layout(hoverlabel=dict(font_size=HOVER_FONT_SIZE), hovermode="x unified")
    return fig

def make_dual_line(left_label, left_col, right_label, right_col):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=weekly["year_week"], y=weekly[left_col],
            name=left_label, mode="lines+markers",
            hovertemplate="%{x}<br>" + (
                f"{left_label}: %{{y:{PCT_FORMAT}}}<extra></extra>" if left_col == "利润率"
                else f"{left_label}: %{{y:{AXIS_TICK_FORMAT}}}<extra></extra>"
            )
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=weekly["year_week"], y=weekly[right_col],
            name=right_label, mode="lines+markers",
            hovertemplate="%{x}<br>" + (
                f"{right_label}: %{{y:{PCT_FORMAT}}}<extra></extra>" if right_col == "利润率"
                else f"{right_label}: %{{y:{AXIS_TICK_FORMAT}}}<extra></extra>"
            )
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=f"{left_label} vs {right_label}（按周）",
        hoverlabel=dict(font_size=HOVER_FONT_SIZE),
        hovermode="x unified",
        legend_title_text=""
    )

    if left_col == "利润率":
        fig.update_yaxes(tickformat=PCT_FORMAT, secondary_y=False, title_text=left_label)
    else:
        fig.update_yaxes(tickformat=AXIS_TICK_FORMAT, secondary_y=False, title_text=left_label)

    if right_col == "利润率":
        fig.update_yaxes(tickformat=PCT_FORMAT, secondary_y=True, title_text=right_label)
    else:
        fig.update_yaxes(tickformat=AXIS_TICK_FORMAT, secondary_y=True, title_text=right_label)

    return fig

# =========================
# "周报表"（趋势图下方：只显示当前选择的指标 + 环比）
# =========================
def build_wow_table_single(col: str, label: str):
    tmp = weekly[["year_week", col]].copy()
    tmp["环比值"] = wow_delta(tmp[col])
    tmp["环比%"] = wow_pct(tmp[col])
    # format
    out = tmp.copy()
    if col == "利润率":
        out[label] = out[col].apply(fmt_pct)
        out["环比值"] = out["环比值"].apply(fmt_pct)   # 利润率环比“值”也是百分点变化（这里用百分比显示更直观）
        out["环比%"] = out["环比%"].apply(fmt_pct)
    else:
        out[label] = out[col].apply(fmt_money)
        out["环比值"] = out["环比值"].apply(fmt_money)
        out["环比%"] = out["环比%"].apply(fmt_pct)
    out = out.drop(columns=[col])
    return out

def build_wow_table_dual(left_col, left_label, right_col, right_label):
    cols = ["year_week", left_col, right_col]
    tmp = weekly[cols].copy()

    tmp[f"{left_label}_环比值"] = wow_delta(tmp[left_col])
    tmp[f"{left_label}_环比%"] = wow_pct(tmp[left_col])

    tmp[f"{right_label}_环比值"] = wow_delta(tmp[right_col])
    tmp[f"{right_label}_环比%"] = wow_pct(tmp[right_col])

    out = tmp.copy()

    # left formatting
    if left_col == "利润率":
        out[left_label] = out[left_col].apply(fmt_pct)
        out[f"{left_label}_环比值"] = out[f"{left_label}_环比值"].apply(fmt_pct)
        out[f"{left_label}_环比%"] = out[f"{left_label}_环比%"].apply(fmt_pct)
    else:
        out[left_label] = out[left_col].apply(fmt_money)
        out[f"{left_label}_环比值"] = out[f"{left_label}_环比值"].apply(fmt_money)
        out[f"{left_label}_环比%"] = out[f"{left_label}_环比%"].apply(fmt_pct)

    # right formatting
    if right_col == "利润率":
        out[right_label] = out[right_col].apply(fmt_pct)
        out[f"{right_label}_环比值"] = out[f"{right_label}_环比值"].apply(fmt_pct)
        out[f"{right_label}_环比%"] = out[f"{right_label}_环比%"].apply(fmt_pct)
    else:
        out[right_label] = out[right_col].apply(fmt_money)
        out[f"{right_label}_环比值"] = out[f"{right_label}_环比值"].apply(fmt_money)
        out[f"{right_label}_环比%"] = out[f"{right_label}_环比%"].apply(fmt_pct)

    out = out.drop(columns=[left_col, right_col])
    return out

# draw chart + wow table
if chart_mode == "单指标":
    metric_label = st.sidebar.selectbox("趋势指标", list(metric_map.keys()), index=0)
    metric_col = metric_map[metric_label]
    fig = make_single_line(metric_label, metric_col)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 周报表（当前指标 + 环比）")
    wow_tbl = build_wow_table_single(metric_col, metric_label)
    st.dataframe(wow_tbl, use_container_width=True)

else:
    keys = list(metric_map.keys())
    left_label = st.sidebar.selectbox("左轴指标", keys, index=0)
    right_default_idx = keys.index("利润") if "利润" in keys else 0
    right_label = st.sidebar.selectbox("右轴指标", keys, index=right_default_idx)

    left_col = metric_map[left_label]
    right_col = metric_map[right_label]

    fig = make_dual_line(left_label, left_col, right_label, right_col)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 周报表（当前双指标 + 环比）")
    wow_tbl = build_wow_table_dual(left_col, left_label, right_col, right_label)
    st.dataframe(wow_tbl, use_container_width=True)

# =========================
# Target module (monthly)
# =========================
st.subheader("目标与达成（按月）")

if df_f["ORDER_PLACED_DT"].notna().any():
    current_month = df_f["ORDER_PLACED_DT"].max().to_period("M")
    df_f = df_f.copy()
    df_f["year_month"] = df_f["ORDER_PLACED_DT"].dt.to_period("M")
    cur = df_f[df_f["year_month"] == current_month].copy()

    sku_month = cur.groupby(["平台", "VENDOR_SKU"], as_index=False).agg(
        本月销量=("QUANTITY", "sum"),
        月度目标=("月度目标_QUANTITY", "first"),
    )
    sku_month["完成率"] = np.where(sku_month["月度目标"] > 0, sku_month["本月销量"] / sku_month["月度目标"], np.nan)

    total_actual = sku_month["本月销量"].sum()
    total_target = sku_month["月度目标"].sum()
    total_rate = (total_actual / total_target) if total_target else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("当前月", str(current_month))
    k2.metric("本月累计销量", f"{total_actual:,.0f}")
    k3.metric("本月目标完成率", f"{total_rate:.1%}" if pd.notna(total_rate) else "—")

    show_top = sku_month.sort_values("完成率", ascending=False).head(20).copy()
    show_top["本月销量"] = show_top["本月销量"].apply(fmt_money)
    show_top["月度目标"] = show_top["月度目标"].apply(fmt_money)
    show_top["完成率"] = show_top["完成率"].apply(fmt_pct)

    st.write("SKU 目标完成率（Top 20）")
    st.dataframe(show_top, use_container_width=True)

else:
    st.info("当前筛选数据没有可用日期，无法计算本月目标达成。")

# =========================
# Cost structure module
# =========================
st.subheader("成本结构（按周）")

tab1, tab2 = st.tabs(["成本金额堆叠", "成本占比（100%堆叠）"])

cost_long = weekly.melt(
    id_vars=["year_week"],
    value_vars=[c for c in COST_COLS if c in weekly.columns],
    var_name="成本项",
    value_name="金额"
)

with tab1:
    fig_cost_amt = px.bar(
        cost_long, x="year_week", y="金额", color="成本项",
        title="每周成本结构（金额堆叠）"
    )
    fig_cost_amt.update_layout(hoverlabel=dict(font_size=HOVER_FONT_SIZE), hovermode="x unified")
    fig_cost_amt.update_yaxes(tickformat=AXIS_TICK_FORMAT)
    fig_cost_amt.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:,.0f}<extra></extra>")
    st.plotly_chart(fig_cost_amt, use_container_width=True)

with tab2:
    cost_share = cost_long.copy()
    total_by_week = cost_share.groupby("year_week")["金额"].transform("sum")
    cost_share["占比"] = np.where(total_by_week != 0, cost_share["金额"] / total_by_week, np.nan)

    fig_cost_pct = px.bar(
        cost_share, x="year_week", y="占比", color="成本项",
        title="每周成本结构（占比 100%堆叠）"
    )
    fig_cost_pct.update_layout(hoverlabel=dict(font_size=HOVER_FONT_SIZE), hovermode="x unified")
    fig_cost_pct.update_yaxes(tickformat=PCT_FORMAT)
    fig_cost_pct.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:.1%}<extra></extra>")
    st.plotly_chart(fig_cost_pct, use_container_width=True)

# =========================
# Single-week snapshot (amount + share + profit)
# =========================
st.subheader("单周快照（金额 + 占比 + 利润）")

week_options = weekly["year_week"].dropna().unique().tolist()
if len(week_options) == 0:
    st.info("当前筛选没有周数据。")
else:
    selected_week = st.selectbox("选择周（year_week）", week_options, index=len(week_options) - 1)
    row = weekly[weekly["year_week"] == selected_week].iloc[0]

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("GMV（￥）", f"{row['gmv（￥）']:,.0f}")
    a2.metric("利润", f"{row['利润']:,.0f}")
    a3.metric("利润率", f"{row['利润率']:.1%}" if pd.notna(row["利润率"]) else "—")
    a4.metric("广告费", f"{row['广告费']:,.0f}")

    cost_items = [c for c in COST_COLS if c in weekly.columns]
    cost_table = pd.DataFrame({
        "成本项": cost_items,
        "金额": [row[c] for c in cost_items],
    })
    cost_sum = cost_table["金额"].sum()
    cost_table["占比"] = np.where(cost_sum != 0, cost_table["金额"] / cost_sum, np.nan)

    # format table
    cost_table_show = cost_table.copy()
    cost_table_show["金额"] = cost_table_show["金额"].apply(fmt_money)
    cost_table_show["占比"] = cost_table_show["占比"].apply(fmt_pct)

    left, right = st.columns([1, 1])
    with left:
        st.write("成本明细（金额 + 占比）")
        st.dataframe(cost_table_show, use_container_width=True)

    with right:
        fig_pie = px.pie(cost_table, names="成本项", values="金额", title="成本占比（环图）", hole=0.45)
        fig_pie.update_traces(textinfo="percent+label",
                              hovertemplate="%{label}: %{value:,.0f}（%{percent}）<extra></extra>")
        fig_pie.update_layout(hoverlabel=dict(font_size=HOVER_FONT_SIZE))
        st.plotly_chart(fig_pie, use_container_width=True)

    # waterfall: GMV -> costs -> profit
    measures = ["absolute"] + ["relative"] * len(cost_items) + ["total"]
    x = ["GMV"] + cost_items + ["利润"]
    y = [row["gmv（￥）"]] + [-row[c] for c in cost_items] + [row["利润"]]

    fig_wf = go.Figure(go.Waterfall(
        name="",
        measure=measures,
        x=x,
        y=y,
        text=[f"{v:,.0f}" for v in y],
        textposition="outside",
        connector={"line": {"width": 1}},
        hovertemplate="%{x}: %{y:,.0f}<extra></extra>"
    ))
    fig_wf.update_layout(title="GMV → 成本 → 利润（瀑布图）", hoverlabel=dict(font_size=HOVER_FONT_SIZE))
    st.plotly_chart(fig_wf, use_container_width=True)

# =========================
# Debug info (optional)
# =========================
with st.expander("（可选）数据路径与字段检查"):
    st.write("SALES_FILE:", str(SALES_FILE))
    st.write("MAP_FILE:", str(MAP_FILE))
    st.write("sales columns:", list(sales.columns))
    st.write("mapping columns:", list(mp.columns))
