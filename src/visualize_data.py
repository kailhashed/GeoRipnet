"""
visualize_data.py  —  GeoRipNet  —  Comprehensive Research Visualizations
Generates all 17 figures from aligned_prices.parquet and adjacency_monthly.parquet.
Run from project root:  python src/visualize_data.py
"""

import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRICE_FILE  = DATA_DIR / "aligned_prices.parquet"
ADJ_FILE    = DATA_DIR / "uncomtrade" / "adjacency_monthly.parquet"
ADJ_CSV     = DATA_DIR / "uncomtrade" / "adjacency_monthly_readable.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
NODES       = ["WTI", "Brent", "OPEC", "ESPO", "Indian"]
NODE_LABELS = ["WTI\n(USA)", "Brent\n(GBR/NOR)", "OPEC\n(SAU)", "ESPO\n(RUS)", "Indian\n(IND)"]
NODE_SHORT  = ["WTI", "Brent", "OPEC", "ESPO", "Indian"]

MODEL_START = "2010-01-01"
MODEL_END   = "2026-03-23"
TRAIN_END   = "2019-12-31"
VAL_START   = "2020-01-01"
VAL_END     = "2021-12-31"
TEST_START  = "2022-01-01"

COLORS = {
    "WTI":    "#1f77b4",
    "Brent":  "#ff7f0e",
    "OPEC":   "#2ca02c",
    "ESPO":   "#d62728",
    "Indian": "#9467bd",
}
NODE_COLORS = [COLORS[n] for n in NODES]

DPI = 150
TITLE_FS  = 13
AXES_FS   = 11
LEGEND_FS = 9

# ── Helpers ────────────────────────────────────────────────────────────────────
saved   = []
failed  = []

def save_fig(name: str):
    path = RESULTS_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    saved.append(str(path))
    print(f"  [OK] {name}")

def log_fail(name: str, e: Exception):
    msg = f"{name}: {e}"
    failed.append(msg)
    print(f"  [FAIL] {msg}")
    traceback.print_exc()
    plt.close("all")

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading price data...")
prices_all = pd.read_parquet(PRICE_FILE)
prices_all.index = pd.to_datetime(prices_all.index)
prices_all = prices_all.sort_index()

prices = prices_all.loc[MODEL_START:MODEL_END].copy()

# Daily log-returns (model window)
log_ret = np.log(prices / prices.shift(1)).dropna()

print(f"  Prices full range : {prices_all.index[0].date()} to {prices_all.index[-1].date()} ({len(prices_all)} rows)")
print(f"  Model window      : {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} rows)")
print(f"  Log-return rows   : {len(log_ret)}")

print("Loading adjacency data...")
adj_df  = pd.read_parquet(ADJ_FILE)
adj_df["period"] = adj_df["period"].astype(int)
adj_df  = adj_df.sort_values("period").reset_index(drop=True)
mat_cols = [c for c in adj_df.columns if c.startswith("col_")]

def get_matrix(period_int: int) -> np.ndarray:
    row = adj_df[adj_df["period"] == period_int]
    if row.empty:
        return np.full((5, 5), np.nan)
    return row[mat_cols].values.reshape(5, 5).copy()

def period_to_date(p: int) -> pd.Timestamp:
    y, m = divmod(p, 100)
    return pd.Timestamp(year=y, month=m, day=1)

adj_dates = [period_to_date(p) for p in adj_df["period"]]

print(f"  Adjacency months  : {adj_df['period'].min()} to {adj_df['period'].max()} ({len(adj_df)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Full history all benchmarks
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/17] fig_price_full_history.png")
try:
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in NODES:
        ax.plot(prices_all.index, prices_all[col], label=col, color=COLORS[col],
                linewidth=0.8, alpha=0.9)
    ax.set_title("Oil Benchmark Prices — Full History (1987–2026)", fontsize=TITLE_FS)
    ax.set_xlabel("Date", fontsize=AXES_FS)
    ax.set_ylabel("Price (USD/bbl)", fontsize=AXES_FS)
    ax.legend(fontsize=LEGEND_FS, ncol=5, loc="upper left")
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    save_fig("fig_price_full_history.png")
except Exception as e:
    log_fail("fig_price_full_history.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Model window all benchmarks
# ══════════════════════════════════════════════════════════════════════════════
print("[2/17] fig_price_model_window.png")
try:
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in NODES:
        ax.plot(prices.index, prices[col], label=col, color=COLORS[col],
                linewidth=1.0, alpha=0.9)
    ax.set_title("Oil Benchmark Prices — Model Window (2010–2026)", fontsize=TITLE_FS)
    ax.set_xlabel("Date", fontsize=AXES_FS)
    ax.set_ylabel("Price (USD/bbl)", fontsize=AXES_FS)
    ax.legend(fontsize=LEGEND_FS, ncol=5, loc="upper left")
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    save_fig("fig_price_model_window.png")
except Exception as e:
    log_fail("fig_price_model_window.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Correlation heatmap of daily log-returns
# ══════════════════════════════════════════════════════════════════════════════
print("[3/17] fig_price_correlation_heatmap.png")
try:
    corr = log_ret.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(corr, dtype=bool)
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.5, vmax=1.0, linewidths=0.5,
        xticklabels=NODES, yticklabels=NODES,
        ax=ax, annot_kws={"size": 10}
    )
    ax.set_title("Pearson Correlation of Daily Log-Returns\n(Model Window 2010–2026)", fontsize=TITLE_FS)
    ax.tick_params(labelsize=AXES_FS)
    save_fig("fig_price_correlation_heatmap.png")
except Exception as e:
    log_fail("fig_price_correlation_heatmap.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Rolling 90-day correlation WTI vs others
# ══════════════════════════════════════════════════════════════════════════════
print("[4/17] fig_price_rolling_correlation.png")
try:
    fig, ax = plt.subplots(figsize=(14, 5))
    other_nodes = [n for n in NODES if n != "WTI"]
    for col in other_nodes:
        roll_corr = log_ret["WTI"].rolling(90).corr(log_ret[col])
        ax.plot(roll_corr.index, roll_corr, label=f"WTI vs {col}",
                color=COLORS[col], linewidth=1.2, alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_title("Rolling 90-Day Pearson Correlation — WTI vs Other Benchmarks (2010–2026)", fontsize=TITLE_FS)
    ax.set_xlabel("Date", fontsize=AXES_FS)
    ax.set_ylabel("Correlation", fontsize=AXES_FS)
    ax.legend(fontsize=LEGEND_FS, ncol=4)
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    save_fig("fig_price_rolling_correlation.png")
except Exception as e:
    log_fail("fig_price_rolling_correlation.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Distribution of daily log-returns (5-panel)
# ══════════════════════════════════════════════════════════════════════════════
print("[5/17] fig_price_distributions.png")
try:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    for i, col in enumerate(NODES):
        ax = axes[i]
        data = log_ret[col].dropna()
        ax.hist(data, bins=80, density=True, color=COLORS[col], alpha=0.5,
                edgecolor="none", label="Hist")
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method=0.15)
        xs = np.linspace(data.min(), data.max(), 300)
        ax.plot(xs, kde(xs), color=COLORS[col], linewidth=2.0, label="KDE")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(col, fontsize=TITLE_FS)
        ax.set_xlabel("Log-Return", fontsize=AXES_FS - 1)
        ax.tick_params(labelsize=AXES_FS - 2)
        mu, sigma = data.mean(), data.std()
        ax.text(0.97, 0.95, f"μ={mu:.4f}\nσ={sigma:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    fig.suptitle("Distribution of Daily Log-Returns — Model Window (2010–2026)", fontsize=TITLE_FS, y=1.02)
    save_fig("fig_price_distributions.png")
except Exception as e:
    log_fail("fig_price_distributions.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — 30-day rolling volatility (5-panel)
# ══════════════════════════════════════════════════════════════════════════════
print("[6/17] fig_price_volatility.png")
try:
    vol = log_ret.rolling(30).std() * np.sqrt(252)  # annualised
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    for i, col in enumerate(NODES):
        ax = axes[i]
        ax.plot(vol.index, vol[col], color=COLORS[col], linewidth=0.9, alpha=0.9)
        ax.fill_between(vol.index, vol[col], alpha=0.2, color=COLORS[col])
        ax.set_ylabel(col, fontsize=AXES_FS, color=COLORS[col])
        ax.tick_params(labelsize=AXES_FS - 1)
        ax.grid(alpha=0.2)
    axes[0].set_title("30-Day Rolling Annualised Volatility of Log-Returns (2010–2026)", fontsize=TITLE_FS)
    axes[-1].set_xlabel("Date", fontsize=AXES_FS)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_fig("fig_price_volatility.png")
except Exception as e:
    log_fail("fig_price_volatility.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Scatter matrix of daily log-returns
# ══════════════════════════════════════════════════════════════════════════════
print("[7/17] fig_price_scatter_matrix.png")
try:
    palette_list = [COLORS[n] for n in NODES]
    g = sns.pairplot(
        log_ret.reset_index(drop=True),
        diag_kind="kde",
        plot_kws=dict(alpha=0.15, s=4, color="#555555"),
        diag_kws=dict(linewidth=1.5),
    )
    # Colour diagonal KDE lines
    for i, ax in enumerate(g.diag_axes):
        for line in ax.get_lines():
            line.set_color(NODE_COLORS[i])
    g.figure.suptitle("Scatter Matrix — Daily Log-Returns (2010–2026)", fontsize=TITLE_FS, y=1.01)
    g.figure.set_size_inches(11, 11)
    plt.tight_layout()
    g.figure.savefig(RESULTS_DIR / "fig_price_scatter_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close(g.figure)
    saved.append(str(RESULTS_DIR / "fig_price_scatter_matrix.png"))
    print("  [OK] fig_price_scatter_matrix.png")
except Exception as e:
    log_fail("fig_price_scatter_matrix.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Train / Val / Test split (5-panel stacked)
# ══════════════════════════════════════════════════════════════════════════════
print("[8/17] fig_price_train_val_test_split.png")
try:
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    t_end = pd.Timestamp(TRAIN_END)
    v_s   = pd.Timestamp(VAL_START)
    v_e   = pd.Timestamp(VAL_END)
    te_s  = pd.Timestamp(TEST_START)
    te_e  = prices.index[-1]
    for i, col in enumerate(NODES):
        ax = axes[i]
        ax.plot(prices.index, prices[col], color=COLORS[col], linewidth=0.9, label=col)
        ax.axvspan(prices.index[0], t_end, alpha=0.08, color="blue", label="Train")
        ax.axvspan(v_s, v_e,   alpha=0.15, color="orange", label="Val")
        ax.axvspan(te_s, te_e, alpha=0.10, color="green",  label="Test")
        for vdate, lbl in [(t_end, "Train/Val"), (v_e, "Val/Test")]:
            ax.axvline(vdate, color="black", linestyle="--", linewidth=0.8)
            ax.text(vdate, ax.get_ylim()[1] * 0.92, lbl, fontsize=7.5,
                    rotation=90, va="top", ha="right", color="black")
        ax.set_ylabel(col, fontsize=AXES_FS, color=COLORS[col])
        ax.tick_params(labelsize=AXES_FS - 1)
        ax.grid(alpha=0.2)
    axes[0].set_title("Train / Validation / Test Split — All Benchmarks (2010–2026)", fontsize=TITLE_FS)
    # Legend once
    patches = [
        mpatches.Patch(color="blue",   alpha=0.3, label="Train (2010–2019)"),
        mpatches.Patch(color="orange", alpha=0.4, label="Val   (2020–2021)"),
        mpatches.Patch(color="green",  alpha=0.35, label="Test  (2022–2026)"),
    ]
    axes[0].legend(handles=patches, fontsize=8, loc="upper left", ncol=3)
    axes[-1].set_xlabel("Date", fontsize=AXES_FS)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_fig("fig_price_train_val_test_split.png")
except Exception as e:
    log_fail("fig_price_train_val_test_split.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — WTI negative price event April 2020
# ══════════════════════════════════════════════════════════════════════════════
print("[9/17] fig_price_wti_negative.png")
try:
    event_date = pd.Timestamp("2020-04-20")
    window_start = event_date - pd.Timedelta(days=60)
    window_end   = event_date + pd.Timedelta(days=60)
    wti_zoom = prices_all["WTI"].loc[window_start:window_end]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wti_zoom.index, wti_zoom.values, color=COLORS["WTI"], linewidth=1.5)
    ax.fill_between(wti_zoom.index, wti_zoom.values, 0,
                    where=(wti_zoom.values < 0), color="red", alpha=0.35, label="Negative price region")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(event_date, color="red", linewidth=1.5, linestyle="--", label=f"Min: −$36.98 ({event_date.date()})")
    ax.scatter([event_date], [wti_zoom.loc[event_date] if event_date in wti_zoom.index
                                else wti_zoom.iloc[(wti_zoom.index - event_date).argmin()]],
               color="red", s=60, zorder=5)
    ax.set_title("WTI Crude — Negative Price Event, April 2020 (±60 Days)", fontsize=TITLE_FS)
    ax.set_xlabel("Date", fontsize=AXES_FS)
    ax.set_ylabel("Price (USD/bbl)", fontsize=AXES_FS)
    ax.legend(fontsize=LEGEND_FS)
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.3)
    save_fig("fig_price_wti_negative.png")
except Exception as e:
    log_fail("fig_price_wti_negative.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — COVID and geopolitical events
# ══════════════════════════════════════════════════════════════════════════════
print("[10/17] fig_price_covid_geopolitical.png")
try:
    geo_start = "2019-01-01"
    geo_end   = "2023-12-31"
    geo_prices = prices_all.loc[geo_start:geo_end]

    events = [
        ("2020-03-18", "COVID crash\n(18 Mar 2020)", "#cc0000"),
        ("2020-04-20", "WTI negative\n(20 Apr 2020)", "#ff6600"),
        ("2022-02-24", "Russia-Ukraine\ninvasion (24 Feb 2022)", "#6600cc"),
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    for col in NODES:
        ax.plot(geo_prices.index, geo_prices[col], label=col, color=COLORS[col],
                linewidth=1.1, alpha=0.9)

    ymin, ymax = ax.get_ylim()
    for date_str, label, color in events:
        vd = pd.Timestamp(date_str)
        ax.axvline(vd, color=color, linewidth=1.4, linestyle="--", alpha=0.85)
        ax.text(vd, ymax * 0.97, label, fontsize=7.5, color=color,
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec=color))

    ax.set_title("Oil Benchmarks — COVID Crash & Geopolitical Shocks (Jan 2019 – Dec 2023)", fontsize=TITLE_FS)
    ax.set_xlabel("Date", fontsize=AXES_FS)
    ax.set_ylabel("Price (USD/bbl)", fontsize=AXES_FS)
    ax.legend(fontsize=LEGEND_FS, ncol=5, loc="upper left")
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.3)
    save_fig("fig_price_covid_geopolitical.png")
except Exception as e:
    log_fail("fig_price_covid_geopolitical.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Annual returns bar chart 2011-2025
# ══════════════════════════════════════════════════════════════════════════════
print("[11/17] fig_price_annual_returns.png")
try:
    annual_ret = {}
    for col in NODES:
        yearly = prices_all[col].resample("YE").last()
        ret = yearly.pct_change() * 100
        annual_ret[col] = ret

    ar_df = pd.DataFrame(annual_ret)
    ar_df.index = ar_df.index.year
    ar_df = ar_df.loc[2011:2025]

    years = ar_df.index.tolist()
    n_nodes = len(NODES)
    x = np.arange(len(years))
    width = 0.15

    fig, ax = plt.subplots(figsize=(18, 6))
    for i, col in enumerate(NODES):
        vals = ar_df[col].values
        bars = ax.bar(x + i * width, vals, width, label=col,
                      color=COLORS[col], alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Annual Returns (%) by Benchmark — 2011–2025", fontsize=TITLE_FS)
    ax.set_xlabel("Year", fontsize=AXES_FS)
    ax.set_ylabel("Annual Return (%)", fontsize=AXES_FS)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(years, fontsize=AXES_FS - 1, rotation=30)
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.legend(fontsize=LEGEND_FS, ncol=5)
    ax.grid(axis="y", alpha=0.3)
    save_fig("fig_price_annual_returns.png")
except Exception as e:
    log_fail("fig_price_annual_returns.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Spread analysis (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
print("[12/17] fig_price_spread_analysis.png")
try:
    spreads = {
        "Brent – WTI":    prices["Brent"] - prices["WTI"],
        "Brent – ESPO":   prices["Brent"] - prices["ESPO"],
        "Brent – Indian": prices["Brent"] - prices["Indian"],
    }
    spread_colors = ["#1a6fba", "#c0392b", "#8e44ad"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (label, series) in enumerate(spreads.items()):
        ax = axes[i]
        ax.plot(series.index, series.values, color=spread_colors[i], linewidth=0.9, alpha=0.9)
        ax.fill_between(series.index, series.values, 0,
                        where=(series.values > 0), color=spread_colors[i], alpha=0.12)
        ax.fill_between(series.index, series.values, 0,
                        where=(series.values < 0), color="red", alpha=0.20)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(label + " (USD/bbl)", fontsize=AXES_FS)
        ax.tick_params(labelsize=AXES_FS - 1)
        ax.grid(alpha=0.2)
        mean_val = series.mean()
        ax.axhline(mean_val, color=spread_colors[i], linewidth=1.0, linestyle=":",
                   label=f"Mean: {mean_val:.2f}")
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_title("Oil Price Spread Analysis — Model Window (2010–2026)", fontsize=TITLE_FS)
    axes[-1].set_xlabel("Date", fontsize=AXES_FS)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_fig("fig_price_spread_analysis.png")
except Exception as e:
    log_fail("fig_price_spread_analysis.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Adjacency heatmap — 3 sample periods
# ══════════════════════════════════════════════════════════════════════════════
print("[13/17] fig_adj_heatmap_sample.png")
try:
    sample_periods = [201901, 202206, 202412]
    period_labels  = ["Jan 2019\n(pre-war baseline)", "Jun 2022\n(war peak)", "Dec 2024\n(latest)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, period, label in zip(axes, sample_periods, period_labels):
        mat = get_matrix(period)
        np.fill_diagonal(mat, np.nan)  # hide diagonal for clarity
        sns.heatmap(
            mat, annot=True, fmt=".3f", cmap="YlOrRd",
            vmin=0.0, vmax=0.8, linewidths=0.5,
            xticklabels=NODE_SHORT, yticklabels=NODE_SHORT,
            ax=ax, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"Adjacency Matrix\n{label}", fontsize=TITLE_FS - 1)
        ax.set_xlabel("To Node", fontsize=AXES_FS - 1)
        ax.set_ylabel("From Node", fontsize=AXES_FS - 1)
        ax.tick_params(labelsize=AXES_FS - 1)

    fig.suptitle("Comtrade Adjacency Matrices (Row-Normalised) — Selected Periods", fontsize=TITLE_FS, y=1.01)
    save_fig("fig_adj_heatmap_sample.png")
except Exception as e:
    log_fail("fig_adj_heatmap_sample.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — RUS→IND edge over time (KEY FIGURE)
# ══════════════════════════════════════════════════════════════════════════════
print("[14/17] fig_adj_rus_ind_edge.png")
try:
    rus_ind = []
    for period in adj_df["period"]:
        mat = get_matrix(period)
        rus_ind.append(mat[3, 4])   # row 3 = ESPO/RUS, col 4 = Indian/IND

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(adj_dates, rus_ind, color="#c0392b", linewidth=2.0, marker="o",
            markersize=4, label="RUS→IND edge weight")
    ax.fill_between(adj_dates, rus_ind, alpha=0.18, color="#c0392b")

    invasion_date = pd.Timestamp("2022-02-24")
    ax.axvline(invasion_date, color="black", linewidth=1.8, linestyle="--",
               label="Russia-Ukraine Invasion (24 Feb 2022)")
    ax.text(invasion_date + pd.Timedelta(days=10), max(rus_ind) * 0.96,
            "Invasion\n24 Feb 2022", fontsize=9, color="black", va="top")

    ax.set_title("Comtrade Adjacency: Russia → India Edge Weight Over Time\n"
                 "(Normalised share of India's oil imports from Russia)", fontsize=TITLE_FS)
    ax.set_xlabel("Date", fontsize=AXES_FS)
    ax.set_ylabel("Row-Normalised Weight (RUS→IND)", fontsize=AXES_FS)
    ax.legend(fontsize=LEGEND_FS)
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    save_fig("fig_adj_rus_ind_edge.png")
except Exception as e:
    log_fail("fig_adj_rus_ind_edge.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15 — All non-diagonal edges over time
# ══════════════════════════════════════════════════════════════════════════════
print("[15/17] fig_adj_all_edges_time.png")
try:
    from_node_colors = {
        0: "#1f77b4",  # WTI
        1: "#ff7f0e",  # Brent
        2: "#2ca02c",  # OPEC
        3: "#d62728",  # ESPO
        4: "#9467bd",  # Indian
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    edge_data = {}
    for fr in range(5):
        for to in range(5):
            if fr == to:
                continue
            vals = [get_matrix(p)[fr, to] for p in adj_df["period"]]
            edge_data[(fr, to)] = vals
            label = f"{NODE_SHORT[fr]}→{NODE_SHORT[to]}"
            lw = 2.0 if (fr == 3 and to == 4) else 0.9
            alpha = 1.0 if (fr == 3 and to == 4) else 0.55
            zorder = 10 if (fr == 3 and to == 4) else 2
            ax.plot(adj_dates, vals,
                    color=from_node_colors[fr],
                    linewidth=lw, alpha=alpha, zorder=zorder,
                    label=label if (fr == 3 and to == 4) else "_nolegend_")

    # Highlight RUS→IND
    ax.plot(adj_dates, edge_data[(3, 4)], color="#c0392b", linewidth=2.5,
            label="RUS→IND (highlighted)", zorder=11)
    ax.axvline(pd.Timestamp("2022-02-24"), color="black", linewidth=1.4,
               linestyle="--", label="Invasion (Feb 2022)", zorder=12)

    # Per-source colour legend
    legend_patches = [mpatches.Patch(color=from_node_colors[i], label=f"From {NODE_SHORT[i]}")
                      for i in range(5)]
    legend_patches.append(mpatches.Patch(color="#c0392b", label="RUS→IND (key)"))
    ax.legend(handles=legend_patches, fontsize=8, ncol=3, loc="upper left")

    ax.set_title("All Non-Diagonal Comtrade Adjacency Edge Weights Over Time\n"
                 "(Colour = source node; bold red = RUS→IND)", fontsize=TITLE_FS)
    ax.set_xlabel("Date", fontsize=AXES_FS)
    ax.set_ylabel("Row-Normalised Weight", fontsize=AXES_FS)
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)
    save_fig("fig_adj_all_edges_time.png")
except Exception as e:
    log_fail("fig_adj_all_edges_time.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 16 — India import share stacked bar
# ══════════════════════════════════════════════════════════════════════════════
print("[16/17] fig_adj_india_imports_share.png")
try:
    # Edges into IND (col 4) from all others (rows 0-3)
    india_sources = {NODE_SHORT[fr]: [] for fr in range(4)}
    for period in adj_df["period"]:
        mat = get_matrix(period)
        # col 4 is IND; use the raw (un-renormalized) perspective:
        # adj_df is row-normalised (from-node perspective), so mat[fr, 4] is
        # fraction of fr's exports that go to IND — we want India's import mix.
        # Use the readable CSV for from_node/to_node perspective if available,
        # otherwise invert: col j of the matrix = weight from row i to j.
        for fr in range(4):
            india_sources[NODE_SHORT[fr]].append(mat[fr, 4])

    # Normalise each month so shares sum to 1
    total = np.array([sum(india_sources[n][t] for n in NODE_SHORT[:4])
                      for t in range(len(adj_df))])
    total = np.where(total == 0, 1, total)
    shares = {n: np.array(india_sources[n]) / total for n in NODE_SHORT[:4]}

    fig, ax = plt.subplots(figsize=(16, 6))
    bottoms = np.zeros(len(adj_df))
    bar_colors = [COLORS[n] for n in NODE_SHORT[:4]]
    bar_dates  = [d.to_pydatetime() for d in adj_dates]

    for n, color in zip(NODE_SHORT[:4], bar_colors):
        ax.bar(bar_dates, shares[n], bottom=bottoms,
               color=color, label=n, width=25, alpha=0.85, edgecolor="none")
        bottoms += shares[n]

    ax.axvline(pd.Timestamp("2022-02-24"), color="black", linewidth=1.5,
               linestyle="--", label="Invasion (Feb 2022)")
    ax.set_title("India's Oil Import Sources — Share of Each Exporter Over Time\n"
                 "(Derived from Comtrade Row-Normalised Adjacency)", fontsize=TITLE_FS)
    ax.set_xlabel("Date", fontsize=AXES_FS)
    ax.set_ylabel("Share of India's Imports", fontsize=AXES_FS)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=LEGEND_FS, ncol=5, loc="upper left")
    ax.tick_params(labelsize=AXES_FS - 1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.grid(axis="y", alpha=0.3)
    save_fig("fig_adj_india_imports_share.png")
except Exception as e:
    log_fail("fig_adj_india_imports_share.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 17 — Network snapshots (3-panel networkx)
# ══════════════════════════════════════════════════════════════════════════════
print("[17/17] fig_adj_network_snapshots.png")
try:
    import networkx as nx

    snapshot_periods = [201901, 202206, 202412]
    snapshot_labels  = ["Jan 2019\n(pre-war)", "Jun 2022\n(war peak)", "Dec 2024\n(latest)"]

    # Fixed circular layout for consistency
    pos_base = {
        0: (0, 1),   # WTI top
        1: (1, 0.5), # Brent right
        2: (0.6, -0.8),  # OPEC lower-right
        3: (-0.6, -0.8), # ESPO lower-left
        4: (-1, 0.5),    # Indian left
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, period, label in zip(axes, snapshot_periods, snapshot_labels):
        mat = get_matrix(period)
        G = nx.DiGraph()
        G.add_nodes_from(range(5))
        threshold = 0.05
        for fr in range(5):
            for to in range(5):
                if fr != to and mat[fr, to] > threshold:
                    G.add_edge(fr, to, weight=mat[fr, to])

        # Node size proportional to out-degree (sum of out-weights)
        out_weights = mat.sum(axis=1)
        np.fill_diagonal(mat, 0)
        node_sizes = 600 + 2500 * (out_weights / out_weights.max())

        # Edge widths and alpha
        edges = G.edges(data=True)
        edge_widths = [d["weight"] * 8 for _, _, d in edges]
        edge_colors = [NODE_COLORS[u] for u, _, _ in G.edges()]

        # Highlight RUS→IND
        edge_style = ["dashed" if (u == 3 and v == 4) else "solid"
                      for u, v, _ in G.edges()]

        nx.draw_networkx_nodes(G, pos_base, ax=ax,
                               node_color=NODE_COLORS, node_size=node_sizes,
                               alpha=0.9)
        nx.draw_networkx_labels(G, pos_base, ax=ax,
                                labels={i: NODE_SHORT[i] for i in range(5)},
                                font_size=9, font_color="white", font_weight="bold")
        # Draw edges individually for style control
        for (u, v, d), width, color, style in zip(
                G.edges(data=True), edge_widths, edge_colors, edge_style):
            nx.draw_networkx_edges(
                G, pos_base, edgelist=[(u, v)], ax=ax,
                width=width, edge_color=[color], alpha=0.75,
                arrows=True, arrowsize=15,
                connectionstyle="arc3,rad=0.15",
                style=style
            )

        ax.set_title(f"Network: {label}", fontsize=TITLE_FS - 1)
        ax.axis("off")

        # Weight legend box
        info_lines = []
        if G.has_edge(3, 4):
            w = mat[3, 4]
            info_lines.append(f"RUS→IND: {w:.3f}")
        if info_lines:
            ax.text(0.02, 0.02, "\n".join(info_lines), transform=ax.transAxes,
                    fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round", fc="lightyellow", ec="orange", alpha=0.85))

    fig.suptitle("Comtrade Trade-Flow Network — Directed Weighted Graphs\n"
                 "(Edge width = row-normalised weight; node size = out-degree strength)",
                 fontsize=TITLE_FS, y=0.98)
    save_fig("fig_adj_network_snapshots.png")
except Exception as e:
    log_fail("fig_adj_network_snapshots.png", e)

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"SAVED ({len(saved)}/{len(saved)+len(failed)}):")
for p in saved:
    print(f"  {p}")

if failed:
    print(f"\nFAILED ({len(failed)}):")
    for f in failed:
        print(f"  {f}")
else:
    print("\nAll figures generated successfully.")
print("=" * 65)
