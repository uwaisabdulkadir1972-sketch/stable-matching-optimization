"""
visualise.py
============
Reads results and trace CSVs produced by stable_matching.jl and generates charts
for both the final outcome and the Gale-Shapley process itself.

Usage
-----
  pip install pandas matplotlib plotly kaleido
  python visualise.py
"""

import os
import sys
import subprocess
from html import escape

for pkg in ["pandas", "matplotlib", "plotly", "kaleido", "pillow"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

UNIS = ["NUS", "NTU", "SMU", "SIT", "SUTD", "SUSS"]
CAP = {"NUS": 5500, "NTU": 2800, "SMU": 2600, "SIT": 3500, "SUTD": 600, "SUSS": 1200}
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS_UNI = ["#8B0000", "#C0392B", "#E74C3C", "#922B21", "#641E16", "#4A0404"]
RANK_COLORS = ["#1a3a5c", "#1f5c8b", "#2e86c1", "#7fb3d3", "#aed6f1", "#d6eaf8"]
SUTD_RED = "#8B0000"
SUTD_BLUE = "#1a3a5c"
ACCENT_GOLD = "#C48A00"
BG_COLOR = "#FAFAFA"
GRID_COLOR = "#D7DBDD"

plt.rcParams.update(
    {
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B3B6B7",
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.8,
    }
)


def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found.")
        return None
    return pd.read_csv(path)


def load_results(path: str) -> pd.DataFrame | None:
    df = load_csv(path)
    if df is None:
        return None
    df["choice_rank"] = pd.to_numeric(df["choice_rank"], errors="coerce")
    if "unmatched" in df.columns:
        df["unmatched"] = df["unmatched"].astype(bool)
    return df


def load_trace(path: str) -> pd.DataFrame | None:
    df = load_csv(path)
    if df is None:
        return None
    numeric_cols = ["round", "student_id", "choice_rank", "university_rank_of_student", "displaced_student_id", "seats", "tentative_holds"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rank_palette(n: int) -> list[str]:
    if n <= len(RANK_COLORS):
        return RANK_COLORS[:n]
    extra = plt.cm.Blues(np.linspace(0.35, 0.95, n))
    return [matplotlib.colors.to_hex(c) for c in extra]


def plot_fill_rate(df: pd.DataFrame, label: str = "GS-DA", suffix: str = "gs"):
    n_admitted = df[~df["unmatched"]].shape[0]
    fills = {u: (df["assigned_uni"] == u).sum() for u in UNIS}

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"University Fill Rates - {label}", fontsize=14, fontweight="bold", color=SUTD_RED, y=1.01)

    y = np.arange(len(UNIS))
    w = 0.38
    caps = [CAP[u] for u in UNIS]
    fill = [fills[u] for u in UNIS]
    pct = [100.0 * fill[i] / caps[i] for i in range(len(UNIS))]

    ax.barh(y + w / 2, caps, w, color="#D0D3D4", label="Capacity")
    ax.barh(y - w / 2, fill, w, color=SUTD_RED, label="Admitted", alpha=0.9)

    for i, (f, p) in enumerate(zip(fill, pct)):
        ax.text(f + max(caps) * 0.01, i - w / 2, f"{f:,} ({p:.1f}%)", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(UNIS, fontsize=11)
    ax.set_xlabel("Number of students", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Total admitted: {n_admitted:,} / {len(df):,} ({100 * n_admitted / max(len(df), 1):.1f}%)", fontsize=10, color="#555555")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"fill_rate_{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_choice_rank(df: pd.DataFrame, label: str = "GS-DA", suffix: str = "gs"):
    admitted = df[~df["unmatched"] & df["choice_rank"].notna()]
    counts = admitted["choice_rank"].value_counts().sort_index()
    total = len(admitted)
    ranks = list(range(1, int(max(counts.index.max() if not counts.empty else 0, 1)) + 1))
    colors = rank_palette(len(ranks))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Student Choice-Rank Distribution - {label}", fontsize=14, fontweight="bold", color=SUTD_RED, y=1.01)

    vals = [counts.get(r, 0) for r in ranks]
    pcts = [100.0 * v / max(total, 1) for v in vals]

    bars = ax.bar(ranks, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(total * 0.01, 1), f"{int(h):,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Student's choice rank (1 = top choice)", fontsize=10)
    ax.set_ylabel("Number of students", fontsize=10)
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"Choice {r}" for r in ranks], fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"choice_rank_{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_rp_by_uni(df: pd.DataFrame, label: str = "GS-DA", suffix: str = "gs"):
    admitted = df[~df["unmatched"]].copy()
    data = [admitted.loc[admitted["assigned_uni"] == u, "rank_points"].values for u in UNIS]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Admitted Students' Rank Points by University - {label}", fontsize=13, fontweight="bold", color=SUTD_RED, y=1.01)

    bp = ax.boxplot(data, patch_artist=True, notch=False, medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS_UNI):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for whisker in bp["whiskers"]:
        whisker.set(color="#888", linewidth=1)
    for cap in bp["caps"]:
        cap.set(color="#888", linewidth=1)
    for flier in bp["fliers"]:
        flier.set(marker=".", color="#aaa", alpha=0.4, markersize=3)

    ax.set_xticklabels(UNIS, fontsize=11)
    ax.set_ylabel("Rank points (lower = stronger)", fontsize=10)
    ax.set_xlabel("University", fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Lower rank points indicate stronger students.", fontsize=9, color="#555555")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"rp_by_uni_{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sankey(df: pd.DataFrame, label: str = "GS-DA", suffix: str = "gs"):
    admitted = df[~df["unmatched"] & df["choice_rank"].notna()].copy()
    admitted["choice_rank"] = admitted["choice_rank"].astype(int)
    max_rank = int(max(admitted["choice_rank"].max() if not admitted.empty else 0, 1))

    rank_nodes = [f"Choice {r}" for r in range(1, max_rank + 1)]
    uni_nodes = UNIS[:]
    nodes = rank_nodes + uni_nodes

    rank_node_colors = [matplotlib.colors.to_rgba(c, 0.85) for c in rank_palette(max_rank)]
    rank_node_colors = [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})" for r, g, b, a in rank_node_colors]
    uni_node_colors = [f"rgba(139,0,0,0.75)" for _ in UNIS]
    node_colors = rank_node_colors + uni_node_colors

    source, target, value, link_color = [], [], [], []
    for r in range(1, max_rank + 1):
        for j, u in enumerate(UNIS):
            cnt = ((admitted["choice_rank"] == r) & (admitted["assigned_uni"] == u)).sum()
            if cnt > 0:
                source.append(r - 1)
                target.append(max_rank + j)
                value.append(int(cnt))
                alpha = 0.35 if r == 1 else 0.18 + 0.55 / max_rank
                link_color.append(f"rgba(139,0,0,{alpha:.2f})")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=22,
                label=nodes,
                color=node_colors,
                hovertemplate="%{label}: %{value:,} students<extra></extra>",
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_color,
                hovertemplate="%{source.label} -> %{target.label}: %{value:,}<extra></extra>",
            ),
        )
    )

    fig.update_layout(
        title=dict(text=f"Sankey: Choice Rank -> Assigned University | {label}", font=dict(size=15, color=SUTD_RED)),
        font_size=12,
        height=520,
        paper_bgcolor=BG_COLOR,
    )

    html_out = os.path.join(OUT_DIR, f"sankey_{suffix}.html")
    fig.write_html(html_out)
    print(f"  Saved: {html_out}")

    try:
        png_out = os.path.join(OUT_DIR, f"sankey_{suffix}.png")
        fig.write_image(png_out, width=1200, height=520, scale=2)
        print(f"  Saved: {png_out}")
    except Exception as e:
        print(f"  [skip] Sankey PNG failed ({e}).")


def plot_comparison(df_gs: pd.DataFrame, df_milp: pd.DataFrame):
    gs_max = df_gs["choice_rank"].max(skipna=True)
    milp_max = df_milp["choice_rank"].max(skipna=True)
    max_rank = int(max(gs_max if pd.notna(gs_max) else 1, milp_max if pd.notna(milp_max) else 1, 1))
    ranks = list(range(1, max_rank + 1))
    gs_vals = [(df_gs["choice_rank"] == r).sum() for r in ranks]
    ml_vals = [(df_milp["choice_rank"] == r).sum() for r in ranks]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("GS-DA vs MILP - Choice-Rank Distribution Comparison", fontsize=13, fontweight="bold", color=SUTD_RED, y=1.01)

    x, w = np.arange(len(ranks)), 0.38
    ax.bar(x - w / 2, gs_vals, w, label="GS-DA", color=SUTD_RED, alpha=0.85)
    ax.bar(x + w / 2, ml_vals, w, label="MILP", color=SUTD_BLUE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Choice {r}" for r in ranks])
    ax.set_ylabel("Number of students")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "comparison_gs_vs_milp.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_round_activity(trace: pd.DataFrame, suffix: str = "gs"):
    if trace is None or trace.empty:
        print("  [skip] No trace data for round activity.")
        return

    round_summary = (
        trace.groupby("round")
        .agg(
            proposals=("student_id", "count"),
            holds=("status", lambda s: (s == "held").sum()),
            rejections=("status", lambda s: (s == "rejected").sum()),
            displaced=("displaced_student_id", lambda s: (s.fillna(0) > 0).sum()),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Gale-Shapley Round Activity", fontsize=14, fontweight="bold", color=SUTD_RED, y=1.01)

    ax.plot(round_summary["round"], round_summary["proposals"], marker="o", linewidth=2.5, color=SUTD_RED, label="Proposals")
    ax.plot(round_summary["round"], round_summary["rejections"], marker="o", linewidth=2.0, color=SUTD_BLUE, label="Rejections")
    ax.plot(round_summary["round"], round_summary["displaced"], marker="o", linewidth=2.0, color=ACCENT_GOLD, label="Students displaced")

    ax.set_xlabel("Round")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_xticks(round_summary["round"])
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"round_activity_{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_round_acceptance_rate(trace: pd.DataFrame, suffix: str = "gs"):
    if trace is None or trace.empty:
        print("  [skip] No trace data for acceptance-rate chart.")
        return

    round_summary = (
        trace.groupby("round")
        .agg(
            proposals=("student_id", "count"),
            held=("status", lambda s: (s == "held").sum()),
        )
        .reset_index()
    )
    round_summary["acceptance_rate"] = 100.0 * round_summary["held"] / round_summary["proposals"].clip(lower=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Acceptance Rate by Proposal Round", fontsize=14, fontweight="bold", color=SUTD_RED, y=1.01)

    bars = ax.bar(round_summary["round"], round_summary["acceptance_rate"], color=SUTD_BLUE, alpha=0.85)
    for bar, rate in zip(bars, round_summary["acceptance_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Round")
    ax.set_ylabel("Held after proposal (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(round_summary["round"])
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"round_acceptance_rate_{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_university_pressure(trace: pd.DataFrame, suffix: str = "gs"):
    if trace is None or trace.empty:
        print("  [skip] No trace data for university-pressure chart.")
        return

    pressure = trace.pivot_table(index="proposed_uni", columns="round", values="student_id", aggfunc="count", fill_value=0)
    pressure = pressure.reindex(UNIS).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("University Pressure by Round", fontsize=14, fontweight="bold", color=SUTD_RED, y=1.02)

    im = ax.imshow(pressure.values, aspect="auto", cmap="Reds")
    ax.set_yticks(np.arange(len(pressure.index)))
    ax.set_yticklabels(pressure.index)
    ax.set_xticks(np.arange(len(pressure.columns)))
    ax.set_xticklabels(pressure.columns.astype(int))
    ax.set_xlabel("Round")
    ax.set_ylabel("University")
    ax.grid(False)

    for i in range(pressure.shape[0]):
        for j in range(pressure.shape[1]):
            val = int(pressure.iloc[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, shrink=0.88, label="Proposals")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"university_pressure_{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_group_audit(df: pd.DataFrame, suffix: str = "gs"):
    if df is None or "group" not in df.columns:
        print("  [skip] No group column for fairness audit.")
        return

    summary = (
        df.groupby("group")
        .agg(applicants=("student_id", "count"), admitted=("unmatched", lambda s: (~s).sum()))
        .reset_index()
        .sort_values("admitted", ascending=False)
    )
    summary["admission_rate"] = 100.0 * summary["admitted"] / summary["applicants"].clip(lower=1)

    palette = [SUTD_RED, SUTD_BLUE, ACCENT_GOLD, "#7D3C98", "#117A65", "#AF601A"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Group Admission Audit", fontsize=14, fontweight="bold", color=SUTD_RED, y=1.01)

    bars = ax.bar(summary["group"], summary["admission_rate"], color=palette[: len(summary)])
    for bar, admitted, applicants in zip(bars, summary["admitted"], summary["applicants"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{admitted}/{applicants}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Admission rate (%)")
    ax.set_ylim(0, 105)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"group_audit_{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def build_round_hold_snapshot(trace: pd.DataFrame) -> pd.DataFrame:
    if trace is None or trace.empty:
        return pd.DataFrame(columns=["round"] + UNIS)

    hold_counts = {u: 0 for u in UNIS}
    snapshots = []

    for round_id in sorted(trace["round"].dropna().astype(int).unique()):
        round_rows = trace.loc[trace["round"] == round_id].copy()
        for _, row in round_rows.iterrows():
            uni = row["proposed_uni"]
            if row["status"] == "held":
                if int(row.get("displaced_student_id", 0) or 0) > 0:
                    hold_counts[uni] = hold_counts.get(uni, 0)
                else:
                    hold_counts[uni] = hold_counts.get(uni, 0) + 1
        snapshots.append({"round": round_id, **hold_counts})

    return pd.DataFrame(snapshots)


def plot_round_animation(trace: pd.DataFrame, suffix: str = "gs"):
    if trace is None or trace.empty:
        print("  [skip] No trace data for round animation.")
        return

    snapshots = build_round_hold_snapshot(trace)
    if snapshots.empty:
        print("  [skip] No round snapshots available for animation.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    max_cap = max(CAP.values())
    colors = dict(zip(UNIS, COLORS_UNI))

    def draw_frame(frame_idx: int):
        ax.clear()
        row = snapshots.iloc[frame_idx]
        values = [int(row[u]) for u in UNIS]
        pct = [100.0 * values[i] / CAP[UNIS[i]] for i in range(len(UNIS))]

        ax.barh(UNIS, [CAP[u] for u in UNIS], color="#E5E7E9", label="Capacity")
        ax.barh(UNIS, values, color=[colors[u] for u in UNIS], alpha=0.92, label="Tentative holds")

        for i, u in enumerate(UNIS):
            ax.text(values[i] + max_cap * 0.01, i, f"{values[i]:,} ({pct[i]:.1f}%)", va="center", fontsize=9)

        ax.set_xlim(0, max_cap * 1.08)
        ax.set_xlabel("Students tentatively held")
        ax.set_title(f"Gale-Shapley Animation - Round {int(row['round'])}", color=SUTD_RED, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(snapshots), interval=1200, repeat_delay=1500)

    gif_out = os.path.join(OUT_DIR, f"round_animation_{suffix}.gif")
    try:
        ani.save(gif_out, writer="pillow", dpi=140)
        print(f"  Saved: {gif_out}")
    except Exception as e:
        print(f"  [skip] Round animation failed ({e}).")
    finally:
        plt.close(fig)


def plot_model_flow_diagram(suffix: str = "gs"):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("How the Flexible Admissions Model Works", fontsize=16, fontweight="bold", color=SUTD_RED, y=0.98)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    boxes = [
        (0.7, 5.5, 2.2, 1.25, "#1a3a5c", "Applicants\npartial rankings"),
        (3.4, 5.5, 2.2, 1.25, "#2e86c1", "Eligibility\nconstraints"),
        (6.1, 5.5, 2.2, 1.25, "#C48A00", "University\npriority scores"),
        (8.8, 5.5, 2.2, 1.25, "#8B0000", "Dynamic batches\nnew applicants"),
        (2.1, 2.2, 2.8, 1.5, "#117A65", "Preference builder\nacceptable schools only"),
        (5.0, 2.2, 2.8, 1.5, "#7D3C98", "Gale-Shapley rounds\npropose, hold, reject"),
        (7.9, 2.2, 2.8, 1.5, "#922B21", "Outputs\nmatches, trace, audits"),
    ]

    for x, y, w, h, color, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=0.14",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.9,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    arrows = [
        ((2.9, 5.5), (3.4, 3.7)),
        ((4.5, 5.5), (4.1, 3.7)),
        ((7.2, 5.5), (6.3, 3.7)),
        ((9.9, 5.5), (6.8, 3.7)),
        ((4.9, 2.95), (5.0, 2.95)),
        ((7.8, 2.95), (7.9, 2.95)),
    ]

    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2.0, color="#566573"))

    ax.text(2.0, 1.1, "Students can rank only the schools they want.", fontsize=10, color="#34495E")
    ax.text(5.0, 1.1, "Later rounds can rerun globally or lock earlier admits.", fontsize=10, color="#34495E")
    ax.text(8.2, 1.1, "Bias can be audited using group outcome charts.", fontsize=10, color="#34495E")

    out = os.path.join(OUT_DIR, f"model_flow_{suffix}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def write_dashboard(df_gs: pd.DataFrame, trace_gs: pd.DataFrame | None, suffix: str = "gs"):
    total_students = len(df_gs)
    admitted = int((~df_gs["unmatched"]).sum()) if "unmatched" in df_gs.columns else 0
    unmatched = total_students - admitted
    avg_choice = (
        float(df_gs.loc[~df_gs["unmatched"] & df_gs["choice_rank"].notna(), "choice_rank"].mean())
        if "choice_rank" in df_gs.columns and admitted > 0
        else float("nan")
    )
    rounds = int(trace_gs["round"].max()) if trace_gs is not None and not trace_gs.empty else 0
    proposals = int(len(trace_gs)) if trace_gs is not None else 0
    avg_choice_label = f"{avg_choice:.2f}" if avg_choice == avg_choice else "n/a"

    group_html = ""
    if "group" in df_gs.columns:
        group_summary = (
            df_gs.groupby("group")
            .agg(applicants=("student_id", "count"), admitted=("unmatched", lambda s: (~s).sum()))
            .reset_index()
        )
        rows = []
        for _, row in group_summary.iterrows():
            rate = 100.0 * row["admitted"] / max(row["applicants"], 1)
            rows.append(
                f"<tr><td>{escape(str(row['group']))}</td><td>{int(row['applicants'])}</td><td>{int(row['admitted'])}</td><td>{rate:.1f}%</td></tr>"
            )
        group_html = """
        <div class="card">
          <h3>Fairness Snapshot</h3>
          <table>
            <thead><tr><th>Group</th><th>Applicants</th><th>Admitted</th><th>Rate</th></tr></thead>
            <tbody>
        """ + "".join(rows) + """
            </tbody>
          </table>
        </div>
        """

    dashboard = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stable Matching Dashboard</title>
  <style>
    :root {{
      --bg: #f5f1eb;
      --panel: #fffaf5;
      --ink: #2a211c;
      --muted: #6e625a;
      --accent: #8b0000;
      --accent-2: #1a3a5c;
      --line: #dfd5ca;
      --warm: #c48a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Trebuchet MS", serif;
      background:
        radial-gradient(circle at top left, rgba(196,138,0,0.12), transparent 28%),
        radial-gradient(circle at bottom right, rgba(139,0,0,0.10), transparent 30%),
        var(--bg);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(139,0,0,0.96), rgba(26,58,92,0.96));
      color: white;
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 20px 40px rgba(42,33,28,0.12);
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 2rem;
      letter-spacing: 0.03em;
    }}
    .hero p {{
      margin: 0;
      max-width: 850px;
      color: rgba(255,255,255,0.88);
      line-height: 1.5;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
      margin: 20px 0 26px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: 0 10px 24px rgba(42,33,28,0.05);
    }}
    .stat .k {{
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--accent);
    }}
    .stat .l {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-top: 4px;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .tab-btn {{
      border: 0;
      background: #eadfd3;
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 16px;
      cursor: pointer;
      font-weight: 700;
    }}
    .tab-btn.active {{
      background: var(--accent);
      color: white;
    }}
    .tab {{
      display: none;
    }}
    .tab.active {{
      display: block;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 12px 28px rgba(42,33,28,0.06);
    }}
    .card h3 {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 1.05rem;
    }}
    .card p {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.45;
    }}
    img, iframe {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
    }}
    iframe {{
      min-height: 560px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      color: var(--accent-2);
    }}
    .note {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    @media (max-width: 720px) {{
      .hero h1 {{ font-size: 1.6rem; }}
      .wrap {{ padding: 18px 14px 30px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Stable Matching Admissions Dashboard</h1>
      <p>This dashboard explains both the final admissions result and how the flexible Gale-Shapley process unfolds round by round. It supports partial rankings, dynamic applicant batches, and fairness auditing.</p>
    </section>

    <section class="stats">
      <div class="stat"><div class="k">{total_students:,}</div><div class="l">Applicants</div></div>
      <div class="stat"><div class="k">{admitted:,}</div><div class="l">Admitted</div></div>
      <div class="stat"><div class="k">{unmatched:,}</div><div class="l">Unmatched</div></div>
      <div class="stat"><div class="k">{avg_choice_label}</div><div class="l">Average admitted choice rank</div></div>
      <div class="stat"><div class="k">{rounds:,}</div><div class="l">Proposal rounds</div></div>
      <div class="stat"><div class="k">{proposals:,}</div><div class="l">Total proposals traced</div></div>
    </section>

    <nav class="tabs">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="process">Process</button>
      <button class="tab-btn" data-tab="fairness">Fairness</button>
      <button class="tab-btn" data-tab="explainer">Explainer</button>
    </nav>

    <section id="overview" class="tab active">
      <div class="grid">
        <div class="card">
          <h3>Fill Rates</h3>
          <p>How many seats each university filled under the final matching.</p>
          <img src="fill_rate_{suffix}.png" alt="University fill rates">
        </div>
        <div class="card">
          <h3>Choice Outcomes</h3>
          <p>How often students received their first, second, or later ranked school.</p>
          <img src="choice_rank_{suffix}.png" alt="Choice-rank distribution">
        </div>
        <div class="card">
          <h3>Rank Points by University</h3>
          <p>Distribution of admitted student rank points across universities.</p>
          <img src="rp_by_uni_{suffix}.png" alt="Rank points by university">
        </div>
        <div class="card">
          <h3>Flow From Preference to Assignment</h3>
          <p>Interactive Sankey showing how admitted students moved from their choice rank to the assigned university.</p>
          <iframe src="sankey_{suffix}.html" title="Sankey diagram"></iframe>
        </div>
      </div>
    </section>

    <section id="process" class="tab">
      <div class="grid">
        <div class="card">
          <h3>Round-by-Round Animation</h3>
          <p>The tentative holds evolve as proposals and rejections propagate through the market.</p>
          <img src="round_animation_{suffix}.gif" alt="Round animation">
        </div>
        <div class="card">
          <h3>Proposal Activity</h3>
          <p>Each round shows how many proposals were made and how much churn happened through rejections and displacements.</p>
          <img src="round_activity_{suffix}.png" alt="Round activity">
        </div>
        <div class="card">
          <h3>Acceptance Rate</h3>
          <p>Later rounds usually get tougher as popular schools fill up and students fall back to lower choices.</p>
          <img src="round_acceptance_rate_{suffix}.png" alt="Acceptance rate by round">
        </div>
        <div class="card">
          <h3>University Pressure Heatmap</h3>
          <p>This highlights which universities receive the most pressure in each round.</p>
          <img src="university_pressure_{suffix}.png" alt="University pressure heatmap">
        </div>
      </div>
    </section>

    <section id="fairness" class="tab">
      <div class="grid">
        <div class="card">
          <h3>Group Audit Chart</h3>
          <p>Admission rates by group make it easier to spot potential disparities that deserve investigation.</p>
          <img src="group_audit_{suffix}.png" alt="Group admission audit">
        </div>
        {group_html}
      </div>
      <div class="note">This audit is descriptive, not a guarantee of fairness. If you need stronger protections, the next step is to add explicit fairness constraints or policy rules into the matching model.</div>
    </section>

    <section id="explainer" class="tab">
      <div class="grid">
        <div class="card">
          <h3>Model Flow</h3>
          <p>The remodel separates applicant scores, eligibility, preference building, and matching so the system can handle partial rankings and later applicant waves.</p>
          <img src="model_flow_{suffix}.png" alt="Model flow diagram">
        </div>
        <div class="card">
          <h3>What This Model Can Explain</h3>
          <p>Use the dashboard to tell four connected stories: who got in, how the rounds evolved, where competition was strongest, and whether different groups saw different outcomes.</p>
          <p>For dynamic admissions, you can also export round-specific result and trace files from Julia and reuse this same visual pattern for each batch.</p>
        </div>
      </div>
    </section>
  </div>

  <script>
    const buttons = document.querySelectorAll('.tab-btn');
    const tabs = document.querySelectorAll('.tab');
    buttons.forEach((btn) => {{
      btn.addEventListener('click', () => {{
        buttons.forEach((b) => b.classList.remove('active'));
        tabs.forEach((t) => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
      }});
    }});
  </script>
</body>
</html>
"""

    out = os.path.join(OUT_DIR, f"dashboard_{suffix}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(dashboard)
    print(f"  Saved: {out}")


def main():
    print("\nVisualise.py - Stable Matching Charts")
    print("=" * 50)

    df_gs = load_results("results_gs.csv")
    df_milp = load_results("results_milp.csv")
    trace_gs = load_trace("trace_gs.csv")

    if df_gs is None:
        print("ERROR: results_gs.csv not found. Run stable_matching.jl first.")
        return

    print("\n[Outcome charts]")
    plot_fill_rate(df_gs, label="Gale-Shapley", suffix="gs")
    plot_choice_rank(df_gs, label="Gale-Shapley", suffix="gs")
    plot_rp_by_uni(df_gs, label="Gale-Shapley", suffix="gs")
    plot_sankey(df_gs, label="Gale-Shapley", suffix="gs")
    plot_group_audit(df_gs, suffix="gs")

    print("\n[Process charts]")
    plot_round_activity(trace_gs, suffix="gs")
    plot_round_acceptance_rate(trace_gs, suffix="gs")
    plot_university_pressure(trace_gs, suffix="gs")
    plot_round_animation(trace_gs, suffix="gs")
    plot_model_flow_diagram(suffix="gs")
    write_dashboard(df_gs, trace_gs, suffix="gs")

    if df_milp is not None:
        print("\n[Comparison chart]")
        plot_comparison(df_gs, df_milp)

    print(f"\nAll charts saved to: ./{OUT_DIR}/\n")


if __name__ == "__main__":
    main()
