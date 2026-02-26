#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from pybaseball import statcast  # <-- NEW

st.set_page_config(page_title="Pitcher Changes", layout="wide")


# ---------- DATA FETCHING ----------
@st.cache_data(ttl=60 * 60 * 6)  # cache for 6 hours
def fetch_statcast(start_date: str, end_date: str) -> pd.DataFrame:
    df = statcast(start_dt=start_date, end_dt=end_date)

    # Keep only MLB regular season games if column exists
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()

    return df


@st.cache_data
def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    else:
        df["game_date"] = pd.NaT

    # Core fields
    core = [
        "game_date", "pitcher", "player_name", "pitch_type",
        "release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z",
        "balls", "strikes"
    ]
    for c in core:
        if c not in df.columns:
            df[c] = np.nan

    # Movement in inches
    df["hb_in"] = df["pfx_x"] * 12
    df["ivb_in"] = df["pfx_z"] * 12

    # Pitcher handedness
    if "p_throws" in df.columns and "pitcher_hand" not in df.columns:
        df["pitcher_hand"] = df["p_throws"]
    elif "pitcher_hand" not in df.columns:
        df["pitcher_hand"] = np.nan

    # Batter handedness (heatmap filter only)
    if "stand" in df.columns and "batter_hand" not in df.columns:
        df["batter_hand"] = df["stand"]
    elif "batter_hand" not in df.columns:
        df["batter_hand"] = np.nan

    # Count state BEFORE pitch
    df["count_state"] = np.select(
        [
            df["balls"] > df["strikes"],
            df["balls"] < df["strikes"],
            df["balls"] == df["strikes"]
        ],
        ["Behind", "Ahead", "Even"],
        default="Unknown"
    )

    # Pitcher team from home/away + inning_topbot
    if "team" not in df.columns:
        df["team"] = np.nan
    if {"home_team", "away_team", "inning_topbot"}.issubset(df.columns):
        top_mask = df["inning_topbot"].astype(str).str.upper().str.startswith("T")
        bot_mask = df["inning_topbot"].astype(str).str.upper().str.startswith("B")
        df.loc[top_mask, "team"] = df.loc[top_mask, "home_team"]
        df.loc[bot_mask, "team"] = df.loc[bot_mask, "away_team"]

    # Arm angle / slot
    if "arm_angle" in df.columns:
        df["arm_angle_deg"] = pd.to_numeric(df["arm_angle"], errors="coerce")
    else:
        if {"release_pos_x", "release_pos_z"}.issubset(df.columns):
            x = pd.to_numeric(df["release_pos_x"], errors="coerce")
            z = pd.to_numeric(df["release_pos_z"], errors="coerce")
            df["arm_angle_deg"] = np.degrees(np.arctan2(z, np.abs(x)))
        else:
            df["arm_angle_deg"] = np.nan

    # Times Through the Order (TTO)
    if {"game_pk", "at_bat_number", "pitcher"}.issubset(df.columns):
        pa = (
            df.drop_duplicates(["game_pk", "pitcher", "at_bat_number"])
              .sort_values(["game_pk", "pitcher", "at_bat_number"], kind="mergesort")
              .copy()
        )
        pa["bf_index"] = pa.groupby(["game_pk", "pitcher"]).cumcount() + 1

        df = df.merge(
            pa[["game_pk", "pitcher", "at_bat_number", "bf_index"]],
            on=["game_pk", "pitcher", "at_bat_number"],
            how="left"
        )

        df["tto"] = np.select(
            [
                df["bf_index"].between(1, 9),
                df["bf_index"].between(10, 18),
                df["bf_index"].between(19, 27),
                df["bf_index"] >= 28,
            ],
            ["1st time", "2nd time", "3rd time", "4th+"],
            default="Unknown"
        )
    else:
        df["bf_index"] = np.nan
        df["tto"] = "Unknown"

    return df


def window_split(df: pd.DataFrame, last_days: int, baseline_days: int):
    max_date = df["game_date"].max()
    recent_start = max_date - timedelta(days=last_days)
    baseline_start = recent_start - timedelta(days=baseline_days)

    recent = df[df["game_date"] >= recent_start].copy()
    baseline = df[(df["game_date"] < recent_start) & (df["game_date"] >= baseline_start)].copy()

    return recent, baseline, recent_start.date(), baseline_start.date(), max_date.date()


def summarize(df: pd.DataFrame, min_pitches: int) -> pd.DataFrame:
    g = df.groupby(["pitcher", "player_name", "pitch_type", "pitcher_hand", "team"], dropna=False)
    out = g.agg(
        pitches=("pitch_type", "size"),
        velo=("release_speed", "mean"),
        ivb=("ivb_in", "mean"),
        hb=("hb_in", "mean"),
        arm_angle=("arm_angle_deg", "mean"),
    ).reset_index()
    return out[out["pitches"] >= min_pitches].copy()


def diff_table(recent_sum: pd.DataFrame, base_sum: pd.DataFrame) -> pd.DataFrame:
    merged = recent_sum.merge(
        base_sum,
        on=["pitcher", "player_name", "pitch_type", "pitcher_hand", "team"],
        how="inner",
        suffixes=("_recent", "_base")
    )
    merged["dVelo"] = merged["velo_recent"] - merged["velo_base"]
    merged["dIVB"] = merged["ivb_recent"] - merged["ivb_base"]
    merged["dHB"] = merged["hb_recent"] - merged["hb_base"]
    merged["dArmAngle"] = merged["arm_angle_recent"] - merged["arm_angle_base"]
    merged["pitches_recent"] = merged["pitches_recent"].astype(int)
    merged["pitches_base"] = merged["pitches_base"].astype(int)
    return merged


def shares_by(df_recent: pd.DataFrame, df_base: pd.DataFrame, col: str) -> pd.DataFrame:
    r = df_recent.groupby(col).size()
    b = df_base.groupby(col).size()
    r = (r / r.sum()).rename("recent") if r.sum() else r.rename("recent")
    b = (b / b.sum()).rename("baseline") if b.sum() else b.rename("baseline")
    out = pd.concat([r, b], axis=1).fillna(0)
    out["diff_pp"] = (out["recent"] - out["baseline"]) * 100
    return out.reset_index().sort_values("diff_pp", ascending=False)


def apply_batter_filter_for_heatmaps(d: pd.DataFrame, batter_filter: str) -> pd.DataFrame:
    if batter_filter == "All":
        return d
    if "batter_hand" not in d.columns:
        return d
    if batter_filter == "vs RHB":
        return d[d["batter_hand"] == "R"]
    return d[d["batter_hand"] == "L"]


def _smooth_2d(H: np.ndarray, passes: int = 2) -> np.ndarray:
    if H.size == 0:
        return H
    A = H.astype(float).copy()
    for _ in range(passes):
        A = (
            A
            + np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
            + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
            + np.roll(np.roll(A, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(A, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(A, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(A, -1, axis=0), -1, axis=1)
        ) / 9.0
    return A


def location_heatmap_working(df_pt: pd.DataFrame, title: str, zmax=None):
    d = df_pt.dropna(subset=["plate_x", "plate_z"]).copy()
    if d.empty:
        st.info(f"No location data for {title}")
        return None

    x_min, x_max = -2.0, 2.0
    z_min, z_max_plot = 0.5, 4.5
    d = d[(d["plate_x"].between(x_min, x_max)) & (d["plate_z"].between(z_min, z_max_plot))]
    if d.empty:
        st.info(f"No location data in plotting window for {title}")
        return None

    bins = 40
    xedges = np.linspace(x_min, x_max, bins + 1)
    yedges = np.linspace(z_min, z_max_plot, bins + 1)
    H, _, _ = np.histogram2d(d["plate_x"], d["plate_z"], bins=[xedges, yedges])

    Hs = _smooth_2d(H, passes=3)
    if Hs.sum() > 0:
        Hs = (Hs / Hs.sum()) * 100.0

    xcent = (xedges[:-1] + xedges[1:]) / 2
    ycent = (yedges[:-1] + yedges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xcent, y=ycent, z=Hs.T,
        colorscale="RdBu_r",
        zmin=0, zmax=zmax,
        hovertemplate="plate_x=%{x:.2f}<br>plate_z=%{y:.2f}<br>share=%{z:.2f}%<extra></extra>"
    ))

    fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                  line=dict(width=3, color="black"), fillcolor="rgba(0,0,0,0)")

    fig.update_layout(title=title, height=450,
                      xaxis=dict(range=[x_min, x_max], title="plate_x", zeroline=False),
                      yaxis=dict(range=[z_min, z_max_plot], title="plate_z", zeroline=False))
    st.plotly_chart(fig, use_container_width=True)
    return float(Hs.max()) if Hs.size else None


# ---------- UI ----------
st.title("Pitcher Changes (Shiny-style, Python/Streamlit)")

with st.sidebar:
    st.header("Savant Pull")
    default_start = date.today() - timedelta(days=120)
    start_dt = st.date_input("Start date", value=default_start)
    end_dt = st.date_input("End date", value=date.today())

    if start_dt > end_dt:
        st.error("Start date must be <= end date")
        st.stop()

    st.header("Windows")
    last_days = st.slider("Recent window (days)", 7, 45, 21)
    baseline_days = st.slider("Baseline window length (days)", 14, 180, 60)

    st.header("Filters")
    min_pitches = st.slider("Min pitches per pitch type (per window)", 10, 300, 50, step=10)

    st.header("Heatmap Batter Hand Only")
    batter_filter_hm = st.radio("Heatmaps vs", ["All", "vs RHB", "vs LHB"])

df_raw = fetch_statcast(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
df = prep_data(df_raw)

recent, baseline, recent_start, baseline_start, max_date = window_split(df, last_days, baseline_days)

st.caption(
    f"Comparing **Recent:** {recent_start} → {max_date}  vs  "
    f"**Baseline:** {baseline_start} → {(pd.to_datetime(recent_start) - pd.Timedelta(days=1)).date()}"
)

recent_sum = summarize(recent, min_pitches=min_pitches)
base_sum = summarize(baseline, min_pitches=min_pitches)
changes = diff_table(recent_sum, base_sum)

all_pitches = sorted([p for p in df["pitch_type"].dropna().unique().tolist()])
pitch_filter = st.selectbox("Leaderboard pitch filter", ["All"] + all_pitches, index=0)
changes_view = changes if pitch_filter == "All" else changes[changes["pitch_type"] == pitch_filter].copy()

st.subheader("Leaderboards: Biggest Changes (Recent vs Baseline)")
metric = st.selectbox("Metric", ["dVelo", "dIVB", "dHB", "dArmAngle"], index=1)
n_show = st.slider("Rows", 10, 100, 30)

show_cols = [
    "player_name", "team", "pitcher_hand", "pitch_type",
    metric,
    "velo_recent", "velo_base",
    "ivb_recent", "ivb_base",
    "hb_recent", "hb_base",
    "arm_angle_recent", "arm_angle_base",
    "pitches_recent", "pitches_base",
]

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Biggest Gains")
    st.dataframe(changes_view.sort_values(metric, ascending=False).head(n_show)[show_cols], use_container_width=True)
with c2:
    st.markdown("### Biggest Drops")
    st.dataframe(changes_view.sort_values(metric, ascending=True).head(n_show)[show_cols], use_container_width=True)

st.divider()
st.subheader("Pitcher Deep Dive")

pitchers = df[["pitcher", "player_name"]].dropna().drop_duplicates().sort_values("player_name")
pitcher_name = st.selectbox("Select pitcher", pitchers["player_name"].tolist())
pitcher_id = int(pitchers.loc[pitchers["player_name"] == pitcher_name, "pitcher"].iloc[0])

p_all = df[df["pitcher"] == pitcher_id].copy()
p_recent = recent[recent["pitcher"] == pitcher_id].copy()
p_base = baseline[baseline["pitcher"] == pitcher_id].copy()

pitch_types = sorted([p for p in p_all["pitch_type"].dropna().unique().tolist()])
pitch_type = st.selectbox("Pitch type", pitch_types)

p_recent_pt = p_recent[p_recent["pitch_type"] == pitch_type].copy()
p_base_pt = p_base[p_base["pitch_type"] == pitch_type].copy()

def mean_or_nan(s):
    return float(np.nanmean(s)) if len(s) else np.nan

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Recent pitches", f"{len(p_recent_pt):,}")
m2.metric("Base pitches", f"{len(p_base_pt):,}")
rv, bv = mean_or_nan(p_recent_pt["release_speed"]), mean_or_nan(p_base_pt["release_speed"])
m3.metric("Velo (mph)", f"{rv:.2f}", f"{(rv - bv):+.2f}")
ri, bi = mean_or_nan(p_recent_pt["ivb_in"]), mean_or_nan(p_base_pt["ivb_in"])
m4.metric("iVB-ish (in)", f"{ri:.2f}", f"{(ri - bi):+.2f}")
rh, bh = mean_or_nan(p_recent_pt["hb_in"]), mean_or_nan(p_base_pt["hb_in"])
m5.metric("HB (in)", f"{rh:.2f}", f"{(rh - bh):+.2f}")
ra, ba = mean_or_nan(p_recent_pt["arm_angle_deg"]), mean_or_nan(p_base_pt["arm_angle_deg"])
m6.metric("Arm angle (deg)", f"{ra:.2f}", f"{(ra - ba):+.2f}")
recent_usage = (p_recent_pt.shape[0] / p_recent.shape[0]) if p_recent.shape[0] else np.nan
base_usage = (p_base_pt.shape[0] / p_base.shape[0]) if p_base.shape[0] else np.nan
m7.metric("Usage share", f"{recent_usage * 100:.1f}%", f"{(recent_usage - base_usage) * 100:+.1f} pp")

st.markdown("### Usage Splits (Recent vs Baseline)")
u1, u2 = st.columns(2)
with u1:
    st.write("**By count state**")
    st.dataframe(shares_by(p_recent_pt, p_base_pt, "count_state"), use_container_width=True)
with u2:
    st.write("**By times through order (TTO)**")
    st.dataframe(shares_by(p_recent_pt, p_base_pt, "tto"), use_container_width=True)

st.divider()
st.subheader("Location Heatmaps")

p_recent_pt_hm = apply_batter_filter_for_heatmaps(p_recent_pt, batter_filter_hm)
p_base_pt_hm = apply_batter_filter_for_heatmaps(p_base_pt, batter_filter_hm)

h1, h2 = st.columns(2)
with h1:
    z1 = location_heatmap_working(p_recent_pt_hm, f"{pitcher_name} {pitch_type} — Recent ({batter_filter_hm})")
with h2:
    location_heatmap_working(p_base_pt_hm, f"{pitcher_name} {pitch_type} — Baseline ({batter_filter_hm})", zmax=z1)

st.divider()
st.subheader("Trend over time (rolling)")

roll_days = st.slider("Rolling window (days)", 3, 30, 7)
metric2 = st.selectbox("Trend metric", ["release_speed", "ivb_in", "hb_in", "arm_angle_deg"], index=1)

trend = p_all[p_all["pitch_type"] == pitch_type].dropna(subset=["game_date"]).copy()
trend = trend.sort_values("game_date")
trend["date"] = trend["game_date"].dt.date

daily = trend.groupby("date")[metric2].mean().reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily[f"{metric2}_roll"] = daily[metric2].rolling(roll_days, min_periods=max(2, roll_days // 2)).mean()

fig = px.line(daily, x="date", y=[metric2, f"{metric2}_roll"],
              title=f"{pitcher_name} {pitch_type} — {metric2} (daily + rolling)")
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)
