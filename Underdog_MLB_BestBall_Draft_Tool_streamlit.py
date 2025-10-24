# === Streamlit version of your Best Ball Draft Tool ===
# Save this as app.py in your repo to run on Streamlit

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import random

# --- App config ---
st.set_page_config(page_title="MLB Best Ball Draft Tool", layout="centered")
st.title("⚾ MLB Best Ball Draft Helper")

# --- File paths ---
AVERAGES_XLSX = "2024_2025_bestball_ud_finals_averages.xlsx"
USAGE_XLSX = "2024_2025_bestball_ud_finals_roster_constructions.xlsx"
POINTS_SEASON_XLSX = "2024_2025_season_points_by_round_pos_comparison.xlsx"
POINTS_CHAMP_XLSX = "2024_2025_champ_points_by_round_pos_comparison.xlsx"

# --- Constants ---
TOTAL_ROUNDS = 20
RANDOMNESS_EPS = 0.10
RNG_SEED = 42
WEIGHT_SHAPE_USAGE = 0.80
WEIGHT_ROUND_MATCH = 0.20
ROUND_MATCH_ALPHA = 0.8
POS_ORDER = ["P", "IF", "OF", "DH"]
DEFAULT_BOUNDS = {"P": (6, 10), "IF": (5, 9), "OF": (4, 9), "DH": (0, 2)}

# --- Functions and classes (from your code) ---
# Copy and paste your class DraftHelper and all helper functions here (unchanged)
# (They’re already great, just remove ipywidgets and display parts)

# 👇 Insert DraftHelper class and parse_*/load_* functions HERE
# (For space, I skipped them in this message but will include them in your canvas)

# --- Load data ---
avg_path = Path(AVERAGES_XLSX)
usage_path = Path(USAGE_XLSX)
total_in_file, bounds, _ = load_constraints(avg_path)
usage = parse_usage(usage_path)
round_priors_base = parse_round_priors(avg_path, total_rounds=TOTAL_ROUNDS)
pos_value_priors = parse_positional_values(
    Path(POINTS_SEASON_XLSX), Path(POINTS_CHAMP_XLSX), total_rounds=TOTAL_ROUNDS
)

round_priors = []
for i in range(TOTAL_ROUNDS):
    combined = {}
    for pos in POS_ORDER:
        base_val = round_priors_base[i].get(pos, 0)
        score_val = pos_value_priors[i].get(pos, 0)
        combined[pos] = base_val * (0.5 + 0.5 * score_val)
    total = sum(combined.values()) or 1
    for pos in combined:
        combined[pos] /= total
    round_priors.append(combined)

helper = DraftHelper(TOTAL_ROUNDS, bounds, usage, round_priors)
picks = []
draft_score = 0.0
pick_history = []

# --- UI: Draft Interaction ---
st.markdown("### 🧠 Draft Assistant")

if len(picks) < TOTAL_ROUNDS:
    counts = helper.counts_from_picks(picks)
    df = helper.next_pick_improvements(counts)
    if df.empty:
        st.warning("Draft complete or no valid picks remaining.")
    else:
        options = df["position"].tolist()
        selection = st.selectbox("Select your next pick:", options)
        if st.button("Submit Pick"):
            picks.append(selection)
            df["normalized"] = (df["pick_score"] - df["pick_score"].min()) / (
                df["pick_score"].max() - df["pick_score"].min() + 1e-9
            )
            selected_score = df.loc[df["position"] == selection, "normalized"].iloc[0]
            pts = selected_score * (100 / TOTAL_ROUNDS)
            draft_score += pts
            pick_history.append({
                "Round": len(picks),
                "Position": selection,
                "Points": round(pts, 2),
                "Total Score": round(draft_score, 2),
            })
            st.rerun()

# --- Display Results ---
if picks:
    st.subheader("📊 Draft Progress")
    st.write("**Current Roster:**", helper.counts_from_picks(picks))
    st.write("**Score:**", f"{draft_score:.1f} / 100")
    st.table(pd.DataFrame(pick_history))

# --- Reset Button ---
if st.button("Reset Draft"):
    picks.clear()
    draft_score = 0.0
    pick_history.clear()
    st.rerun()