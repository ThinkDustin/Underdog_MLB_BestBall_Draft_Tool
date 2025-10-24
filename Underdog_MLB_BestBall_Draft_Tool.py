import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === MLB Best Ball Draft Helper — PRIOR + ELITE SHAPE ALIGNMENT (single clean cell) ===
# Files (edit if needed):
AVERAGES_XLSX = "2024_2025_bestball_ud_finals_averages.xlsx"
USAGE_XLSX    = "2024_2025_bestball_ud_finals_roster_constructions.xlsx"
POINTS_SEASON_XLSX = "2024_2025_season_points_by_round_pos_comparison.xlsx"
POINTS_CHAMP_XLSX  = "2024_2025_champ_points_by_round_pos_comparison.xlsx"

# Rounds:
TOTAL_ROUNDS   = 20

# Shape prob smoothing (softens usage weights slightly):
RANDOMNESS_EPS = 0.10
RNG_SEED       = 42

# Final-construction scoring
WEIGHT_SHAPE_USAGE = 0.80
WEIGHT_ROUND_MATCH = 0.20
ROUND_MATCH_ALPHA  = 0.8

# Pick scoring weights
W_PRIOR       = 0.50
W_ELITE       = 0.40
W_DELTA_BEST  = 0.10

# ---------------------------------------------------------------------------
import pandas as pd, numpy as np
from pathlib import Path
from IPython.display import display, clear_output
import ipywidgets as widgets

POS_ORDER = ["P","IF","OF","DH"]
DEFAULT_BOUNDS = {"P": (6,10), "IF": (5,9), "OF": (4,9), "DH": (0,2)}

# ---------------- Parse optional constraints ----------------
def load_constraints(xl_path: Path):
    try:
        xls = pd.ExcelFile(xl_path)
    except Exception:
        return TOTAL_ROUNDS, DEFAULT_BOUNDS, False
    if "Constraints" not in xls.sheet_names:
        return TOTAL_ROUNDS, DEFAULT_BOUNDS, False
    df = pd.read_excel(xl_path, sheet_name="Constraints")
    cols = {str(c).strip().lower(): c for c in df.columns}
    if not {"position","min","max"}.issubset(cols.keys()):
        return TOTAL_ROUNDS, DEFAULT_BOUNDS, False
    bounds = {}
    for _, r in df.iterrows():
        pos = str(r[cols["position"]]).strip().upper()
        try:
            lo = int(r[cols["min"]]); hi = int(r[cols["max"]])
        except Exception: continue
        if pos and lo <= hi: bounds[pos] = (lo, hi)
    total_slots = TOTAL_ROUNDS
    if "total_slots" in cols:
        s = df[cols["total_slots"]].dropna()
        if not s.empty:
            try: total_slots = int(s.iloc[0])
            except: pass
    if not bounds: bounds = DEFAULT_BOUNDS
    for p in POS_ORDER: bounds.setdefault(p,(0,0))
    return total_slots, bounds, True

# ---------------- Enumerate valid final shapes ----------------
def enumerate_roster_constructions(total_slots: int, bounds: dict):
    positions = list(bounds.keys())
    rows = []
    min_sum = sum(lo for lo,_ in bounds.values())
    max_sum = sum(hi for _,hi in bounds.values())
    if not (min_sum <= total_slots <= max_sum): return []

    def backtrack(i, chosen, remaining):
        if i == len(positions):
            if remaining == 0:
                row = {p:v for p,v in zip(positions, chosen)}
                row["TOTAL"] = sum(chosen)
                row["LABEL"] = "{} ({})".format("-".join(str(row.get(p,0)) for p in POS_ORDER),
                                                "-".join(POS_ORDER))
                rows.append(row)
            return
        p = positions[i]
        lo, hi = bounds[p]
        min_rest = sum(bounds[q][0] for q in positions[i+1:])
        max_rest = sum(bounds[q][1] for q in positions[i+1:])
        for x in range(lo, hi+1):
            rem_after = remaining - x
            if rem_after < min_rest or rem_after > max_rest: continue
            backtrack(i+1, chosen+[x], rem_after)
    backtrack(0, [], total_slots)
    return rows

# ---------------- Parse usage/popularity ----------------
def parse_usage(xl_path: Path):
    df = pd.read_excel(xl_path, sheet_name=0)
    comp_col, pct_col = None, None
    for c in df.columns:
        cl = str(c).strip().lower()
        if comp_col is None and any(k in cl for k in ["comp","label","shape"]):
            comp_col = c
        if pct_col is None and any(k in cl for k in ["%","pct","avg","share","weight","usage"]):
            pct_col = c
    if comp_col is None: comp_col = df.columns[0]
    if pct_col is None: pct_col = df.columns[1]
    parts = df[comp_col].astype(str).str.extract(r"^\s*(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)\s*$").rename(columns={0:"P",1:"IF",2:"OF",3:"DH"})
    mask = parts.notna().all(axis=1)
    parts = parts[mask].astype(int)
    usage = pd.concat([parts, df.loc[mask, pct_col].rename("weight_raw")], axis=1)
    usage["weight_raw"] = pd.to_numeric(usage["weight_raw"], errors="coerce")
    usage = usage.dropna(subset=["weight_raw"])
    if usage["weight_raw"].max() > 1.5: usage["weight_raw"] /= 100.0
    return usage[["P","IF","OF","DH","weight_raw"]]

# ---------------- Parse round priors ----------------
def parse_round_priors(xl_path: Path, total_rounds: int = TOTAL_ROUNDS):
    df = pd.read_excel(xl_path, sheet_name=0)
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"p_avg","if_avg","of_avg","dh_avg"}:
            rename_map[c] = c.split("_")[0].upper(); continue
        if cl.startswith("p_"): rename_map[c] = "P"
        elif cl.startswith("if_"): rename_map[c] = "IF"
        elif cl.startswith("of_"): rename_map[c] = "OF"
        elif cl.startswith("dh_"): rename_map[c] = "DH"
    df = df.rename(columns=rename_map)
    keep = [c for c in POS_ORDER if c in df.columns]
    df = df[keep].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if df.max(numeric_only=True).max() > 1.5: df /= 100.0
    if len(df) == 1:
        priors = [df.iloc[0].to_dict()] * total_rounds
    else:
        row_sums = df.sum(axis=1).replace(0,1.0)
        df = df.div(row_sums, axis=0)
        priors = [row.to_dict() for _,row in df.iterrows()]
        if len(priors) < total_rounds:
            priors += [priors[-1]] * (total_rounds - len(priors))
        elif len(priors) > total_rounds:
            priors = priors[:total_rounds]
    return priors

# ---------------- Parse positional value curves ----------------
def parse_positional_values(season_path: Path, champ_path: Path, total_rounds: int, w_season: float = 0.7, w_champ: float = 0.3):
    def _read_points(fp):
        df = pd.read_excel(fp, sheet_name=0)
        rename_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl.startswith("p"): rename_map[c] = "P"
            elif cl.startswith("if"): rename_map[c] = "IF"
            elif cl.startswith("of"): rename_map[c] = "OF"
            elif cl.startswith("dh"): rename_map[c] = "DH"
        df = df.rename(columns=rename_map)
        keep = [c for c in POS_ORDER if c in df.columns]
        df = df[keep].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if df.max(numeric_only=True).max() > 1.5: df /= 100.0
        return df

    season_df = _read_points(season_path)
    champ_df  = _read_points(champ_path)

    # normalize each round so each round sums to 1 (✅ fixed version)
    for df in [season_df, champ_df]:
        row_sums = df.sum(axis=1).replace(0, 1).to_numpy()
        df = df.div(row_sums[:, None], axis=0)

    # weighted average
    blended = (season_df * w_season) + (champ_df * w_champ)

    # extend or trim to match total rounds
    priors = [row.to_dict() for _, row in blended.iterrows()]
    if len(priors) < total_rounds:
        priors += [priors[-1]] * (total_rounds - len(priors))
    elif len(priors) > total_rounds:
        priors = priors[:total_rounds]

    return priors



# ---------------- Helper class ----------------
class DraftHelper:
    def __init__(self, total_slots, bounds, usage_df, round_priors,
                 randomness_eps=RANDOMNESS_EPS, seed=RNG_SEED):
        self.total = int(total_slots)
        self.bounds = bounds
        self.usage = usage_df.copy()
        self.round_priors = round_priors or []
        self.rng = np.random.default_rng(seed)
        valid = pd.DataFrame(enumerate_roster_constructions(self.total, self.bounds))
        merged = valid.merge(self.usage, on=POS_ORDER, how="left").fillna(0.0)
        # --- Strengthen elite finals shape weighting ---
        tw_usage = merged["weight_raw"].sum()
        if tw_usage > 0:
            # Power transform (amplifies peaks — makes elite shapes stand out)
            probs_base = (merged["weight_raw"] / tw_usage) ** 1.6
        else:
            probs_base = np.ones(len(merged)) / len(merged)

        # Normalize back to sum=1
        probs_base /= probs_base.sum()

        # Use this as both elite-pop and final probability foundation
        elite_pop = probs_base.copy()

        # Randomness: softer blend, still respecting elite weights
        uniform = np.ones_like(probs_base) / len(probs_base)
        prob_final = (1.0 - randomness_eps * 0.5) * probs_base + (randomness_eps * 0.5) * uniform

        # Normalize again
        prob_final /= prob_final.sum()

        merged["prob_final"] = prob_final
        merged["elite_pop"]  = elite_pop

        self.valid = merged
        exp_counts = {p: 0.0 for p in POS_ORDER}
        for r in range(self.total):
            pr = self.round_priors[r] if r < len(self.round_priors) else {p:0.0 for p in POS_ORDER}
            for p in POS_ORDER: exp_counts[p] += pr.get(p,0.0)
        self.expected_counts_from_rounds = exp_counts

        # ✅ ideal elite mix
        self.target_counts = {
            p: float((self.valid[p] * self.valid["prob_final"]).sum())
            for p in POS_ORDER
        }

    def counts_from_picks(self, picks):
        c = {p:0 for p in POS_ORDER}
        for pos in picks:
            if pos in c: c[pos]+=1
        return c

    def feasible_shapes_from_state(self, counts):
        taken = sum(counts.values()); rem = self.total - taken
        mask = np.ones(len(self.valid), bool)
        for p in POS_ORDER: mask &= (self.valid[p] >= counts.get(p,0))
        fill_needed = self.valid[POS_ORDER].sum(axis=1) - taken
        mask &= (fill_needed == rem)
        feas = self.valid.loc[mask].copy()
        if not feas.empty:
            feas["prob_final_norm"] = feas["prob_final"] / feas["prob_final"].sum()
        return feas

    def add_construction_scores(self, df_shapes):
        out = df_shapes.copy()
        wr = out["weight_raw"].fillna(0.0) if "weight_raw" in out.columns else pd.Series(0, index=out.index)
        if wr.max() > wr.min():
            score_usage = (wr - wr.min()) / (wr.max() - wr.min() + 1e-12)
        else:
            score_usage = pd.Series(np.zeros(len(out)), index=out.index)
        tgt = self.expected_counts_from_rounds
        l1 = (out["P"].sub(tgt.get("P",0)).abs()
              + out["IF"].sub(tgt.get("IF",0)).abs()
              + out["OF"].sub(tgt.get("OF",0)).abs()
              + out["DH"].sub(tgt.get("DH",0)).abs())
        score_round = np.exp(-ROUND_MATCH_ALPHA * l1)
        out["score_usage"] = score_usage
        out["score_round"] = score_round
        out["score_combo"] = WEIGHT_SHAPE_USAGE*score_usage + WEIGHT_ROUND_MATCH*score_round
        return out

    def state_scores(self, counts):
        feas = self.feasible_shapes_from_state(counts)
        if feas.empty or feas["prob_final"].sum() < 1e-6:
            return pd.DataFrame({
                "position": ["⚠️ None"],
                "pick_score": [0.0]
            })
        best = feas["prob_final"].max()
        exp = (feas["prob_final"]*feas["elite_pop"]).sum()
        return best, exp, feas

    def _round_prior_val(self, round_idx0, pos):
        if 0 <= round_idx0 < len(self.round_priors):
            return self.round_priors[round_idx0].get(pos,0.0)
        return 0.0

    def next_pick_improvements(self, counts):
        """
        Data-driven with soft draft-capital awareness.
        Prioritizes early hitter balance but protects elite finals shapes.
        """
        best_now, exp_now, _ = self.state_scores(counts)
        taken = sum(counts.values())
        if taken >= self.total:
            return pd.DataFrame()

        rows = []

        # round "phase" from 0 (early) to 1 (late)
        phase = taken / self.total

        # balance reference
        cur_ratio = {p: counts[p] / max(1, taken) for p in POS_ORDER}
        tgt_ratio = {p: self.target_counts[p] / self.total for p in POS_ORDER}

        for p in POS_ORDER:
            cur = counts.get(p, 0)
            lo, hi = self.bounds[p]
            if cur >= hi:
                continue

            nxt = counts.copy()
            nxt[p] = cur + 1
            feas_next = self.feasible_shapes_from_state(nxt)
            if feas_next.empty or feas_next["prob_final"].sum() < 1e-6:
                continue

            best_next, exp_next, _ = self.state_scores(nxt)
            improvement = exp_next - exp_now
            prior_val = self._round_prior_val(taken, p)

            # early = hitters slightly boosted; pitchers slightly delayed
            cap_factor = 1.0
            if p in ["IF", "OF"]:
                cap_factor += 0.4 * (1 - phase)   # +40% early
            elif p == "P":
                cap_factor -= 0.25 * (1 - phase)  # -25% early
            elif p == "DH":
                cap_factor -= 0.4 * (1 - phase)   # DH minimal early
            cap_factor = max(0.5, cap_factor)

            # positional balance adjustment
            ratio_gap = max(0, tgt_ratio[p] - cur_ratio[p])
            bal_factor = 1.0 + 0.5 * ratio_gap

            # combine all data-driven signals
            raw_score = (
                0.55 * improvement +
                0.30 * prior_val +
                0.15 * ratio_gap
            ) * cap_factor * bal_factor

            # 🧠 stronger taper for pitchers once you have 7+
            if p == "P":
                p_count = counts.get("P", 0)
                # Gradually reduce pitcher attractiveness after 7
                if p_count >= 7:
                    # linear dropoff: 7 → 0.6x, 8 → 0.45x, 9+ → 0.3x
                    fade = max(0.3, 1.2 - 0.15 * p_count)
                    raw_score *= fade

            # 🧩 balance factor between IF and OF to avoid overstacking one
            if p in ["IF", "OF"]:
                # how far each is from its ideal ratio
                if_gap = tgt_ratio["IF"] - cur_ratio["IF"]
                of_gap = tgt_ratio["OF"] - cur_ratio["OF"]
                # if we're overweight IF, give OF a small lift
                if if_gap < of_gap:
                    if p == "OF":
                        bal_factor *= 1.15
                    elif p == "IF":
                        bal_factor *= 0.9



            rows.append({"position": p, "pick_score": max(raw_score, 1e-6)})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["pick_score"] /= df["pick_score"].max()
        df["pick_score"] *= 5.0
        df["pick_score"] = df["pick_score"].clip(lower=0.25)

        df.attrs["best_now"] = best_now
        df.attrs["exp_now"] = exp_now
        return df.sort_values("pick_score", ascending=False).reset_index(drop=True)



# ---------------- Instantiate ----------------
avg_path = Path(AVERAGES_XLSX)
usage_path = Path(USAGE_XLSX)
total_in_file, bounds, _ = load_constraints(avg_path)
usage = parse_usage(usage_path)
# Combine base priors (draft tendency) with positional value curves (scoring)
round_priors_base = parse_round_priors(avg_path, total_rounds=TOTAL_ROUNDS)
pos_value_priors  = parse_positional_values(
    Path(POINTS_SEASON_XLSX),
    Path(POINTS_CHAMP_XLSX),
    total_rounds=TOTAL_ROUNDS,
    w_season=0.7,
    w_champ=0.3
)

# Merge both: combine usage frequency (tendencies) and positional scoring (output)
round_priors = []
for i in range(TOTAL_ROUNDS):
    combined = {}
    for pos in POS_ORDER:
        base_val = round_priors_base[i].get(pos, 0)
        score_val = pos_value_priors[i].get(pos, 0)
        # simple product to weigh both: draft tendency × positional value
        combined[pos] = base_val * (0.5 + 0.5 * score_val)
    # normalize
    total = sum(combined.values()) or 1
    for pos in combined: combined[pos] /= total
    round_priors.append(combined)

helper = DraftHelper(TOTAL_ROUNDS, bounds, usage, round_priors)

import random

def load_draft_quotes(filepath="draft_quotes.txt"):
    """
    Reads themed or funny draft quotes grouped by category in [brackets].
    Returns a dict like {'elite': [...], 'strong': [...], ...}
    """
    quotes = {"elite": [], "strong": [], "decent": [], "poor": []}
    current = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("[elite]"):
                    current = "elite"
                elif line.lower().startswith("[strong]"):
                    current = "strong"
                elif line.lower().startswith("[decent]"):
                    current = "decent"
                elif line.lower().startswith("[poor]"):
                    current = "poor"
                elif current:
                    quotes[current].append(line)
    except FileNotFoundError:
        # Fallback if file not found
        quotes["elite"] = ["Elite: You nailed it!"]
        quotes["strong"] = ["Strong: Well done."]
        quotes["decent"] = ["Decent: Room for improvement."]
        quotes["poor"] = ["Poor: Maybe next season."]
    return quotes

draft_quotes = load_draft_quotes("draft_quotes.txt")

# === CLEAN SINGLE-CELL INTERACTIVE DRAFT UI ===
picks = []

def interactive_draft_ui(helper):
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    out = widgets.Output(
        layout={
            "border": "2px solid #ddd",
            "padding": "10px 20px",
            "background": "white",
            "max_width": "720px",
            "margin": "0 auto",  # center it
            "box_shadow": "0 0 8px rgba(0,0,0,0.1)",
        }
    )


    dd = widgets.Dropdown(description="Pick:")
    btn_submit = widgets.Button(description="Submit", button_style="success")
    btn_undo = widgets.Button(description="Undo", button_style="warning")
    btn_reset = widgets.Button(description="Reset", button_style="danger")

    draft_score = {"total": 0.0}
    pick_history = []  # 🆕 Track round-by-round progress

    def refresh():
        with out:
            clear_output(wait=True)
            counts = helper.counts_from_picks(picks)
            round_num = len(picks) + 1
            best_now, exp_now, feas = helper.state_scores(counts)
            df = helper.next_pick_improvements(counts)

            # === END OF DRAFT ===
            if df.empty or "position" not in df.columns or "pick_score" not in df.columns:
                final_score = draft_score["total"]

                # --- Determine Tier and Quote ---
                tier, title, summary = "", "", ""
                if final_score >= 90:
                    tier, title, summary = "elite", "🏆 Elite Draft!", "You nailed the construction."
                elif final_score >= 75:
                    tier, title, summary = "strong", "💪 Strong Draft!", "Well-balanced decisions overall."
                elif final_score >= 60:
                    tier, title, summary = "decent", "🤔 Decent Draft", "A few suboptimal choices, but competitive."
                else:
                    tier, title, summary = "poor", "🚨 Poor Draft", "Your structure diverged from elite builds."

                quote_list = draft_quotes.get(tier, [])
                fun_line = random.choice(quote_list) if quote_list else ""

                # 🏆 --- Feedback banner (TOP) ---
                feedback = f"""
                <div style='text-align:center;margin-bottom:1em;'>
                    <div style='font-size:1.8em;font-weight:bold;color:#2a7;'>{title}</div>
                    <div style='font-size:1em;color:#555;margin-top:0.8em;'><i>{fun_line}</i></div>
                </div>
                """


                # --- Compact summary table (left-aligned, narrower) ---
                history_table = """
                <b>Draft Summary:</b><br>
                <table style='width:100%;border-collapse:collapse;font-size:0.9em;text-align:left;'>
                <tr>
                    <th style='width:15%;border-bottom:1px solid #ccc;'>Round</th>
                    <th style='width:25%;border-bottom:1px solid #ccc;'>Position</th>
                    <th style='width:25%;border-bottom:1px solid #ccc;'>Points</th>
                    <th style='width:35%;border-bottom:1px solid #ccc;'>Total</th>
                </tr>
                """
                for row in pick_history:
                    history_table += (
                        f"<tr>"
                        f"<td style='padding:2px 4px;border-bottom:1px solid #eee;'>{row['Round']}</td>"
                        f"<td style='padding:2px 4px;border-bottom:1px solid #eee;'>{row['Position']}</td>"
                        f"<td style='padding:2px 4px;border-bottom:1px solid #eee;'>{row['Points']}</td>"
                        f"<td style='padding:2px 4px;border-bottom:1px solid #eee;'>{row['Total Score']}</td>"
                        f"</tr>"
                    )
                history_table += "</table><br>"

                # --- 🧩 Final display layout ---
                display(widgets.HTML(f"""
                {feedback}
                <div style='font-size:1.1em;'>
                <b>Final Roster:</b> {counts}<br>
                <b>Total Rounds:</b> {helper.total}<br>
                <b>Final Score:</b> {final_score:.1f} / 100<br><br>
                {history_table}
                </div>
                """))
                return


            # === NORMAL ROUND DISPLAY ===
            top = df.iloc[0]
            top_score = top["pick_score"]

            # Normalize for scoring
            df["normalized"] = (df["pick_score"] - df["pick_score"].min()) / (df["pick_score"].max() - df["pick_score"].min() + 1e-9)
            points_per_round = 100 / helper.total
            top_points = df["normalized"].iloc[0] * points_per_round

            header = f"""
            <div style='font-size:1.1em;'>
              <b>Round:</b> {round_num}/{helper.total} &nbsp;
              <b>Roster:</b> {counts}<br>
              <b>Current Score:</b> {draft_score["total"]:.1f} / 100<br>
              💡 <b style='color:#2a7;'>Recommended Pick:</b> {top['position']} (+{top_points:.2f} points)
            </div><hr>
            """

            # 🆕 Display full draft progress table (compact layout)
            progress_html = ""
            if pick_history:
                progress_html += """
                <hr><b>Draft Progress:</b><br>
                <table style='width:100%;border-collapse:collapse;font-size:0.9em;text-align:left;'>
                <tr>
                    <th style='width:15%;border-bottom:1px solid #ccc;'>Round</th>
                    <th style='width:25%;border-bottom:1px solid #ccc;'>Position</th>
                    <th style='width:25%;border-bottom:1px solid #ccc;'>Points</th>
                    <th style='width:35%;border-bottom:1px solid #ccc;'>Total</th>
                </tr>
                """
                for row in pick_history:
                    progress_html += (
                        f"<tr>"
                        f"<td style='padding:2px 4px;border-bottom:1px solid #eee;'>{row['Round']}</td>"
                        f"<td style='padding:2px 4px;border-bottom:1px solid #eee;'>{row['Position']}</td>"
                        f"<td style='padding:2px 4px;border-bottom:1px solid #eee;'>{row['Points']}</td>"
                        f"<td style='padding:2px 4px;border-bottom:1px solid #eee;'>{row['Total Score']}</td>"
                        f"</tr>"
                    )
                progress_html += "</table><hr>"

            # Build other options
            lines = []
            for _, row in df.iterrows():
                if row["position"] == top["position"]:
                    continue
                pts = row["normalized"] * (100 / helper.total)
                if pts >= 3.5:
                    color = "#2a7"
                elif pts <= 1.5:
                    color = "#b22"
                else:
                    color = "#e67e22"
                lines.append(f"<span style='color:{color};'>• {row['position']} (+{pts:.2f} pts)</span>")

            display(widgets.HTML(
                header + 
                "<b>Other Options:</b><br>" + "<br>".join(lines) +
                progress_html  # 👈 now progress shows below the other options
            ))


            dd.options = df["position"].tolist()
            dd.value = dd.options[0]

    def on_submit(_):
        pos = dd.value
        ok, msg = True, ""
        c = helper.counts_from_picks(picks)
        lo, hi = helper.bounds[pos]
        if c[pos] >= hi:
            ok, msg = False, f"'{pos}' exceeds max ({hi})"
        if ok:
            counts_next = c.copy()
            counts_next[pos] += 1
            if helper.feasible_shapes_from_state(counts_next).empty:
                ok, msg = False, f"'{pos}' leads to no valid finals."

        if ok:
            picks.append(pos)
            df = helper.next_pick_improvements(c)
            if not df.empty and "pick_score" in df.columns:
                df["normalized"] = (df["pick_score"] - df["pick_score"].min()) / (df["pick_score"].max() - df["pick_score"].min() + 1e-9)
                selected_score = df.loc[df["position"] == pos, "normalized"].iloc[0]
                pts = selected_score * (100 / helper.total)
                draft_score["total"] += pts
                pick_history.append({
                    "Round": len(picks),
                    "Position": pos,
                    "Points": round(pts, 2),
                    "Total Score": round(draft_score["total"], 2)
                })
        else:
            with out:
                clear_output(wait=True)
                display(widgets.HTML(f"<p style='color:red;'>❌ {msg}</p>"))

        refresh()

    def on_undo(_):
        if picks:
            picks.pop()
            draft_score["total"] = 0
            pick_history.clear()
            temp_picks = picks.copy()
            picks.clear()
            for p in temp_picks:
                c = helper.counts_from_picks(picks)
                df = helper.next_pick_improvements(c)
                if not df.empty and "pick_score" in df.columns:
                    df["normalized"] = (df["pick_score"] - df["pick_score"].min()) / (df["pick_score"].max() - df["pick_score"].min() + 1e-9)
                    sc = df.loc[df["position"] == p, "normalized"].iloc[0]
                    pts = sc * (100 / helper.total)
                    draft_score["total"] += pts
                    pick_history.append({
                        "Round": len(picks) + 1,
                        "Position": p,
                        "Points": round(pts, 2),
                        "Total Score": round(draft_score["total"], 2)
                    })
                picks.append(p)
        refresh()

    def on_reset(_):
        picks.clear()
        draft_score["total"] = 0
        pick_history.clear()
        refresh()

    btn_submit.on_click(on_submit)
    btn_undo.on_click(on_undo)
    btn_reset.on_click(on_reset)

    # --- Style the dropdown label (“Pick →”, bigger font + taller box) ---
    dd.description = "Pick →"
    dd.style = {
        "description_width": "70px",   # space for label
        "font_weight": "700",          # bold label
    }
    dd.layout = widgets.Layout(
        width="180px",                 # dropdown width
        height="36px",                 # align with buttons
        margin="0 8px 0 0",            # right spacing
        align_self="center",
        font_size="42px"               # increase label font size
    )

    # --- Manually bump label font size via HTML/CSS injection ---
    from IPython.display import HTML
    display(HTML("""
    <style>
    .widget-label { font-size: 16px !important; font-weight: 600 !important; }
    </style>
    """))


    # --- Match all buttons to same height ---
    btn_submit.layout = widgets.Layout(height="36px", width="100px")
    btn_undo.layout   = widgets.Layout(height="36px", width="100px")
    btn_reset.layout  = widgets.Layout(height="36px", width="100px")

    # --- Create aligned top toolbar ---
    toolbar = widgets.HBox(
        [dd, btn_submit, btn_undo, btn_reset],
        layout=widgets.Layout(
            width="100%",
            justify_content="flex-start",
            align_items="center",
            gap="10px",
            padding="4px 0"
        )
    )

        # --- Unified consistent width for toolbar + output ---
    main_width = 500  # ✅ adjust this number to your perfect fit (try 660–700 range)

    # --- Style dropdown ---
    dd.description = "Pick →"
    dd.style = {
        "description_width": "70px",
        "font_weight": "600",
    }
    dd.layout = widgets.Layout(
        width="180px",
        height="36px",
        margin="0 8px 0 0",
        align_self="center",
        font_size="16px"
    )

    # --- Buttons (equal widths + same height) ---
    btn_submit.layout = widgets.Layout(height="36px", width="100px")
    btn_undo.layout   = widgets.Layout(height="36px", width="100px")
    btn_reset.layout  = widgets.Layout(height="36px", width="100px")

    # --- Toolbar (fixed pixel width, centered) ---
    toolbar = widgets.HBox(
        [dd, btn_submit, btn_undo, btn_reset],
        layout=widgets.Layout(
            width=f"{main_width}px",            # ✅ fixed exact width
            justify_content="flex-start",
            align_items="center",
            gap="10px",
            padding="4px 0",
            margin="0 auto"
        )
    )

    # --- Output panel (exact same width as toolbar) ---
    out.layout = widgets.Layout(
        border="2px solid #ddd",
        padding="10px 18px",                   # ✅ smaller padding = no extra width bleed
        background_color="white",
        width=f"{main_width}px",               # ✅ identical to toolbar width
        box_shadow="0 0 8px rgba(0,0,0,0.1)",
        margin="0 auto",
        align_self="center"
    )

    # --- Wrapper ---
    inner_box = widgets.VBox(
        [toolbar, out],
        layout=widgets.Layout(
            width=f"{main_width}px",
            align_items="center",
            margin="0 auto"
        )
    )

    main_container = widgets.VBox(
        [inner_box],
        layout=widgets.Layout(
            width="100%",
            align_items="center",
            margin="0 auto",
            background_color="white"
        )
    )

    display(main_container)
    refresh()

interactive_draft_ui(helper)