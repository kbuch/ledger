# File: pages/monte_carlo.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from mcs import (
    simulate_normal,
    simulate_block,
    simulate_garch,
    simulate_fhs_fast,
    compute_rolling,
    percentile_strs,
    calibrate_weight_combined,
)
import os, io

st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")
st.title("🔢 Monte Carlo Simulator")

# ──────────────────────────────────────────────────────────────────────
# 📁 Data Manager: shared upload + save/load/delete
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

if "trade_bytes" not in st.session_state:
    st.session_state.trade_bytes = None
    st.session_state.trade_name  = None

st.sidebar.header("📁 Data Manager")
file_uploader = st.sidebar.file_uploader(
    "Upload trade-log CSV", type="csv", key="trade_uploader"
)
if file_uploader is not None:
    st.session_state.trade_bytes = file_uploader.getvalue()
    st.session_state.trade_name  = file_uploader.name

if st.session_state.trade_name:
    st.sidebar.markdown(f"**Uploaded:** {st.session_state.trade_name}")

save_name_mc = st.sidebar.text_input(
    "Save as (no .csv)", key="save_name_mc"
)
if st.sidebar.button("Save uploaded log", key="save_button_mc"):
    if not st.session_state.trade_bytes:
        st.sidebar.warning("No file to save.")
    elif not save_name_mc.strip():
        st.sidebar.warning("Please enter a name.")
    else:
        path = os.path.join(DATA_DIR, f"{save_name_mc}.csv")
        with open(path, "wb") as f:
            f.write(st.session_state.trade_bytes)
        st.sidebar.success(f"Saved as {save_name_mc}.csv")

st.sidebar.subheader("Saved Logs")
saved_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
selected_saved = st.sidebar.selectbox(
    "Load saved log", [""] + saved_files, key="selected_saved_file_mc"
)
if selected_saved and st.sidebar.button("Delete selected log", key="delete_button_mc"):
    os.remove(os.path.join(DATA_DIR, selected_saved))
    st.sidebar.success(f"Deleted {selected_saved}")
    st.experimental_rerun()

if selected_saved:
    uploaded = os.path.join(DATA_DIR, selected_saved)
elif st.session_state.trade_bytes:
    uploaded = io.BytesIO(st.session_state.trade_bytes)
else:
    uploaded = None

if not uploaded:
    st.info("Please upload a CSV with your P/L data (e.g. columns 'Date Opened' and 'P/L').")
    st.stop()
# ──────────────────────────────────────────────────────────────────────

# Column names
date_col = st.text_input(
    "Date column name", value="Date Opened", key="mc_date_col"
)
pl_col   = st.text_input(
    "P/L column name",   value="P/L",         key="mc_pl_col"
)

# Load and validate
df = pd.read_csv(uploaded)
if date_col not in df or pl_col not in df:
    st.error(f"CSV must contain '{date_col}' and '{pl_col}'.")
    st.stop()

# Aggregate daily P/L
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df['Date'] = df[date_col].dt.date
daily = df.groupby('Date')[pl_col].sum()

# Fill missing calendar days
days = pd.date_range(daily.index.min(), daily.index.max(), freq='D').date
daily = daily.reindex(days, fill_value=0)
daily_returns = daily.values
N = len(daily_returns)
st.markdown(f"**Loaded {N:,} days of P/L data**")

# 2) UI controls (all keys set for persistence)
col1, col2, col3 = st.columns(3)
with col1:
    method = st.selectbox(
        "Simulation method",
        ['normal','block','garch','fhs','rolling'],
        help="normal, block, garch, fhs (with calibration), rolling-window",
        key="mc_method",
    )
    trials = st.number_input(
        "Number of trials", value=100000, min_value=1, step=1000,
        key="mc_trials",
    )
    interval_opts = ['30d','60d','90d','180d','1y','all']
    intervals = st.multiselect(
        "Interval(s)", interval_opts, default=['30d'], key="mc_intervals"
    )

with col2:
    seed = st.number_input(
        "Random seed (0 = none)", value=0, min_value=0,
        key="mc_seed",
    )
    if seed > 0:
        np.random.seed(int(seed))

    if method == 'fhs':
        tail_low  = st.number_input(
            "Tail weight (worst 10%)", value=1.0, min_value=1.0, step=0.1,
            key="mc_tail_low",
        )
        tail_high = st.number_input(
            "Tail weight (best 10%)",  value=1.0, min_value=1.0, step=0.1,
            key="mc_tail_high",
        )
        calibrate_only    = st.checkbox("Calibrate only",    key="mc_calibrate_only")
        calibrate_and_run = st.checkbox("Calibrate & run",   key="mc_calibrate_and_run")

with col3:
    run = st.button("▶️ Run", key="mc_run")

# 3) Run or re-display simulations
if run or "mc_prev_results" in st.session_state:
    # If new run, compute & stash results
    if run:
        results = {}
        for ref in intervals:
            # resolve draws for this interval
            if ref == 'all':
                draws = N
            elif ref == '1y':
                draws = min(365, N)
            else:
                draws = int(ref[:-1])

            # possibly pre-fit GARCH
            if method == 'fhs':
                am = arch_model(
                    daily_returns, mean='Constant', vol='Garch', p=1, q=1,
                    dist='normal', rescale=False
                )
                res = am.fit(disp='off')
                p = res.params
                mu         = p.get('mu', p.get('Const', 0.0))
                omega      = p['omega']
                alpha      = p['alpha[1]']
                beta       = p['beta[1]']
                uncond_var = omega / (1 - alpha - beta)
                std_resid  = res.std_resid[np.isfinite(res.std_resid)]

            # run simulation
            if method == 'normal':
                cum, dd = simulate_normal(daily_returns, trials, draws)
            elif method == 'block':
                cum, dd = simulate_block(daily_returns, trials, draws)
            elif method == 'garch':
                cum, dd = simulate_garch(daily_returns, trials, draws)
            elif method == 'rolling':
                cum, dd = compute_rolling(daily_returns, draws)
            else:  # fhs
                # calibration step
                if calibrate_only or calibrate_and_run:
                    best_w, (p1, dd99) = calibrate_weight_combined(
                        mu, omega, alpha, beta,
                        uncond_var, std_resid,
                        daily_returns, draws,
                        trials
                    )
                    st.write(
                        f"Calibrated tail_weight_low = {best_w:.3f} — "
                        f"1% P/L = {p1:.2f}, 99% DD = {dd99:.2f}"
                    )
                if calibrate_only and not calibrate_and_run:
                    continue
                use_low  = best_w if calibrate_and_run else tail_low
                use_high = 1.0   if calibrate_and_run else tail_high
                cum, dd = simulate_fhs_fast(
                    mu, omega, alpha, beta,
                    uncond_var, std_resid,
                    trials, draws,
                    tail_weight_low=use_low,
                    tail_weight_high=use_high,
                )

            # percentile table
            pct_cum = percentile_strs(cum)
            pct_dd  = percentile_strs(dd)
            df_pct = pd.DataFrame({
                "Percentile": list(pct_cum.keys()),
                "P/L":         list(pct_cum.values()),
                "Drawdown":    [pct_dd[k] for k in pct_cum.keys()],
            })

            results[ref] = {
                "cum": cum.tolist(),
                "dd": dd.tolist(),
                "df_pct": df_pct,
            }

        # save for later re-display
        st.session_state.mc_prev_results = {
            "settings": {
                "method": method,
                "trials": trials,
                "intervals": intervals,
                "seed": seed,
                "tail_low":  locals().get("tail_low", None),
                "tail_high": locals().get("tail_high", None),
                "calibrate_only":    locals().get("calibrate_only", None),
                "calibrate_and_run": locals().get("calibrate_and_run", None),
            },
            "results": results,
        }

    # now display (new or previous) results:
    prev = st.session_state.mc_prev_results
    for ref, data in prev["results"].items():
        cum = np.array(data["cum"])
        dd  = np.array(data["dd"])
        df_pct = data["df_pct"]

        # compute p_loss
        p_loss = np.mean(cum < 0) * 100
        # recalculate how many days we drew for this interval
        if ref == 'all':
            draws_i = N
        elif ref == '1y':
            draws_i = min(365, N)
        else:
            draws_i = int(ref[:-1])

        st.markdown(f"---\n## Interval: {ref}  ({draws_i} days)\n")
        st.write(f"**Probability of losing period:** {p_loss:.2f}%")

        st.subheader("P&L & Drawdown Percentiles")
        st.table(df_pct)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Equity Curve Distribution")
            plt.figure(figsize=(6,3))
            plt.hist(cum, bins=50, edgecolor='k')
            plt.xlabel("Total P/L")
            plt.ylabel("Frequency")
            st.pyplot(plt.gcf())
        with c2:
            st.subheader("Max Drawdown Distribution")
            plt.figure(figsize=(6,3))
            plt.hist(dd, bins=50, edgecolor='k')
            plt.xlabel("Max Drawdown")
            plt.ylabel("Frequency")
            st.pyplot(plt.gcf())

        res = pd.DataFrame({"cum_pnl": cum, "max_drawdown": dd})
        csv = res.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"💾 Download results ({ref})",
            data=csv,
            file_name=f"mc_results_{ref}.csv"
        )

else:
    st.info("Adjust parameters above and click ▶️ Run")
