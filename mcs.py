#!/usr/bin/env python3
import argparse
import sys
import psutil
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arch import arch_model


def percentile_strs(values, levels=[1, 5, 10, 50, 90, 95, 99]):
    pct = np.percentile(values, levels)
    return {f"{lvl}%": pct[i] for i, lvl in enumerate(levels)}


def compute_drawdown(series):
    cum = np.cumsum(series)
    peaks = np.maximum.accumulate(cum)
    drawdowns = peaks - cum
    return drawdowns.max()


def get_optimal_block_size(draws, dtype_size=4, mem_fraction=0.5, max_block=100_000):
    """
    Determine how many trials we can simulate at once without exceeding
    a fraction of available RAM.

    draws:        number of timesteps per trial
    dtype_size:   bytes per sample (float32=4, float64=8)
    mem_fraction: fraction of total free RAM to consume at most
    max_block:    absolute upper cap to avoid pathological values
    """
    mem = psutil.virtual_memory()
    avail = mem.available * mem_fraction
    # each trial block requires draws * dtype_size bytes
    bs = int(avail // (draws * dtype_size))
    return max(1, min(bs, max_block))


def simulate_normal(daily_returns, trials, draws):
    """
    Monte Carlo by sampling returns with replacement in adaptive blocks.
    Uses float32 and only holds (block_size × draws) in memory at once.
    """
    daily = daily_returns.astype(np.float32)
    cum_pnl = np.empty(trials, dtype=np.float32)
    max_dd  = np.empty(trials, dtype=np.float32)
    block_size = get_optimal_block_size(draws)
    for start in range(0, trials, block_size):
        end = min(start + block_size, trials)
        size = end - start
        block = np.random.choice(daily, size=(size, draws), replace=True)
        cum_pnl[start:end] = block.sum(axis=1, dtype=np.float32)
        for i in range(size):
            path = block[i]
            cum = np.cumsum(path, dtype=np.float32)
            peaks = np.maximum.accumulate(cum)
            max_dd[start + i] = np.max(peaks - cum)
        del block; gc.collect()
    return cum_pnl, max_dd


def simulate_block(daily_returns, trials, draws):
    N = len(daily_returns)
    max_start = N - draws
    cum_pnl = np.empty(trials)
    max_dd  = np.empty(trials)
    for i in range(trials):
        start = np.random.randint(0, max_start + 1)
        window = daily_returns[start:start+draws]
        cum = np.cumsum(window)
        cum_pnl[i] = cum[-1]
        peaks = np.maximum.accumulate(cum)
        max_dd[i]  = np.max(peaks - cum)
    return cum_pnl, max_dd


def simulate_garch(daily_returns, trials, draws):
    """
    GARCH(1,1) with normal innovations.
    """
    am = arch_model(daily_returns, mean='Constant', vol='Garch', p=1, q=1,
                    dist='normal', rescale=False)
    res = am.fit(disp='off')
    p = res.params
    mu, omega = p.get('mu', p.get('Const',0.0)), p['omega']
    alpha, beta = p['alpha[1]'], p['beta[1]']
    uncond_var = omega/(1-alpha-beta)
    eps = np.random.normal(size=(trials, draws))
    sims = np.zeros((trials, draws))
    h = np.full(trials, uncond_var)
    for t in range(draws):
        sigma = np.sqrt(h)
        r = mu + sigma * eps[:,t]
        sims[:,t] = r
        h = omega + alpha*(r-mu)**2 + beta*h
    cum_pnl = sims.sum(axis=1)
    max_dd  = np.array([compute_drawdown(s) for s in sims])
    return cum_pnl, max_dd


def simulate_fhs(daily_returns, trials, draws,
                 tail_weight_low=1.0, tail_weight_high=1.0):
    """
    Filtered Historical Simulation with optional tail-weighting.
    - tail_weight_low: factor to oversample the worst 10%% residuals.
    - tail_weight_high: factor to oversample the best 10%% residuals.
    """
    am = arch_model(daily_returns, mean='Constant', vol='Garch', p=1, q=1,
                    dist='normal', rescale=False)
    res = am.fit(disp='off')
    p = res.params
    mu, omega = p.get('mu', p.get('Const',0.0)), p['omega']
    alpha, beta = p['alpha[1]'], p['beta[1]']
    uncond_var = omega/(1-alpha-beta)
    std_resid = res.std_resid[np.isfinite(res.std_resid)]
    low_thr = np.percentile(std_resid, 10)
    high_thr = np.percentile(std_resid, 90)
    if tail_weight_low>1 or tail_weight_high>1:
        w = np.ones_like(std_resid)
        w[std_resid<low_thr]  *= tail_weight_low
        w[std_resid>high_thr] *= tail_weight_high
        w /= w.sum()
    else:
        w = None
    sims = np.zeros((trials, draws))
    h = np.full(trials, uncond_var)
    for t in range(draws):
        if w is not None:
            eps = np.random.choice(std_resid, size=trials, replace=True, p=w)
        else:
            eps = np.random.choice(std_resid, size=trials, replace=True)
        sigma = np.sqrt(h)
        r = mu + sigma * eps
        sims[:,t] = r
        h = omega + alpha*(r-mu)**2 + beta*h
    cum_pnl = sims.sum(axis=1)
    max_dd  = np.array([compute_drawdown(s) for s in sims])
    return cum_pnl, max_dd


def simulate_fhs_fast(mu, omega, alpha, beta, uncond_var,
                      std_resid, trials, draws,
                      tail_weight_low=1.0, tail_weight_high=1.0):
    """
    Fast FHS using pre-fitted GARCH params + standardized residuals.
    """
    # build mixture weights once
    low_thr  = np.percentile(std_resid, 10)
    high_thr = np.percentile(std_resid, 90)
    if tail_weight_low > 1 or tail_weight_high > 1:
        w = np.ones_like(std_resid)
        w[std_resid <  low_thr]  *= tail_weight_low
        w[std_resid >  high_thr] *= tail_weight_high
        w /= w.sum()
    else:
        w = None

    sims = np.zeros((trials, draws))
    h    = np.full(trials, uncond_var)

    for t in range(draws):
        if w is not None:
            eps = np.random.choice(std_resid, size=trials, replace=True, p=w)
        else:
            eps = np.random.choice(std_resid, size=trials, replace=True)
        sigma = np.sqrt(h)
        r     = mu + sigma * eps
        sims[:, t] = r
        h = omega + alpha * (r - mu)**2 + beta * h

    cum_pnl = sims.sum(axis=1)
    max_dd  = np.array([compute_drawdown(path) for path in sims])
    return cum_pnl, max_dd


def compute_rolling(daily_returns, draws):
    """
    Streaming rolling‐window P/L + max‐drawdown.
    Uses float32 and only a single window in memory at once.
    """
    daily = daily_returns.astype(np.float32)
    N = len(daily)
    M = N - draws + 1
    cum_pnl = np.empty(M, dtype=np.float32)
    max_dd  = np.empty(M, dtype=np.float32)
    for i in range(M):
        window = daily[i:i+draws]
        cum_pnl[i] = window.sum(dtype=np.float32)
        max_dd[i]  = compute_drawdown(window)
    del daily; gc.collect()
    return cum_pnl, max_dd


def print_summary(label, values, dds, trials=None):
    pnl_pct = percentile_strs(values)
    dd_pct = percentile_strs(dds)
    prob_neg = np.mean(values < 0)
    print(f"--- {label} ({len(values)} samples) ---")
    print("Cumulative P/L percentiles:")
    for k, v in pnl_pct.items(): print(f" {k}: {v:,.2f}")
    print(f"Probability of losing period: {prob_neg:.2%}")
    print("Max drawdown percentiles:")
    for k, v in dd_pct.items(): print(f" {k}: {v:,.2f}")
    print()


def plot_and_save(arr, title, xlabel, fname):
    plt.hist(arr, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def calibrate_weight_combined(mu, omega, alpha, beta, uncond_var, std_resid,
                              daily_returns, draws,
                              full_trials, calib_trials=5000,
                              low=1.0, high=5.0, steps=41):
    """
    Find a single tail_weight_low ∈ [low, high] that best balances
    1% cumulative P/L and 99% drawdown via a combined relative‐error score.
    Uses simulate_fhs_fast with calib_trials for speed.
    Returns (best_weight, (sim_pnl1, sim_dd99)).
    """
    # 1) compute historical targets
    hist_pnl, hist_dd = compute_rolling(daily_returns, draws)
    target_p1   = np.percentile(hist_pnl, 1)
    target_dd99 = np.percentile(hist_dd, 99)

    # 2) grid search
    ws        = np.linspace(low, high, steps)
    best_w    = low
    best_err  = float('inf')
    best_stats = (None, None)

    for w in ws:
        sim_pnl, sim_dd = simulate_fhs_fast(
            mu, omega, alpha, beta, uncond_var, std_resid,
            calib_trials, draws,
            tail_weight_low=w,
            tail_weight_high=1.0
        )
        s_p1   = np.percentile(sim_pnl, 1)
        s_dd99 = np.percentile(sim_dd, 99)

        # combined relative error
        err = abs(s_p1   - target_p1)   / abs(target_p1) + \
              abs(s_dd99 - target_dd99) / abs(target_dd99)

        if err < best_err:
            best_err   = err
            best_w     = w
            best_stats = (s_p1, s_dd99)

    return best_w, best_stats


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo and bootstrap simulation of daily P/L"
    )
    parser.add_argument("csv_file", help="Path to trade-log CSV file")
    parser.add_argument(
        "-t", "--trials", type=int, default=100000,
        help="Number of Monte Carlo trials"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--method",
        choices=['normal', 'block', 'rolling', 'garch', 'fhs'],
        default='normal',
        help="Simulation method: 'normal', 'block', 'rolling', 'garch', or 'fhs'"
    )
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="(FHS only) calibrate tail weights for P/L & DD and exit"
    )
    parser.add_argument(
        "--calibrate-run", action="store_true",
        help="(FHS only) calibrate tail weights and then run combined FHS"
    )
    parser.add_argument(
        "--tail-weight-low", type=float, default=1.0,
        help="(FHS) low-tail weight factor (>1)"
    )
    parser.add_argument(
        "--tail-weight-high", type=float, default=1.0,
        help="(FHS) high-tail weight factor (>1)"
    )
    parser.add_argument(
        "--date-col", default="Date Opened",
        help="CSV column name for entry dates"
    )
    parser.add_argument(
        "--pl-col", default="P/L",
        help="CSV column name for profit/loss values"
    )
    parser.add_argument(
        "--charts", action="store_true",
        help="Save histogram and drawdown charts for each interval"
    )
    parser.add_argument(
        "-i", "--interval", nargs='+',
        default=['30d', '60d', '90d', '180d', '1y', 'all'],
        choices=['30d', '60d', '90d', '180d', '1y', 'all'],
        help="Intervals: 30d, 60d, 90d, 180d, 1y, or all-time"
    )
    args = parser.parse_args()

    # Load & validate CSV
    df = pd.read_csv(args.csv_file)
    if args.date_col not in df.columns or args.pl_col not in df.columns:
        print(f"Error: CSV must contain '{args.date_col}' and '{args.pl_col}' columns.", file=sys.stderr)
        sys.exit(1)

    # Aggregate P/L by date
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors='coerce')
    df = df.dropna(subset=[args.date_col])
    df['Date'] = df[args.date_col].dt.date
    daily = df.groupby('Date')[args.pl_col].sum()

    # Reindex calendar days
    start, end = daily.index.min(), daily.index.max()
    days = pd.date_range(start, end, freq='D').date
    daily = daily.reindex(days, fill_value=0)

    daily_returns = daily.values
    N = len(daily_returns)

    if args.seed is not None:
        np.random.seed(args.seed)

    days_map = {'30d': 30, '60d': 60, '90d': 90, '180d': 180, '1y': 365}

    am   = arch_model(
        daily_returns,
        mean='Constant', vol='Garch', p=1, q=1,
        dist='normal', rescale=False
    )
    res  = am.fit(disp='off')
    p    = res.params
    mu   = p.get('mu', p.get('Const', 0.0))
    omega= p['omega']
    alpha= p['alpha[1]']
    beta = p['beta[1]']
    uncond_var = omega / (1 - alpha - beta)
    std_resid  = res.std_resid[np.isfinite(res.std_resid)]

    # ————— Multi‐objective calibration/workflow in main() —————
    if args.method == 'fhs' and (args.calibrate_only or args.calibrate_run):
        # pick reference window for calibration
        ref = 'all' if 'all' in args.interval else args.interval[0]
        draws_ref = N if ref == 'all' else min(days_map[ref], N)

        # grid-search for best low-tail weight
        w, (sim_p1, sim_dd99) = calibrate_weight_combined(
            mu, omega, alpha, beta, uncond_var, std_resid,
            daily_returns, draws_ref,
            full_trials=args.trials,
            calib_trials=5000,
            low=1.0, high=5.0, steps=41
        )

        print(
            f"Calibrated tail_weight_low={w:.3f} (based on {ref}): "
            f"sim 1% P/L={sim_p1:.2f}, sim 99% DD={sim_dd99:.2f}"
        )

        if args.calibrate_only:
            return

        # now run full-scale FHS (full args.trials) on every interval
        for label in args.interval:
            draws = N if label == 'all' else min(days_map[label], N)
            vals, dds = simulate_fhs(
                daily_returns, args.trials, draws,
                tail_weight_low=w,
                tail_weight_high=1.0
            )
            print_summary(f"FHS combined tails {label}", vals, dds, args.trials)
        return

    # default behavior: run once with --tail-weight-low/high or no calibration
    print(f"Data span: {start} to {end} ({N} days)")
    print(f"Method: {args.method}, Trials: {args.trials}, Intervals: {args.interval}")
    if args.method == 'fhs':
        print(f"  tail_weight_low={args.tail_weight_low}, tail_weight_high={args.tail_weight_high}")
    print()

    for label in args.interval:
        draws = N if label == 'all' else min(days_map[label], N)

        if args.method == 'normal':
            values, dds = simulate_normal(daily_returns, args.trials, draws)
        elif args.method == 'block':
            values, dds = simulate_block(daily_returns, args.trials, draws)
        elif args.method == 'rolling':
            values, dds = compute_rolling(daily_returns, draws)
        elif args.method == 'garch':
            values, dds = simulate_garch(daily_returns, args.trials, draws)
        else:  # fhs
            values, dds = simulate_fhs(
                daily_returns, args.trials, draws,
                tail_weight_low=args.tail_weight_low,
                tail_weight_high=args.tail_weight_high
            )

        print_summary(f"Interval {label}", values, dds,
                      None if args.method == 'rolling' else args.trials)

        if args.charts:
            plot_and_save(values, f"Returns {label}", 'P/L ($)', f"hist_{args.method}_{label}.png")
            plot_and_save(dds, f"Drawdowns {label}", 'Drawdown ($)', f"dd_{args.method}_{label}.png")
            print(f"Charts saved for {label}\n")

    try:
        del df, daily_returns
    except NameError:
        pass
    gc.collect()
    print("Simulation complete.")


if __name__ == '__main__':
    main()
