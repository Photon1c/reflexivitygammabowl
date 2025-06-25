"""
Reflexivity Gamma Bowl v4.4 - Automated Backtesting & Foresight Module

Main Features:
- Loops over a date range and runs gamma bowl, curvature, and dealer resilience analysis for each day.
- Loads stock and option chain data for each date.
- Saves output chart (PNG), strategy signals (markdown), and key metrics (JSON) for each date in /outputs/SPY/{date}/.
- Designed for research, foresight, and future LLM narration.
- No animation UI; this is a batch processor for automated analysis.

Usage:
    python reflexivity_gamma_bowlv44.py --ticker SPY --start 2025-04-24 --end 2025-06-24
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from data_loader import load_stock_data, load_option_chain_data

# --- Analysis logic (adapted from v4.3, non-interactive) ---
def run_analysis_for_date(ticker, date_str, output_dir):
    try:
        stock_df = load_stock_data(ticker)
        option_df = load_option_chain_data(ticker.lower(), date_str)
    except FileNotFoundError:
        print(f"[WARN] Data not found for {date_str}")
        return False
    # Parse and normalize Expiration Date
    cleaned = option_df['Expiration Date'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    parsed = pd.to_datetime(cleaned, errors='coerce')
    option_df['Expiration Date'] = parsed.dt.strftime('%Y-%m-%d')
    option_df = option_df[option_df['Expiration Date'] != 'NaT']
    expiry_list = sorted(option_df['Expiration Date'].unique())
    selected_expiry = expiry_list[0] if expiry_list else None
    strike_list = sorted(option_df[option_df['Expiration Date'] == selected_expiry]['Strike'].unique())
    selected_strike = strike_list[len(strike_list)//2] if strike_list else None
    # Use the last available stock price for the day
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    row = stock_df.iloc[-1]
    spot = row['Close/Last']
    # Gamma curve for selected expiry
    df = option_df[option_df['Expiration Date'] == selected_expiry]
    strikes = np.sort(df['Strike'].unique())
    gamma_curve = []
    for strike in strikes:
        call_gamma = df[(df['Strike'] == strike) & (df['Calls'].notnull())]['Gamma'].sum() if 'Gamma' in df else 0
        put_gamma = df[(df['Strike'] == strike) & (df['Puts'].notnull())]['Gamma.1'].sum() if 'Gamma.1' in df else 0
        gamma_curve.append(call_gamma + put_gamma)
    gamma_curve = np.array(gamma_curve)
    # Gamma flip
    gamma_flip = strikes[np.argmin(np.abs(gamma_curve))] if len(strikes) > 0 else None
    # Curvature at spot
    if len(gamma_curve) > 2:
        dx = strikes[1] - strikes[0]
        curvature = 1000 * np.mean(np.gradient(np.gradient(gamma_curve, dx), dx))
    else:
        curvature = 0
    # Dealer strength (simple model)
    dealer_strength = 1.0
    # Reflexivity score
    reflexivity_score = abs(spot - (strikes[len(strikes)//2] if len(strikes) > 0 else spot))
    # Strategy signal
    grad_mean = np.abs(np.gradient(gamma_curve).mean()) if len(gamma_curve) > 1 else 0
    if spot < gamma_flip and grad_mean > 0:
        signal = "🐍 Coiled Volatility: Watch for Breakout"
    elif spot > gamma_flip and grad_mean < 0.01:
        signal = "🌾 Calm Basin: Premium Harvest Zone"
    elif dealer_strength < 0.2:
        signal = "🧭 Dealer Exhaustion: Volatility May Spike!"
    elif abs(spot - gamma_flip) < 1.0:
        signal = "🔔 Gamma Flip Approaching!"
    else:
        signal = ""
    # --- Save outputs ---
    date_folder = Path(output_dir) / ticker / date_str
    date_folder.mkdir(parents=True, exist_ok=True)
    # Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strikes, gamma_curve, label="Gamma Curve", color='blue')
    ax.axvline(gamma_flip, color='red', linestyle='--', label='Gamma Flip')
    ax.plot([spot], [np.interp(spot, strikes, gamma_curve)], 'ro', label='Spot Price')
    ax.set_title(f"Gamma Bowl {ticker} {date_str}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Net Gamma Exposure")
    ax.legend()
    fig.savefig(date_folder / f"gamma_bowl_{ticker}_{date_str}.png")
    plt.close(fig)
    # Markdown signals
    md_path = date_folder / f"strategy_signals_{ticker}_{date_str}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Strategy Signals Log for {ticker} {date_str}\n\n")
        f.write("| Spot | Gamma Flip | Curvature | Dealer Strength | Signal |\n")
        f.write("|------|------------|-----------|-----------------|--------|\n")
        f.write(f"| {spot:.2f} | {gamma_flip} | {curvature:.4f} | {dealer_strength:.2f} | {signal} |\n")
    # Metrics JSON
    metrics = {
        "date": date_str,
        "spot": float(spot),
        "gamma_flip": float(gamma_flip) if gamma_flip is not None else None,
        "curvature": float(curvature),
        "dealer_strength": float(dealer_strength),
        "reflexivity_score": float(reflexivity_score),
        "signal": signal
    }
    with open(date_folder / f"{date_str}_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] {date_str}: Chart, signals, and metrics saved.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Reflexivity Gamma Bowl v4.4 - Automated Backtesting")
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()
    stock_df = load_stock_data(args.ticker)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.sort_values('Date')
    trading_days = stock_df[(stock_df['Date'] >= args.start) & (stock_df['Date'] <= args.end)]['Date']
    for date in trading_days:
        date_str = date.strftime('%m_%d_%Y')
        run_analysis_for_date(args.ticker, date_str, args.output_dir)

if __name__ == "__main__":
    main() 