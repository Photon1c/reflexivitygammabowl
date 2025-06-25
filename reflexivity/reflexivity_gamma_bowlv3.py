"""
Reflexivity Gamma Bowl v3

Main Features:
- Loads historical stock and option chain data for a given ticker and option date.
- Real-time synchronized animation: one frame per row of historical data.
- Gamma bowl is dynamically generated from option chain data (nearest expiry).
- Time-aware playback: spot price and gamma curve evolve with real data.
- Instability zone overlays and basic narrative strategy signals.
- Markdown export of strategy signals and PNG export of the chart.

How this version differs from v2/v4:
- First to synchronize animation with real historical data and option chain.
- Gamma bowl is not synthetic, but built from actual option data.
- No interactive UI controls or multi-subplot analytics (see v4+ for those).
- Designed as a real-time monitor for gamma/spot dynamics.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from pathlib import Path
from data_loader import load_stock_data, load_option_chain_data

class ReflexivityGammaBowlV3:
    def __init__(self, ticker='SPY', option_date=None, playback_speed=1, export_dir='reflexivity/logs'):
        self.ticker = ticker
        self.option_date = option_date
        self.playback_speed = playback_speed
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._load_data()
        self._align_data()
        self.trade_log = []
        self.strategy_signals = []

    def _load_data(self):
        self.stock_df = load_stock_data(self.ticker)
        self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date'])
        self.stock_df = self.stock_df.sort_values('Date').reset_index(drop=True)
        self.option_df = load_option_chain_data(self.ticker.lower(), date=self.option_date)
        self.option_df['Expiration Date'] = pd.to_datetime(self.option_df['Expiration Date'])

    def _align_data(self):
        # For now, align by date only (expand to minute if available)
        self.dates = self.stock_df['Date'].unique()
        # Option chain may have multiple expirations per date; we'll use the nearest expiry
        self.expiries = self.option_df['Expiration Date'].unique()
        self.nearest_expiry = min(self.expiries) if len(self.expiries) > 0 else None

    def _gamma_curve_for_date(self, date):
        # Filter option chain for the current date and nearest expiry
        df = self.option_df[self.option_df['Expiration Date'] == self.nearest_expiry]
        # Group by strike, sum gamma for calls and puts
        strikes = df['Strike'].unique()
        strikes = np.sort(strikes)
        gamma_curve = []
        for strike in strikes:
            call_gamma = df[(df['Strike'] == strike) & (df['Calls'].notnull())]['Gamma'].sum()
            put_gamma = df[(df['Strike'] == strike) & (df['Puts'].notnull())]['Gamma'].sum()
            gamma_curve.append(call_gamma + put_gamma)
        gamma_curve = np.array(gamma_curve)
        return strikes, gamma_curve

    def _find_gamma_flip(self, strikes, gamma_curve):
        # Gamma flip: strike where net gamma crosses zero or is minimized
        idx = np.argmin(np.abs(gamma_curve))
        return strikes[idx]

    def animate(self, savefig=False, figpath=None, html_export=False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.4)
        n_frames = len(self.stock_df)
        line, = ax1.plot([], [], label="Gamma Curve", color='blue')
        point, = ax1.plot([], [], 'ro', markersize=8, label="Spot Price")
        gamma_flip_line = ax1.axvline(0, color='red', linestyle='--', label='Gamma Flip')
        gamma_flip_text = ax1.text(0, 1.0, 'Gamma Flip', color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')
        timestamp_text = ax1.text(0.98, 0.02, '', transform=ax1.transAxes, ha='right', va='bottom', fontsize=10, color='black')
        narrative_annot = ax1.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green', visible=False)
        instability_poly = None
        ax1.set_title("Real-Time Gamma Reflexivity Bowl (v3)")
        ax1.set_xlabel("Strike")
        ax1.set_ylabel("Net Gamma Exposure")
        ax1.legend()
        line2, = ax2.plot([], [], color="green", label="Spot Price Time Series")
        ax2.set_title("Spot Price Over Time")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Spot Price")
        ax2.legend()
        spot_history = []
        last_strategy_signal = {'text': '', 'frame': -1}
        def update(frame):
            row = self.stock_df.iloc[frame]
            date = row['Date']
            spot = row['Close/Last']
            strikes, gamma_curve = self._gamma_curve_for_date(date)
            gamma_flip = self._find_gamma_flip(strikes, gamma_curve)
            # Update gamma curve
            line.set_data(strikes, gamma_curve)
            ax1.set_xlim(strikes.min(), strikes.max())
            ax1.set_ylim(gamma_curve.min() - 0.1, gamma_curve.max() + 0.1)
            # Update spot price marker
            point.set_data([spot], [np.interp(spot, strikes, gamma_curve)])
            # Update gamma flip marker
            gamma_flip_line.set_xdata([gamma_flip, gamma_flip])
            gamma_flip_text.set_position((gamma_flip, gamma_curve.max() * 0.9))
            # Instability zone: shade where gamma is negative
            for collection in ax1.collections[:]:
                collection.remove()
            ax1.fill_between(strikes, 0, gamma_curve, where=gamma_curve < 0, alpha=0.2, color='red', label="⚠️ Instability Zone")
            # Strategy signals
            strategy_signal = None
            if spot < gamma_flip and np.gradient(gamma_curve).mean() > 0:
                strategy_signal = "🐍 Coiled Volatility: Watch for Breakout"
            elif spot > gamma_flip and np.abs(np.gradient(gamma_curve).mean()) < 0.01:
                strategy_signal = "🌾 Calm Basin: Premium Harvest Zone"
            # Narrative annotation
            y = np.interp(spot, strikes, gamma_curve)
            if strategy_signal:
                narrative_annot.set_text(strategy_signal)
                narrative_annot.xy = (spot, y)
                narrative_annot.set_position((spot + 2, y + 0.5))
                narrative_annot.set_color('green' if 'Calm' in strategy_signal or 'Harvest' in strategy_signal else 'orange')
                narrative_annot.set_visible(True)
                last_strategy_signal['text'] = strategy_signal
                last_strategy_signal['frame'] = frame
                self.strategy_signals.append({'frame': frame, 'date': date, 'spot': spot, 'signal': strategy_signal})
            elif last_strategy_signal['frame'] >= 0 and frame - last_strategy_signal['frame'] < 30:
                narrative_annot.set_text(last_strategy_signal['text'])
                narrative_annot.xy = (spot, y)
                narrative_annot.set_position((spot + 2, y + 0.5))
                narrative_annot.set_visible(True)
            else:
                narrative_annot.set_visible(False)
            # Update timestamp
            timestamp_text.set_text(date.strftime('%Y-%m-%d'))
            # Update spot price history
            spot_history.append(spot)
            line2.set_data(range(len(spot_history)), spot_history)
            ax2.set_xlim(0, n_frames)
            ax2.set_ylim(min(spot_history) - 1, max(spot_history) + 1)
            return line, point, gamma_flip_line, gamma_flip_text, timestamp_text, line2, narrative_annot
        ani = FuncAnimation(fig, update, frames=len(self.stock_df), interval=200, blit=True)
        plt.show()
        # Export figure if requested
        if savefig:
            if not figpath:
                figpath = self.export_dir / f"reflexivity_bowlv3_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(figpath)
            print(f"Figure saved to {figpath}")
        # Export strategy signals as markdown
        if self.strategy_signals:
            md_path = self.export_dir / f"reflexivity_signals_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(md_path, 'w') as f:
                f.write(f"# Strategy Signals Log for {self.ticker}\n\n")
                f.write("| Frame | Date | Spot | Signal |\n")
                f.write("|-------|------------|--------|-----------------------------|\n")
                for s in self.strategy_signals:
                    f.write(f"| {s['frame']} | {s['date'].strftime('%Y-%m-%d')} | {s['spot']:.2f} | {s['signal']} |\n")
            print(f"Strategy signals markdown saved to {md_path}")

    @staticmethod
    def cli():
        parser = argparse.ArgumentParser(description="Reflexivity Gamma Bowl v3 - Real-Time Monitor")
        parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
        parser.add_argument('--option_date', type=str, default=None, help='Option chain date (YYYY-MM-DD)')
        parser.add_argument('--playback_speed', type=int, default=1, help='Playback speed (frames per update)')
        parser.add_argument('--savefig', action='store_true', help='Save the final figure as PNG')
        parser.add_argument('--export_signals', action='store_true', help='Export strategy signals as markdown')
        args = parser.parse_args()
        cli = ReflexivityGammaBowlV3(
            ticker=args.ticker,
            option_date=args.option_date,
            playback_speed=args.playback_speed
        )
        cli.animate(savefig=args.savefig)

if __name__ == "__main__":
    ReflexivityGammaBowlV3.cli() 