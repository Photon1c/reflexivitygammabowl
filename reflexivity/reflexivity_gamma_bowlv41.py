"""
Reflexivity Gamma Bowl v4.1

Main Features:
- Loads historical stock and option chain data for a given ticker and option date.
- Multi-subplot analytics: gamma curve, spot price time series, curvature tracker, and dealer strength gradient.
- Interactive matplotlib UI: sliders for gamma pressure and sentiment tilt, lock hill checkbox.
- Expanded narrative strategy module and overlays for instability zones.
- Dealer resilience gradient subplot and live reflexivity score.
- Markdown export of strategy signals and PNG export of the chart.

How this version differs from v2/v3/v4:
- First to introduce dealer strength gradient and expanded narrative overlays.
- Multi-subplot layout for deeper analytics (4 panels).
- No expiry/strike dropdowns (see v4.2+ for those).
- Designed as an advanced research/teaching tool for reflexivity and dealer dynamics.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, CheckButtons
from datetime import datetime
from pathlib import Path
from data_loader import load_stock_data, load_option_chain_data

class ReflexivityGammaBowlV41:
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
        # Interactive parameters
        self.base_a = 0.01
        self.sentiment_bias = 0.0
        self.reflexivity_threshold = 0.5
        self.dealer_strength = 1.0
        self.dealer_decay = 0.002
        self.hill_locked = False
        self.default_spot = 600
        self.x = self.default_spot
        self.v = 0.0
        self.x0 = self.default_spot
        self.frame_count = 0
        self.reflexivity_score = 0.0
        self.instability_logged = False
        self.instability_start_frame = None
        self.reflexivity_log = []
        self.curvature_history = []
        self.dealer_strength_history = []

    def _load_data(self):
        self.stock_df = load_stock_data(self.ticker)
        self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date'])
        self.stock_df = self.stock_df.sort_values('Date').reset_index(drop=True)
        self.option_df = load_option_chain_data(self.ticker.lower(), date=self.option_date)
        self.option_df['Expiration Date'] = pd.to_datetime(self.option_df['Expiration Date'])

    def _align_data(self):
        self.dates = self.stock_df['Date'].unique()
        self.expiries = self.option_df['Expiration Date'].unique()
        self.nearest_expiry = min(self.expiries) if len(self.expiries) > 0 else None

    def _gamma_curve_for_date(self, date):
        df = self.option_df[self.option_df['Expiration Date'] == self.nearest_expiry]
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
        idx = np.argmin(np.abs(gamma_curve))
        return strikes[idx]

    def animate(self, savefig=False, figpath=None, export_signals=False):
        fig, axs = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        plt.subplots_adjust(bottom=0.35, hspace=0.5)
        ax1, ax2, ax3, ax4 = axs
        n_frames = len(self.stock_df)
        line, = ax1.plot([], [], label="Gamma Curve", color='blue')
        point, = ax1.plot([], [], 'ro', markersize=8, label="Spot Price")
        gamma_flip_line = ax1.axvline(0, color='red', linestyle='--', label='Gamma Flip')
        gamma_flip_text = ax1.text(0, 1.0, 'Gamma Flip', color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')
        timestamp_text = ax1.text(0.98, 0.02, '', transform=ax1.transAxes, ha='right', va='bottom', fontsize=10, color='black')
        narrative_annot = ax1.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green', visible=False)
        ref_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax1.set_title("Reflexivity Gamma Bowl v4.1 (Curvature, Dealer Resilience, Narrative)")
        ax1.set_xlabel("Strike")
        ax1.set_ylabel("Net Gamma Exposure")
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        # Spot price time series
        line2, = ax2.plot([], [], color="green", label="Spot Price Time Series")
        ax2.set_title("Spot Price Over Time")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Spot Price")
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
        # Curvature tracker
        line3, = ax3.plot([], [], color="purple", label="Gamma Curvature (2nd Derivative)")
        ax3.set_title("Curvature Evolution")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Curvature")
        ax3.legend(loc='upper right', bbox_to_anchor=(1, 1))
        # Dealer resilience gradient
        line4, = ax4.plot([], [], color="orange", label="Dealer Strength")
        ax4.set_title("Dealer Resilience Gradient")
        ax4.set_xlabel("Frame")
        ax4.set_ylabel("Strength")
        ax4.legend(loc='upper right', bbox_to_anchor=(1, 1))
        spot_history = []
        last_strategy_signal = {'text': '', 'frame': -1}
        # Sliders and checkboxes
        ax_gamma = plt.axes([0.15, 0.22, 0.65, 0.03])
        slider_gamma = Slider(ax_gamma, 'Gamma Pressure', 0.001, 0.05, valinit=self.base_a)
        ax_sentiment = plt.axes([0.15, 0.16, 0.65, 0.03])
        slider_sentiment = Slider(ax_sentiment, 'Sentiment Tilt', -0.1, 0.1, valinit=0.0)
        ax_check = plt.axes([0.85, 0.05, 0.12, 0.08])
        check = CheckButtons(ax_check, ['Lock Hill'], [False])
        def update(frame):
            self.frame_count = frame
            row = self.stock_df.iloc[frame]
            date = row['Date']
            spot = row['Close/Last']
            # Interactive controls
            a = slider_gamma.val
            self.sentiment_bias = slider_sentiment.val
            self.hill_locked = check.get_status()[0]
            # Gamma bowl
            strikes, gamma_curve = self._gamma_curve_for_date(date)
            gamma_flip = self._find_gamma_flip(strikes, gamma_curve)
            # Dealer strength decay
            if not self.hill_locked and self.dealer_strength > 0:
                self.x0 += 0.02 * (spot - self.x0)
                self.dealer_strength -= self.dealer_decay
            if self.dealer_strength <= 0:
                self.hill_locked = True
            # Instability zone
            for collection in ax1.collections[:]:
                collection.remove()
            ax1.fill_between(strikes, 0, gamma_curve, where=gamma_curve < 0, alpha=0.2, color='red', label="⚠️ Instability Zone")
            # Gamma curve
            line.set_data(strikes, gamma_curve)
            ax1.set_xlim(strikes.min(), strikes.max())
            ax1.set_ylim(gamma_curve.min() - 0.1, gamma_curve.max() + 0.1)
            # Spot price marker
            y = np.interp(spot, strikes, gamma_curve)
            point.set_data([spot], [y])
            # Gamma flip marker
            gamma_flip_line.set_xdata([gamma_flip, gamma_flip])
            gamma_flip_text.set_position((gamma_flip, gamma_curve.max() * 0.9))
            # Curvature tracker
            dx = strikes[1] - strikes[0] if len(strikes) > 1 else 1
            curvature = np.mean(np.gradient(np.gradient(gamma_curve, dx), dx))
            self.curvature_history.append(curvature)
            line3.set_data(range(len(self.curvature_history)), self.curvature_history)
            ax3.set_xlim(0, n_frames)
            ax3.set_ylim(min(self.curvature_history + [0]) - 0.01, max(self.curvature_history + [0]) + 0.01)
            # Annotate inflection points
            if len(self.curvature_history) > 2:
                if np.sign(self.curvature_history[-1]) != np.sign(self.curvature_history[-2]):
                    ax3.annotate('Inflection', xy=(frame, self.curvature_history[-1]), xytext=(frame, self.curvature_history[-1]+0.01),
                                 arrowprops=dict(facecolor='purple', shrink=0.05), fontsize=8, color='purple')
            # Dealer resilience gradient
            self.dealer_strength_history.append(self.dealer_strength)
            line4.set_data(range(len(self.dealer_strength_history)), self.dealer_strength_history)
            ax4.set_xlim(0, n_frames)
            ax4.set_ylim(0, 1.05)
            # Shade low-resilience region
            if self.dealer_strength < 0.2:
                ax4.axhspan(0, 0.2, color='red', alpha=0.2)
            # Spot price time series
            spot_history.append(spot)
            line2.set_data(range(len(spot_history)), spot_history)
            ax2.set_xlim(0, n_frames)
            ax2.set_ylim(min(spot_history) - 1, max(spot_history) + 1)
            # Strategy signals & narrative
            strategy_signal = None
            # Expanded narrative prompts
            if spot < gamma_flip and np.gradient(gamma_curve).mean() > 0:
                strategy_signal = "🐍 Coiled Volatility: Watch for Breakout"
            elif spot > gamma_flip and np.abs(np.gradient(gamma_curve).mean()) < 0.01:
                strategy_signal = "🌾 Calm Basin: Premium Harvest Zone"
            elif self.dealer_strength < 0.2:
                strategy_signal = "🧭 Dealer Exhaustion: Volatility May Spike!"
            elif abs(spot - gamma_flip) < 1.0:
                strategy_signal = "🔔 Gamma Flip Approaching!"
            # Narrative annotation
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
            # Reflexivity score and dealer strength
            self.reflexivity_score = abs(spot - self.x0)
            ref_text.set_text(f"Reflexivity Score: {self.reflexivity_score:.2f}\nDealer Strength: {self.dealer_strength:.2f}")
            # Timestamp
            timestamp_text.set_text(date.strftime('%Y-%m-%d'))
            return (line, point, gamma_flip_line, gamma_flip_text, timestamp_text, line2, narrative_annot, ref_text, line3, line4)
        ani = FuncAnimation(fig, update, frames=len(self.stock_df), interval=200, blit=True)
        plt.show()
        # Export figure if requested
        if savefig:
            if not figpath:
                figpath = self.export_dir / f"reflexivity_bowlv41_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(figpath)
            print(f"Figure saved to {figpath}")
        # Export strategy signals as markdown
        if export_signals and self.strategy_signals:
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
        parser = argparse.ArgumentParser(description="Reflexivity Gamma Bowl v4.1 - Curvature, Dealer Resilience, Narrative")
        parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
        parser.add_argument('--option_date', type=str, default=None, help='Option chain date (YYYY-MM-DD)')
        parser.add_argument('--playback_speed', type=int, default=1, help='Playback speed (frames per update)')
        parser.add_argument('--savefig', action='store_true', help='Save the final figure as PNG')
        parser.add_argument('--export_signals', action='store_true', help='Export strategy signals as markdown')
        args = parser.parse_args()
        cli = ReflexivityGammaBowlV41(
            ticker=args.ticker,
            option_date=args.option_date,
            playback_speed=args.playback_speed
        )
        cli.animate(savefig=args.savefig, export_signals=args.export_signals)
if __name__ == "__main__":
    ReflexivityGammaBowlV41.cli() 