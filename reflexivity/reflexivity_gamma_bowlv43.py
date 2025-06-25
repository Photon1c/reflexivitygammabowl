"""
Reflexivity Gamma Bowl v4.3

Main Features:
- Loads historical stock and option chain data for a given ticker and option date.
- Robust parsing and handling of Expiration Date and strike selection (synchronized with v4.2).
- Fully synchronized time-series animation: all subplots (gamma, spot price, curvature) share the same time axis.
- Interactive matplotlib UI: sliders for gamma pressure and sentiment tilt, expiry dropdown, lock hill checkbox.
- Multi-subplot analytics: net gamma/dealer strength, spot price time series, curvature tracker (amplified second derivative).
- Instability zone overlays, narrative strategy signals, and live reflexivity score.
- Markdown export of strategy signals and PNG export of the chart.

How this version differs from v4.2 and earlier:
- All subplots are synchronized to the same time axis (true time-series view).
- Spot price chart is a real time series, not just a frame index.
- Curvature sensitivity is further amplified for advanced analytics.
- Most advanced and interactive version for research and presentation.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from datetime import datetime
from pathlib import Path
from data_loader import load_stock_data, load_option_chain_data

class ReflexivityGammaBowlV43:
    def __init__(self, ticker='SPY', option_date=None, playback_speed=1, export_dir='reflexivity/logs'):
        self.ticker = ticker
        self.option_date = option_date
        self.playback_speed = playback_speed
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        # All state variables are instance variables
        self.expiry_list = []
        self.strike_list = []
        self.selected_expiry = None
        self.selected_strike = None
        self.trade_log = []
        self.strategy_signals = []
        self.base_a = 0.01
        self.sentiment_bias = 0.0
        self.reflexivity_threshold = 0.5
        self.dealer_strength = 1.0
        self.dealer_decay_base = 0.002
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
        self._load_data()
        self._align_data()
        print("[DEBUG] (init end) expiry_list:", self.expiry_list)

    def _parse_expiration_dates(self, expiration_series):
        """
        Robustly parse and normalize Expiration Date values from the option chain DataFrame.
        Returns a pandas Series of strings in 'YYYY-MM-DD' format, with invalid/missing dates dropped.
        Adds deep debug output for failed parses and raw values.
        """
        print("[DEBUG] Raw Expiration Date values (first 10):", expiration_series.head(10).tolist())
        print("[DEBUG] repr of first Expiration Date value:", repr(expiration_series.iloc[0]))
        # Aggressively clean whitespace
        cleaned = expiration_series.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        print("[DEBUG] Cleaned Expiration Date values (first 10):", cleaned.head(10).tolist())
        # Try parsing as datetime (flexible)
        parsed = pd.to_datetime(cleaned, errors='coerce')
        print("[DEBUG] After default parse (first 10):", parsed.head(10).tolist())
        # If all are NaT, try explicit format for 'Tue Jun 24 2025'
        if parsed.isna().all():
            parsed = pd.to_datetime(cleaned, errors='coerce', format='%a %b %d %Y')
            print("[DEBUG] After explicit format parse (first 10):", parsed.head(10).tolist())
        # Debug: print which rows failed to parse
        failed = expiration_series[parsed.isna()]
        if not failed.empty:
            print("[DEBUG] Expiration Date rows failed to parse:", failed.tolist())
        # Convert to string format 'YYYY-MM-DD', drop NaT
        parsed_str = parsed.dt.strftime('%Y-%m-%d')
        return parsed_str

    def _load_data(self):
        self.stock_df = load_stock_data(self.ticker)
        self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date'])
        self.stock_df = self.stock_df.sort_values('Date').reset_index(drop=True)
        self.option_df = load_option_chain_data(self.ticker.lower(), date=self.option_date)
        # Parse and normalize Expiration Date, always as string 'YYYY-MM-DD'
        cleaned = self.option_df['Expiration Date'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        parsed = pd.to_datetime(cleaned, errors='coerce')
        self.option_df['Expiration Date'] = parsed.dt.strftime('%Y-%m-%d')
        # Drop rows with invalid dates (string 'NaT')
        self.option_df = self.option_df[self.option_df['Expiration Date'] != 'NaT']
        self.expiry_list = sorted(self.option_df['Expiration Date'].unique())
        print("[DEBUG] Final parsed Expiration Date values (first 10):", self.option_df['Expiration Date'].head(10).tolist())
        print("[DEBUG] Final expiry_list:", self.expiry_list)
        print("[DEBUG] Option DataFrame shape after parsing:", self.option_df.shape)
        self.selected_expiry = self.expiry_list[0] if self.expiry_list else None
        self.strike_list = sorted(self.option_df[self.option_df['Expiration Date'] == self.selected_expiry]['Strike'].unique())
        self.selected_strike = self.strike_list[len(self.strike_list)//2] if self.strike_list else None

    def _align_data(self):
        self.dates = self.stock_df['Date'].unique()

    def _gamma_curve_for_expiry(self, expiry):
        df = self.option_df[self.option_df['Expiration Date'] == expiry]
        strikes = df['Strike'].unique()
        strikes = np.sort(strikes)
        gamma_curve = []
        for strike in strikes:
            call_gamma = df[(df['Strike'] == strike) & (df['Calls'].notnull())]['Gamma'].sum() if 'Gamma' in df else 0
            put_gamma = df[(df['Strike'] == strike) & (df['Puts'].notnull())]['Gamma.1'].sum() if 'Gamma.1' in df else 0
            gamma_curve.append(call_gamma + put_gamma)
        gamma_curve = np.array(gamma_curve)
        return strikes, gamma_curve

    def _find_gamma_flip(self, strikes, gamma_curve):
        idx = np.argmin(np.abs(gamma_curve))
        return strikes[idx]

    def animate(self, savefig=False, figpath=None, export_signals=False):
        print("[DEBUG] (animate, very start) expiry_list:", self.expiry_list)
        print("[DEBUG] (animate) selected_expiry:", self.selected_expiry)
        print("[DEBUG] (animate) strike_list:", self.strike_list)
        print("[DEBUG] (animate) option_df shape:", self.option_df.shape)
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        plt.subplots_adjust(bottom=0.35, hspace=0.5)
        ax1, ax2, ax3 = axs
        if not self.expiry_list or self.selected_expiry is None or not self.strike_list:
            print("[ERROR] No valid expiration dates found in the option chain data. Please check your CSV file and column names.")
            ax1.text(0.5, 0.5, 'No valid Expiry Data Found\nCheck your option chain CSV!',
                     transform=ax1.transAxes, fontsize=16, color='red', ha='center', va='center', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
            ax2.axis('off')
            ax3.axis('off')
            plt.show()
            return
        line, = ax1.plot([], [], label="Gamma Curve", color='blue')
        dealer_line, = ax1.plot([], [], label="Dealer Strength", color='orange', linestyle='--')
        point, = ax1.plot([], [], 'ro', markersize=8, label="Spot Price")
        gamma_flip_line = ax1.axvline(0, color='red', linestyle='--', label='Gamma Flip')
        gamma_flip_text = ax1.text(0, 1.0, 'Gamma Flip', color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')
        timestamp_text = ax1.text(0.98, 0.02, '', transform=ax1.transAxes, ha='right', va='bottom', fontsize=10, color='black')
        narrative_annot = ax1.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green', visible=False)
        ref_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax1.set_title("Reflexivity Gamma Bowl v4.3 (Synchronized, Time Series, Sensitive Curvature)")
        ax1.set_xlabel("Strike")
        ax1.set_ylabel("Net Gamma Exposure / Dealer Strength")
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        # Spot price time series (x-axis: actual timestamps)
        line2, = ax2.plot([], [], color="green", label="Spot Price Time Series")
        ax2.set_title("Spot Price Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Spot Price")
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
        # Curvature tracker (amplified)
        line3, = ax3.plot([], [], color="purple", label="Gamma Curvature (2nd Derivative, x1000)")
        ax3.set_title("Curvature Evolution (x1000)")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Curvature")
        ax3.legend(loc='upper right', bbox_to_anchor=(1, 1))
        spot_history = []
        spot_dates = []
        curvature_dates = []
        last_strategy_signal = {'text': '', 'frame': -1}
        # Sliders and checkboxes
        ax_gamma = plt.axes([0.15, 0.22, 0.65, 0.03])
        slider_gamma = Slider(ax_gamma, 'Gamma Pressure', 0.001, 0.05, valinit=self.base_a)
        ax_sentiment = plt.axes([0.15, 0.16, 0.65, 0.03])
        slider_sentiment = Slider(ax_sentiment, 'Sentiment Tilt', -0.1, 0.1, valinit=0.0)
        ax_check = plt.axes([0.85, 0.05, 0.12, 0.08])
        check = CheckButtons(ax_check, ['Lock Hill'], [False])
        # Dropdowns for expiry
        expiry_labels = self.expiry_list if self.expiry_list else ['No Expiry']
        ax_expiry = plt.axes([0.02, 0.22, 0.1, 0.1])
        expiry_radio = RadioButtons(ax_expiry, expiry_labels, active=0)
        if not self.expiry_list:
            print("[Warning] No expiration dates found in option chain data. UI controls are disabled.")
        def on_expiry_change(label):
            if not self.expiry_list:
                return
            self.selected_expiry = label  # label is already in 'YYYY-MM-DD' format
            self.strike_list = sorted(self.option_df[self.option_df['Expiration Date'] == self.selected_expiry]['Strike'].unique())
            self.selected_strike = self.strike_list[len(self.strike_list)//2] if self.strike_list else None
        expiry_radio.on_clicked(on_expiry_change)
        # Main update function
        def update(frame):
            self.frame_count = frame
            row = self.stock_df.iloc[frame]
            date = row['Date']
            spot = row['Close/Last']
            # Interactive controls
            a = slider_gamma.val
            self.sentiment_bias = slider_sentiment.val
            self.hill_locked = check.get_status()[0]
            # Gamma bowl for selected expiry
            if not self.expiry_list or self.selected_expiry is None:
                strikes, gamma_curve = np.array([0]), np.array([0])
            else:
                strikes, gamma_curve = self._gamma_curve_for_expiry(self.selected_expiry)
            gamma_flip = self._find_gamma_flip(strikes, gamma_curve) if len(strikes) > 1 else 0
            # Dealer strength decay (dynamic)
            reflexivity = abs(spot - self.x0)
            decay = self.dealer_decay_base * (1 + 5 * min(reflexivity, 1))
            if not self.hill_locked and self.dealer_strength > 0:
                self.x0 += 0.02 * (spot - self.x0)
                self.dealer_strength -= decay
            if self.dealer_strength <= 0:
                self.hill_locked = True
            # Dealer strength recovery in calm periods
            if self.hill_locked and reflexivity < self.reflexivity_threshold/2 and self.dealer_strength < 1.0:
                self.dealer_strength += self.dealer_decay_base * 2
                if self.dealer_strength > 0.2:
                    self.hill_locked = False
            # Instability zone
            for collection in ax1.collections[:]:
                collection.remove()
            ax1.fill_between(strikes, 0, gamma_curve, where=gamma_curve < 0, alpha=0.2, color='red', label="⚠️ Instability Zone")
            # Gamma curve
            line.set_data(strikes, gamma_curve)
            # Dealer strength overlay (normalized to strikes)
            dealer_strength_curve = np.ones_like(strikes) * self.dealer_strength
            dealer_line.set_data(strikes, dealer_strength_curve)
            ax1.set_xlim(strikes.min(), strikes.max())
            ax1.set_ylim(min(gamma_curve.min(), 0) - 0.1, max(gamma_curve.max(), 1) + 0.1)
            # Spot price marker
            y = np.interp(spot, strikes, gamma_curve)
            point.set_data([spot], [y])
            # Gamma flip marker
            gamma_flip_line.set_xdata([gamma_flip, gamma_flip])
            gamma_flip_text.set_position((gamma_flip, gamma_curve.max() * 0.9))
            # Curvature tracker (amplified for visibility)
            if len(gamma_curve) > 2:
                dx = strikes[1] - strikes[0]
                curvature = 1000 * np.mean(np.gradient(np.gradient(gamma_curve, dx), dx))
            else:
                curvature = 0
            self.curvature_history.append(curvature)
            curvature_dates.append(date)
            line3.set_data(curvature_dates, self.curvature_history)
            ax3.set_xlim(curvature_dates[0] if curvature_dates else 0, curvature_dates[-1] if curvature_dates else 1)
            ax3.set_ylim(min(self.curvature_history + [0]) - 0.01, max(self.curvature_history + [0]) + 0.01)
            # Spot price time series (x-axis: actual timestamps)
            spot_history.append(spot)
            spot_dates.append(date)
            line2.set_data(spot_dates, spot_history)
            ax2.set_xlim(spot_dates[0] if spot_dates else 0, spot_dates[-1] if spot_dates else 1)
            ax2.set_ylim(min(spot_history) - 1, max(spot_history) + 1)
            # Strategy signals & narrative
            strategy_signal = None
            grad_mean = np.abs(np.gradient(gamma_curve).mean()) if len(gamma_curve) > 1 else 0
            if spot < gamma_flip and grad_mean > 0:
                strategy_signal = "🐍 Coiled Volatility: Watch for Breakout"
            elif spot > gamma_flip and grad_mean < 0.01:
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
            self.reflexivity_score = reflexivity
            ref_text.set_text(f"Reflexivity Score: {self.reflexivity_score:.2f}\nDealer Strength: {self.dealer_strength:.2f}")
            # Timestamp
            timestamp_text.set_text(date.strftime('%Y-%m-%d'))
            return (line, dealer_line, point, gamma_flip_line, gamma_flip_text, timestamp_text, line2, narrative_annot, ref_text, line3)
        ani = FuncAnimation(fig, update, frames=len(self.stock_df), interval=200, blit=True)
        plt.show()
        # Export figure if requested
        if savefig:
            if not figpath:
                figpath = self.export_dir / f"reflexivity_bowlv43_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(figpath)
            print(f"Figure saved to {figpath}")
        # Export strategy signals as markdown
        if export_signals and self.strategy_signals:
            md_path = self.export_dir / f"reflexivity_signals_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# Strategy Signals Log for {self.ticker}\n\n")
                f.write("| Frame | Date | Spot | Signal |\n")
                f.write("|-------|------------|--------|-----------------------------|\n")
                for s in self.strategy_signals:
                    f.write(f"| {s['frame']} | {s['date'].strftime('%Y-%m-%d')} | {s['spot']:.2f} | {s['signal']} |\n")
            print(f"Strategy signals markdown saved to {md_path}")
    @staticmethod
    def cli():
        parser = argparse.ArgumentParser(description="Reflexivity Gamma Bowl v4.3 - Synchronized, Time Series, Sensitive Curvature")
        parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
        parser.add_argument('--option_date', type=str, default=None, help='Option chain date (YYYY-MM-DD)')
        parser.add_argument('--playback_speed', type=int, default=1, help='Playback speed (frames per update)')
        parser.add_argument('--savefig', action='store_true', help='Save the final figure as PNG')
        parser.add_argument('--export_signals', action='store_true', help='Export strategy signals as markdown')
        args = parser.parse_args()
        cli = ReflexivityGammaBowlV43(
            ticker=args.ticker,
            option_date=args.option_date,
            playback_speed=args.playback_speed
        )
        cli.animate(savefig=args.savefig, export_signals=args.export_signals)
if __name__ == "__main__":
    ReflexivityGammaBowlV43.cli() 