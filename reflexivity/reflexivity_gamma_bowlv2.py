"""
Reflexivity Gamma Bowl v2

Main Features:
- Enhanced gamma bowl simulation with both stock and (optionally) option chain data.
- Visualizes instability (reflexive) and stability (basin) zones on the gamma curve.
- Tracks and plots curvature (second derivative) and divergence delta.
- Generates and overlays strategy signals (coiled volatility, calm basin) with narrative annotations.
- Logs trades and strategy signals, with export to markdown report and optional HTML export.
- CLI for animation, report generation, and export options.

How this version differs from v1/v3/v4:
- First to introduce instability/stability zone overlays and curvature tracking.
- Adds narrative strategy signals and trade log/report export.
- No interactive UI controls or expiry/strike selection (see v4+ for those).
- Designed as a bridge between the baseline and the interactive/option-aware versions.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from datetime import datetime
from pathlib import Path
from data_loader import load_stock_data, load_option_chain_data

class ReflexivityGammaBowlV2:
    def __init__(self, ticker='SPY', base_a=0.01, max_gamma=1.0, mass=1.0, dt=0.1, 
                 friction=0.02, noise_level=0.05, gamma_shift_rate=0.02, 
                 reflexivity_threshold=0.5):
        self.ticker = ticker
        self.base_a = base_a
        self.max_gamma = max_gamma
        self.mass = mass
        self.dt = dt
        self.friction = friction
        self.noise_level = noise_level
        self.gamma_shift_rate = gamma_shift_rate
        self.reflexivity_threshold = reflexivity_threshold
        self.trade_log = []
        self.strategy_signals = []
        self._load_data()
        self._reset_state()
        
    def _load_data(self):
        """Load both stock and options data"""
        self.df = load_stock_data(self.ticker)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        try:
            self.options_df = load_option_chain_data(self.ticker.lower())
            self.has_options = True
        except:
            print("Warning: Options data not available. Running in stock-only mode.")
            self.has_options = False

    def _reset_state(self):
        """Initialize or reset simulation state"""
        self.x = float(self.df['Close/Last'].iloc[-1])  # Current price
        self.v = 0.0  # Velocity
        self.x0 = self.x  # Hill center
        self.reflexivity_score = 0.0
        self.prev_reflexivity = 0.0
        self.divergence_delta = 0.0
        self.curvature = 0.0
        self.prev_curvature = 0.0
        self.reflexivity_history = deque(maxlen=300)
        self.divergence_history = deque(maxlen=300)
        self.curvature_history = deque(maxlen=300)
        self.time_points = deque(maxlen=300)
        self.in_position = False
        
    def _compute_gamma_curve(self, x_vals):
        """Compute the gamma exposure curve"""
        return -self.base_a * (x_vals - self.x0) ** 2 + self.max_gamma
        
    def _compute_curvature(self, x_vals, gamma_curve):
        """Compute the second derivative of gamma exposure"""
        dx = x_vals[1] - x_vals[0]
        second_derivative = np.gradient(np.gradient(gamma_curve, dx), dx)
        return np.mean(second_derivative)  # Return average curvature
        
    def _generate_strategy_signal(self, frame):
        """Generate strategy signals based on price, gamma flip, and curvature"""
        if self.x < self.x0 and self.curvature > self.prev_curvature:
            signal = "🐍 Coiled Volatility: Watch for Breakout"
            self.strategy_signals.append({
                'frame': frame,
                'type': 'COILED',
                'price': self.x,
                'curvature': self.curvature,
                'message': signal
            })
            return signal
        elif self.x > self.x0 and abs(self.curvature - self.prev_curvature) < 0.1:
            signal = "🌾 Calm Basin: Premium Harvest Zone"
            self.strategy_signals.append({
                'frame': frame,
                'type': 'CALM',
                'price': self.x,
                'curvature': self.curvature,
                'message': signal
            })
            return signal
        return None

    def animate(self, frames=300, interval=50, savefig=False, figpath=None, html_export=False):
        """Enhanced animation with instability zones, gamma flip marker, and strategy overlay"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.4)
        
        # Setup main gamma bowl plot
        x_vals = np.linspace(self.x - 10, self.x + 10, 400)
        line, = ax1.plot([], [], label="Gamma Curve", color='blue')
        point, = ax1.plot([], [], 'ro', markersize=8, label="Spot Price")
        curvature_line, = ax1.plot([], [], '--', color='purple', label="Curvature")
        gamma_flip_line = ax1.axvline(self.x0, color='red', linestyle='--', label='Gamma Flip')
        gamma_flip_text = ax1.text(self.x0, 1.0, 'Gamma Flip', color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')
        narrative_annot = ax1.annotate('', xy=(self.x, self.max_gamma), xytext=(self.x + 2, self.max_gamma + 0.5),
                                       arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green', visible=False)
        
        # Add shaded regions for instability zones and stabilizing basins
        instability_collection = ax1.fill_between([], [], alpha=0.2, color='red', label="⚠️ Reflexive Instability Zone")
        stability_collection = ax1.fill_between([], [], alpha=0.2, color='green', label="✅ Stabilizing Basin")
        
        ax1.set_xlim(self.x - 10, self.x + 10)
        ax1.set_ylim(-0.5, 1.2)
        ax1.set_title("Enhanced Gamma Reflexivity Bowl")
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Gamma Exposure")
        ax1.legend()

        # Setup divergence delta plot
        line2, = ax2.plot([], [], color="green", label="Divergence Delta (dx/dt)")
        ax2.axhline(y=self.reflexivity_threshold, color="red", linestyle="--", label="Instability Threshold")
        ax2.set_xlim(0, frames)
        ax2.set_ylim(-2, 2)
        ax2.set_title("Divergence Delta Tracker")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Delta")
        ax2.legend()

        # Setup curvature plot
        line3, = ax3.plot([], [], color="purple", label="Gamma Curvature")
        ax3.set_xlim(0, frames)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_title("Curvature Evolution")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Curvature")
        ax3.legend()

        last_strategy_signal = {'text': '', 'frame': -1}

        def update(frame):
            # Update hill position
            self.x0 += self.gamma_shift_rate * (self.x - self.x0)
            
            # Compute gamma curve and zones
            gamma_curve = self._compute_gamma_curve(x_vals)
            line.set_data(x_vals, gamma_curve)
            
            # Update instability zones
            instability_mask = x_vals < self.x0
            stability_mask = x_vals >= self.x0
            for collection in ax1.collections[:]:
                collection.remove()
            ax1.fill_between(x_vals, 0, gamma_curve, where=instability_mask, 
                           alpha=0.2, color='red', label="⚠️ Reflexive Instability Zone")
            ax1.fill_between(x_vals, 0, gamma_curve, where=stability_mask, 
                           alpha=0.2, color='green', label="✅ Stabilizing Basin")
            # Update gamma flip marker and label
            gamma_flip_line.set_xdata([self.x0, self.x0])
            gamma_flip_text.set_position((self.x0, 1.0))
            # Motion update
            force = 2 * self.base_a * (self.x - self.x0)
            noise = np.random.normal(0, self.noise_level)
            acceleration = (-force + noise - self.friction * self.v) / self.mass
            self.v += acceleration * self.dt
            self.x += self.v * self.dt
            # Update point position
            y = -self.base_a * (self.x - self.x0) ** 2 + self.max_gamma
            point.set_data([self.x], [y])
            # Update reflexivity metrics
            self.prev_reflexivity = self.reflexivity_score
            self.reflexivity_score = abs(self.x - self.x0)
            self.divergence_delta = (self.reflexivity_score - self.prev_reflexivity) / self.dt
            # Update curvature
            self.prev_curvature = self.curvature
            self.curvature = self._compute_curvature(x_vals, gamma_curve)
            # Update histories
            self.reflexivity_history.append(self.reflexivity_score)
            self.divergence_history.append(self.divergence_delta)
            self.curvature_history.append(self.curvature)
            self.time_points.append(frame)
            # Update plots
            line2.set_data(range(len(self.divergence_history)), self.divergence_history)
            line3.set_data(range(len(self.curvature_history)), self.curvature_history)
            # Check for signals
            trade_signal = None
            if not self.in_position and self.divergence_delta > self.reflexivity_threshold:
                trade_signal = f"[ENTRY] Frame {frame} | Delta: {self.divergence_delta:.2f}"
                self.in_position = True
                self.trade_log.append({
                    'frame': frame,
                    'type': 'ENTRY',
                    'price': self.x,
                    'delta': self.divergence_delta,
                    'curvature': self.curvature
                })
            elif self.in_position and self.divergence_delta < 0:
                trade_signal = f"[EXIT] Frame {frame} | Delta: {self.divergence_delta:.2f}"
                self.in_position = False
                self.trade_log.append({
                    'frame': frame,
                    'type': 'EXIT',
                    'price': self.x,
                    'delta': self.divergence_delta,
                    'curvature': self.curvature
                })
            # Generate and display strategy signals
            strategy_signal = self._generate_strategy_signal(frame)
            # Narrative annotation logic
            if strategy_signal:
                narrative_annot.set_text(strategy_signal)
                narrative_annot.xy = (self.x, y)
                narrative_annot.set_position((self.x + 2, y + 0.5))
                narrative_annot.set_color('green' if 'Calm' in strategy_signal or 'Harvest' in strategy_signal else 'orange')
                narrative_annot.set_visible(True)
                last_strategy_signal['text'] = strategy_signal
                last_strategy_signal['frame'] = frame
            elif last_strategy_signal['frame'] >= 0 and frame - last_strategy_signal['frame'] < 30:
                # Keep showing the last signal for a few frames
                narrative_annot.set_text(last_strategy_signal['text'])
                narrative_annot.xy = (self.x, y)
                narrative_annot.set_position((self.x + 2, y + 0.5))
                narrative_annot.set_visible(True)
            else:
                narrative_annot.set_visible(False)
            if trade_signal:
                print(trade_signal)
            if strategy_signal:
                print(strategy_signal)
            return line, point, line2, line3, gamma_flip_line, gamma_flip_text, narrative_annot
        ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        plt.show()
        # Export figure if requested
        if savefig:
            if not figpath:
                figpath = f"reflexivity/logs/reflexivity_bowl_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(figpath)
            print(f"Figure saved to {figpath}")
        # Optional HTML export
        if html_export:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            import pandas as pd
            html_path = figpath.replace('.png', '.html') if figpath else f"reflexivity/logs/reflexivity_bowl_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            # Save static image and embed in HTML
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            import base64, io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            # Trade log as HTML
            trade_df = pd.DataFrame(self.trade_log)
            trade_html = trade_df.to_html(index=False) if not trade_df.empty else '<p>No trades logged.</p>'
            with open(html_path, 'w') as f:
                f.write(f'<h1>Reflexivity Gamma Bowl - {self.ticker}</h1>')
                f.write(f'<img src="data:image/png;base64,{img_base64}"/><br>')
                f.write('<h2>Trade Log</h2>')
                f.write(trade_html)
            print(f"HTML report saved to {html_path}")

    def generate_report(self):
        """Generate a detailed markdown report of the simulation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"reflexivity/logs/reflexivity_report_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# Reflexivity Analysis Report - {self.ticker}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 Simulation Parameters\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| Ticker | {self.ticker} |\n")
            f.write(f"| Base Gamma (a) | {self.base_a} |\n")
            f.write(f"| Max Gamma | {self.max_gamma} |\n")
            f.write(f"| Friction | {self.friction} |\n")
            f.write(f"| Reflexivity Threshold | {self.reflexivity_threshold} |\n\n")
            
            f.write("## 📈 Trade Log\n")
            f.write("| Frame | Type | Price | Delta | Curvature |\n")
            f.write("|-------|------|--------|--------|------------|\n")
            for trade in self.trade_log:
                f.write(f"| {trade['frame']:5d} | {trade['type']:5s} | "
                       f"{trade['price']:.2f} | {trade['delta']:.2f} | "
                       f"{trade['curvature']:.2f} |\n")
            f.write("\n")
            
            f.write("## 🎯 Strategy Signals\n")
            f.write("| Frame | Type | Message | Price | Curvature |\n")
            f.write("|-------|------|---------|--------|------------|\n")
            for signal in self.strategy_signals:
                f.write(f"| {signal['frame']:5d} | {signal['type']:6s} | "
                       f"{signal['message']} | {signal['price']:.2f} | "
                       f"{signal['curvature']:.2f} |\n")
            f.write("\n")
            
            f.write("## 📝 Analysis Summary\n")
            f.write(f"- Total Trades: {len(self.trade_log)}\n")
            f.write(f"- Strategy Signals Generated: {len(self.strategy_signals)}\n")
            if self.trade_log:
                avg_delta = np.mean([t['delta'] for t in self.trade_log])
                f.write(f"- Average Divergence Delta: {avg_delta:.2f}\n")
            
            print(f"\nReport generated: {report_path}")

    @staticmethod
    def cli():
        parser = argparse.ArgumentParser(
            description="Enhanced Reflexivity Gamma Bowl CLI - Targeting System for SPY"
        )
        parser.add_argument('--ticker', type=str, default='SPY',
                          help='Ticker symbol (default: SPY)')
        parser.add_argument('--frames', type=int, default=300,
                          help='Number of animation frames')
        parser.add_argument('--threshold', type=float, default=0.5,
                          help='Reflexivity threshold for entry/exit')
        parser.add_argument('--friction', type=float, default=0.02,
                          help='Friction parameter')
        parser.add_argument('--gamma', type=float, default=1.0,
                          help='Max gamma')
        parser.add_argument('--base_a', type=float, default=0.01,
                          help='Gamma bowl curvature')
        parser.add_argument('--dt', type=float, default=0.1,
                          help='Time step')
        parser.add_argument('--noise', type=float, default=0.05,
                          help='Noise level')
        parser.add_argument('--gamma_shift', type=float, default=0.02,
                          help='Gamma hill shift rate')
        parser.add_argument('--no-plot', action='store_true',
                          help='Run without animation (just generate report)')
        parser.add_argument('--savefig', action='store_true',
                          help='Save the final figure as PNG')
        parser.add_argument('--html', action='store_true',
                          help='Export the final figure and trade log as HTML')
        args = parser.parse_args()
        cli = ReflexivityGammaBowlV2(
            ticker=args.ticker,
            base_a=args.base_a,
            max_gamma=args.gamma,
            mass=1.0,
            dt=args.dt,
            friction=args.friction,
            noise_level=args.noise,
            gamma_shift_rate=args.gamma_shift,
            reflexivity_threshold=args.threshold
        )
        if not args.no_plot:
            cli.animate(frames=args.frames, savefig=args.savefig, html_export=args.html)
        cli.generate_report()

if __name__ == "__main__":
    ReflexivityGammaBowlV2.cli() 