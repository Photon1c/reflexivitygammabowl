"""
Reflexivity Gamma Bowl CLI (Original/Baseline Version)

Main Features:
- Command-line interface for simulating a simple gamma bowl reflexivity system.
- Loads historical stock data for a given ticker (default: SPY).
- Simulates a marble-in-bowl system with a moving gamma hill, using basic physics (curvature, friction, noise).
- Animates the gamma exposure curve and spot price as a 2-panel matplotlib animation.
- Tracks and logs entry/exit signals based on divergence delta (reflexivity threshold crossings).
- Outputs a trade log table after simulation.

How this version differs from v2/v3/v4:
- No option chain or expiry selection; only spot price and a synthetic gamma bowl.
- No interactive UI controls, dropdowns, or sliders.
- No dealer strength, instability overlays, or curvature tracking.
- No markdown export or multi-subplot analytics.
- Designed as a minimal, educational baseline for reflexivity simulation and signal generation.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from data_loader import load_stock_data

class ReflexivityGammaBowlCLI:
    def __init__(self, ticker='SPY', base_a=0.01, max_gamma=1.0, mass=1.0, dt=0.1, friction=0.02, noise_level=0.05, gamma_shift_rate=0.02, reflexivity_threshold=0.5):
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
        self._load_data()
        self._reset_state()

    def _load_data(self):
        self.df = load_stock_data(self.ticker)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)

    def _reset_state(self):
        self.x = float(self.df['Close/Last'].iloc[0]) if 'Close/Last' in self.df else 600.0
        self.v = 0.0
        self.x0 = self.x
        self.reflexivity_score = 0.0
        self.prev_reflexivity = 0.0
        self.divergence_delta = 0.0
        self.reflexivity_history = deque(maxlen=300)
        self.divergence_history = deque(maxlen=300)
        self.time_points = deque(maxlen=300)
        self.in_position = False

    def animate(self, frames=300, interval=50):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.4)
        x_vals = np.linspace(self.x - 10, self.x + 10, 400)
        line, = ax1.plot([], [], label="Gamma Curve")
        point, = ax1.plot([], [], 'ro', markersize=8, label="Spot Price")
        ax1.set_xlim(self.x - 10, self.x + 10)
        ax1.set_ylim(-0.5, 1.2)
        ax1.set_title("Gamma Reflexivity Bowl")
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Gamma Exposure")
        ax1.legend()
        line2, = ax2.plot([], [], color="green", label="Divergence Delta (dx/dt)")
        ax2.axhline(y=self.reflexivity_threshold, color="red", linestyle="--", label="Instability Threshold")
        ax2.set_xlim(0, frames)
        ax2.set_ylim(-2, 2)
        ax2.set_title("Divergence Delta Tracker")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Delta")
        ax2.legend()

        def update(frame):
            # Shift hill toward marble
            self.x0 += self.gamma_shift_rate * (self.x - self.x0)
            gamma_curve = -self.base_a * (x_vals - self.x0) ** 2 + self.max_gamma
            line.set_data(x_vals, gamma_curve)
            # Motion update
            force = 2 * self.base_a * (self.x - self.x0)
            noise = np.random.normal(0, self.noise_level)
            acceleration = (-force + noise - self.friction * self.v) / self.mass
            self.v += acceleration * self.dt
            self.x += self.v * self.dt
            y = -self.base_a * (self.x - self.x0) ** 2 + self.max_gamma
            point.set_data([self.x], [y])
            # Reflexivity and divergence
            self.prev_reflexivity = self.reflexivity_score
            self.reflexivity_score = abs(self.x - self.x0)
            self.divergence_delta = (self.reflexivity_score - self.prev_reflexivity) / self.dt
            self.reflexivity_history.append(self.reflexivity_score)
            self.divergence_history.append(self.divergence_delta)
            self.time_points.append(frame)
            line2.set_data(range(len(self.divergence_history)), self.divergence_history)
            # Entry/exit logic
            signal = None
            if not self.in_position and self.divergence_delta > self.reflexivity_threshold:
                signal = f"[ENTRY] Frame {frame} | Delta: {self.divergence_delta:.2f}"
                self.in_position = True
                self.trade_log.append({'frame': frame, 'type': 'ENTRY', 'delta': self.divergence_delta})
            elif self.in_position and self.divergence_delta < 0:
                signal = f"[EXIT] Frame {frame} | Delta: {self.divergence_delta:.2f}"
                self.in_position = False
                self.trade_log.append({'frame': frame, 'type': 'EXIT', 'delta': self.divergence_delta})
            if signal:
                print(signal)
            return line, point, line2

        ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        plt.show()

    def print_trade_log(self):
        if not self.trade_log:
            print("No trades logged.")
            return
        print("\nTrade Log:")
        print("| Frame | Type  | Delta  |")
        print("|-------|-------|--------|")
        for trade in self.trade_log:
            print(f"| {trade['frame']:5d} | {trade['type']:5s} | {trade['delta']:6.2f} |")

    @staticmethod
    def cli():
        parser = argparse.ArgumentParser(description="Reflexivity Gamma Bowl CLI - Targeting System for SPY")
        parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
        parser.add_argument('--frames', type=int, default=300, help='Number of animation frames')
        parser.add_argument('--threshold', type=float, default=0.5, help='Reflexivity threshold for entry/exit')
        parser.add_argument('--friction', type=float, default=0.02, help='Friction parameter')
        parser.add_argument('--gamma', type=float, default=1.0, help='Max gamma')
        parser.add_argument('--base_a', type=float, default=0.01, help='Gamma bowl curvature')
        parser.add_argument('--dt', type=float, default=0.1, help='Time step')
        parser.add_argument('--noise', type=float, default=0.05, help='Noise level')
        parser.add_argument('--gamma_shift', type=float, default=0.02, help='Gamma hill shift rate')
        parser.add_argument('--no-plot', action='store_true', help='Run without animation (just log signals)')
        args = parser.parse_args()
        cli = ReflexivityGammaBowlCLI(
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
            cli.animate(frames=args.frames)
        cli.print_trade_log()

if __name__ == "__main__":
    ReflexivityGammaBowlCLI.cli()
