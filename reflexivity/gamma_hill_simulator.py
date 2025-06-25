import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, CheckButtons

class GammaHillSimulator:
    def __init__(self):
        self.base_a = 0.01
        self.max_gamma = 1.0
        self.mass = 1.0
        self.dt = 0.1
        self.friction = 0.02
        self.noise_level = 0.05
        self.gamma_shift_rate = 0.02
        self.reflexivity_threshold = 0.5
        self.dealer_strength = 1.0
        self.dealer_decay = 0.002
        self.sentiment_bias = 0.0
        self.default_spot = 600

        self.x = self.default_spot
        self.v = 0.0
        self.x0 = self.default_spot
        self.hill_locked = False
        self.reflexivity_score = 0.0
        self.instability_logged = False
        self.instability_start_frame = None
        self.frame_count = 0
        self.reflexivity_log = []

        self.shock_frame = 180  # simulate news shock at this frame

        self.setup_plot()

    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.45)

        self.x_vals = np.linspace(self.default_spot - 20, self.default_spot + 20, 400)
        self.line, = self.ax.plot([], [], label="Gamma Curve")
        self.point, = self.ax.plot([], [], 'ro', markersize=8, label="Marble")
        self.ref_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes,
                                     fontsize=10, verticalalignment='top',
                                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        self.ax.set_xlim(self.default_spot - 20, self.default_spot + 20)
        self.ax.set_ylim(-0.5, 1.2)
        self.ax.set_title("Reflexive Gamma Hill with Stress Events")
        self.ax.set_xlabel("Stock Price (Strike Axis)")
        self.ax.set_ylabel("Gamma Exposure")
        self.ax.legend()

        self.ax_gamma = plt.axes([0.15, 0.3, 0.65, 0.03])
        self.slider_gamma = Slider(self.ax_gamma, 'Gamma Pressure', 0.001, 0.05, valinit=self.base_a)

        self.ax_check = plt.axes([0.8, 0.12, 0.15, 0.1])
        self.check = CheckButtons(self.ax_check, ['Lock Hill'], [False])

        self.ax_sentiment = plt.axes([0.15, 0.23, 0.65, 0.03])
        self.slider_sentiment = Slider(self.ax_sentiment, 'Sentiment Tilt', -0.1, 0.1, valinit=0.0)

    def compute_reflexivity(self):
        return abs(self.x - self.x0)

    def log_instability(self):
        print(f"[!] Instability Detected at frame {self.frame_count}")
        print(f"    Marble Position: {self.x:.2f}")
        print(f"    Hill Center (x0): {self.x0:.2f}")
        print(f"    Reflexivity Score: {self.reflexivity_score:.2f}")
        return True

    def print_dashboard(self):
        print(f"[Frame {self.frame_count}]")
        print(f"  ↳ Marble Price: {self.x:.2f}")
        print(f"  ↳ Gamma Hill Peak: {self.x0:.2f}")
        print(f"  ↳ Reflexivity Score: {self.reflexivity_score:.2f}")
        print(f"  ↳ Dealer Strength: {self.dealer_strength:.2f}")
        print(f"  ↳ Hill Locked: {self.hill_locked}")
        print("-" * 60)

    def update(self, frame):
        self.frame_count += 1
        a = self.slider_gamma.val
        self.hill_locked = self.check.get_status()[0]
        self.sentiment_bias = self.slider_sentiment.val

        if not self.hill_locked and self.dealer_strength > 0:
            self.x0 += self.gamma_shift_rate * (self.x - self.x0)
            self.dealer_strength -= self.dealer_decay
        if self.dealer_strength <= 0:
            self.hill_locked = True

        gamma_curve = -a * (self.x_vals - self.x0)**2 + self.max_gamma
        self.line.set_data(self.x_vals, gamma_curve)

        force = 2 * a * (self.x - self.x0)
        noise = np.random.normal(0, self.noise_level)

        # Panic behavior if dealer exhausted and reflexivity is high
        if self.dealer_strength <= 0 and self.reflexivity_score > self.reflexivity_threshold:
            noise += np.random.normal(0, 0.2)

        # Simulated external shock
        if self.frame_count == self.shock_frame:
            print("[⚡] External Shock: Major news event triggered.")
            noise += np.random.normal(0, 0.5)

        sentiment_push = self.sentiment_bias
        acceleration = (-force + noise + sentiment_push - self.friction * self.v) / self.mass
        self.v += acceleration * self.dt
        self.x += self.v * self.dt

        y = -a * (self.x - self.x0)**2 + self.max_gamma
        # Change color if unstable
        if self.reflexivity_score > self.reflexivity_threshold:
            self.point.set_color('orange')
        else:
            self.point.set_color('red')
        self.point.set_data([self.x], [y])

        self.reflexivity_score = self.compute_reflexivity()
        self.ref_text.set_text(
            f"Reflexivity Score: {self.reflexivity_score:.2f}\\nDealer Strength: {self.dealer_strength:.2f}")

        # Log data
        self.reflexivity_log.append({
            "frame": self.frame_count,
            "marble": self.x,
            "hill": self.x0,
            "reflexivity": self.reflexivity_score,
            "dealer_strength": self.dealer_strength
        })

        if self.reflexivity_score > self.reflexivity_threshold and not self.instability_logged:
            self.instability_logged = self.log_instability()
            self.instability_start_frame = self.frame_count
        elif self.reflexivity_score <= self.reflexivity_threshold:
            self.instability_logged = False

        if self.frame_count % 25 == 0:
            self.print_dashboard()

        return self.line, self.point, self.ref_text

    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, frames=1000, interval=50, blit=True)
        plt.show()

    def export_log(self, path='reflexivity_log.csv'):
        import pandas as pd
        df = pd.DataFrame(self.reflexivity_log)
        df.to_csv(path, index=False)
        print(f"[✓] Reflexivity log saved to: {path}")