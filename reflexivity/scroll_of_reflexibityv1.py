import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from data_loader import load_stock_data

# --- Load stock data ---
spy_df = load_stock_data("SPY")
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_df['Close/Last'] = spy_df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
spy_df = spy_df.sort_values("Date").reset_index(drop=True)

# --- Simulation Parameters ---
base_a = 0.01
max_gamma = 1.0
mass = 1.0
dt = 0.1
friction = 0.02
noise_level = 0.05
gamma_shift_rate = 0.02
reflexivity_threshold = 0.5

# --- Initial State ---
x = 600.0  # Starting spot price
v = 0.0
x0 = 600.0  # Initial gamma hill center
reflexivity_score = 0.0
prev_reflexivity = 0.0
divergence_delta = 0.0
reflexivity_history = deque(maxlen=300)
divergence_history = deque(maxlen=300)
time_points = deque(maxlen=300)

# --- Plot Setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

x_vals = np.linspace(590, 610, 400)
line, = ax1.plot([], [], label="Gamma Curve")
point, = ax1.plot([], [], 'ro', markersize=8, label="Spot Price")
ax1.set_xlim(590, 610)
ax1.set_ylim(-0.5, 1.2)
ax1.set_title("Gamma Reflexivity Bowl")
ax1.set_xlabel("Price")
ax1.set_ylabel("Gamma Exposure")
ax1.legend()

line2, = ax2.plot([], [], color="green", label="Divergence Delta (dx/dt)")
ax2.axhline(y=0.5, color="red", linestyle="--", label="Instability Threshold")
ax2.set_xlim(0, 300)
ax2.set_ylim(-2, 2)
ax2.set_title("Divergence Delta Tracker")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Delta")
ax2.legend()

# --- Animation Update ---
def update(frame):
    global x, v, x0, reflexivity_score, prev_reflexivity, divergence_delta

    # Shift hill toward marble
    x0 += gamma_shift_rate * (x - x0)

    # Gamma bowl shape
    gamma_curve = -base_a * (x_vals - x0) ** 2 + max_gamma
    line.set_data(x_vals, gamma_curve)

    # Motion update
    force = 2 * base_a * (x - x0)
    noise = np.random.normal(0, noise_level)
    acceleration = (-force + noise - friction * v) / mass
    v += acceleration * dt
    x += v * dt

    y = -base_a * (x - x0) ** 2 + max_gamma
    point.set_data([x], [y])

    # Reflexivity and divergence
    prev_reflexivity = reflexivity_score
    reflexivity_score = abs(x - x0)
    divergence_delta = (reflexivity_score - prev_reflexivity) / dt

    reflexivity_history.append(reflexivity_score)
    divergence_history.append(divergence_delta)
    time_points.append(frame)

    line2.set_data(range(len(divergence_history)), divergence_history)

    # Optional alert
    if divergence_delta > reflexivity_threshold:
        print(f"[⚠️] Reflexivity Surge at Frame {frame} | Delta: {divergence_delta:.2f}")

    return line, point, line2

ani = FuncAnimation(fig, update, frames=300, interval=50, blit=True)
plt.show()
