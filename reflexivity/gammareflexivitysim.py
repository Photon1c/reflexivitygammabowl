"""
Gamma Reflexivity Simulator

This script runs a real-time animated simulation of a price particle ("marble") on top of a gamma curve 
("hill") used in options market modeling. The simulation captures key reflexive behaviors:

- A gamma hill that dynamically reshapes based on dealer hedging behavior.
- Reflexivity score: distance between current price and gamma center.
- Instability logging when reflexivity exceeds a threshold.
- Dealer exhaustion mechanic that stops the hill from tracking the price.
- Sentiment slider that adds directional bias (bullish or bearish).
- Interactive controls for gamma pressure and locking the hill.

This tool is meant to illustrate how reflexivity and feedback loops manifest in market microstructure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, CheckButtons
from gamma_hill_simulator import GammaHillSimulator

# Run the modular simulator
sim = GammaHillSimulator()
sim.run()
sim.export_log()
