import sys
import os
sys.path.append(os.path.dirname(__file__))
from debate_engine import generate_debate

if __name__ == "__main__":
    # Example context for testing
    context = {
        "spot": 606.78,
        "gamma_flip": 607,
        "curvature": 0.1234,
        "dealer_strength": 0.85,
        "reflexivity_score": 2.5,
        "signal": "🐍 Coiled Volatility: Watch for Breakout"
    }
    print("Generating debate transcript...\n")
    transcript = generate_debate(context, tone="Analytical")
    print(transcript) 