import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI client
client = OpenAI()

# Default panel of thinkers
DEFAULT_PANEL = [
    "Adam Smith",
    "Theophrastus",
    "Jim Simons",
    "Albert Einstein",
    "Isaac Newton"
]

# Path to debate prompt template (relative to this file)
DEBATE_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "debate.json")
# Directory to save debate logs (relative to this file)
DEBATE_LOG_DIR = os.path.join(os.path.dirname(__file__), "log", "debates")
os.makedirs(DEBATE_LOG_DIR, exist_ok=True)

def generate_debate(context_dict, tone="Analytical", panel=None):
    """
    Generate a debate transcript among a panel of thinkers using the OpenAI API.
    context_dict: dict with keys like spot, gamma_flip, curvature, reflexivity, etc.
    tone: Debate tone (Analytical, Satirical, Historical, AI-Augmented)
    panel: List of panelist names (default: DEFAULT_PANEL)
    Returns: debate transcript as string
    """
    if panel is None:
        panel = DEFAULT_PANEL
    # Load prompt template
    with open(DEBATE_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = json.load(f)["prompt"]
    # Format the prompt
    context_lines = [f"{k}: {v}" for k, v in context_dict.items()]
    context_str = "\n".join(context_lines)
    panel_str = ", ".join(panel)
    prompt = prompt_template.format(
        context=context_str,
        panel=panel_str,
        tone=tone
    )
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a panel of renowned thinkers debating market reflexivity."},
            {"role": "user", "content": prompt}
        ]
    )
    transcript = response.choices[0].message.content
    # Save transcript to log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(DEBATE_LOG_DIR, f"debate_{timestamp}.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return transcript 