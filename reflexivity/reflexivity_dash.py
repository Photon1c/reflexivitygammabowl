"""
reflexivity_dash.py

Dash UI Playground for Reflexivity Gamma Bowl v4.5+
- Dropdowns: Ticker, Expiry (populated from manifest)
- Sliders: Gamma Pressure, Sentiment Tilt
- Multi-panel chart area: Gamma Bowl, Curvature, Dealer Gradient
- Text Card: 'Narrator Signal' (LLM-pluggable)
- Playback controls: Step, Play/Pause, Export (placeholders)
- Designed for LLM agent integration and future extensibility
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import json
import os
from pathlib import Path
from debate_engine import generate_debate

# --- Load manifest ---
MANIFEST_PATH = Path(__file__).parent / 'reflexivity_manifest.json'
if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        MANIFEST = json.load(f)
else:
    MANIFEST = {}

# --- LLM Narrator Stub ---
def narrate_reflexivity_state(metrics: dict) -> str:
    """
    Inputs: { 'reflexivity_score': X, 'dealer_strength': Y, 'spot': Z, 'curvature': W }
    Returns: A string strategy narration (for now, hardcoded prompt).
    TODO: Swap for OpenAI or local LLM call.
    """
    if not metrics:
        return "No metrics available."
    return f"📈 Spot: {metrics.get('spot', '?')}, Curvature: {metrics.get('curvature', '?')}, Dealer: {metrics.get('dealer_strength', '?')}, Reflexivity: {metrics.get('reflexivity_score', '?')}\n\nStrategy: {metrics.get('signal', 'No signal.')}"

# --- App Layout ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

ticker_options = [{'label': t, 'value': t} for t in MANIFEST.keys()]
def get_expiry_options(ticker):
    return [{'label': d, 'value': d} for d in sorted(MANIFEST.get(ticker, {}).keys())]

def get_metrics(ticker, expiry):
    entry = MANIFEST.get(ticker, {}).get(expiry, {})
    if entry and os.path.exists(entry['metrics']):
        with open(entry['metrics'], 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_plot_path(ticker, expiry):
    entry = MANIFEST.get(ticker, {}).get(expiry, {})
    return entry.get('plot') if entry else None

app.layout = dbc.Container([
    html.H2("Reflexivity Gamma Bowl Playground v4.5"),
    dbc.Row([
        dbc.Col([
            html.Label("Ticker"),
            dcc.Dropdown(id='ticker-dropdown', options=ticker_options, value=ticker_options[0]['value'] if ticker_options else None),
        ], width=2),
        dbc.Col([
            html.Label("Expiry"),
            dcc.Dropdown(id='expiry-dropdown'),
        ], width=2),
        dbc.Col([
            html.Label("Gamma Pressure"),
            dcc.Slider(id='gamma-slider', min=0.001, max=0.05, step=0.001, value=0.01, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        ], width=3),
        dbc.Col([
            html.Label("Sentiment Tilt"),
            dcc.Slider(id='sentiment-slider', min=-0.1, max=0.1, step=0.01, value=0.0, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        ], width=3),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.Div(id='chart-area'),
            # TODO: Add multi-panel charts (gamma bowl, curvature, dealer gradient)
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Narrator Signal (LLM-Ready)"),
                dbc.CardBody([
                    html.Pre(id='narrator-signal', style={"whiteSpace": "pre-line"})
                ])
            ]),
            html.Br(),
            # --- Debate Button and Output ---
            dbc.Button("💬 Generate Thinker Debate", id="debate-button", color="primary", className="mb-2"),
            dcc.Textarea(id="debate-output", style={"width": "100%", "height": "300px", "marginBottom": "1rem"}),
            # Playback controls (placeholders)
            dbc.ButtonGroup([
                dbc.Button("⏮️ Step Back", id='step-back', n_clicks=0),
                dbc.Button("▶️ Play", id='play', n_clicks=0),
                dbc.Button("⏭️ Step Forward", id='step-forward', n_clicks=0),
                dbc.Button("💾 Export Frame", id='export-frame', n_clicks=0),
            ], size="md"),
        ], width=4),
    ]),
    html.Hr(),
    html.Div("Future: LLM agent, narrator.py, and data_router.py integration hooks go here.", style={"color": "#888"}),
], fluid=True)

# --- Callbacks ---
@app.callback(
    Output('expiry-dropdown', 'options'),
    Output('expiry-dropdown', 'value'),
    Input('ticker-dropdown', 'value')
)
def update_expiry_dropdown(ticker):
    options = get_expiry_options(ticker)
    value = options[0]['value'] if options else None
    return options, value

@app.callback(
    Output('chart-area', 'children'),
    Output('narrator-signal', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('expiry-dropdown', 'value'),
    Input('gamma-slider', 'value'),
    Input('sentiment-slider', 'value'),
)
def update_chart_and_narrator(ticker, expiry, gamma, sentiment):
    metrics = get_metrics(ticker, expiry)
    plot_path = get_plot_path(ticker, expiry)
    chart = []
    if plot_path and os.path.exists(plot_path):
        chart.append(html.Img(src='data:image/png;base64,' + encode_image(plot_path), style={"width": "100%", "border": "1px solid #ccc"}))
    else:
        chart.append(html.Div("No chart available for this selection."))
    # TODO: In the future, update chart based on gamma/sentiment sliders (LLM/data_router integration)
    narration = narrate_reflexivity_state(metrics)
    return chart, narration

def encode_image(image_path):
    import base64
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# --- Debate Callback ---
@app.callback(
    Output("debate-output", "value"),
    Input("debate-button", "n_clicks"),
    State('ticker-dropdown', 'value'),
    State('expiry-dropdown', 'value'),
)
def debate_callback(n_clicks, ticker, expiry):
    if not n_clicks:
        return ""
    metrics = get_metrics(ticker, expiry)
    if not metrics:
        return "No metrics available for debate."
    return generate_debate(metrics)

if __name__ == "__main__":
    app.run(debug=True) 