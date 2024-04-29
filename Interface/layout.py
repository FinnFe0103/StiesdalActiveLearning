from dash import Dash, html, dcc
import dash_daq as daq
import pandas as pd

initial_sensors = [
    'foundation_origin xy FloaterOffset [m]',
    'foundation_origin Rxy FloaterTilt [deg]',
    'foundation_origin Rz FloaterYaw [deg]',
    'foundation_origin z FloaterHeave [m]',
    'foundation_origin Mooring GXY Resultant force [kN]',
    'MooringLine1 Effective tension Fairlead [kN]',
    'MooringLine2 Effective tension Fairlead [kN]',
    'MooringLine3 Effective tension Fairlead [kN]',
    'MooringLine4 Effective tension Fairlead [kN]',
    'MooringLine5 Effective tension Fairlead [kN]',
    'GE14-220 GXY acceleration [m/s^2]',
    'CEN_E3 Resultant bending moment ArcLength=2.72 [kN.m]',
]

def create_layout(app: Dash, sensors: pd.DataFrame) -> html.Div:
    return html.Div([
        html.H1(app.title, style={'textAlign': 'center'}),
        html.Div([
            html.Div(
                children=[
                    # Container for Training Steps Slider
                    html.Div([
                        html.Label('Training Steps', style={'textAlign': 'center'}),
                        dcc.Slider(
                            min=1,
                            max=20,
                            step=1,
                            value=10,
                            marks={i: str(i) for i in range(1, 21)},
                            id='training-steps-slider',
                        )
                    # Decreased left and right padding for the slider
                    ], style={'width': '100%', 'paddingLeft': '5%', 'paddingRight': '5%', 'marginBottom': '20px'}),  # Line Changed

                    # Container for Prediction Steps Slider
                    html.Div([
                        html.Label('Prediction Steps', style={'textAlign': 'center'}),
                        dcc.Slider(
                            min=1,
                            max=5,
                            step=1,
                            value=3,
                            marks={i: str(i) for i in range(1, 6)},
                            id='prediction-steps-slider',
                        )
                    # Decreased left and right padding for the slider
                    ], style={'width': '100%', 'paddingLeft': '5%', 'paddingRight': '5%', 'marginBottom': '20px'}),  # Line Changed

                    # Container for Sensor Selection Dropdown
                    html.Div([
                        html.Label('Sensor Selection', style={'textAlign': 'center'}),
                        dcc.Dropdown(
                            sensors['name'].unique(),
                            initial_sensors,
                            id='dropdown-selection',
                            multi=True,
                        )
                    # Decreased left and right padding for the dropdown
                    ], style={'width': '100%', 'paddingLeft': '5%', 'paddingRight': '5%'}),  # Line Changed

                ],
                style={
                    "width": "20%",
                    "height": "calc(100vh - 20px)",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "flex-start",
                    "borderRight": "2px solid #d6d6d6",
                    "padding": "10px",
                    "boxSizing": "border-box",
                }
            ),
            html.Div(
                dcc.Graph(id='graph-content'),
                style={"width": "80%", "height": "calc(100vh - 20px)"}
            ),
        ], style={"display": "flex", "height": "100vh", "marginTop": "20px"})
    ])
