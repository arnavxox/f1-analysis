from dash import dcc, html
import dash_bootstrap_components as dbc

def create_race_selector():
    return dbc.Card([
        dbc.CardHeader("Select Race", className="bg-primary text-white"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Season:"),
                    dcc.Dropdown(
                        id='year-selector',
                        options=[{'label': str(y), 'value': y} for y in range(2018, 2025)],
                        value=2023,
                        clearable=False,
                        style={"color": "black"}  # Fix text color issue
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Grand Prix:"),
                    dcc.Dropdown(
                        id='race-selector',
                        clearable=False,
                        style={"color": "black"}  # Fix text color issue
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Load Data", 
                              id='load-data-button',
                              color="primary", 
                              className="mt-3 w-100")
                ], width=12)
            ])
        ])
    ])
