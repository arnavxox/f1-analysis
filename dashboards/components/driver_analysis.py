from dash import dcc, html
import dash_bootstrap_components as dbc

def create_driver_analysis():
    return dbc.Card([
        dbc.CardHeader("Driver Comparison", className="bg-primary text-white"),
        dbc.CardBody([
            html.P("Select drivers to compare telemetry and performance", className="text-muted mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Drivers:"),
                    dcc.Dropdown(
                        id='driver-selector',
                        multi=True,
                        placeholder="Select drivers to compare",
                        style={"color": "black"}  # Fix text color issue
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Analysis Type:", className="mt-3"),
                    dbc.RadioItems(
                        id='analysis-type-selector',
                        options=[
                            {'label': 'Lap Time Comparison', 'value': 'lap_time'},
                            {'label': 'Telemetry Comparison', 'value': 'telemetry'},
                            {'label': 'Tire Strategy', 'value': 'strategy'}
                        ],
                        value='lap_time',
                        inline=True
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='driver-comparison-graph')
                ], width=12, className="mt-3")
            ])
        ])
    ])
