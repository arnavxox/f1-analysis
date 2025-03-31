import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import fastf1 as ff1
import plotly.express as px
import plotly.graph_objects as go
from modules import F1DataCache, TelemetryAnalyzer, StrategyAnalyzer, VisualizationEngine, F1WikiScraper

def create_race_dashboard(cache):
    """
    Creates a complete F1 race analysis dashboard application
    
    Parameters:
    -----------
    cache : F1DataCache
        The cache instance for storing and retrieving data
        
    Returns:
    --------
    app : dash.Dash
        The configured Dash application
    """
    # Initialize Dash app with Bootstrap dark theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    
    # Define app layout
    app.layout = dbc.Container([
        # Header row
        dbc.Row([
            dbc.Col([
                html.H1("F1 Race Analysis Dashboard", className="text-center my-4"),
                html.P("Comprehensive analysis of Formula 1 race data", 
                       className="text-center text-muted mb-4"),
                dcc.Store(id='session-data'),
                html.Div(id='hidden-div', style={'display': 'none'})
            ], width=12)
        ]),
        
        # Race selection controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Race Selection", className="bg-primary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Season:", style={"color": "black"}),
                                dcc.Dropdown(
                                    id='year-dropdown',
                                    options=[{'label': str(y), 'value': y} for y in range(2018, 2025)],
                                    value=2023,
                                    clearable=False,
                                    style={"color": "black"}
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Grand Prix:", style={"color": "black"}),
                                dcc.Dropdown(
                                    id='race-dropdown',
                                    placeholder="Select a race",
                                    clearable=False,
                                    style={"color": "black"}
                                )
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Load Race Data", 
                                          id='load-button', 
                                          color="primary", 
                                          className="mt-3 w-100")
                            ], width=12)
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Loading indicator
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-indicator",
                    type="circle",
                    children=[html.Div(id="loading-output")]
                )
            ], width=12, className="text-center")
        ]),
        
        # Race information
        dbc.Row([
            dbc.Col([
                html.Div(id="race-info-container")
            ], width=12)
        ], className="mb-4"),
        
        #Race wiki
        dbc.Row([
            dbc.Col([
                html.Div(id="wiki-info-container")
            ], width=12)
        ], className="mb-4"),

        
        # Analysis tabs 
        dbc.Tabs([
            dbc.Tab(label="Telemetry Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Driver Analysis", className="mt-3 mb-2"),
                        html.P("Select a driver to view detailed telemetry data", className="text-muted"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Primary Driver:", style={"color": "black"}),
                                dcc.Dropdown(
                                    id='telemetry-driver-dropdown',
                                    placeholder="Select driver",
                                    style={"color": "black"}
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Lap Selection:", style={"color": "black"}),
                                dcc.Dropdown(
                                    id='lap-selection-dropdown',
                                    options=[{'label': 'Fastest Lap', 'value': 'fastest'}],
                                    value='fastest',
                                    style={"color": "black"}
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Compare with:", style={"color": "black"}),
                                dcc.Dropdown(
                                    id='compare-driver-dropdown',
                                    placeholder="Select driver for comparison (optional)",
                                    style={"color": "black"}
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Compare Lap:", style={"color": "black"}),
                                dcc.Dropdown(
                                    id='compare-lap-dropdown',
                                    options=[{'label': 'Fastest Lap', 'value': 'fastest'}],
                                    value='fastest',
                                    disabled=True,
                                    style={"color": "black"}
                                )
                            ], width=6)
                        ])
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='telemetry-graph')
                    ], width=12)
                ])
            ]),
            dbc.Tab(label="Strategy Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Tire Strategy", className="mt-3 mb-2"),
                        dbc.Tabs([
                            dbc.Tab(label="Tire Timeline", children=[
                                dcc.Graph(id='tire-timeline-graph')
                            ]),
                            dbc.Tab(label="Lap Time by Compound", children=[
                                dcc.Graph(id='strategy-graph')
                            ]),
                            dbc.Tab(label="Pit Stops", children=[
                                dcc.Graph(id='pitstop-graph')
                            ])
                        ])
                    ], width=12)
                ])
            ]),
            dbc.Tab(label="Race Progress", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Position Changes", className="mt-3 mb-2"),
                        dcc.Graph(id='position-changes-graph')
                    ], width=12)
                ])
            ]),
            dbc.Tab(label="Lap Time Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Lap Time Distribution", className="mt-3 mb-2"),
                        dcc.Graph(id='laptimes-graph')
                    ], width=12)
                ])
            ])
        ], className="mt-4"),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P("F1 Analysis Tool by arnavxox", className="text-center text-muted")
            ], width=12)
        ], className="mt-4")
    ], fluid=True)

    # Register callbacks
    @app.callback(
        Output('race-dropdown', 'options'),
        Input('year-dropdown', 'value')
    )
    def update_race_options(year):
        """Update race dropdown based on selected year"""
        try:
            schedule = ff1.get_event_schedule(year)
            return [{'label': e.EventName, 'value': e.EventName} 
                   for _, e in schedule.iterrows()]
        except Exception as e:
            print(f"Error loading race options: {e}")
            return []

    @app.callback(
        Output('race-dropdown', 'value'),
        Input('race-dropdown', 'options')
    )
    def set_default_race(options):
        """Set default race when options change"""
        if options and len(options) > 0:
            return options[0]['value']
        return None

    @app.callback(
        [Output('session-data', 'data'),
         Output('loading-output', 'children')],
        Input('load-button', 'n_clicks'),
        [State('year-dropdown', 'value'),
         State('race-dropdown', 'value')]
    )
    def load_race_data(n_clicks, year, race):
        """Load race data when user clicks the load button"""
        if n_clicks is None or not race:
            return dash.no_update, dash.no_update
            
        try:
            # Enable caching to speed up repeated data accesses
            ff1.Cache.enable_cache('f1_cache')
            
            # Load session data
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            # Return session data for use in other callbacks
            return {
                'year': year,
                'race': race,
                'drivers': session.drivers.tolist() if hasattr(session.drivers, 'tolist') else list(session.drivers),
                'loaded': True
            }, ""
        except Exception as e:
            print(f"Error loading race data: {e}")
            return dash.no_update, f"Error: Could not load race data - {str(e)}"

    @app.callback(
        Output('telemetry-driver-dropdown', 'options'),
        Output('compare-driver-dropdown', 'options'),
        Input('session-data', 'data')
    )
    def update_driver_options(session_data):
        if not session_data or not session_data.get('loaded'):
            return [], []
            
        try:
            session = ff1.get_session(session_data['year'], session_data['race'], 'R')
            session.load()
            
            # Get driver information from results directly
            options = []
            for _, driver_data in session.results.iterrows():
                driver = driver_data['DriverNumber']
                abbr = driver_data['Abbreviation']
                options.append({'label': f"{abbr} ({driver})", 'value': driver})
            
            # Add empty option for comparison dropdown
            compare_options = [{'label': 'None', 'value': ''}] + options
            
            return options, compare_options
        except Exception as e:
            print(f"Error updating driver options: {e}")
            return [], []

    @app.callback(
        Output('telemetry-driver-dropdown', 'value'),
        Input('telemetry-driver-dropdown', 'options')
    )
    def set_default_driver(options):
        """Set default driver when options change"""
        if options and len(options) > 0:
            return options[0]['value']
        return None

    @app.callback(
        Output('lap-selection-dropdown', 'options'),
        Input('session-data', 'data'),
        Input('telemetry-driver-dropdown', 'value')
    )
    def update_lap_options(session_data, driver):
        """Update lap selection dropdown based on selected driver"""
        if not session_data or not session_data.get('loaded') or not driver:
            return [{'label': 'Fastest Lap', 'value': 'fastest'}]
            
        try:
            # Get available laps for the selected driver
            analyzer = TelemetryAnalyzer(cache)
            lap_options = analyzer.get_available_laps(
                session_data['year'], 
                session_data['race'], 
                driver
            )
            return lap_options
        except Exception as e:
            print(f"Error updating lap options: {e}")
            return [{'label': 'Fastest Lap', 'value': 'fastest'}]

    @app.callback(
        Output('compare-lap-dropdown', 'options'),
        Output('compare-lap-dropdown', 'disabled'),
        Input('session-data', 'data'),
        Input('compare-driver-dropdown', 'value')
    )
    def update_compare_lap_options(session_data, compare_driver):
        """Update comparison lap dropdown based on selected comparison driver"""
        if not session_data or not session_data.get('loaded') or not compare_driver:
            return [{'label': 'Fastest Lap', 'value': 'fastest'}], True
            
        try:
            # Get available laps for the comparison driver
            analyzer = TelemetryAnalyzer(cache)
            lap_options = analyzer.get_available_laps(
                session_data['year'], 
                session_data['race'], 
                compare_driver
            )
            return lap_options, False
        except Exception as e:
            print(f"Error updating comparison lap options: {e}")
            return [{'label': 'Fastest Lap', 'value': 'fastest'}], True

    @app.callback(
        Output('race-info-container', 'children'),
        Input('session-data', 'data')
    )
    def update_race_info(session_data):
        """Update race information display"""
        if not session_data or not session_data.get('loaded'):
            return html.Div()
            
        try:
            year = session_data['year']
            race = session_data['race']
            
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            winner = session.results.iloc[0]
            winner_abbr = winner['Abbreviation']
            
            # Try different ways to get circuit name
            circuit_name = "Unknown Circuit"
            try:
                circuit_name = session.event['CircuitName']
            except:
                try:
                    circuit_name = session.event.CircuitName
                except:
                    try:
                        circuit_name = session.event_name
                    except:
                        pass
            
            return dbc.Card([
                dbc.CardHeader(f"{race} Grand Prix {year}", className="bg-success text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Race Winner"),
                            html.P(f"{winner_abbr} ({winner['TeamName']})", className="lead")
                        ], width=4),
                        dbc.Col([
                            html.H5("Laps"),
                            html.P(f"{len(session.laps['LapNumber'].unique())}", className="lead")
                        ], width=4),
                        dbc.Col([
                            html.H5("Circuit"),
                            html.P(f"{circuit_name}", className="lead")
                        ], width=4)
                    ])
                ])
            ])
        except Exception as e:
            print(f"Error updating race info: {e}")
            return html.Div(f"Error loading race information: {str(e)}")


    @app.callback(
        Output('telemetry-graph', 'figure'),
        [Input('session-data', 'data'),
        Input('telemetry-driver-dropdown', 'value'),
        Input('lap-selection-dropdown', 'value'),
        Input('compare-driver-dropdown', 'value'),
        Input('compare-lap-dropdown', 'value')]
    )
    def update_telemetry(session_data, driver, lap_selection, compare_driver, compare_lap):
        # Add a debug print
        print(f"Updating telemetry: driver={driver}, lap={lap_selection}, compare={compare_driver}, compare_lap={compare_lap}")
        
        if not session_data or not session_data.get('loaded') or not driver:
            return go.Figure().update_layout(title="No data available")
                
        try:
            year = session_data['year']
            race = session_data['race']
            
            # Get telemetry data
            analyzer = TelemetryAnalyzer(cache)
            fig = analyzer.get_speed_trace(
                year, race, driver, 
                lap_selection=lap_selection,
                compare_driver=compare_driver if compare_driver else None,
                compare_lap=compare_lap if compare_driver and compare_lap else None
            )
            return fig
        except Exception as e:
            print(f"Error updating telemetry: {e}")
            fig = go.Figure()
            fig.update_layout(title=f"Error loading telemetry data: {str(e)}")
            return fig


    @app.callback(
        Output('strategy-graph', 'figure'),
        Input('session-data', 'data')
    )
    def update_strategy(session_data):
        """Update strategy graph with tire compounds"""
        if not session_data or not session_data.get('loaded'):
            return go.Figure().update_layout(title="No data available")
            
        try:
            year = session_data['year']
            race = session_data['race']
            
            # Get strategy data
            analyzer = StrategyAnalyzer(cache)
            fig = analyzer.plot_tire_strategy(year, race)
            return fig
        except Exception as e:
            print(f"Error updating strategy: {e}")
            fig = go.Figure()
            fig.update_layout(title=f"Error loading strategy data: {str(e)}")
            return fig

    @app.callback(
        Output('tire-timeline-graph', 'figure'),
        Input('session-data', 'data')
    )
    def update_tire_timeline(session_data):
        """Update tire strategy timeline visualization"""
        if not session_data or not session_data.get('loaded'):
            return go.Figure().update_layout(title="No data available")
            
        try:
            year = session_data['year']
            race = session_data['race']
            
            # Get tire timeline visualization
            analyzer = StrategyAnalyzer(cache)
            fig = analyzer.visualize_tire_strategy(year, race)
            return fig
        except Exception as e:
            print(f"Error updating tire timeline: {e}")
            fig = go.Figure()
            fig.update_layout(title=f"Error loading tire strategy data: {str(e)}")
            return fig

    # @app.callback(
    #     Output('pitstop-graph', 'figure'),
    #     Input('session-data', 'data')
    # )
    # def update_pitstop_analysis(session_data):
    #     """Update pit stop performance visualization"""
    #     if not session_data or not session_data.get('loaded'):
    #         return go.Figure().update_layout(title="No data available")
            
    #     try:
    #         year = session_data['year']
    #         race = session_data['race']
            
    #         # Get pit stop analysis
    #         analyzer = StrategyAnalyzer(cache)
    #         pit_figs = analyzer.plot_pit_stop_performance(year, race)
            
    #         # Return the first figure (driver duration)
    #         if 'driver_duration' in pit_figs:
    #             return pit_figs['driver_duration']
    #         return pit_figs.get(list(pit_figs.keys())[0])
    #     except Exception as e:
    #         print(f"Error updating pit stop analysis: {e}")
    #         fig = go.Figure()
    #         fig.update_layout(title=f"Error loading pit stop data: {str(e)}")
    #         return fig

    @app.callback(
        Output('position-changes-graph', 'figure'),
        Input('session-data', 'data')
    )
    def update_position_changes(session_data):
        """Update position changes graph"""
        if not session_data or not session_data.get('loaded'):
            return go.Figure().update_layout(title="No data available")
            
        try:
            year = session_data['year']
            race = session_data['race']
            
            # Load session
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            # Create visualization
            viz = VisualizationEngine()
            fig = viz.create_position_changes(session)
            return fig
        except Exception as e:
            print(f"Error updating position changes: {e}")
            fig = go.Figure()
            fig.update_layout(title=f"Error loading position data: {str(e)}")
            return fig

    @app.callback(
        Output('laptimes-graph', 'figure'),
        Input('session-data', 'data')
    )
    def update_laptimes(session_data):
        """Update lap times distribution graph"""
        if not session_data or not session_data.get('loaded'):
            return go.Figure().update_layout(title="No data available")
            
        try:
            year = session_data['year']
            race = session_data['race']
            
            # Load session
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            # Create visualization
            viz = VisualizationEngine()
            fig = viz.create_pace_distribution(session)
            return fig
        except Exception as e:
            print(f"Error updating lap times: {e}")
            fig = go.Figure()
            fig.update_layout(title=f"Error loading lap time data: {str(e)}")
            return fig
        
        
    @app.callback(
        Output('wiki-info-container', 'children'),
        Input('session-data', 'data')
    )
    def update_wiki_info(session_data):
        """Update Wikipedia information"""
        if not session_data or not session_data.get('loaded'):
            return html.Div()
            
        try:
            year = session_data['year']
            race = session_data['race']
            
            # Get wiki data
            scraper = F1WikiScraper('./wiki_cache')
            race_info = scraper.get_race_summary(year, race)
            
            # Create info card
            info_rows = []
            for key, value in race_info.items():
                if key in ['Date', 'Circuit', 'Winner', 'Pole position', 'Fastest lap']:
                    info_rows.append(
                        html.Tr([
                            html.Td(key, className="font-weight-bold"),
                            html.Td(value)
                        ])
                    )
            
            return dbc.Card([
                dbc.CardHeader("Race Information", className="bg-info text-white"),
                dbc.CardBody([
                    html.Table(html.Tbody(info_rows), className="table table-sm")
                ])
            ])
        except Exception as e:
            print(f"Error getting wiki info: {e}")
            return html.Div()


    return app
