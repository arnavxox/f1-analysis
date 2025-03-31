import fastf1 as ff1
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Tuple

class TelemetryAnalyzer:
    """Advanced telemetry analysis with circuit-aware processing"""
    
    def __init__(self, cache):
        self.cache = cache
        
    def get_available_laps(self, year: int, race: str, driver: str) -> List[Dict]:
        """Get a list of all available laps for the driver in the race"""
        try:
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            # Get driver abbreviation from results
            driver_abbr = None
            for _, driver_data in session.results.iterrows():
                if str(driver_data['DriverNumber']) == str(driver):
                    driver_abbr = driver_data['Abbreviation']
                    break
            
            display_name = driver_abbr if driver_abbr else driver
            
            # Get all laps for the driver
            driver_laps = session.laps.pick_drivers(driver)
            if driver_laps.empty:
                return [{'label': 'No laps available', 'value': -1}]
            
            # Create list of laps with their lap times
            lap_options = []
            
            # Add fastest lap option
            fastest_lap = driver_laps.pick_fastest()
            if not fastest_lap.empty:
                lap_options.append({
                    'label': f"Fastest Lap: {display_name} - Lap {fastest_lap['LapNumber']} ({fastest_lap['LapTime']})",
                    'value': f"{fastest_lap['LapNumber']}_fastest"
                })
            
            # Add all individual laps
            for _, lap in driver_laps.iterrows():
                lap_num = lap['LapNumber']
                lap_time = lap['LapTime']
                if pd.notnull(lap_time):
                    lap_options.append({
                        'label': f"Lap {lap_num} - {display_name} ({lap_time})",
                        'value': str(lap_num)
                    })
            
            return lap_options
        except Exception as e:
            print(f"Error getting available laps: {str(e)}")
            return [{'label': f"Error: {str(e)}", 'value': -1}]

    
    def get_driver_abbreviation(self, session, driver_number: str) -> str:
        """Get driver abbreviation from driver number"""
        try:
            # Access driver abbreviation from results
            for _, driver_data in session.results.iterrows():
                if str(driver_data['DriverNumber']) == str(driver_number):
                    return driver_data['Abbreviation']
            return driver_number  # Return original if not found
        except Exception:
            return driver_number  # Return original on error

    
    def get_lap_by_selection(self, session, driver: str, lap_selection: str) -> pd.Series:
        """Get lap data based on selection (fastest or specific lap number)"""
        driver_laps = session.laps.pick_drivers(driver)
        
        if lap_selection.endswith('_fastest'):
            return driver_laps.pick_fastest()
        else:
            try:
                lap_number = int(lap_selection)
                return driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
            except:
                # Fallback to fastest lap if something goes wrong
                return driver_laps.pick_fastest()
    
    def get_speed_trace(self, year: int, race: str, driver: str, 
                       lap_selection: str = 'fastest',
                       compare_driver: str = None,
                       compare_lap: str = None) -> go.Figure:
        """
        Generate annotated speed trace with corner markers
        
        Parameters:
        -----------
        year: int
            Season year
        race: str
            Race name
        driver: str
            Primary driver to analyze
        lap_selection: str
            Selected lap (lap number or 'fastest')
        compare_driver: str, optional
            Second driver to compare with
        compare_lap: str, optional
            Selected lap for comparison driver
            
        Returns:
        --------
        fig: go.Figure
            Plotly figure with telemetry data
        """
        cache_key = f"speed_trace_{year}_{race}_{driver}_{lap_selection}_{compare_driver}_{compare_lap}"
        cached = self.cache.get('telemetry', key=cache_key)
        if cached:
            return cached
        
        try:
            # Load session data
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            # Get driver abbreviations for better display
            driver_abbr = self.get_driver_abbreviation(session, driver)
            
            # Get lap data
            if lap_selection == 'fastest':
                lap = session.laps.pick_drivers(driver).pick_fastest()
            else:
                lap = self.get_lap_by_selection(session, driver, lap_selection)
            
            # Get telemetry data
            telemetry = lap.get_telemetry()
            circuit = session.get_circuit_info()
            
            # Create figure with multiple subplots for different metrics
            if compare_driver:
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=(
                        "Speed Comparison", 
                        "Throttle/Brake Comparison",
                        "Gear Shifts"
                    ),
                    row_heights=[0.4, 0.3, 0.3]
                )
            else:
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=(
                        "Speed Trace", 
                        "Throttle/Brake Application",
                        "Gear Shifts"
                    ),
                    row_heights=[0.4, 0.3, 0.3]
                )
            
            # Add primary driver's data
            # Speed trace
            fig.add_trace(go.Scatter(
                x=telemetry['Distance'],
                y=telemetry['Speed'],
                mode='lines',
                name=f"{driver_abbr} Speed",
                line=dict(color='#FF1801', width=2),
                hovertemplate='%{y:.1f} km/h<br>Distance: %{x:.0f}m'
            ), row=1, col=1)
            
            # Throttle/Brake
            if 'Throttle' in telemetry.columns:
                fig.add_trace(go.Scatter(
                    x=telemetry['Distance'],
                    y=telemetry['Throttle'],
                    mode='lines',
                    name=f"{driver_abbr} Throttle",
                    line=dict(color='#00FF00', width=1.5),
                    hovertemplate='Throttle: %{y:.0f}%<br>Distance: %{x:.0f}m'
                ), row=2, col=1)
            
            if 'Brake' in telemetry.columns:
                fig.add_trace(go.Scatter(
                    x=telemetry['Distance'],
                    y=telemetry['Brake'] * 100,  # Scale to percentage
                    mode='lines',
                    name=f"{driver_abbr} Brake",
                    line=dict(color='#FF0000', width=1.5),
                    hovertemplate='Brake: %{y:.0f}%<br>Distance: %{x:.0f}m'
                ), row=2, col=1)
                
            # Gear shifts
            if 'nGear' in telemetry.columns:
                fig.add_trace(go.Scatter(
                    x=telemetry['Distance'],
                    y=telemetry['nGear'],
                    mode='lines+markers',
                    name=f"{driver_abbr} Gear",
                    line=dict(color='#00FFFF', width=1.5),
                    marker=dict(size=4),
                    hovertemplate='Gear: %{y:.0f}<br>Distance: %{x:.0f}m'
                ), row=3, col=1)
            
            # Add comparison driver if requested
            if compare_driver and compare_lap:
                comp_driver_abbr = self.get_driver_abbreviation(session, compare_driver)
                
                # Get comparison lap data
                if compare_lap == 'fastest':
                    comp_lap = session.laps.pick_drivers(compare_driver).pick_fastest()
                else:
                    comp_lap = self.get_lap_by_selection(session, compare_driver, compare_lap)
                
                comp_telemetry = comp_lap.get_telemetry()
                
                # Speed comparison
                fig.add_trace(go.Scatter(
                    x=comp_telemetry['Distance'],
                    y=comp_telemetry['Speed'],
                    mode='lines',
                    name=f"{comp_driver_abbr} Speed",
                    line=dict(color='#0000FF', width=2, dash='dash'),
                    hovertemplate='%{y:.1f} km/h<br>Distance: %{x:.0f}m'
                ), row=1, col=1)
                
                # Throttle/Brake
                if 'Throttle' in comp_telemetry.columns:
                    fig.add_trace(go.Scatter(
                        x=comp_telemetry['Distance'],
                        y=comp_telemetry['Throttle'],
                        mode='lines',
                        name=f"{comp_driver_abbr} Throttle",
                        line=dict(color='#00AA00', width=1.5, dash='dash'),
                        hovertemplate='Throttle: %{y:.0f}%<br>Distance: %{x:.0f}m'
                    ), row=2, col=1)
                
                if 'Brake' in comp_telemetry.columns:
                    fig.add_trace(go.Scatter(
                        x=comp_telemetry['Distance'],
                        y=comp_telemetry['Brake'] * 100,  # Scale to percentage
                        mode='lines',
                        name=f"{comp_driver_abbr} Brake",
                        line=dict(color='#AA0000', width=1.5, dash='dash'),
                        hovertemplate='Brake: %{y:.0f}%<br>Distance: %{x:.0f}m'
                    ), row=2, col=1)
                
                # Gear shifts
                if 'nGear' in comp_telemetry.columns:
                    fig.add_trace(go.Scatter(
                        x=comp_telemetry['Distance'],
                        y=comp_telemetry['nGear'],
                        mode='lines+markers',
                        name=f"{comp_driver_abbr} Gear",
                        line=dict(color='#0000AA', width=1.5, dash='dash'),
                        marker=dict(size=4),
                        hovertemplate='Gear: %{y:.0f}<br>Distance: %{x:.0f}m'
                    ), row=3, col=1)
            
            # Annotate corners on the speed trace
            max_speed = telemetry['Speed'].max()
            for _, corner in circuit.corners.iterrows():
                # Add corner marker on speed plot
                fig.add_vline(
                    x=corner['Distance'], 
                    line=dict(color='gray', dash='dash'),
                    opacity=0.5,
                    row=1, col=1
                )
                fig.add_annotation(
                    x=corner['Distance'],
                    y=max_speed * 0.9,
                    text=f"T{corner['Number']}",
                    showarrow=False,
                    textangle=90,
                    font=dict(size=8, color='white'),
                    row=1, col=1
                )
                
                # Add light corner markers on other plots
                fig.add_vline(
                    x=corner['Distance'], 
                    line=dict(color='gray', dash='dash'),
                    opacity=0.3,
                    row=2, col=1
                )
                fig.add_vline(
                    x=corner['Distance'], 
                    line=dict(color='gray', dash='dash'),
                    opacity=0.3,
                    row=3, col=1
                )
            
            # Update y-axis ranges
            fig.update_yaxes(title_text="Speed (km/h)", range=[0, max_speed * 1.1], row=1, col=1)
            fig.update_yaxes(title_text="Throttle/Brake (%)", range=[-5, 105], row=2, col=1)
            fig.update_yaxes(title_text="Gear", range=[0, 9], dtick=1, row=3, col=1)
            
            # Update x-axis
            fig.update_xaxes(title_text="Distance (m)", row=3, col=1)
            
            # Update layout
            lap_info = f"Lap {lap['LapNumber']} ({lap['LapTime']})" if pd.notnull(lap['LapTime']) else f"Lap {lap['LapNumber']}"
            
            title_text = f"{driver_abbr} Telemetry - {race} {year} - {lap_info}"
            if compare_driver:
                comp_lap_info = f"Lap {comp_lap['LapNumber']} ({comp_lap['LapTime']})" if pd.notnull(comp_lap['LapTime']) else f"Lap {comp_lap['LapNumber']}"
                title_text += f" vs {comp_driver_abbr} - {comp_lap_info}"
            
            fig.update_layout(
                title=title_text,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=800,
                margin=dict(l=60, r=40, t=100, b=60)
            )
            
            self.cache.set(fig, 'telemetry', key=cache_key)
            return fig
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                annotations=[
                    dict(
                        text=f"Failed to load telemetry data: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig

    def analyze_sectors(self, year: int, race: str) -> dict:
        """Compare sector times across drivers"""
        try:
            session = ff1.get_session(year, race, 'Q')
            session.load()
            
            sector_data = {}
            for driver_num in session.drivers:
                # Get driver abbreviation
                driver_abbr = self.get_driver_abbreviation(session, driver_num)
                
                # Get sector times
                laps = session.laps.pick_drivers(driver_num)
                sectors = {
                    'Sector1': laps['Sector1Time'].dropna().mean().total_seconds() if not laps['Sector1Time'].dropna().empty else None,
                    'Sector2': laps['Sector2Time'].dropna().mean().total_seconds() if not laps['Sector2Time'].dropna().empty else None,
                    'Sector3': laps['Sector3Time'].dropna().mean().total_seconds() if not laps['Sector3Time'].dropna().empty else None
                }
                # Store with abbreviation as key
                sector_data[driver_abbr] = sectors
                
            return sector_data
        except Exception as e:
            print(f"Error analyzing sectors: {str(e)}")
            return {}
