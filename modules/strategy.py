import fastf1 as ff1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union, Tuple

class StrategyAnalyzer:
    """Tire strategy and pit stop analysis engine with comprehensive race strategy insights"""
    
    def __init__(self, cache):
        self.cache = cache
    
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

    
    def analyze_stints(self, year: int, race: str) -> Dict[str, List[Dict]]:
        """Analyze tire stints for all drivers with detailed degradation metrics"""
        cache_key = f"stint_analysis_{year}_{race}"
        cached = self.cache.get('strategy', key=cache_key)
        if cached:
            return cached
            
        try:
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            stint_data = {}
            for driver in session.drivers:
                # Get driver abbreviation
                driver_abbr = self.get_driver_abbreviation(session, driver)
                
                # Get driver laps
                driver_laps = session.laps.pick_drivers(driver)
                stints = []
                
                for stint_num in driver_laps['Stint'].unique():
                    stint_laps = driver_laps[driver_laps['Stint'] == stint_num]
                    if len(stint_laps) < 3:  # Skip short stints
                        continue
                    
                    compound = stint_laps['Compound'].iloc[0]
                    lap_times = stint_laps['LapTime'].dt.total_seconds()
                    
                    # Skip if lap times are empty
                    if len(lap_times.dropna()) < 3:
                        continue
                    
                    # Calculate degradation rate
                    x = np.array(stint_laps['LapNumber'])
                    y = lap_times.dropna().values
                    
                    if len(y) >= 3:  # Need at least 3 points for meaningful fit
                        coeffs = np.polyfit(x[:len(y)], y, 2)
                        poly = np.poly1d(coeffs)
                        
                        # Calculate metrics
                        degradation_rate = coeffs[0]  # Coefficient of x²
                        avg_laptime = np.mean(y)
                        min_laptime = np.min(y)
                        consistency = np.std(y)
                        
                        # Calculate theoretical time loss over stint
                        stint_length = x.max() - x.min() + 1
                        time_loss = poly(x.min() + stint_length) - poly(x.min())
                        
                        stints.append({
                            'driver': driver,
                            'driver_abbr': driver_abbr,
                            'stint': int(stint_num),
                            'compound': compound,
                            'start_lap': int(x.min()),
                            'end_lap': int(x.max()),
                            'laps': int(stint_length),
                            'degradation_rate': float(degradation_rate),
                            'avg_laptime': float(avg_laptime),
                            'min_laptime': float(min_laptime),
                            'consistency': float(consistency),
                            'time_loss': float(time_loss),
                            'coefficients': coeffs.tolist()
                        })
                
                if stints:  # Only add drivers with valid stints
                    stint_data[driver_abbr] = stints
            
            self.cache.set(stint_data, 'strategy', key=cache_key)
            return stint_data
            
        except Exception as e:
            print(f"Error analyzing stints: {str(e)}")
            return {}

    def analyze_pit_stops(self, year: int, race: str) -> Dict[str, Any]:
        """Analyze pit stop performance for all drivers"""
        cache_key = f"pitstop_analysis_{year}_{race}"
        cached = self.cache.get('strategy', key=cache_key)
        if cached:
            return cached
            
        try:
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            pit_data = []
            team_pit_times = {}
            
            for driver in session.drivers:
                driver_abbr = self.get_driver_abbreviation(session, driver)
                driver_laps = session.laps.pick_drivers(driver)
                
                # Get team name
                team_name = None
                for _, result in session.results.iterrows():
                    if str(result['DriverNumber']) == str(driver):
                        team_name = result['TeamName']
                        break
                
                # Find pit stops
                for _, lap in driver_laps.iterrows():
                    if pd.notnull(lap['PitInTime']) and pd.notnull(lap['PitOutTime']):
                        # Calculate pit stop duration
                        pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
                        
                        # Skip unrealistic pit times
                        if pit_duration < 1 or pit_duration > 120:
                            continue
                        
                        # Determine compounds
                        current_compound = lap['Compound']
                        
                        # Try to find next compound
                        next_lap_idx = driver_laps[driver_laps['LapNumber'] > lap['LapNumber']].index
                        next_compound = current_compound
                        if not next_lap_idx.empty:
                            next_lap = driver_laps.loc[next_lap_idx[0]]
                            if pd.notnull(next_lap['Compound']):
                                next_compound = next_lap['Compound']
                        
                        # Add to pit data
                        pit_data.append({
                            'Driver': driver,
                            'DriverAbbr': driver_abbr,
                            'Team': team_name,
                            'Lap': lap['LapNumber'],
                            'PitDuration': pit_duration,
                            'FromCompound': current_compound,
                            'ToCompound': next_compound
                        })
                        
                        # Track team pit times
                        if team_name:
                            if team_name not in team_pit_times:
                                team_pit_times[team_name] = []
                            team_pit_times[team_name].append(pit_duration)
            
            # Calculate team statistics
            team_stats = []
            for team, times in team_pit_times.items():
                if times:
                    team_stats.append({
                        'Team': team,
                        'AvgPitTime': np.mean(times),
                        'MinPitTime': np.min(times),
                        'MaxPitTime': np.max(times),
                        'StdPitTime': np.std(times),
                        'NumPitStops': len(times)
                    })
            
            # Create DataFrames
            pit_df = pd.DataFrame(pit_data)
            team_df = pd.DataFrame(team_stats).sort_values('AvgPitTime')
            
            result = {
                'pit_stops': pit_df,
                'team_stats': team_df
            }
            
            self.cache.set(result, 'strategy', key=cache_key)
            return result
            
        except Exception as e:
            print(f"Error analyzing pit stops: {str(e)}")
            return {'pit_stops': pd.DataFrame(), 'team_stats': pd.DataFrame()}

    def visualize_tire_strategy(self, year: int, race: str) -> go.Figure:
        """Create comprehensive tire strategy visualization with stint timelines"""
        cache_key = f"tire_strategy_viz_{year}_{race}"
        cached = self.cache.get('strategy', key=cache_key)
        if cached:
            return cached
            
        try:
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            # Get driver info for abbreviations and team assignment
            driver_info = {}
            for _, result in session.results.iterrows():
                driver_num = str(result['DriverNumber'])
                driver_info[driver_num] = {
                    'abbr': self.get_driver_abbreviation(session, driver_num),
                    'team': result['TeamName'],
                    'position': result['Position']
                }
            
            # Sort drivers by final position
            sorted_drivers = sorted(driver_info.keys(), 
                                 key=lambda x: driver_info[x]['position'])
            
            # Define compounds and their colors
            compound_colors = {
                'SOFT': 'rgb(255, 0, 0)',      # Red
                'MEDIUM': 'rgb(255, 255, 0)',  # Yellow
                'HARD': 'rgb(255, 255, 255)',  # White
                'INTERMEDIATE': 'rgb(0, 255, 0)',  # Green
                'WET': 'rgb(0, 0, 255)',       # Blue
                None: 'rgb(200, 200, 200)'     # Gray for unknown
            }
            
            # Create figure
            fig = go.Figure()
            
            # Add data for each driver
            y_positions = {}  # Track y-position for each driver
            y_pos = 1
            
            for driver in sorted_drivers:
                driver_abbr = driver_info[driver]['abbr']
                team = driver_info[driver]['team']
                y_positions[driver] = y_pos
                
                # Get laps for this driver
                driver_laps = session.laps.pick_drivers(driver)
                
                # Add rectangles for each stint
                for stint in driver_laps['Stint'].unique():
                    stint_laps = driver_laps[driver_laps['Stint'] == stint]
                    if len(stint_laps) < 2:  # Skip very short stints
                        continue
                    
                    # Get stint details
                    start_lap = stint_laps['LapNumber'].min()
                    end_lap = stint_laps['LapNumber'].max()
                    compound = stint_laps['Compound'].iloc[0]
                    
                    # Add stint as a colored rectangle
                    fig.add_shape(
                        type="rect",
                        x0=start_lap - 0.4,
                        x1=end_lap + 0.4,
                        y0=y_pos - 0.4,
                        y1=y_pos + 0.4,
                        fillcolor=compound_colors.get(compound, 'rgb(200, 200, 200)'),
                        line=dict(color="black", width=1),
                        opacity=0.8
                    )
                    
                    # Add pit stop marker
                    if stint > 1:
                        fig.add_shape(
                            type="line",
                            x0=start_lap - 0.4,
                            x1=start_lap - 0.4,
                            y0=y_pos - 0.5,
                            y1=y_pos + 0.5,
                            line=dict(color="black", width=2)
                        )
                
                # Add driver label
                fig.add_annotation(
                    x=0,
                    y=y_pos,
                    text=f"{driver_abbr} ({team})",
                    showarrow=False,
                    xanchor="right",
                    xshift=-10,
                    font=dict(size=10)
                )
                
                y_pos += 1
            
            # Add legend for compounds
            legend_items = []
            for compound, color in compound_colors.items():
                if compound:  # Skip None
                    legend_items.append(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(size=10, color=color),
                            name=compound,
                            showlegend=True
                        )
                    )
            
            for item in legend_items:
                fig.add_trace(item)
            
            # Format the chart
            max_lap = session.laps['LapNumber'].max()
            
            fig.update_layout(
                title=f"Tire Strategy - {race} {year}",
                xaxis=dict(
                    title="Lap Number",
                    range=[0, max_lap + 1],
                    dtick=5,
                    gridcolor="rgba(200, 200, 200, 0.2)"
                ),
                yaxis=dict(
                    title="",
                    range=[0, len(sorted_drivers) + 1],
                    showticklabels=False,
                    gridcolor="rgba(200, 200, 200, 0.2)"
                ),
                plot_bgcolor="rgba(0, 0, 0, 0.05)",
                height=max(500, 100 + (30 * len(sorted_drivers))),  # Dynamic height based on drivers
                margin=dict(l=120, r=40, t=50, b=50),
                legend=dict(
                    title="Tire Compounds",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                hovermode="closest"
            )
            
            self.cache.set(fig, 'strategy', key=cache_key)
            return fig
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                annotations=[
                    dict(
                        text=f"Failed to load strategy data: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig

    def plot_tire_strategy(self, year: int, race: str) -> go.Figure:
        """Visualize tire strategies across teams (colored by compound)"""
        cache_key = f"tire_lap_strategy_{year}_{race}"
        cached = self.cache.get('strategy', key=cache_key)
        if cached:
            return cached
            
        try:
            session = ff1.get_session(year, race, 'R')
            session.load()
            
            # Create Plotly figure
            fig = go.Figure()
            
            compound_colors = {
                'SOFT': 'red',
                'MEDIUM': 'yellow',
                'HARD': 'white',
                'INTERMEDIATE': 'green',
                'WET': 'blue'
            }
            
            # Track which compounds we've already added to legend
            shown_compounds = set()
            
            for driver in session.drivers:
                # Get driver abbreviation
                driver_abbr = self.get_driver_abbreviation(session, driver)
                
                # Get laps for this driver
                laps = session.laps.pick_drivers(driver)
                
                # Process each stint
                for stint in laps['Stint'].unique():
                    stint_laps = laps[laps['Stint'] == stint]
                    if len(stint_laps) < 2:  # Skip very short stints
                        continue
                        
                    compound = stint_laps['Compound'].iloc[0]
                    compound_name = f"{compound}"
                    
                    # Only show in legend if we haven't seen this compound before
                    show_in_legend = compound_name not in shown_compounds
                    if show_in_legend:
                        shown_compounds.add(compound_name)
                    
                    # Create hover text with lap time and gap info
                    hover_texts = []
                    for _, lap in stint_laps.iterrows():
                        lap_text = f"Driver: {driver_abbr}<br>Lap: {lap['LapNumber']}"
                        
                        if pd.notnull(lap['LapTime']):
                            lap_text += f"<br>Lap Time: {lap['LapTime']}"
                        
                        if pd.notnull(lap['Compound']):
                            lap_text += f"<br>Compound: {lap['Compound']}"
                            
                        hover_texts.append(lap_text)
                    
                    # Add trace for this stint
                    fig.add_trace(go.Scatter(
                        x=stint_laps['LapNumber'],
                        y=stint_laps['LapTime'].dt.total_seconds(),
                        mode='lines+markers',
                        name=compound_name if show_in_legend else None,
                        legendgroup=compound_name,
                        showlegend=show_in_legend,
                        line=dict(color=compound_colors.get(compound, 'gray')),
                        marker=dict(size=4),
                        hovertext=hover_texts,
                        hoverinfo='text',
                        opacity=0.7,  # Make lines slightly transparent
                        customdata=[[driver_abbr, compound]] * len(stint_laps)
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"Lap Time by Tire Compound - {race} {year}",
                xaxis_title="Lap Number",
                yaxis_title="Lap Time (seconds)",
                legend=dict(
                    title="Tire Compounds",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_color="black"
                ),
                plot_bgcolor="rgba(0, 0, 0, 0.05)"
            )
            
            self.cache.set(fig, 'strategy', key=cache_key)
            return fig
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                annotations=[
                    dict(
                        text=f"Failed to load strategy data: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig

    def plot_pit_stop_performance(self, year: int, race: str) -> Dict[str, go.Figure]:
        """Create pit stop performance visualizations"""
        cache_key = f"pitstop_viz_{year}_{race}"
        cached = self.cache.get('strategy', key=cache_key)
        if cached:
            return cached
            
        try:
            # Get pit stop data
            pit_data = self.analyze_pit_stops(year, race)
            
            if pit_data['pit_stops'].empty:
                raise ValueError("No pit stop data available")
                
            pit_stops = pit_data['pit_stops']
            team_stats = pit_data['team_stats']
            
            # Create result dictionary
            result_figures = {}
            
            # 1. Pit stop duration by driver
            if not pit_stops.empty:
                # Sort by average pit duration
                driver_avg = pit_stops.groupby('DriverAbbr')['PitDuration'].mean().reset_index()
                driver_avg = driver_avg.sort_values('PitDuration')
                driver_order = driver_avg['DriverAbbr'].tolist()
                
                # Create box plot
                fig_duration = px.box(
                    pit_stops,
                    x='DriverAbbr',
                    y='PitDuration',
                    color='Team',
                    category_orders={"DriverAbbr": driver_order},
                    title="Pit Stop Duration by Driver",
                    labels={
                        'DriverAbbr': 'Driver',
                        'PitDuration': 'Pit Stop Duration (seconds)',
                        'Team': 'Team'
                    },
                    points="all"
                )
                
                # Update layout
                fig_duration.update_layout(
                    xaxis={'categoryorder': 'array', 'categoryarray': driver_order},
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_color="black"
                    ),
                    plot_bgcolor="rgba(0, 0, 0, 0.05)"
                )
                
                # Customize hover template
                fig_duration.update_traces(
                    hovertemplate="Driver: %{x}<br>Duration: %{y:.2f}s<br>Team: %{marker.color}<extra></extra>"
                )
                
                result_figures['driver_duration'] = fig_duration
            
            # 2. Team pit stop performance
            if not team_stats.empty:
                # Create bar chart with error bars
                fig_team = go.Figure()
                
                # Sort teams by average pit time
                team_stats = team_stats.sort_values('AvgPitTime')
                
                # Add bars for average pit time
                fig_team.add_trace(go.Bar(
                    x=team_stats['Team'],
                    y=team_stats['AvgPitTime'],
                    error_y=dict(
                        type='data',
                        array=team_stats['StdPitTime'],
                        visible=True
                    ),
                    marker_color='lightblue',
                    name='Average Pit Time'
                ))
                
                # Add markers for fastest pit stops
                fig_team.add_trace(go.Scatter(
                    x=team_stats['Team'],
                    y=team_stats['MinPitTime'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='green'
                    ),
                    name='Fastest Pit Stop'
                ))
                
                # Update layout
                fig_team.update_layout(
                    title="Team Pit Stop Performance",
                    xaxis_title="Team",
                    yaxis_title="Pit Stop Duration (seconds)",
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_color="black"
                    ),
                    plot_bgcolor="rgba(0, 0, 0, 0.05)"
                )
                
                # Add pit stop count annotations
                for i, team in enumerate(team_stats['Team']):
                    count = team_stats.iloc[i]['NumPitStops']
                    fig_team.add_annotation(
                        x=team,
                        y=team_stats.iloc[i]['AvgPitTime'] + team_stats.iloc[i]['StdPitTime'] + 0.5,
                        text=f"n={count}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                
                result_figures['team_performance'] = fig_team
            
            # 3. Pit stop timing during race
            if not pit_stops.empty:
                fig_timing = px.scatter(
                    pit_stops,
                    x='Lap',
                    y='PitDuration',
                    color='Team',
                    symbol='ToCompound',
                    hover_data=['DriverAbbr', 'FromCompound', 'ToCompound'],
                    title="Pit Stop Timing During Race",
                    labels={
                        'Lap': 'Lap Number',
                        'PitDuration': 'Pit Stop Duration (seconds)',
                        'Team': 'Team',
                        'ToCompound': 'New Compound'
                    },
                    size_max=10
                )
                
                # Update layout
                fig_timing.update_layout(
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_color="black"
                    ),
                    plot_bgcolor="rgba(0, 0, 0, 0.05)"
                )
                
                # Customize hover template
                fig_timing.update_traces(
                    hovertemplate="Driver: %{customdata[0]}<br>Lap: %{x}<br>Duration: %{y:.2f}s<br>From: %{customdata[1]}<br>To: %{customdata[2]}<extra></extra>"
                )
                
                result_figures['race_timing'] = fig_timing
            
            self.cache.set(result_figures, 'strategy', key=cache_key)
            return result_figures
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                annotations=[
                    dict(
                        text=f"Failed to load pit stop data: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return {'error': fig}
    
    def visualize_tire_degradation(self, year: int, race: str) -> go.Figure:
        """Create tire degradation comparison visualization"""
        cache_key = f"tire_degradation_{year}_{race}"
        cached = self.cache.get('strategy', key=cache_key)
        if cached:
            return cached
            
        try:
            # Get stint analysis data
            stint_data = self.analyze_stints(year, race)
            
            if not stint_data:
                raise ValueError("No stint data available")
            
            # Create figure
            fig = go.Figure()
            
            # Set of markers for different compounds
            compound_markers = {
                'SOFT': 'circle',
                'MEDIUM': 'square',
                'HARD': 'diamond',
                'INTERMEDIATE': 'triangle-up',
                'WET': 'cross'
            }
            
            # Color for teams
            teams_seen = set()
            color_idx = 0
            team_colors = {}
            
            # Process each driver's stints
            for driver_abbr, stints in stint_data.items():
                # Skip if no stints
                if not stints:
                    continue
                
                # Get team for coloring
                session = ff1.get_session(year, race, 'R')
                session.load()
                
                driver = stints[0]['driver']
                team = None
                
                for _, result in session.results.iterrows():
                    if str(result['DriverNumber']) == str(driver):
                        team = result['TeamName']
                        break
                
                # Assign color to team
                if team and team not in teams_seen:
                    teams_seen.add(team)
                    # Colors will cycle for different teams
                    team_colors[team] = color_idx
                    color_idx += 1
                
                # Process each stint
                for stint in stints:
                    # Calculate fitted lap times
                    start_lap = stint['start_lap']
                    end_lap = stint['end_lap']
                    compound = stint['compound']
                    coeffs = stint['coefficients']
                    
                    # Create polynomial function
                    poly = np.poly1d(coeffs)
                    
                    # Calculate lap times over stint
                    lap_range = np.linspace(start_lap, end_lap, 100)
                    fitted_times = poly(lap_range)
                    
                    # Normalize to show degradation relative to first lap
                    norm_times = fitted_times - fitted_times[0]
                    
                    # Add trace for fitted degradation curve
                    hover_text = [
                        f"Driver: {driver_abbr}<br>" +
                        f"Team: {team}<br>" +
                        f"Compound: {compound}<br>" +
                        f"Lap {int(lap)}<br>" +
                        f"Degradation: {time:.3f}s"
                        for lap, time in zip(lap_range, norm_times)
                    ]
                    
                    # Legend name based on team and stint details
                    legend_name = f"{driver_abbr} - {compound} (L{start_lap}-{end_lap})"
                    
                    # For visibility, only show driver in legend once
                    show_legend = True
                    
                    fig.add_trace(go.Scatter(
                        x=lap_range - start_lap,  # Normalize to laps into stint
                        y=norm_times,
                        mode='lines',
                        name=legend_name,
                        line=dict(
                            width=2,
                            dash='solid'
                        ),
                        marker=dict(
                            symbol=compound_markers.get(compound, 'circle'),
                            size=8
                        ),
                        hovertext=hover_text,
                        hoverinfo='text',
                        showlegend=show_legend
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"Tire Degradation Analysis - {race} {year}",
                xaxis_title="Laps into Stint",
                yaxis_title="Time Loss (seconds)",
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_color="black"
                ),
                plot_bgcolor="rgba(0, 0, 0, 0.05)",
                legend=dict(
                    title="Driver - Compound (Stint Range)",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1.1
                )
            )
            
            # Add zero reference line
            fig.add_shape(
                type="line",
                x0=0,
                x1=max(stint['end_lap'] - stint['start_lap'] for driver_stints in stint_data.values() for stint in driver_stints),
                y0=0,
                y1=0,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dash"
                )
            )
            
            self.cache.set(fig, 'strategy', key=cache_key)
            return fig
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                annotations=[
                    dict(
                        text=f"Failed to load tire degradation data: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig
