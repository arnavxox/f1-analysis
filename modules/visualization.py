import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class VisualizationEngine:
    """Advanced visualization factory for F1 race data analysis"""
    
    @staticmethod
    def get_driver_abbreviation(session, driver: str) -> str:
        """Convert driver number to abbreviation"""
        try:
            # Replace get_driver_info with direct access to drivers info
            for _, driver_data in session.results.iterrows():
                if str(driver_data['DriverNumber']) == str(driver):
                    return driver_data['Abbreviation']
            return driver  # Return original if not found
        except Exception:
            return driver  # Return original on error

    
    @staticmethod
    def create_position_changes(session) -> go.Figure:
        """Interactive position changes chart with driver abbreviations"""
        try:
            # Create a dataframe to hold all position data
            all_pos_data = []
            
            # Get driver abbreviations dictionary
            abbr_dict = {}
            for _, driver_data in session.results.iterrows():
                abbr_dict[str(driver_data['DriverNumber'])] = driver_data['Abbreviation']
                        
            # Process each driver
            for driver in session.drivers:
                driver_laps = session.laps.pick_drivers(driver)
                
                if not driver_laps.empty:
                    # Get driver abbreviation
                    driver_abbr = abbr_dict.get(str(driver), driver)
                    
                    # Extract lap positions
                    for _, lap in driver_laps.iterrows():
                        if pd.notnull(lap['Position']):
                            all_pos_data.append({
                                'Driver': driver_abbr,  # Use abbreviation instead of number
                                'DriverNumber': driver,
                                'Lap': lap['LapNumber'],
                                'Position': lap['Position'],
                                'LapTime': lap['LapTime'].total_seconds() if pd.notnull(lap['LapTime']) else None
                            })
            
            # Convert to DataFrame
            df = pd.DataFrame(all_pos_data)
            
            if df.empty:
                raise ValueError("No valid position data found")
            
            # Create the position changes figure
            fig = px.line(df, x='Lap', y='Position', color='Driver',
                        hover_data=['LapTime'],
                        title="Position Changes During Race",
                        labels={'Position': 'Race Position', 'Lap': 'Lap Number'})
            
            # Customize the figure
            fig.update_yaxes(autorange="reversed", title_text="Position",
                           tickmode='linear', tick0=1, dtick=1)
            fig.update_xaxes(title_text="Lap Number", 
                           tickmode='linear', tick0=1, dtick=5)
            
            # Add grid lines for easier position tracking
            fig.update_layout(
                yaxis=dict(
                    gridcolor='rgba(128,128,128,0.3)',
                    gridwidth=1,
                ),
                xaxis=dict(
                    gridcolor='rgba(128,128,128,0.3)',
                    gridwidth=1,
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_color="black"
                ),
                hovermode="closest"
            )
            
            # Add custom hover template
            for trace in fig.data:
                trace.hovertemplate = 'Driver: %{fullData.name}<br>Lap: %{x}<br>Position: %{y}<extra></extra>'
            
            return fig
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating position chart: {str(e)}",
                annotations=[dict(
                    text=f"Failed to generate position changes chart: {str(e)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
            return fig
    
    @staticmethod
    def create_pace_distribution(session) -> go.Figure:
        """Enhanced violin plot of lap time distribution with driver abbreviations"""
        try:
            # Get clean laps (not in/out laps)
            clean_laps = session.laps.pick_quicklaps().reset_index()
            
            if clean_laps.empty:
                raise ValueError("No valid lap time data found")
            
            # Convert lap times to seconds
            clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
            
            # Get driver abbreviations
            abbr_dict = {}
            for _, driver_data in session.results.iterrows():
                abbr_dict[str(driver_data['DriverNumber'])] = driver_data['Abbreviation']
                        
            # Replace driver numbers with abbreviations
            clean_laps['DriverAbbr'] = clean_laps['Driver'].map(lambda x: abbr_dict.get(str(x), x))
            
            # Calculate additional statistics for each driver
            driver_stats = clean_laps.groupby('DriverAbbr')['LapTimeSeconds'].agg([
                ('Median', 'median'),
                ('Min', 'min'),
                ('Max', 'max'),
                ('Count', 'count')
            ]).reset_index()
            
            # Sort by median lap time
            driver_stats = driver_stats.sort_values('Median')
            driver_order = driver_stats['DriverAbbr'].tolist()
            
            # Create violin plot with box plot overlay
            fig = px.violin(clean_laps, x='DriverAbbr', y='LapTimeSeconds',
                          box=True, points="outliers",
                          category_orders={"DriverAbbr": driver_order},
                          title="Race Pace Distribution by Driver",
                          labels={
                              'DriverAbbr': 'Driver',
                              'LapTimeSeconds': 'Lap Time (seconds)'
                          },
                          color='DriverAbbr')
            
            # Add median line
            for driver, stats in driver_stats.iterrows():
                driver_name = stats['DriverAbbr']
                median_time = stats['Median']
                
                # Add annotation for median time
                fig.add_annotation(
                    x=driver_name,
                    y=median_time,
                    text=f"{median_time:.3f}s",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="black", size=10),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            
            # Improve hover information
            fig.update_traces(
                hovertemplate='Driver: %{x}<br>Lap Time: %{y:.3f}s<extra></extra>'
            )
            
            # Update layout
            fig.update_layout(
                showlegend=False,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_color="black"
                )
            )
            
            return fig
        
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating pace distribution chart: {str(e)}",
                annotations=[dict(
                    text=f"Failed to generate pace distribution chart: {str(e)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
            return fig

    @staticmethod
    def create_team_comparison(session) -> go.Figure:
        """Enhanced team performance visualization with multiple metrics"""
        try:
            # Group data by team
            team_laps = session.laps.groupby('Team')
            
            # Create a dataframe for team metrics
            team_metrics = []
            
            for team_name, laps in team_laps:
                # Skip teams with too few laps
                if len(laps) < 3:
                    continue
                
                # Calculate metrics
                avg_lap_time = laps['LapTime'].dropna().mean().total_seconds()
                best_lap_time = laps['LapTime'].dropna().min().total_seconds()
                avg_position = laps['Position'].mean()
                best_position = laps['Position'].min()
                
                # Get most used compound
                compounds = laps['Compound'].value_counts()
                preferred_compound = compounds.index[0] if not compounds.empty else "Unknown"
                
                # Get drivers
                drivers = laps['Driver'].unique()
                driver_abbrs = []
                for d in drivers:
                    try:
                        abbr = VisualizationEngine.get_driver_abbreviation(session, d)
                        driver_abbrs.append(abbr)
                    except:
                        driver_abbrs.append(d)
                
                driver_text = ", ".join(driver_abbrs)
                
                # Standardize lap times for comparison (lower is better)
                # Invert so that faster times = higher score
                normalized_lap_time = 100 - min(100, (avg_lap_time / best_lap_time - 1) * 1000)
                
                # Add to metrics list
                team_metrics.append({
                    'Team': team_name,
                    'Drivers': driver_text,
                    'AvgLapTime': avg_lap_time,
                    'BestLapTime': best_lap_time,
                    'AvgPosition': avg_position,
                    'BestPosition': best_position,
                    'PreferredCompound': preferred_compound,
                    'PaceScore': normalized_lap_time,
                    'PositionScore': 20 - min(20, avg_position),  # Higher is better
                    'CompoundScore': 70 if preferred_compound == 'HARD' else 
                                  (85 if preferred_compound == 'MEDIUM' else 
                                   (100 if preferred_compound == 'SOFT' else 50))  # Arbitrary score
                })
            
            # Convert to DataFrame
            team_df = pd.DataFrame(team_metrics)
            
            if team_df.empty:
                raise ValueError("No valid team data found")
            
            # Create radar chart for comprehensive team comparison
            fig = go.Figure()
            
            for _, team in team_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[team['PaceScore'], team['PositionScore'], team['CompoundScore']],
                    theta=['Pace', 'Position', 'Tire Strategy'],
                    fill='toself',
                    name=f"{team['Team']} ({team['Drivers']})"
                ))
            
            # Update layout
            fig.update_layout(
                title="Team Performance Comparison",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_color="black"
                )
            )
            
            return fig
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating team comparison chart: {str(e)}",
                annotations=[dict(
                    text=f"Failed to generate team comparison chart: {str(e)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
            return fig
    
    @staticmethod
    def create_lap_time_progression(session) -> go.Figure:
        """New: Create a lap time progression chart showing lap time trends over race distance"""
        try:
            # Get all laps data
            laps_df = session.laps
            
            if laps_df.empty:
                raise ValueError("No lap data found")
            
            # Convert lap times to seconds
            laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
            
            # Get driver abbreviations
            abbr_dict = {}
            for _, driver_data in session.results.iterrows():
                abbr_dict[str(driver_data['DriverNumber'])] = driver_data['Abbreviation']
            
            # Create figure
            fig = go.Figure()
            
            # Add trace for each driver
            for driver in session.drivers:
                driver_abbr = abbr_dict.get(str(driver), driver)
                driver_laps = laps_df[laps_df['Driver'] == driver]
                
                if not driver_laps.empty:
                    # Add line trace
                    fig.add_trace(go.Scatter(
                        x=driver_laps['LapNumber'],
                        y=driver_laps['LapTimeSeconds'],
                        mode='lines+markers',
                        name=driver_abbr,
                        connectgaps=False,
                        marker=dict(size=6),
                        hovertemplate=(
                            f"Driver: {driver_abbr}<br>" +
                            "Lap: %{x}<br>" +
                            "Time: %{y:.3f}s<br>" +
                            "Compound: %{customdata}<extra></extra>"
                        ),
                        customdata=driver_laps['Compound']
                    ))
            
            # Update layout
            fig.update_layout(
                title="Lap Time Evolution Throughout Race",
                xaxis_title="Lap Number",
                yaxis_title="Lap Time (seconds)",
                hovermode="closest",
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_color="black"
                )
            )
            
            return fig
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating lap time progression chart: {str(e)}",
                annotations=[dict(
                    text=f"Failed to generate lap time progression chart: {str(e)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
            return fig
    
    @staticmethod
    def create_race_summary(session) -> Dict[str, go.Figure]:
        """New: Create a comprehensive set of summary visualizations for the race"""
        try:
            # Summary figures dictionary
            summary_figures = {}
            
            # Get lap and driver data
            laps_df = session.laps
            results_df = session.results
            
            # Get driver abbreviations
            abbr_dict = {}
            for _, driver_data in session.results.iterrows():
                abbr_dict[str(driver_data['DriverNumber'])] = driver_data['Abbreviation']
                        
            # 1. Final positions bar chart
            if not results_df.empty:
                # Convert driver numbers to abbreviations
                results_df['DriverAbbr'] = results_df['DriverNumber'].astype(str).map(
                    lambda x: abbr_dict.get(x, x))
                
                # Sort by final position
                sorted_results = results_df.sort_values('Position')
                
                # Create bar chart of final positions
                fig_results = px.bar(
                    sorted_results,
                    x='DriverAbbr',
                    y='Points',
                    color='TeamName',
                    title="Race Results and Points",
                    labels={'DriverAbbr': 'Driver', 'Points': 'Championship Points'},
                    text='Position'
                )
                
                fig_results.update_traces(
                    texttemplate='P%{text}',
                    textposition='outside'
                )
                
                fig_results.update_layout(
                    xaxis={'categoryorder': 'array', 
                          'categoryarray': sorted_results['DriverAbbr']},
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_color="black"
                    )
                )
                
                summary_figures['results'] = fig_results
            
            # 2. Fastest lap comparison
            if not laps_df.empty:
                # Get fastest lap for each driver
                fastest_laps = []
                
                for driver in session.drivers:
                    driver_laps = laps_df[laps_df['Driver'] == driver]
                    if not driver_laps.empty and not driver_laps['LapTime'].dropna().empty:
                        fastest_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
                        
                        fastest_laps.append({
                            'Driver': abbr_dict.get(str(driver), driver),
                            'LapNumber': fastest_lap['LapNumber'],
                            'LapTime': fastest_lap['LapTime'].total_seconds(),
                            'Compound': fastest_lap['Compound']
                        })
                
                fastest_df = pd.DataFrame(fastest_laps)
                fastest_df = fastest_df.sort_values('LapTime')
                
                # Calculate gap to fastest
                if not fastest_df.empty:
                    fastest_time = fastest_df['LapTime'].min()
                    fastest_df['Gap'] = fastest_df['LapTime'] - fastest_time
                    
                    # Create horizontal bar chart of fastest laps
                    fig_fastest = px.bar(
                        fastest_df,
                        x='LapTime',
                        y='Driver',
                        color='Compound',
                        title="Fastest Lap Comparison",
                        labels={'LapTime': 'Lap Time (seconds)', 'Driver': 'Driver'},
                        orientation='h',
                        hover_data=['LapNumber', 'Gap']
                    )
                    
                    fig_fastest.update_traces(
                        hovertemplate=(
                            "Driver: %{y}<br>" +
                            "Lap Time: %{x:.3f}s<br>" +
                            "Lap Number: %{customdata[0]}<br>" +
                            "Gap to Fastest: +%{customdata[1]:.3f}s<br>" +
                            "Compound: %{marker.color}<extra></extra>"
                        )
                    )
                    
                    fig_fastest.update_layout(
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=12,
                            font_color="black"
                        ),
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    summary_figures['fastest_laps'] = fig_fastest
            
            return summary_figures
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating race summary: {str(e)}",
                annotations=[dict(
                    text=f"Failed to generate race summary: {str(e)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
            return {'error': fig}
