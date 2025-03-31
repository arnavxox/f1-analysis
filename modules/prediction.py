import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Union, Tuple, Optional, Any
import fastf1 as ff1
import logging

class F1PredictionModel:
    """Advanced machine learning model for Formula 1 race outcome predictions"""
    
    def __init__(self, cache=None):
        """
        Initialize the prediction model
        
        Parameters:
        -----------
        cache : F1DataCache, optional
            Cache instance for storing and retrieving model results
        """
        self.cache = cache
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        # Default features for different prediction types
        self.features = {
            'race': [
                'GridPosition',
                'QualifyingPosition',
                'FP1Position', 
                'FP2Position', 
                'FP3Position',
                'PreviousRacePosition',
                'DriverExperience',
                'TeamPerformance'
            ],
            'qualifying': [
                'FP1Position',
                'FP2Position',
                'FP3Position',
                'PreviousQualifyingPosition',
                'DriverExperience',
                'TeamPerformance'
            ]
        }
        
        # Target variables for different prediction types
        self.targets = {
            'race': 'RacePosition',
            'qualifying': 'QualifyingPosition'
        }
        
        # Dictionary to store trained models for different prediction tasks
        self.trained_models = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

def get_driver_abbreviation(self, session, driver_number: str) -> str:
    """
    Convert driver number to abbreviation
    
    Parameters:
    -----------
    session : fastf1.core.Session
        FastF1 session object
    driver_number : str
        Driver number to convert
        
    Returns:
    --------
    str
        Driver abbreviation
    """
    try:
        abbr_dict = {}
        for _, driver_data in session.results.iterrows():
            abbr_dict[str(driver_data['DriverNumber'])] = driver_data['Abbreviation']
        return abbr_dict.get(str(driver_number), driver_number)  # Return abbreviation if found, otherwise original
    except Exception as e:
        self.logger.warning(f"Could not get driver abbreviation: {str(e)}")
        return driver_number  # Return original on error

    def _calculate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional derived features for better prediction
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with raw features
            
        Returns:
        --------
        pd.DataFrame
            Data with added derived features
        """
        df = data.copy()
        
        # Calculate practice session average position (when available)
        practice_cols = ['FP1Position', 'FP2Position', 'FP3Position']
        available_cols = [col for col in practice_cols if col in df.columns]
        
        if available_cols:
            df['PracticeAvgPosition'] = df[available_cols].mean(axis=1)
        
        # Calculate qualifying vs practice delta (when available)
        if 'QualifyingPosition' in df.columns and 'PracticeAvgPosition' in df.columns:
            df['QualifyingDelta'] = df['PracticeAvgPosition'] - df['QualifyingPosition']
        
        # Convert any string positions (like 'DNF', 'DNS') to numerical values
        position_cols = [col for col in df.columns if 'Position' in col]
        for col in position_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def preprocess_data(self, historical_data: pd.DataFrame, prediction_type: str = 'race') -> pd.DataFrame:
        """
        Clean and prepare training data
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Raw historical race data
        prediction_type : str
            Type of prediction (race or qualifying)
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data ready for model training
        """
        try:
            data = historical_data.copy()
            
            # Convert string positions (DNF, DNS, etc.) to NaN
            position_cols = [col for col in data.columns if 'Position' in col]
            for col in position_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    
            # Convert times to seconds if present
            time_cols = [col for col in data.columns if 'Time' in col]
            for col in time_cols:
                if col in data.columns and data[col].dtype == 'object':
                    try:
                        data[col] = pd.to_timedelta(data[col]).dt.total_seconds()
                    except:
                        # If conversion fails, try a different approach or set to NaN
                        data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Add derived features
            data = self._calculate_derived_features(data)
            
            # Handle missing values
            for col in data.columns:
                if col in position_cols and data[col].isna().any():
                    # Fill missing positions with a high value (back of grid)
                    data[col].fillna(data[col].max() + 1, inplace=True)
                elif data[col].dtype.kind in 'if' and data[col].isna().any():
                    # For numerical columns, fill with median
                    data[col].fillna(data[col].median(), inplace=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing error: {str(e)}")
            raise ValueError(f"Failed to preprocess data: {str(e)}")

    def select_model(self, model_type: str = 'random_forest', **kwargs) -> Any:
        """
        Select and configure a prediction model
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'gradient_boosting', 'ridge')
        **kwargs : dict
            Additional parameters for the model
            
        Returns:
        --------
        model : sklearn model
            Configured model object
        """
        if model_type == 'random_forest':
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)
            return RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            n_estimators = kwargs.get('n_estimators', 100)
            learning_rate = kwargs.get('learning_rate', 0.1)
            return GradientBoostingRegressor(
                n_estimators=n_estimators, 
                learning_rate=learning_rate,
                random_state=42
            )
        elif model_type == 'ridge':
            alpha = kwargs.get('alpha', 1.0)
            return Ridge(alpha=alpha, random_state=42)
        else:
            self.logger.warning(f"Unknown model type '{model_type}', defaulting to RandomForest")
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, historical_data: pd.DataFrame, prediction_type: str = 'race', 
             model_type: str = 'random_forest', **kwargs) -> Dict[str, Any]:
        """
        Train the prediction model
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical race data for training
        prediction_type : str
            Type of prediction (race or qualifying)
        model_type : str
            Type of model to use
        **kwargs : dict
            Additional parameters for model configuration
            
        Returns:
        --------
        dict
            Dictionary with model evaluation metrics
        """
        cache_key = f"model_training_{prediction_type}_{model_type}"
        if self.cache:
            cached_result = self.cache.get('prediction', key=cache_key)
            if cached_result:
                self.model = cached_result['model']
                self.feature_importance = cached_result['feature_importance']
                return cached_result['metrics']
        
        try:
            # Get relevant features and target for this prediction type
            features = self.features.get(prediction_type, self.features['race'])
            target = self.targets.get(prediction_type, self.targets['race'])
            
            # Check if target exists in data
            if target not in historical_data.columns:
                raise ValueError(f"Target variable '{target}' not found in data")
            
            # Filter to only use columns that exist in the data
            available_features = [f for f in features if f in historical_data.columns]
            
            if not available_features:
                raise ValueError("No valid features found in data")
                
            self.logger.info(f"Training {prediction_type} model with features: {available_features}")
            
            # Preprocess data
            data = self.preprocess_data(historical_data, prediction_type)
            
            # Select features and target
            X = data[available_features]
            y = data[target]
            
            # Apply feature scaling
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Select and train model
            self.model = self.select_model(model_type, **kwargs)
            self.model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_scaled, y, 
                                       cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'Feature': available_features,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=False)
            
            # Create metrics dictionary
            metrics = {
                'r2_score': r2,
                'rmse': rmse,
                'cv_rmse': cv_rmse,
                'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None
            }
            
            # Store trained model in dictionary
            self.trained_models[prediction_type] = {
                'model': self.model,
                'features': available_features,
                'scaler': self.scaler
            }
            
            # Cache results
            if self.cache:
                self.cache.set({
                    'model': self.model,
                    'feature_importance': self.feature_importance,
                    'metrics': metrics
                }, 'prediction', key=cache_key)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            return {
                'error': str(e),
                'r2_score': 0.0,
                'rmse': float('inf')
            }

    def predict(self, current_data: pd.DataFrame, prediction_type: str = 'race') -> pd.DataFrame:
        """
        Generate predictions using the trained model
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            Current race weekend data
        prediction_type : str
            Type of prediction (race or qualifying)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions
        """
        try:
            # Check if a model is trained for this prediction type
            if prediction_type not in self.trained_models:
                raise ValueError(f"No trained model available for {prediction_type} prediction")
                
            model_info = self.trained_models[prediction_type]
            model = model_info['model']
            features = model_info['features']
            scaler = model_info['scaler']
            
            # Preprocess data
            data = self.preprocess_data(current_data, prediction_type)
            
            # Check if all required features are available
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features in prediction data: {missing_features}")
            
            # Extract features
            X = data[features]
            
            # Scale features using the same scaler as training
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Add predictions to the data
            result = data.copy()
            result['PredictedPosition'] = predictions.round().astype(int)
            
            # Add prediction error columns
            target = self.targets.get(prediction_type)
            if target in result.columns:
                result['PredictionError'] = result['PredictedPosition'] - result[target]
            
            # Select relevant columns for output
            output_cols = ['Driver']
            if 'DriverAbbr' in result.columns:
                output_cols.append('DriverAbbr')
            if 'Team' in result.columns:
                output_cols.append('Team')
                
            output_cols += ['PredictedPosition']
            
            if target in result.columns:
                output_cols += [target, 'PredictionError']
            
            return result[output_cols]
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return pd.DataFrame({
                'Error': [str(e)]
            })

    def visualize_feature_importance(self) -> go.Figure:
        """
        Generate feature importance visualization
        
        Returns:
        --------
        go.Figure
            Plotly figure with feature importance visualization
        """
        try:
            if self.feature_importance is None:
                raise ValueError("No feature importance data available. Train a model first.")
                
            # Sort by importance
            sorted_importance = self.feature_importance.sort_values('Importance', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                sorted_importance,
                y='Feature',
                x='Importance',
                orientation='h',
                title='Feature Importance',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                color='Importance'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_color="black"
                ),
                plot_bgcolor="rgba(0, 0, 0, 0.05)"
            )
            
            return fig
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                annotations=[
                    dict(
                        text=f"Failed to visualize feature importance: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig

    def visualize_predictions(self, predictions: pd.DataFrame) -> go.Figure:
        """
        Create visualization of predicted vs actual positions
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            DataFrame with predictions
            
        Returns:
        --------
        go.Figure
            Plotly figure with prediction visualization
        """
        try:
            if 'PredictedPosition' not in predictions.columns:
                raise ValueError("No predictions found in data")
                
            # Create a copy to avoid modifying the original
            df = predictions.copy()
            
            # Check if we have actual positions
            has_actual = False
            actual_col = None
            
            for col in df.columns:
                if 'Position' in col and col != 'PredictedPosition' and 'Predicted' not in col:
                    has_actual = True
                    actual_col = col
                    break
            
            # Choose visualization based on available data
            if has_actual:
                # Create comparison plot of predicted vs actual
                driver_col = 'DriverAbbr' if 'DriverAbbr' in df.columns else 'Driver'
                
                # Sort by actual position
                df = df.sort_values(actual_col)
                
                # Create scatter plot with 45-degree line
                fig = go.Figure()
                
                # Add a diagonal line representing perfect prediction
                max_pos = max(df['PredictedPosition'].max(), df[actual_col].max()) + 1
                fig.add_trace(go.Scatter(
                    x=[1, max_pos],
                    y=[1, max_pos],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='Perfect Prediction'
                ))
                
                # Add the actual predictions
                fig.add_trace(go.Scatter(
                    x=df[actual_col],
                    y=df['PredictedPosition'],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color='blue',
                        opacity=0.7
                    ),
                    text=df[driver_col],
                    textposition='top center',
                    name='Predictions'
                ))
                
                # Update layout
                fig.update_layout(
                    title='Predicted vs Actual Positions',
                    xaxis_title=f'Actual Position ({actual_col})',
                    yaxis_title='Predicted Position',
                    xaxis=dict(
                        tickmode='linear',
                        tick0=1,
                        dtick=1,
                        range=[0.5, max_pos]
                    ),
                    yaxis=dict(
                        tickmode='linear',
                        tick0=1,
                        dtick=1,
                        range=[0.5, max_pos]
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_color="black"
                    ),
                    plot_bgcolor="rgba(0, 0, 0, 0.05)"
                )
                
            else:
                # Create bar chart of predictions only
                driver_col = 'DriverAbbr' if 'DriverAbbr' in df.columns else 'Driver'
                
                # Sort by predicted position
                df = df.sort_values('PredictedPosition')
                
                # Create bar chart
                fig = px.bar(
                    df,
                    x=driver_col,
                    y='PredictedPosition',
                    color='PredictedPosition',
                    title='Predicted Race Positions',
                    labels={
                        driver_col: 'Driver',
                        'PredictedPosition': 'Predicted Position'
                    }
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Driver',
                    yaxis_title='Predicted Position',
                    yaxis=dict(
                        autorange='reversed',  # Lower positions (1st, 2nd) at the top
                        tickmode='linear',
                        tick0=1,
                        dtick=1
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_color="black"
                    ),
                    plot_bgcolor="rgba(0, 0, 0, 0.05)"
                )
            
            return fig
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                annotations=[
                    dict(
                        text=f"Failed to visualize predictions: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig

    def get_historical_data(self, years: List[int], prediction_type: str = 'race') -> pd.DataFrame:
        """
        Extract historical data from multiple seasons for model training
        
        Parameters:
        -----------
        years : List[int]
            List of years to extract data from
        prediction_type : str
            Type of prediction (race or qualifying)
            
        Returns:
        --------
        pd.DataFrame
            Historical data for training
        """
        cache_key = f"historical_data_{'_'.join(map(str, years))}_{prediction_type}"
        if self.cache:
            cached_data = self.cache.get('prediction', key=cache_key)
            if cached_data is not None:
                return cached_data
                
        try:
            combined_data = []
            
            for year in years:
                self.logger.info(f"Processing historical data from {year}...")
                
                # Get events for the year
                schedule = ff1.get_event_schedule(year)
                
                for _, event in schedule.iterrows():
                    race_name = event['EventName']
                    try:
                        # Get session based on prediction type
                        if prediction_type == 'race':
                            session = ff1.get_session(year, race_name, 'R')
                        else:  # qualifying
                            session = ff1.get_session(year, race_name, 'Q')
                            
                        session.load()
                        
                        # Extract results
                        results = session.results.copy()
                        
                        # Add driver abbreviations
                        results['DriverAbbr'] = results['DriverNumber'].apply(
                            lambda x: self.get_driver_abbreviation(session, x))
                        
                        # Add race information
                        results['Year'] = year
                        results['RaceName'] = race_name
                        
                        combined_data.append(results)
                        
                    except Exception as e:
                        self.logger.warning(f"Could not load data for {race_name} {year}: {str(e)}")
                        continue
            
            if not combined_data:
                raise ValueError("No historical data could be loaded")
                
            historical_df = pd.concat(combined_data, ignore_index=True)
            
            # Cache the data
            if self.cache:
                self.cache.set(historical_df, 'prediction', key=cache_key)
                
            return historical_df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {str(e)}")
            return pd.DataFrame()
