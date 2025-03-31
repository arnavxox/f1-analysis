import dash
from dash import Dash
import dash_bootstrap_components as dbc
from modules.data_cache import F1DataCache
from dashboards.race_dashboard import create_race_dashboard

def create_app():
    # Initialize cache with reasonable defaults
    cache = F1DataCache(cache_dir='./f1_cache', max_memory_items=1000)
    
    # Create the dashboard application
    app = create_race_dashboard(cache)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
