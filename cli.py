import click
from modules.data_cache import F1DataCache
from modules.telemetry import TelemetryAnalyzer

@click.group()
def cli():
    """F1 Analysis CLI Tool"""
    pass

@cli.command()
@click.option('--year', type=int, required=True)
@click.option('--race', type=str, required=True)
@click.option('--driver', type=str, required=True)
def generate_speedtrace(year, race, driver):
    """Generate speed trace image for a driver"""
    cache = F1DataCache('./f1_cache')  # Only change - specify cache directory
    analyzer = TelemetryAnalyzer(cache)
    
    fig = analyzer.get_speed_trace(year, race, driver)
    # Changed from fig.savefig to write_image since we now use Plotly figures
    fig.write_image(f"{driver}_{race}_{year}_speed.png")
    click.echo(f"Saved speed trace to {driver}_{race}_{year}_speed.png")

if __name__ == '__main__':
    cli()
