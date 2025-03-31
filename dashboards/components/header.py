import dash_bootstrap_components as dbc

def create_header():
    return dbc.Navbar(
        [
            dbc.Container([
                dbc.NavbarBrand("F1 Analysis Pro", href="#"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Telemetry", href="#telemetry")),
                    dbc.NavItem(dbc.NavLink("Strategy", href="#strategy")),
                    dbc.NavItem(dbc.NavLink("Race Progress", href="#progress")),
                    dbc.NavItem(dbc.NavLink("Lap Analysis", href="#laptimes")),
                    dbc.NavItem(dbc.NavLink("Compare", href="#compare"))
                ], className="ml-auto")
            ])
        ],
        color="dark",
        dark=True,
        sticky="top"
    )
