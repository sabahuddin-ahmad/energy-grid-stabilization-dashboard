# Step 1: Import necessary libraries
from pathlib import Path
from typing import Optional

import pandas as pd
from dash import Dash, html, dcc, Output, Input, State
import plotly.express as px

mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"


COUNTRY_REGION_CITY_DATA = {
    "Canada": {
        "Alberta": [
            {"city": "Calgary", "lat": 51.0447, "lon": -114.0719},
            {"city": "Edmonton", "lat": 53.5461, "lon": -113.4938},
        ],
        "British Columbia": [
            {"city": "Vancouver", "lat": 49.2827, "lon": -123.1207},
            {"city": "Victoria", "lat": 48.4284, "lon": -123.3656},
            {"city": "Kelowna", "lat": 49.8880, "lon": -119.4960},
        ],
        "Manitoba": [
            {"city": "Winnipeg", "lat": 49.8951, "lon": -97.1384},
        ],
        "New Brunswick": [
            {"city": "Fredericton", "lat": 45.9636, "lon": -66.6431},
            {"city": "Saint John", "lat": 45.2733, "lon": -66.0633},
        ],
        "Newfoundland and Labrador": [
            {"city": "St. John's", "lat": 47.5615, "lon": -52.7126},
        ],
        "Nova Scotia": [
            {"city": "Halifax", "lat": 44.6488, "lon": -63.5752},
            {"city": "Sydney", "lat": 46.1368, "lon": -60.1942},
        ],
        "Ontario": [
            {"city": "Toronto", "lat": 43.6532, "lon": -79.3832},
            {"city": "Ottawa", "lat": 45.4215, "lon": -75.6972},
            {"city": "Hamilton", "lat": 43.2557, "lon": -79.8711},
            {"city": "London", "lat": 42.9849, "lon": -81.2453},
        ],
        "Prince Edward Island": [
            {"city": "Charlottetown", "lat": 46.2382, "lon": -63.1311},
        ],
        "Quebec": [
            {"city": "Montreal", "lat": 45.5017, "lon": -73.5673},
            {"city": "Quebec City", "lat": 46.8139, "lon": -71.2080},
            {"city": "Gatineau", "lat": 45.4765, "lon": -75.7013},
        ],
        "Saskatchewan": [
            {"city": "Saskatoon", "lat": 52.1332, "lon": -106.6700},
            {"city": "Regina", "lat": 50.4452, "lon": -104.6189},
        ],
        "Yukon": [
            {"city": "Whitehorse", "lat": 60.7212, "lon": -135.0568},
        ],
        "Northwest Territories": [
            {"city": "Yellowknife", "lat": 62.4540, "lon": -114.3718},
        ],
        "Nunavut": [
            {"city": "Iqaluit", "lat": 63.7467, "lon": -68.5169},
        ],
    },
    "United States": {
        "California": [
            {"city": "San Francisco", "lat": 37.7749, "lon": -122.4194},
            {"city": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        ],
        "New York": [
            {"city": "New York City", "lat": 40.7128, "lon": -74.0060},
            {"city": "Buffalo", "lat": 42.8864, "lon": -78.8784},
        ],
        "Texas": [
            {"city": "Austin", "lat": 30.2672, "lon": -97.7431},
            {"city": "Houston", "lat": 29.7604, "lon": -95.3698},
        ],
        "Washington": [
            {"city": "Seattle", "lat": 47.6062, "lon": -122.3321},
        ],
    },
    "United Kingdom": {
        "England": [
            {"city": "London", "lat": 51.5074, "lon": -0.1278},
            {"city": "Manchester", "lat": 53.4808, "lon": -2.2426},
        ],
        "Scotland": [
            {"city": "Edinburgh", "lat": 55.9533, "lon": -3.1883},
            {"city": "Glasgow", "lat": 55.8642, "lon": -4.2518},
        ],
        "Wales": [
            {"city": "Cardiff", "lat": 51.4816, "lon": -3.1791},
        ],
    },
    "Australia": {
        "New South Wales": [
            {"city": "Sydney", "lat": -33.8688, "lon": 151.2093},
        ],
        "Victoria": [
            {"city": "Melbourne", "lat": -37.8136, "lon": 144.9631},
        ],
        "Queensland": [
            {"city": "Brisbane", "lat": -27.4698, "lon": 153.0251},
        ],
    },
}

CITY_RECORDS = [
    {
        "Country": country,
        "Region": region,
        "City": entry["city"],
        "Latitude": entry["lat"],
        "Longitude": entry["lon"],
    }
    for country, regions in COUNTRY_REGION_CITY_DATA.items()
    for region, cities in regions.items()
    for entry in cities
]

MAP_LOCATIONS_DF = pd.DataFrame(CITY_RECORDS)

COUNTRY_OPTIONS = (
    [{"label": "All Countries", "value": "All"}]
    + [
        {"label": country, "value": country}
        for country in sorted(COUNTRY_REGION_CITY_DATA.keys())
    ]
)

px.set_mapbox_access_token(mapbox_access_token)


def get_region_options(country: str):
    if country == "All":
        regions = MAP_LOCATIONS_DF["Region"].unique()
    else:
        regions = MAP_LOCATIONS_DF.loc[
            MAP_LOCATIONS_DF["Country"] == country, "Region"
        ].unique()
    region_options = [
        {"label": region, "value": region}
        for region in sorted(regions)
    ]
    if region_options:
        return [{"label": "All Regions", "value": "All"}] + region_options
    return region_options


def get_city_options(country: str, region: str):
    cities = MAP_LOCATIONS_DF.copy()
    if country != "All":
        cities = cities[cities["Country"] == country]
    if region != "All":
        cities = cities[cities["Region"] == region]

    city_options = [
        {"label": city, "value": city}
        for city in sorted(cities["City"].unique())
    ]
    if city_options:
        return [{"label": "All Cities", "value": "All"}] + city_options
    return city_options


def build_map_figure(
    country: str = "All", region: str = "All", city: Optional[str] = None
):
    filtered = MAP_LOCATIONS_DF.copy()
    if country != "All":
        filtered = filtered[filtered["Country"] == country].copy()
    if region != "All":
        filtered = filtered[filtered["Region"] == region].copy()
    if city and city != "All":
        filtered = filtered[filtered["City"] == city].copy()

    if filtered.empty:
        filtered = MAP_LOCATIONS_DF

    center_lat = filtered["Latitude"].mean()
    center_lon = filtered["Longitude"].mean()

    if city and city != "All":
        zoom = 6
    elif region != "All":
        zoom = 4.5
    elif country != "All":
        zoom = 4.5
    else:
        zoom = 2.5

    fig = px.scatter_mapbox(
        filtered,
        lat="Latitude",
        lon="Longitude",
        hover_name="City",
        hover_data={"Country": True, "Region": True},
        color="Country",
        size_max=20,
        zoom=zoom,
        height=500,
    )

    fig.update_layout(
        mapbox=dict(
            center={"lat": center_lat, "lon": center_lon},
            style="carto-darkmatter",
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#FFFFFF"),
    )

    return fig


INITIAL_REGION_OPTIONS = get_region_options("All")
INITIAL_CITY_OPTIONS = get_city_options("All", "All")
INITIAL_MAP_FIG = build_map_figure()

# Step 2: Load your dataset from the local data directory
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "PUB_Demand_2025_v275.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Expected dataset at {DATA_FILE}. Place the CSV file in the data/ folder."
    )

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

# Step 3: Prepare demand vs date data
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
if df["Date"].isna().all():
    raise ValueError("Failed to parse Date values from the dataset.")

demand_columns = [col for col in ("Market Demand", "Ontario Demand") if col in df.columns]
if not demand_columns:
    raise ValueError("Expected demand columns ('Market Demand', 'Ontario Demand') not found in dataset.")

daily_demand = (
    df.groupby("Date", as_index=False)[demand_columns]
    .mean()
    .sort_values("Date")
)
long_demand = daily_demand.melt(
    id_vars="Date",
    value_vars=demand_columns,
    var_name="Demand Type",
    value_name="Demand",
)

# Step 4: Create the Plotly Graph
# Line chart of demand vs date
fig = px.line(
    long_demand,
    x="Date",
    y="Demand",
    color="Demand Type",
    markers=True,
    title="Electricity Demand vs Date",
    labels={"Date": "Date", "Demand": "Demand (MW)", "Demand Type": "Series"},
)
fig.update_layout(
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="#FFFFFF"),
    legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="#FFFFFF")),
    xaxis=dict(gridcolor="#333333", zerolinecolor="#555555"),
    yaxis=dict(gridcolor="#333333", zerolinecolor="#555555"),
)
fig.update_traces(line=dict(width=3))

# Step 4: Initialize the Dash App
app = Dash(__name__)

# Step 5: Define the App Layout
# The layout defines the structure of your dashboard.
# It uses HTML components (html.Div, html.H1, etc.) and Dash Core Components (dcc.Graph).
app.layout = html.Div(children=[
    html.H1(
        children=[
            "GridSense",
            html.Span(
                "by",
                style={
                    'font-size': '0.6em',
                    'margin-left': '0.5rem',
                    'font-weight': 'normal'
                },
            ),
            html.Img(
                src='/assets/QuanTech.png',
                style={
                    'width': '220px',
                    'margin-left': '0.5rem',
                    'background': 'transparent'
                },
            ),
        ],
        style={
            'textAlign': 'center',
            'font-size': 40,
            'padding': '40px 10px 40px 10px',
            'color': '#FFFFFF',
            'fontFamily': 'sans-serif',
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'margin': '0 auto',
            'gap': '0.5rem'
        },
    ),
    
    html.Div(children='''
        Daily electricity demand derived from PUB_Demand_2025_v275.csv.
    '''),

    html.Div(
        [
            dcc.Graph(
                id='world-map',
                figure=INITIAL_MAP_FIG,
                style={
                    'height': '600px',
                    'width': '100%'
                }
            ),
            html.Div(
                [
                    html.Label(
                        "Country",
                        htmlFor='country-dropdown',
                        style={'color': '#FFFFFF', 'fontWeight': '600'}
                    ),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=COUNTRY_OPTIONS,
                        value='All',
                        clearable=False,
                        style={'marginBottom': '15px'}
                    ),
                    html.Label(
                        "Region / Province",
                        htmlFor='region-dropdown',
                        style={'color': '#FFFFFF', 'fontWeight': '600'}
                    ),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=INITIAL_REGION_OPTIONS,
                        value='All',
                        clearable=False,
                        style={'marginBottom': '15px'}
                    ),
                    html.Label(
                        "City",
                        htmlFor='city-dropdown',
                        style={'color': '#FFFFFF', 'fontWeight': '600'}
                    ),
                    dcc.Dropdown(
                        id='city-dropdown',
                        options=INITIAL_CITY_OPTIONS,
                        value='All',
                        clearable=False,
                        style={'marginBottom': '15px'}
                    ),
                    html.P(
                        "Drag the map or zoom to explore anywhere in the world.",
                        style={'color': '#EEEEEE', 'fontSize': '0.9rem'}
                    ),
                ],
                style={
                    'position': 'absolute',
                    'top': '20px',
                    'left': '20px',
                    'maxWidth': '320px',
                    'background': 'rgba(0, 0, 0, 0.35)',
                    'padding': '20px',
                    'borderRadius': '12px',
                    'backdropFilter': 'blur(6px)',
                    'boxShadow': '0 8px 24px rgba(0, 0, 0, 0.3)',
                    'zIndex': 10,
                }
            ),
        ],
        style={
            'position': 'relative',
            'width': '100%',
            'marginTop': '30px',
        }
    ),
    dcc.Graph(
        id='demand-line-chart',
        figure=fig
    )
], style={"background-color": "#000000", 'padding': '10px 10px 10px 10px'})



# app.layout = html.Div(
#     children=[
#         html.Div(
#             className="row",
#             children=[
#                 # Column for user controls
#                 html.Div(
#                     className="four columns div-user-controls",
#                     children=[
#                         html.A(
#                             html.Img(
#                                 className="logo",
#                                 src=app.get_asset_url("dash-logo-new.png"),
#                             ),
#                             href="https://plotly.com/dash/",
#                         ),
#                         html.H2("DASH - UBER DATA APP"),
#                         html.P(
#                             """Select different days using the date picker or by selecting
#                             different time frames on the histogram."""
#                         ),
#                         html.Div(
#                             className="div-for-dropdown",
#                             children=[
#                                 dcc.Dropdown(
#                                     id="date-picker",
#                                     options=[
#                                         {"label": i, "value": i}
#                                         for i in [
#                                             "2014-04-01",
#                                             "2014-04-02",
#                                             "2014-04-03",
#                                             "2014-04-04",
#                                             "2014-04-05",
#                                             "2014-04-06",
#                                             "2014-04-07",
#                                         ]
#                                     ],
#                                     style={"border": "0px solid black"},
#                                 )
#                             ],
#                         ),
#                         # Change to side-by-side for mobile layout
#                         html.Div(
#                             className="row",
#                             children=[
#                                 html.Div(
#                                     className="div-for-dropdown",
#                                     children=[
#                                         # Dropdown for locations on map
#                                         dcc.Dropdown(
#                                             id="location-dropdown",
#                                             options=[
#                                                 'Calgary', 'Edmonton', 'Halifax', 'Montreal', 'Ottawa', 'Quebec City', 'Toronto', 'Vancouver', 'Victoria', 'Winnipeg'
#                                             ],
#                                             placeholder="Select a location",
#                                         )
#                                     ],
#                                 ),
#                                 html.Div(
#                                     className="div-for-dropdown",
#                                     children=[
#                                         # Dropdown to select times
#                                         dcc.Dropdown(
#                                             id="bar-selector",
#                                             options=[
#                                                 {
#                                                     "label": str(n) + ":00",
#                                                     "value": str(n),
#                                                 }
#                                                 for n in range(24)
#                                             ],
#                                             multi=True,
#                                             placeholder="Select certain hours",
#                                         )
#                                     ],
#                                 ),
#                             ],
#                         ),
#                         html.P(id="total-rides"),
#                         html.P(id="total-rides-selection"),
#                         html.P(id="date-value"),
#                         dcc.Markdown(
#                             """
#                             Source: [FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response/tree/master/uber-trip-data)

#                             Links: [Source Code](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-uber-rides-demo) | [Enterprise Demo](https://plotly.com/get-demo/)
#                             """
#                         ),
#                     ],
#                 ),
#                 # Column for app graphs and plots
#                 html.Div(
#                     className="eight columns div-for-charts bg-grey",
#                     children=[
#                         dcc.Graph(id="map-graph"),
#                         html.Div(
#                             className="text-padding",
#                             children=[
#                                 "Select any of the bars on the histogram to section data by time."
#                             ],
#                         ),
#                         dcc.Graph(id="histogram"),
#                     ],
#                 ),
#             ],
#         )
#     ]
# )


# Interactivity callbacks
@app.callback(
    [Output('region-dropdown', 'options'), Output('region-dropdown', 'value')],
    Input('country-dropdown', 'value'),
    State('region-dropdown', 'value'),
)
def update_region_dropdown(country_value, current_region):
    options = get_region_options(country_value)
    option_values = {opt['value'] for opt in options}

    if current_region in option_values:
        new_value = current_region
    elif 'All' in option_values:
        new_value = 'All'
    elif options:
        new_value = options[0]['value']
    else:
        new_value = None

    return options, new_value


@app.callback(
    [Output('city-dropdown', 'options'), Output('city-dropdown', 'value')],
    Input('country-dropdown', 'value'),
    Input('region-dropdown', 'value'),
    State('city-dropdown', 'value'),
)
def update_city_dropdown(country_value, region_value, current_city):
    options = get_city_options(country_value, region_value or 'All')
    option_values = {opt['value'] for opt in options}

    if current_city in option_values:
        new_value = current_city
    elif 'All' in option_values:
        new_value = 'All'
    elif options:
        new_value = options[0]['value']
    else:
        new_value = None

    return options, new_value


@app.callback(
    Output('world-map', 'figure'),
    Input('country-dropdown', 'value'),
    Input('region-dropdown', 'value'),
    Input('city-dropdown', 'value'),
)
def update_map_figure(country_value, region_value, city_value):
    return build_map_figure(country_value, region_value or 'All', city_value)


# Step 6: Run the Dashboard
if __name__ == '__main__':
    app.run_server(debug=True)
