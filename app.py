# Step 1: Import necessary libraries
from pathlib import Path
from typing import Optional
import pandas as pd
from dash import Dash, html, dcc, Output, Input, State
import plotly.graph_objects as go


# mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

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

# Simplified rectangular polygons to visually differentiate Canadian provinces/territories on the
# map. These are coarse placeholders that can be replaced with high-resolution GeoJSON data when
# available.
CANADA_PROVINCE_POLYGONS = {
    "British Columbia": [
        (60.0, -139.0),
        (48.3, -139.0),
        (48.3, -114.0),
        (60.0, -114.0),
        (60.0, -139.0),
    ],
    "Alberta": [
        (60.0, -120.0),
        (49.0, -120.0),
        (49.0, -110.0),
        (60.0, -110.0),
        (60.0, -120.0),
    ],
    "Saskatchewan": [
        (60.0, -110.0),
        (49.0, -110.0),
        (49.0, -101.0),
        (60.0, -101.0),
        (60.0, -110.0),
    ],
    "Manitoba": [
        (60.0, -101.0),
        (49.0, -101.0),
        (49.0, -94.0),
        (60.0, -94.0),
        (60.0, -101.0),
    ],
    "Ontario": [
        (56.0, -95.0),
        (42.0, -95.0),
        (42.0, -74.0),
        (56.0, -74.0),
        (56.0, -95.0),
    ],
    "Quebec": [
        (62.0, -79.0),
        (45.0, -79.0),
        (45.0, -57.0),
        (62.0, -57.0),
        (62.0, -79.0),
    ],
    "New Brunswick": [
        (48.0, -69.0),
        (44.8, -69.0),
        (44.8, -63.0),
        (48.0, -63.0),
        (48.0, -69.0),
    ],
    "Nova Scotia": [
        (47.2, -66.0),
        (43.4, -66.0),
        (43.4, -60.0),
        (47.2, -60.0),
        (47.2, -66.0),
    ],
    "Prince Edward Island": [
        (47.0, -64.5),
        (46.0, -64.5),
        (46.0, -62.0),
        (47.0, -62.0),
        (47.0, -64.5),
    ],
    "Newfoundland and Labrador": [
        (60.0, -61.0),
        (47.0, -61.0),
        (47.0, -52.0),
        (60.0, -52.0),
        (60.0, -61.0),
    ],
    "Yukon": [
        (69.0, -141.0),
        (60.0, -141.0),
        (60.0, -129.0),
        (69.0, -129.0),
        (69.0, -141.0),
    ],
    "Northwest Territories": [
        (75.0, -129.0),
        (60.0, -129.0),
        (60.0, -102.0),
        (75.0, -102.0),
        (75.0, -129.0),
    ],
    "Nunavut": [
        (83.0, -110.0),
        (62.0, -110.0),
        (62.0, -65.0),
        (83.0, -65.0),
        (83.0, -110.0),
    ],
}

CANADA_PROVINCE_COLORS = {
    "British Columbia": "#1f77b4",
    "Alberta": "#ff7f0e",
    "Saskatchewan": "#2ca02c",
    "Manitoba": "#d62728",
    "Ontario": "#9467bd",
    "Quebec": "#8c564b",
    "New Brunswick": "#e377c2",
    "Nova Scotia": "#7f7f7f",
    "Prince Edward Island": "#bcbd22",
    "Newfoundland and Labrador": "#17becf",
    "Yukon": "#f1c40f",
    "Northwest Territories": "#e67e22",
    "Nunavut": "#16a085",
}


def normalize_region_value(region: Optional[str]) -> Optional[str]:
    if not region or region in {"All", "All Regions"}:
        return "All"
    return region


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


def build_map_figure(
    country: str = "All", region: str = "All"
):
    fig = go.Figure()

    if country == "Canada":
        normalized_region = normalize_region_value(region)
        selected_province = (
            normalized_region if normalized_region and normalized_region != "All" else None
        )
        for province, polygon in CANADA_PROVINCE_POLYGONS.items():
            latitudes = [coord[0] for coord in polygon]
            longitudes = [coord[1] for coord in polygon]
            color = CANADA_PROVINCE_COLORS.get(province, "#2c3e50")
            is_selected = selected_province == province or selected_province is None

            fig.add_trace(
                go.Scattermapbox(
                    lat=latitudes,
                    lon=longitudes,
                    mode="lines",
                    fill="toself",
                    hoverinfo="text",
                    text=province,
                    name=province,
                    fillcolor=color,
                    line=dict(
                        width=2 if selected_province == province else 1,
                        color="#0F2530",
                    ),
                    opacity=0.75 if is_selected else 0.25,
                )
            )

        if selected_province and selected_province in CANADA_PROVINCE_POLYGONS:
            province_coords = CANADA_PROVINCE_POLYGONS[selected_province]
            center_lat = sum(coord[0] for coord in province_coords) / len(province_coords)
            center_lon = sum(coord[1] for coord in province_coords) / len(province_coords)
            zoom = 4.2
        else:
            center_lat = 56.0
            center_lon = -96.0
            zoom = 3.2

        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center={"lat": center_lat, "lon": center_lon},
                zoom=zoom,
                accesstoken=None,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="#FFFFFF"),
            showlegend=False,
        )
    else:
        filtered = MAP_LOCATIONS_DF.copy()
        if country and country != "All":
            filtered = filtered[filtered["Country"] == country]
        if region and region != "All":
            filtered = filtered[filtered["Region"] == region]

        if filtered.empty:
            filtered = MAP_LOCATIONS_DF

        hover_text = (
            filtered["City"]
            + " — "
            + filtered["Region"]
            + ", "
            + filtered["Country"]
        )

        fig.add_trace(
            go.Scattermapbox(
                lat=filtered["Latitude"],
                lon=filtered["Longitude"],
                mode="markers",
                marker=dict(size=9, color="#00B5FF"),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        center_lat = filtered["Latitude"].mean()
        center_lon = filtered["Longitude"].mean()
        zoom = 1.8 if (country == "All" or not country) else 4.0

        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center={"lat": center_lat, "lon": center_lon},
                zoom=zoom,
                accesstoken=None,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="#FFFFFF"),
            showlegend=False,
        )

    return fig


INITIAL_REGION_OPTIONS = get_region_options("Canada")
INITIAL_MAP_FIG = build_map_figure("Canada", "All")

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

if "Load" not in df.columns:
    if "Market Demand" in df.columns:
        df = df.rename(columns={"Market Demand": "Load"})
    else:
        raise ValueError("Expected a 'Market Demand' or 'Load' column in the dataset.")

BASE_DEMAND_DF = df.copy()

PROVINCE_FOLDERS = {
    "Alberta": DATA_DIR / "Alberta",
    "British Columbia": DATA_DIR / "British_Columbia",
    "Manitoba": DATA_DIR / "Manitoba",
    "New Brunswick": DATA_DIR / "New_Brunswick",
    "Newfoundland and Labrador": DATA_DIR / "Newfoundland_and_Labrador",
    "Nova Scotia": DATA_DIR / "Nova_Scotia",
    "Ontario": DATA_DIR / "Ontario",
    "Prince Edward Island": DATA_DIR / "Prince_Edward_Island",
    "Quebec": DATA_DIR / "Quebec",
    "Saskatchewan": DATA_DIR / "Saskatchewan",
    "Yukon": DATA_DIR / "Yukon",
    "Northwest Territories": DATA_DIR / "Northwest_Territories",
    "Nunavut": DATA_DIR / "Nunavut",
}

for folder_path in PROVINCE_FOLDERS.values():
    folder_path.mkdir(exist_ok=True)


def load_region_dataframe(country: str, region: Optional[str]) -> pd.DataFrame:
    """Load a region-specific dataset if one exists, otherwise fall back to the base file."""

    normalized_region = normalize_region_value(region)

    if country == "Canada" and normalized_region and normalized_region != "All":
        target_folder = PROVINCE_FOLDERS.get(normalized_region)
        if target_folder and target_folder.exists():
            csv_candidates = sorted(target_folder.glob("*.csv"))
            for csv_path in reversed(csv_candidates):
                try:
                    region_df = pd.read_csv(csv_path, encoding="utf-8-sig")
                    region_df["Date"] = pd.to_datetime(region_df["Date"], errors="coerce")
                    if region_df["Date"].notna().any():
                        if "Load" not in region_df.columns and "Market Demand" in region_df.columns:
                            region_df = region_df.rename(columns={"Market Demand": "Load"})
                        return region_df
                except Exception:
                    continue

    return BASE_DEMAND_DF.copy()


def prepare_daily_demand(country: str, region: Optional[str]) -> pd.DataFrame:
    normalized_region = normalize_region_value(region)
    dataset = load_region_dataframe(country, normalized_region)
    dataset = dataset.dropna(subset=["Date"]).copy()

    if "Load" not in dataset.columns:
        raise ValueError("The selected dataset must contain a 'Load' column.")

    daily = (
        dataset.groupby("Date", as_index=False)["Load"]
        .mean()
        .sort_values("Date")
    )

    if daily.empty:
        daily["Grid Capacity"] = pd.Series(dtype="float64")
        return daily

    daily["Grid Capacity"] = (
        daily["Load"].rolling(window=7, min_periods=1).mean() * 1.05
    )

    return daily


def build_demand_figure(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if daily.empty:
        fig.update_layout(
            title="Load vs Grid Capacity",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="#FFFFFF"),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Load"],
            mode="lines+markers",
            line=dict(color="#00B5FF", width=3),
            marker=dict(size=6),
            name="Load",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Load: %{y:.0f} MW<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Grid Capacity"],
            mode="lines",
            line=dict(color="#F39C12", width=2, dash="dot"),
            name="Grid Capacity",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Grid Capacity: %{y:.0f} MW<extra></extra>",
        )
    )

    fig.update_layout(
        title="Load vs Grid Capacity",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#FFFFFF"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(color="#FFFFFF")),
        xaxis=dict(gridcolor="#333333", zerolinecolor="#555555"),
        yaxis=dict(gridcolor="#333333", zerolinecolor="#555555"),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#021F2C", font=dict(color="#FFFFFF")),
    )

    return fig


def describe_scope(context: Optional[dict]) -> str:
    if not context:
        return "All data"

    country = context.get("country", "All")
    region = context.get("region", "All")

    if country == "Canada":
        if region and region != "All":
            return f"{region}, Canada"
        return "Canada"

    if country == "All":
        if region and region != "All":
            return region
        return "Global"

    if region and region != "All":
        return f"{region}, {country}"

    return country


def build_detail_panel(
    daily: pd.DataFrame, click_data: Optional[dict], context: Optional[dict]
) -> html.Div:
    scope_label = describe_scope(context)

    if daily.empty:
        return html.Div(
            f"No demand data available for {scope_label}.",
            style={
                "color": "#FFFFFF",
                "textAlign": "center",
                "fontSize": "24px",
                "fontWeight": "600",
                "marginTop": "16px",
            },
        )

    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily = daily.dropna(subset=["Date"])

    if daily.empty:
        return html.Div(
            f"No demand data available for {scope_label}.",
            style={
                "color": "#FFFFFF",
                "textAlign": "center",
                "fontSize": "24px",
                "fontWeight": "600",
                "marginTop": "16px",
            },
        )

    if click_data and click_data.get("points"):
        selected_point = click_data["points"][0]
        selected_date = pd.to_datetime(selected_point["x"], errors="coerce")
        if not pd.isna(selected_date):
            selected_date = selected_date.tz_localize(None) if getattr(selected_date, "tzinfo", None) else selected_date
            selected_date = selected_date.normalize()
        show_hint = False
    else:
        selected_date = daily["Date"].max()
        show_hint = True

    if pd.isna(selected_date) or selected_date not in daily["Date"].values:
        nearest_idx = daily["Date"].sub(selected_date).abs().idxmin()
        selected_date = daily.loc[nearest_idx, "Date"]

    row = daily.loc[daily["Date"] == selected_date].iloc[0]
    load_value = row.get("Load", float("nan"))
    grid_capacity_value = row.get("Grid Capacity", float("nan"))

    if pd.isna(grid_capacity_value):
        status = "Capacity Unknown"
        status_color = "#BDC3C7"
        difference_text = "Capacity data unavailable"
    else:
        difference = grid_capacity_value - load_value
        status = "Surplus" if difference >= 0 else "Deficit"
        status_color = "#2ECC71" if difference >= 0 else "#E74C3C"
        difference_text = f"{status} {difference:+,.0f} MW"

    info_blocks = [
        html.Div(
            scope_label,
            style={
                "fontSize": "22px",
                "color": "#8FA9B5",
                "marginBottom": "12px",
                "textTransform": "uppercase",
                "letterSpacing": "0.08em",
            },
        )
    ]

    if show_hint:
        info_blocks.append(
            html.Div(
                "Click on the plot to explore specific forecasting information.",
                style={
                    "fontSize": "20px",
                    "color": "#8FA9B5",
                    "marginBottom": "8px",
                },
            )
        )

    info_blocks.extend(
        [
            html.Div(
                selected_date.strftime("%B %d, %Y"),
                style={
                    "fontSize": "38px",
                    "fontWeight": "700",
                    "letterSpacing": "0.04em",
                },
            ),
            html.Div(
                [
                    html.Span(
                        f"Load: {load_value:,.0f} MW",
                        style={"marginRight": "24px"},
                    ),
                    html.Span(f"Grid Capacity: {grid_capacity_value:,.0f} MW"),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "gap": "16px",
                    "fontSize": "24px",
                    "marginTop": "16px",
                },
            ),
            html.Div(
                difference_text,
                style={
                    "marginTop": "18px",
                    "fontSize": "34px",
                    "fontWeight": "700",
                    "color": status_color,
                },
            ),
        ]
    )

    return html.Div(
        info_blocks,
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
        },
    )


INITIAL_DAILY = prepare_daily_demand("Canada", "All")
INITIAL_CONTEXT = {"country": "Canada", "region": "All"}
INITIAL_FIGURE = build_demand_figure(INITIAL_DAILY)
INITIAL_DETAILS = build_detail_panel(INITIAL_DAILY, None, INITIAL_CONTEXT)
INITIAL_DAILY_EXPORT = INITIAL_DAILY.copy()
if not INITIAL_DAILY_EXPORT.empty:
    INITIAL_DAILY_EXPORT["Date"] = INITIAL_DAILY_EXPORT["Date"].dt.strftime("%Y-%m-%d")

# Step 4: Initialize the Dash App
app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "GridSense"
server = app.server

# Step 5: Define the App Layout
# The layout defines the structure of your dashboard.
# It uses HTML components (html.Div, html.H1, etc.) and Dash Core Components (dcc.Graph).
app.layout = html.Div(
    id="root",
    style={
        "backgroundColor": "#001821",
        "minHeight": "100vh",
        "padding": "0 16px 32px",
    },
    children=[
        dcc.Store(
            id="daily-demand-store",
            data={
                "daily": INITIAL_DAILY_EXPORT.to_dict("records"),
                "context": INITIAL_CONTEXT,
            },
        ),
        html.H1(
            children=[
                "GridSense",
                html.Span(
                    "by",
                    style={
                        "fontSize": "0.6em",
                        "marginLeft": "0.5rem",
                        "marginTop": "1.5rem",
                        "fontWeight": "normal",
                        "fontStyle": "italic",
                    },
                ),
                html.Img(
                    src="/assets/QuanTech.png",
                    style={
                        "width": "220px",
                        "marginLeft": "0.8rem",
                        "marginBottom": "1.2rem",
                        "background": "transparent",
                    },
                ),
            ],
            style={
                "textAlign": "center",
                "fontSize": 40,
                "padding": "28px 10px",
                "color": "#FFFFFF",
                "fontFamily": "sans-serif",
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "margin": "0 auto",
                "gap": "0.5rem",
                "fontWeight": "bold",
                "position": "sticky",
                "top": 0,
                "zIndex": 1000,
                "backgroundColor": "#001821",
                "boxShadow": "0 6px 18px rgba(0, 0, 0, 0.35)",
            },
        ),
        html.Div(
            style={"maxWidth": "1200px", "margin": "0 auto"},
            children=[
                html.Div(
                    [
                        dcc.Graph(
                            id="world-map",
                            figure=INITIAL_MAP_FIG,
                            style={"height": "600px", "width": "100%"},
                            config={"scrollZoom": True, "displayModeBar": True},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Country",
                                    htmlFor="country-dropdown",
                                    style={"color": "#FFFFFF", "fontWeight": "600"},
                                ),
                                dcc.Dropdown(
                                    id="country-dropdown",
                                    options=COUNTRY_OPTIONS,
                                    value="All",
                                    clearable=False,
                                    style={"marginBottom": "15px"},
                                ),
                                html.Label(
                                    "Region / Province",
                                    htmlFor="region-dropdown",
                                    style={"color": "#FFFFFF", "fontWeight": "600"},
                                ),
                                dcc.Dropdown(
                                    id="region-dropdown",
                                    options=INITIAL_REGION_OPTIONS,
                                    value="All",
                                    clearable=False,
                                    style={"marginBottom": "15px"},
                                ),
                                html.P(
                                    "Drag the map or zoom to explore anywhere in the world.",
                                    style={"color": "#EEEEEE", "fontSize": "1.2rem"},
                                ),
                            ],
                            style={
                                "position": "absolute",
                                "top": "20px",
                                "left": "20px",
                                "maxWidth": "320px",
                                "background": "rgba(0, 0, 0, 0.35)",
                                "padding": "20px",
                                "borderRadius": "12px",
                                "backdropFilter": "blur(6px)",
                                "boxShadow": "0 8px 24px rgba(0, 0, 0, 0.3)",
                                "zIndex": 10,
                            },
                        ),
                    ],
                    style={
                        "position": "relative",
                        "width": "100%",
                        "marginTop": "30px",
                    },
                ),
                dcc.Graph(
                    id="demand-line-chart",
                    figure=INITIAL_FIGURE,
                    config={"displayModeBar": False},
                    style={"marginTop": "32px"},
                ),
                html.Div(
                    id="market-click-details",
                    children=INITIAL_DETAILS,
                    style={
                        "color": "#FFFFFF",
                        "textAlign": "center",
                        "marginTop": "24px",
                        "fontSize": "26px",
                        "fontWeight": "600",
                    },
                ),
                html.Div(
                    style={"marginTop": "36px"},
                    children=[
                        html.Label(
                            "Recommendations",
                            htmlFor="recommendations-box",
                            style={"color": "#FFFFFF", "fontWeight": "600", "fontSize": "20px"},
                        ),
                        dcc.Textarea(
                            id="recommendations-box",
                            placeholder="Capture operational recommendations, mitigation steps, or notes here...",
                            style={
                                "width": "100%",
                                "height": "180px",
                                "backgroundColor": "#021F2C",
                                "color": "#FFFFFF",
                                "border": "1px solid #0A3A4A",
                                "borderRadius": "8px",
                                "padding": "14px",
                                "fontSize": "17px",
                                "lineHeight": "1.5",
                                "marginTop": "12px",
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
)



@app.callback(
    [Output("demand-line-chart", "figure"), Output("daily-demand-store", "data")],
    Input("country-dropdown", "value"),
    Input("region-dropdown", "value"),
)
def update_demand_chart(country_value, region_value):
    normalized_region = normalize_region_value(region_value)
    country_scope = country_value or "All"
    daily = prepare_daily_demand(country_scope, normalized_region)
    figure = build_demand_figure(daily)
    daily_export = daily.copy()
    if not daily_export.empty:
        daily_export["Date"] = daily_export["Date"].dt.strftime("%Y-%m-%d")
    store_payload = {
        "daily": daily_export.to_dict("records"),
        "context": {
            "country": country_scope,
            "region": normalized_region,
        },
    }
    return figure, store_payload


@app.callback(
    [Output("region-dropdown", "options"), Output("region-dropdown", "value")],
    Input("country-dropdown", "value"),
    State("region-dropdown", "value"),
)
def update_region_dropdown(country_value, current_region):
    options = get_region_options(country_value)
    option_values = {opt["value"] for opt in options}

    if current_region in option_values:
        new_value = current_region
    elif "All" in option_values:
        new_value = "All"
    elif options:
        new_value = options[0]["value"]
    else:
        new_value = None

    return options, new_value


@app.callback(
    Output("world-map", "figure"),
    Input("country-dropdown", "value"),
    Input("region-dropdown", "value"),
)
def update_map_figure(country_value, region_value):
    return build_map_figure(country_value, normalize_region_value(region_value))


@app.callback(
    Output("market-click-details", "children"),
    Input("demand-line-chart", "clickData"),
    Input("daily-demand-store", "data"),
)
def update_market_details(click_data, daily_data):
    if isinstance(daily_data, dict):
        daily_records = daily_data.get("daily", [])
        context = daily_data.get("context")
    else:
        daily_records = daily_data or []
        context = None

    daily_df = pd.DataFrame(daily_records)
    if not daily_df.empty and "Date" in daily_df.columns:
        daily_df["Date"] = pd.to_datetime(daily_df["Date"], errors="coerce")

    return build_detail_panel(daily_df, click_data, context)


# Step 6: Run the Dashboard
if __name__ == '__main__':
    app.run_server(debug=False)
