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
    }
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
DATA_FILE = DATA_DIR / "qrc_predictions_all_original_units.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Expected dataset at {DATA_FILE}. Place the CSV file in the data/ folder."
    )

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")


def standardize_prediction_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction CSV columns so downstream plotting is consistent."""

    df = raw_df.rename(
        columns={
            "y_truth": "y_true",
            "truth": "y_true",
            "prediction": "y_pred",
            "pred": "y_pred",
        }
    ).copy()

    if "timestamp" not in df.columns:
        if "t" in df.columns:
            df["timestamp"] = df["t"]
        else:
            raise ValueError("Dataset must contain a 'timestamp' or 't' column.")

    df["set"] = (
        df.get("set", "unknown")
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    for error_column in ("error", "abs_error"):
        if error_column in df.columns:
            df[error_column] = pd.to_numeric(df[error_column], errors="coerce")

    base_date = pd.Timestamp("2025-01-01")
    weeks_offset = pd.to_numeric(df["timestamp"], errors="coerce")
    if weeks_offset.notna().any():
        timedelta_offset = pd.to_timedelta(weeks_offset, unit="W", errors="coerce")
        df["time_axis"] = base_date + timedelta_offset
        df.loc[timedelta_offset.isna(), "time_axis"] = pd.NaT
    else:
        parsed_time = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed_time.notna().any():
            df["time_axis"] = parsed_time.dt.tz_localize(None)
        else:
            df["time_axis"] = weeks_offset

    def _format_timestamp_label(value):
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M")
        if pd.isna(value):
            return "unknown"
        try:
            numeric_value = float(value)
            if numeric_value.is_integer():
                return str(int(numeric_value))
            return str(numeric_value)
        except (TypeError, ValueError):
            return str(value)

    df["timestamp_label"] = df["time_axis"].apply(_format_timestamp_label)

    df = df.dropna(subset=["time_axis", "y_true", "y_pred"])

    return df


# Step 3: Prepare demand vs date data
BASE_DEMAND_DF = standardize_prediction_dataframe(df)

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
                    region_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
                    region_df = standardize_prediction_dataframe(region_raw)
                    if not region_df.empty:
                        return region_df
                except Exception:
                    continue

    return BASE_DEMAND_DF.copy()


def prepare_daily_demand(country: str, region: Optional[str]) -> pd.DataFrame:
    normalized_region = normalize_region_value(region)
    dataset = load_region_dataframe(country, normalized_region).copy()

    if dataset.empty:
        return dataset

    dataset = dataset.sort_values("time_axis").reset_index(drop=True)
    dataset["Date"] = dataset["time_axis"]
    dataset["Load"] = dataset["y_true"]
    dataset["Prediction"] = dataset["y_pred"]

    return dataset


def build_demand_figure(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if daily.empty:
        fig.update_layout(
            title="Actual vs Predicted Load",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="#FFFFFF"),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    set_style = {
        "train": dict(dash="solid", opacity=1.0),
        "test": dict(dash="dash", opacity=1.0),
        "validation": dict(dash="dot", opacity=1.0),
        "unknown": dict(dash="dot", opacity=0.8),
    }

    for set_name, group in daily.groupby(daily["set"].str.lower()):
        style = set_style.get(set_name, set_style["unknown"])
        legend_suffix = set_name.title() if set_name else "Unknown"

        fig.add_trace(
            go.Scatter(
                x=group["time_axis"],
                y=group["y_true"],
                mode="lines",
                line=dict(color="#00B5FF", width=2.5, dash=style["dash"]),
                opacity=style["opacity"],
                name=f"Actual ({legend_suffix})",
                hovertemplate=(
                    "Timestamp: %{text}<br>Actual: %{y:,.0f} MW"
                    "<br>Set: " + legend_suffix + "<extra></extra>"
                ),
                text=group["timestamp_label"],
            )
        )

        fig.add_trace(
            go.Scatter(
                x=group["time_axis"],
                y=group["y_pred"],
                mode="lines",
                line=dict(color="#F39C12", width=2, dash=style["dash"]),
                opacity=style["opacity"],
                name=f"Prediction ({legend_suffix})",
                hovertemplate=(
                    "Timestamp: %{text}<br>Prediction: %{y:,.0f} MW"
                    "<br>Set: " + legend_suffix + "<extra></extra>"
                ),
                text=group["timestamp_label"],
            )
        )

    segments = []
    if "set" in daily.columns and not daily.empty:
        current_label = daily["set"].iloc[0]
        start_value = daily["time_axis"].iloc[0]
        for idx in range(1, len(daily)):
            row_label = daily["set"].iloc[idx]
            if row_label != current_label:
                end_value = daily["time_axis"].iloc[idx - 1]
                segments.append((current_label, start_value, end_value))
                current_label = row_label
                start_value = daily["time_axis"].iloc[idx]
        segments.append((current_label, start_value, daily["time_axis"].iloc[-1]))

    for label, start, end in segments:
        if pd.isna(start) or pd.isna(end):
            continue
        fill = "rgba(46, 204, 113, 0.05)" if label.lower() == "train" else "rgba(231, 76, 60, 0.05)"
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=fill,
            layer="below",
            line_width=0,
        )

        midpoint = None
        if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            midpoint = start + (end - start) / 2
        elif isinstance(start, (int, float)) and isinstance(end, (int, float)):
            midpoint = (start + end) / 2

        if midpoint is not None:
            fig.add_annotation(
                x=midpoint,
                y=1.04,
                xref="x",
                yref="paper",
                text=label.title(),
                showarrow=False,
                font=dict(color="#FFFFFF", size=12),
                align="center",
                bgcolor="rgba(0, 0, 0, 0.4)",
                borderpad=4,
            )

    fig.update_layout(
        title="Actual vs Predicted Load",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#FFFFFF"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(color="#FFFFFF")),
        xaxis=dict(gridcolor="#333333", zerolinecolor="#555555", title="Timestamp"),
        yaxis=dict(gridcolor="#333333", zerolinecolor="#555555", title="Load (MW)"),
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
    if "time_axis" not in daily.columns:
        return html.Div(
            "Timestamp information is missing from the dataset.",
            style={
                "color": "#FFFFFF",
                "textAlign": "center",
                "fontSize": "24px",
                "fontWeight": "600",
                "marginTop": "16px",
            },
        )

    time_series = daily["time_axis"]
    is_datetime_axis = pd.api.types.is_datetime64_any_dtype(time_series)

    def parse_target(value):
        if value is None:
            return None
        if is_datetime_axis:
            target = pd.to_datetime(value, errors="coerce")
            return target.tz_localize(None) if isinstance(target, pd.Timestamp) else target
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if click_data and click_data.get("points"):
        selected_point = click_data["points"][0]
        target_value = parse_target(selected_point.get("x"))
        show_hint = False
    else:
        target_value = time_series.iloc[-1]
        show_hint = True

    if target_value is None or (pd.isna(target_value) if is_datetime_axis else pd.isna(target_value)):
        target_idx = time_series.last_valid_index()
    else:
        try:
            if is_datetime_axis:
                deltas = (time_series - target_value).abs()
            else:
                deltas = (time_series.astype(float) - float(target_value)).abs()
            target_idx = deltas.idxmin()
        except Exception:
            target_idx = time_series.last_valid_index()

    row = daily.loc[target_idx]

    actual_value = row.get("y_true", float("nan"))
    prediction_value = row.get("y_pred", float("nan"))
    signed_error = row.get("error")
    if pd.isna(signed_error):
        if pd.notna(actual_value) and pd.notna(prediction_value):
            signed_error = prediction_value - actual_value
    absolute_error = row.get("abs_error")
    if pd.isna(absolute_error) and pd.notna(signed_error):
        absolute_error = abs(signed_error)

    set_label = str(row.get("set", "unknown")).title()

    def format_timestamp(value, fallback: str = "Unknown timestamp") -> str:
        if isinstance(value, pd.Timestamp):
            return value.strftime("%B %d, %Y %H:%M")
        if pd.isna(value):
            return fallback
        return str(value)

    timestamp_text = row.get("timestamp_label")
    if not timestamp_text or timestamp_text == "unknown":
        timestamp_text = format_timestamp(row.get("time_axis"))

    def format_value(value):
        return f"{value:,.0f} MW" if pd.notna(value) else "—"

    if pd.isna(signed_error):
        status_text = "Prediction error unavailable"
        status_color = "#BDC3C7"
    else:
        status_text = "Overprediction" if signed_error >= 0 else "Underprediction"
        status_color = "#E67E22" if signed_error >= 0 else "#3498DB"

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
                "Click on the plot to explore specific forecast points.",
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
                timestamp_text,
                style={
                    "fontSize": "38px",
                    "fontWeight": "700",
                    "letterSpacing": "0.04em",
                },
            ),
            html.Div(
                f"{set_label}",
                style={
                    "fontSize": "20px",
                    "fontWeight": "600",
                    "color": "#FFFFFF",
                    "marginTop": "8px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        "Actual Load",
                        style={
                            "fontSize": "18px",
                            "color": "#8FA9B5",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        format_value(actual_value),
                        style={
                            "fontSize": "30px",
                            "fontWeight": "600",
                            "color": "#00B5FF",
                        },
                    ),
                ],
                style={"marginTop": "18px"},
            ),
            html.Div(
                [
                    html.Div(
                        "Model Prediction",
                        style={
                            "fontSize": "18px",
                            "color": "#8FA9B5",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        format_value(prediction_value),
                        style={
                            "fontSize": "30px",
                            "fontWeight": "600",
                            "color": "#F39C12",
                        },
                    ),
                ],
                style={"marginTop": "18px"},
            ),
            html.Div(
                status_text,
                style={
                    "fontSize": "22px",
                    "fontWeight": "600",
                    "color": status_color,
                    "marginTop": "20px",
                },
            ),
            html.Div(
                (
                    f"Signed Error: {signed_error:+,.0f} MW"
                    if pd.notna(signed_error)
                    else "Signed Error: —"
                ),
                style={
                    "fontSize": "18px",
                    "color": "#8FA9B5",
                    "marginTop": "6px",
                },
            ),
            html.Div(
                (
                    f"Absolute Error: {absolute_error:,.0f} MW"
                    if pd.notna(absolute_error)
                    else "Absolute Error: —"
                ),
                style={
                    "fontSize": "18px",
                    "color": "#8FA9B5",
                    "marginTop": "4px",
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
if not INITIAL_DAILY_EXPORT.empty and "Date" in INITIAL_DAILY_EXPORT.columns:
    date_series = INITIAL_DAILY_EXPORT["Date"]
    if pd.api.types.is_datetime64_any_dtype(date_series):
        INITIAL_DAILY_EXPORT["Date"] = date_series.dt.strftime("%Y-%m-%d %H:%M")
    else:
        INITIAL_DAILY_EXPORT["Date"] = date_series.astype(str)

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
    if not daily_export.empty and "Date" in daily_export.columns:
        date_series = daily_export["Date"]
        if pd.api.types.is_datetime64_any_dtype(date_series):
            daily_export["Date"] = date_series.dt.strftime("%Y-%m-%d %H:%M")
        else:
            daily_export["Date"] = date_series.astype(str)
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
