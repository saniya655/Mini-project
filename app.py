# -*- coding: utf-8 -*-
"""
Australia Wildfire Analytics - Flask Backend
"""

from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import timedelta, date as dt_date
import urllib.request
import json

app = Flask(__name__)

# ── globals ───────────────────────────────────────────────────────────────────
predicted_df = None
model        = None
full_df      = None
met_df       = None


# ── data loading & model training ─────────────────────────────────────────────
def load_and_train():
    global predicted_df, model, full_df
    try:
        df       = pd.read_csv('Data/australia_fire.csv')
        df       = df.dropna()
        df_clean = df[df['confidence'] > 50].copy()
        df_clean['acq_date'] = pd.to_datetime(df_clean['acq_date'])

        features = ["latitude", "longitude", "brightness", "bright_t31", "confidence"]
        X = df_clean[features]
        y = df_clean["frp"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred              = model.predict(X_test)
        pred_df             = X_test.copy()
        pred_df['predicted_frp'] = y_pred
        pred_df['acq_date'] = df_clean.loc[X_test.index, 'acq_date']

        predicted_df = pred_df
        full_df      = df_clean
        print("✅ Model trained successfully.")
    except Exception as e:
        print(f"❌ Data Loading Error: {e}")


def load_met():
    global met_df
    try:
        met_df = pd.read_csv('Data/metrological_data_australia.csv')
        print("✅ Agency dataset loaded.")
    except Exception:
        np.random.seed(42)
        agencies = ['NSW RFS', 'CFA Victoria', 'DFES WA', 'NSWSES', 'FBA QLD']
        states   = ['NSW', 'VIC', 'WA', 'QLD', 'SA', 'TAS']
        met_df   = pd.DataFrame({
            'agency':         np.random.choice(agencies, 200),
            'state':          np.random.choice(states, 200),
            'units_deployed': np.random.randint(5, 120, 200),
            'date':           pd.date_range('2019-11-01', periods=200, freq='D')[:200],
        })
        print("⚠️  Using synthetic agency data.")


# ── date-filter helper ────────────────────────────────────────────────────────
def parse_date_filter(df, date_col='acq_date'):
    """
    Returns (filtered_df, out_of_range: bool).
    out_of_range=True when a filter was applied but produced zero rows.
    """
    start_str  = request.args.get('start', '').strip()
    end_str    = request.args.get('end',   '').strip()
    filtered   = df.copy()
    filtered[date_col] = pd.to_datetime(filtered[date_col])
    has_filter = bool(start_str or end_str)

    if start_str:
        try:
            filtered = filtered[filtered[date_col] >= pd.to_datetime(start_str)]
        except Exception:
            pass
    if end_str:
        try:
            filtered = filtered[filtered[date_col] <= pd.to_datetime(end_str)]
        except Exception:
            pass

    out_of_range = has_filter and filtered.empty
    return filtered, out_of_range


# ── Open-Meteo helpers ────────────────────────────────────────────────────────
FORECAST_LOCATIONS = [
    {"name": "Central AU", "lat": -25.27, "lon": 133.77},
    {"name": "NSW",        "lat": -32.00, "lon": 148.00},
    {"name": "Victoria",   "lat": -37.00, "lon": 144.00},
    {"name": "Queensland", "lat": -23.00, "lon": 144.00},
    {"name": "WA",         "lat": -26.00, "lon": 121.00},
]

def fetch_openmeteo_forecast(start_date_str=None):
    """
    Fetch 7-day weather from Open-Meteo.
    If start_date_str is given and is in the future, use it as start_date.
    Open-Meteo free tier supports up to 16-day forecasts.
    Returns list of 7 day-dicts or None on failure.
    """
    all_days = {}
    today    = dt_date.today()

    # Work out how many days ahead the requested start date is
    forecast_days = 7
    start_offset  = 0
    if start_date_str:
        try:
            req_start   = pd.to_datetime(start_date_str).date()
            start_offset = (req_start - today).days
            if start_offset < 0:
                start_offset = 0          # past date → use today
            # We need 7 days from that offset
            forecast_days = min(16, start_offset + 7)  # Open-Meteo max = 16
        except Exception:
            pass

    for loc in FORECAST_LOCATIONS:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={loc['lat']}&longitude={loc['lon']}"
            "&daily=temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,windspeed_10m_max,relativehumidity_2m_max"
            "&timezone=Australia%2FSydney"
            f"&forecast_days={forecast_days}"
        )
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            daily    = data.get("daily", {})
            dates    = daily.get("time", [])
            temp_max = daily.get("temperature_2m_max", [])
            temp_min = daily.get("temperature_2m_min", [])
            precip   = daily.get("precipitation_sum", [])
            wind     = daily.get("windspeed_10m_max", [])
            humidity = daily.get("relativehumidity_2m_max", [])

            # Slice to the 7 days starting at start_offset
            dates    = dates[start_offset:start_offset + 7]
            temp_max = temp_max[start_offset:start_offset + 7]
            temp_min = temp_min[start_offset:start_offset + 7]
            precip   = precip[start_offset:start_offset + 7]
            wind     = wind[start_offset:start_offset + 7]
            humidity = humidity[start_offset:start_offset + 7]

            for i, d in enumerate(dates):
                all_days.setdefault(d, []).append({
                    "temp_max": temp_max[i] if i < len(temp_max) else None,
                    "temp_min": temp_min[i] if i < len(temp_min) else None,
                    "precip":   precip[i]   if i < len(precip)   else None,
                    "wind":     wind[i]     if i < len(wind)     else None,
                    "humidity": humidity[i] if i < len(humidity)  else None,
                })
        except Exception as e:
            print(f"⚠️  Open-Meteo failed for {loc['name']}: {e}")

    if not all_days:
        return None, start_offset

    aggregated = []
    for date_str in sorted(all_days.keys())[:7]:
        rows = all_days[date_str]
        def avg(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else 0.0
        aggregated.append({
            "date":     date_str,
            "temp_max": avg("temp_max"),
            "temp_min": avg("temp_min"),
            "precip":   avg("precip"),
            "wind":     avg("wind"),
            "humidity": avg("humidity"),
        })
    return aggregated, start_offset


def weather_to_frp(weather_days, full_df_ref, model_ref):
    """Convert weather features → RF model → FRP predictions."""
    b_mean = float(full_df_ref['brightness'].mean())
    b_std  = float(full_df_ref['brightness'].std())
    t_mean = float(full_df_ref['bright_t31'].mean())

    results = []
    for day in weather_days:
        temp_max = day["temp_max"]
        precip   = day["precip"]
        wind     = day["wind"]
        humidity = day["humidity"]

        temp_factor = max(0.0, min(1.0, (temp_max - 15) / 35.0))
        brightness  = b_mean + temp_factor * b_std * 1.5
        bright_t31  = t_mean + temp_factor * b_std * 0.8
        dryness     = max(0.0, 1.0 - (precip / 10.0) - (humidity / 200.0))
        confidence  = 50 + dryness * 50

        lat, lon    = -25.27, 133.77
        frp_pred    = model_ref.predict([[lat, lon, brightness, bright_t31, confidence]])[0]
        wind_factor = 1.0 + (wind / 100.0)
        frp_final   = round(float(frp_pred) * wind_factor, 2)

        try:
            dt_obj     = pd.to_datetime(day["date"])
            date_label = dt_obj.strftime("%d-%m-%Y")
            weekday    = dt_obj.strftime("%a")
        except Exception:
            date_label = day["date"]
            weekday    = ""

        results.append({
            "date":              date_label,
            "weekday":           weekday,
            "estimated_avg_frp": frp_final,
            "temp_max":          round(temp_max, 1),
            "precip":            round(precip,   1),
            "wind":              round(wind,     1),
            "humidity":          round(humidity, 1),
            "source":            "Open-Meteo + RF Model",
        })
    return results


def statistical_fallback(start_date_str=None):
    """
    RF-based estimate using synthetic but realistic weather inputs.
    Used when Open-Meteo is unreachable.
    Respects a start_date if provided.
    """
    b_mean = float(full_df['brightness'].mean())
    b_std  = float(full_df['brightness'].std())
    t_mean = float(full_df['bright_t31'].mean())

    today = pd.Timestamp.now(tz='Australia/Sydney').normalize().tz_localize(None)
    if start_date_str:
        try:
            req = pd.to_datetime(start_date_str).normalize()
            start = req if req >= today else today
        except Exception:
            start = today
    else:
        start = today

    rng     = np.random.default_rng(seed=int(start.timestamp()) % (2**31))
    results = []
    for i in range(7):
        next_date   = start + timedelta(days=i + 1)
        temp_max    = float(rng.uniform(25, 42))
        precip      = float(rng.uniform(0, 5))
        wind        = float(rng.uniform(10, 60))
        humidity    = float(rng.uniform(15, 55))

        temp_factor = max(0.0, min(1.0, (temp_max - 15) / 35.0))
        brightness  = b_mean + temp_factor * b_std * 1.5
        bright_t31  = t_mean + temp_factor * b_std * 0.8
        dryness     = max(0.0, 1.0 - (precip / 10.0) - (humidity / 200.0))
        confidence  = 50 + dryness * 50

        frp_pred    = model.predict([[-25.27, 133.77, brightness, bright_t31, confidence]])[0]
        frp_final   = round(float(frp_pred) * (1.0 + wind / 100.0), 2)

        results.append({
            "date":              next_date.strftime('%d-%m-%Y'),
            "weekday":           next_date.strftime('%a'),
            "estimated_avg_frp": frp_final,
            "temp_max":          round(temp_max, 1),
            "precip":            round(precip, 1),
            "wind":              round(wind, 1),
            "humidity":          round(humidity, 1),
            "source":            "Statistical estimate (weather API unavailable)",
        })
    return results


# ── ML-generated rows for future date ranges ──────────────────────────────────
def generate_future_chart_rows(start_str, end_str):
    """
    When the user picks a date range outside the CSV, generate daily ML
    predictions so charts and tables still show meaningful data.
    Uses the same weather_to_frp pipeline as the forecast.
    Returns a dict matching the /api/stats charts shape.
    """
    try:
        start = pd.to_datetime(start_str)
        end   = pd.to_datetime(end_str)
    except Exception:
        start = pd.Timestamp.now()
        end   = start + timedelta(days=6)

    # Cap to 90 days so it doesn't hang
    if (end - start).days > 90:
        end = start + timedelta(days=90)

    b_mean = float(full_df['brightness'].mean())
    b_std  = float(full_df['brightness'].std())
    t_mean = float(full_df['bright_t31'].mean())
    c_mean = float(full_df['confidence'].mean())

    days   = pd.date_range(start, end, freq='D')
    rng    = np.random.default_rng(seed=42)

    conf_labels, conf_data   = [], []
    br_labels,   br_data, bt31_data = [], [], []
    frp_labels,  frp_data   = [], []
    cnt_labels,  cnt_data   = [], []

    for day in days:
        label = day.strftime('%Y-%m-%d')
        temp_max  = float(rng.uniform(25, 42))
        precip    = float(rng.uniform(0, 5))
        wind      = float(rng.uniform(10, 60))
        humidity  = float(rng.uniform(15, 55))

        temp_factor = max(0.0, min(1.0, (temp_max - 15) / 35.0))
        brightness  = b_mean + temp_factor * b_std * 1.5
        bright_t31  = t_mean + temp_factor * b_std * 0.8
        dryness     = max(0.0, 1.0 - (precip / 10.0) - (humidity / 200.0))
        confidence  = 50 + dryness * 50

        frp_pred  = model.predict([[-25.27, 133.77, brightness, bright_t31, confidence]])[0]
        frp_final = round(float(frp_pred) * (1.0 + wind / 100.0), 2)

        # Simulate ~10-60 fire detections per day scaled by FRP
        fire_count = int(rng.integers(10, 60) * (frp_final / max(frp_pred, 1)))
        fire_count = max(1, fire_count)

        conf_labels.append(label); conf_data.append(round(confidence, 2))
        br_labels.append(label);   br_data.append(round(brightness, 2)); bt31_data.append(round(bright_t31, 2))
        frp_labels.append(label);  frp_data.append(frp_final)
        cnt_labels.append(label);  cnt_data.append(fire_count)

    return {
        'conf_trend':   {'labels': conf_labels,  'data': conf_data},
        'bright_trend': {'labels': br_labels, 'brightness': br_data, 'bright_t31': bt31_data},
        'frp_trend':    {'labels': frp_labels,   'data': frp_data},
        'count_trend':  {'labels': cnt_labels,   'data': cnt_data},
    }


def build_charts_from(df):
    """Build chart payload from a historical dataframe."""
    df = df.copy()
    df['date_only'] = pd.to_datetime(df['acq_date']).dt.date

    conf_trend   = df.groupby('date_only')['confidence'].mean().reset_index()
    bright_trend = df.groupby('date_only').agg({'brightness': 'max', 'bright_t31': 'max'}).reset_index()
    frp_trend    = df.groupby('date_only')['predicted_frp'].sum().reset_index()
    count_trend  = df.groupby('date_only').size().reset_index(name='count')

    for t in [conf_trend, bright_trend, frp_trend, count_trend]:
        t['date_only'] = t['date_only'].astype(str)

    return {
        'conf_trend':   {'labels': conf_trend['date_only'].tolist(),
                         'data':   conf_trend['confidence'].round(2).tolist()},
        'bright_trend': {'labels':     bright_trend['date_only'].tolist(),
                         'brightness': bright_trend['brightness'].round(2).tolist(),
                         'bright_t31': bright_trend['bright_t31'].round(2).tolist()},
        'frp_trend':    {'labels': frp_trend['date_only'].tolist(),
                         'data':   frp_trend['predicted_frp'].round(2).tolist()},
        'count_trend':  {'labels': count_trend['date_only'].tolist(),
                         'data':   count_trend['count'].tolist()},
    }


def build_agency_charts():
    chart5_labels, chart5_data, chart6_labels, chart6_data = [], [], [], []
    if met_df is not None:
        if 'Agency' in met_df.columns and 'Units_Deployed' in met_df.columns:
            ag = met_df.groupby('Agency')['Units_Deployed'].sum().reset_index()
            chart5_labels = ag['Agency'].tolist()
            chart5_data   = ag['Units_Deployed'].tolist()
        if 'State' in met_df.columns and 'Units_Deployed' in met_df.columns:
            st = met_df.groupby('State')['Units_Deployed'].sum().reset_index()
            chart6_labels = st['State'].tolist()
            chart6_data   = st['Units_Deployed'].tolist()
    return chart5_labels, chart5_data, chart6_labels, chart6_data


# ── routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/fire/data')
def fire_data():
    if predicted_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    df       = predicted_df.copy()
    df['acq_date'] = df['acq_date'].astype(str)
    df = df.round(4)
    total       = len(df)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page        = min(page, total_pages)
    start       = (page - 1) * per_page
    rows        = df.iloc[start:start + per_page].to_dict(orient='records')
    return jsonify({'page': page, 'total_pages': total_pages, 'total_rows': total, 'rows': rows})


@app.route('/api/met/data')
def met_data():
    if met_df is None:
        return jsonify({'error': 'Agency data not loaded'}), 500
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    df       = met_df.copy()
    if 'Event_Date' in df.columns:
        df['Event_Date'] = df['Event_Date'].astype(str)
    total       = len(df)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page        = min(page, total_pages)
    start       = (page - 1) * per_page
    rows        = df.iloc[start:start + per_page].to_dict(orient='records')
    return jsonify({'page': page, 'total_pages': total_pages, 'total_rows': total, 'rows': rows})


@app.route('/api/stats')
def stats():
    if predicted_df is None or full_df is None:
        return jsonify({'error': 'Data not loaded'}), 500

    start_str = request.args.get('start', '').strip()
    end_str   = request.args.get('end',   '').strip()

    tmp,  tmp_oor  = parse_date_filter(predicted_df, date_col='acq_date')
    filt, filt_oor = parse_date_filter(full_df,      date_col='acq_date')

    out_of_range = tmp_oor or filt_oor

    if out_of_range:
        # ── Future/unknown date range → generate ML-predicted chart data ──
        # Use whatever bounds we have; fill missing end with start+6 days
        s = start_str or end_str
        e = end_str   or start_str
        if s and not e:
            e = (pd.to_datetime(s) + timedelta(days=6)).strftime('%Y-%m-%d')
        if e and not s:
            s = (pd.to_datetime(e) - timedelta(days=6)).strftime('%Y-%m-%d')

        charts_data = generate_future_chart_rows(s, e)

        # Summary stats: use full historical dataset as reference
        ref = full_df
        return jsonify({
            'total_fires':    0,           # no real fires in this range
            'avg_confidence': round(float(ref['confidence'].mean()), 1),
            'avg_brightness': round(float(ref['brightness'].mean()), 1),
            'avg_frp':        round(float(ref['frp'].mean()), 2),
            'max_frp':        round(float(ref['frp'].max()), 2),
            'date_range':     {'start': s, 'end': e},
            'out_of_range':   True,
            'charts': {
                **charts_data,
                **dict(zip(
                    ['agency', 'state'],
                    [{'labels': l, 'data': d} for l, d in
                     zip(*[iter(build_agency_charts())] * 2)]
                )),
                'agency': {'labels': build_agency_charts()[0], 'data': build_agency_charts()[1]},
                'state':  {'labels': build_agency_charts()[2], 'data': build_agency_charts()[3]},
            },
        })

    # ── Normal historical range ───────────────────────────────────────────
    charts_data = build_charts_from(tmp)
    a0, a1, a2, a3 = build_agency_charts()

    return jsonify({
        'total_fires':    int(len(filt)),
        'avg_confidence': round(float(filt['confidence'].mean()), 1),
        'avg_brightness': round(float(filt['brightness'].mean()), 1),
        'avg_frp':        round(float(filt['frp'].mean()), 2),
        'max_frp':        round(float(filt['frp'].max()), 2),
        'date_range': {
            'start': str(filt['acq_date'].min().date()),
            'end':   str(filt['acq_date'].max().date()),
        },
        'out_of_range': False,
        'charts': {
            **charts_data,
            'agency': {'labels': a0, 'data': a1},
            'state':  {'labels': a2, 'data': a3},
        },
    })


@app.route('/api/heatmap')
def heatmap():
    if predicted_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    df = predicted_df[['latitude', 'longitude', 'predicted_frp']].dropna()
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)
    points  = df.values.tolist()
    max_frp = float(df['predicted_frp'].max()) if len(df) > 0 else 1.0
    return jsonify({'center': [-25.27, 133.77], 'points': points, 'max_frp': max_frp})


@app.route('/api/predict')
def predict():
    if model is None or full_df is None:
        return jsonify({'error': 'Model not ready'}), 500
    date_str = request.args.get('date', '')
    if not date_str:
        return jsonify({'error': 'No date provided'}), 400
    try:
        target_date = pd.to_datetime(date_str)
    except Exception:
        return jsonify({'error': 'Invalid date format'}), 400

    full_df['acq_date'] = pd.to_datetime(full_df['acq_date'])
    window = full_df[
        (full_df['acq_date'] >= target_date - timedelta(days=15)) &
        (full_df['acq_date'] <= target_date + timedelta(days=15))
    ]
    if window.empty:
        window = full_df.sample(min(300, len(full_df)), random_state=42)

    features = ["latitude", "longitude", "brightness", "bright_t31", "confidence"]
    X_pred   = window[features].dropna()
    if len(X_pred) > 500:
        X_pred = X_pred.sample(500, random_state=42)

    preds     = model.predict(X_pred)
    threshold = np.percentile(preds, 50)
    mask      = preds >= threshold
    lats      = X_pred['latitude'].values[mask]
    lons      = X_pred['longitude'].values[mask]
    frps      = preds[mask]
    max_frp   = frps.max() if len(frps) > 0 else 1.0
    weights   = (frps / max_frp).tolist()
    points    = list(zip(lats.tolist(), lons.tolist(), weights))
    return jsonify({'date': date_str, 'points': points})


@app.route('/api/manual_predict')
def manual_predict():
    if model is None:
        return jsonify({'error': 'Model not ready'}), 500
    try:
        lat        = float(request.args.get('lat',         -37.0))
        lon        = float(request.args.get('lon',         145.0))
        brightness = float(request.args.get('brightness',  300.0))
        bright_t31 = float(request.args.get('bright_t31', 280.0))
        confidence = float(request.args.get('confidence',   80.0))
        result     = model.predict([[lat, lon, brightness, bright_t31, confidence]])
        return jsonify({'predicted_frp': round(float(result[0]), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/future')
def future_forecast():
    """
    7-day FRP forecast starting from an optional ?start=YYYY-MM-DD parameter.
    Always uses Open-Meteo live weather → RF model.
    Falls back to statistical estimate if API is unreachable.
    """
    if full_df is None or model is None:
        return jsonify({'error': 'Data not loaded'}), 500

    start_date_str = request.args.get('start', '').strip() or None

    weather_days, _ = fetch_openmeteo_forecast(start_date_str)

    if weather_days:
        print("✅ Open-Meteo data fetched — running RF forecast.")
        forecast = weather_to_frp(weather_days, full_df, model)
    else:
        print("⚠️  Open-Meteo unavailable — using statistical fallback.")
        forecast = statistical_fallback(start_date_str)

    return jsonify(forecast)


load_and_train()
load_met()

if __name__ == '__main__':
    app.run(debug=False)