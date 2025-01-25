from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import osmnx as ox
import networkx as nx
from datetime import datetime
import pytz
from math import radians, sin, cos, sqrt, atan2
from groq import Groq
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize Groq client
client = Groq(api_key="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")

# Global variables from gg.py
alert_list = []
emergency_vehicles = []
community_reports = []
parking_bookings = []

# School zones data
school_zones = [
    {
        "pos": [13.0827, 80.2707],
        "name": "DAV Public School",
        "radius": 200,
        "address": "Anna Salai, Chennai",
        "timing": "7:30 AM - 3:30 PM"
    },
    {
        "pos": [13.0900, 80.2800],
        "name": "Chennai Public School",
        "radius": 200,
        "address": "OMR Road, Chennai",
        "timing": "8:00 AM - 4:00 PM"
    },
    {
        "pos": [13.0750, 80.2600],
        "name": "St. Joseph's School",
        "radius": 200,
        "address": "T Nagar, Chennai",
        "timing": "8:00 AM - 3:00 PM"
    },
    {
        "pos": [13.0950, 80.2650],
        "name": "Modern School Chennai",
        "radius": 200,
        "address": "Egmore, Chennai",
        "timing": "7:45 AM - 2:45 PM"
    }
]

# Model related functions
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    categorical_cols = ['Zone', 'Vehicle_Type', 'Cause']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df[['Latitude', 'Longitude']] = scaler.fit_transform(df[['Latitude', 'Longitude']])

    X = df.drop(['Accident_Count', 'Fatalities', 'Grievous_Injuries', 'Minor_Injuries'], axis=1)
    y = df['Accident_Count']

    return X, y, label_encoders, scaler

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    
    return model

def save_model_and_objects(model, label_encoders, scaler):
    joblib.dump(model, 'accident_hotspot_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')

def get_route(start_lat, start_lon, end_lat, end_lon):
    G = ox.graph_from_point((start_lat, start_lon), dist=5000, network_type='drive')
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)
    route = nx.shortest_path(G, start_node, end_node, weight='length')
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    return route_coords

def predict_traffic_density(lat, lon):
    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    hour = current_time.hour
    
    # Simple traffic prediction based on time
    # You can replace this with actual ML model predictions
    if hour >= 8 and hour <= 10:  # Morning rush
        return "high"
    elif hour >= 16 and hour <= 19:  # Evening rush
        return "high"
    elif hour >= 11 and hour <= 15:  # Mid-day
        return "medium"
    elif hour >= 6 and hour <= 7:  # Early morning
        return "medium"
    else:  # Late night/early morning
        return "low"

def predict_accident_hotspots_along_route(route_coords, model, label_encoders, scaler):
    hotspots = []
    for lat, lon in route_coords:
        sample = {
            'Zone': 'Thiruvanmiyur',
            'Latitude': lat,
            'Longitude': lon,
            'Vehicle_Type': 'Two Wheeler',
            'Cause': 'Speeding'
        }
        sample_df = pd.DataFrame([sample])
        
        for col in label_encoders:
            sample_df[col] = label_encoders[col].transform(sample_df[col])
        
        sample_df[['Latitude', 'Longitude']] = scaler.transform(sample_df[['Latitude', 'Longitude']])
        accident_count = model.predict(sample_df)
        
        # Add traffic density prediction
        traffic_density = predict_traffic_density(lat, lon)
        
        hotspots.append({
            "latitude": lat,
            "longitude": lon,
            "accident_count": float(accident_count[0]),
            "traffic_density": traffic_density
        })
    return hotspots

def get_accident_hotspots(start_lat, start_lon, end_lat, end_lon):
    model = joblib.load('accident_hotspot_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    
    route_coords = get_route(start_lat, start_lon, end_lat, end_lon)
    hotspots = predict_accident_hotspots_along_route(route_coords, model, label_encoders, scaler)
    return hotspots

# New functions from gg.py
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hotspots')
def hotspots():
    return render_template('hotspots.html')

@app.route('/navigation')
def navigation():
    return render_template('navigation.html')

@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

@app.route('/community')
def community():
    return render_template('community.html')

@app.route('/safety')
def safety():
    return render_template('safety.html')

@app.route('/parking')
def parking():
    return render_template('parking.html')

@app.route('/rewards')
def rewards():
    return render_template('rewards.html')

@app.route('/community-updates')
def community_updates():
    return render_template('community_updates.html')

@app.route('/emergency-alerts')
def emergency_alerts():
    return render_template('emergency_alerts.html')

@app.route('/driver-fatigue')
def driver_fatigue():
    return render_template('driver_fatigue.html')

@app.route('/mechanic-charging')
def mechanic_charging():
    return render_template('mechanic_charging.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == 'test@123' and password == 'test':
            session['logged_in'] = True
            session['user_type'] = 'user'
            return redirect(url_for('home'))
        elif email == 'auth@123' and password == 'auth':
            session['logged_in'] = True
            session['user_type'] = 'authority'
            return redirect(url_for('authority_dashboard'))
        elif email == 'insurance@123' and password == 'insurance':
            session['logged_in'] = True
            session['user_type'] = 'insurance'
            return redirect(url_for('insurance_dashboard'))
        elif email == 'parking@123' and password == 'parking':
            session['logged_in'] = True
            session['user_type'] = 'parking'
            return redirect(url_for('parking_dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/api/hotspots', methods=['POST'])
def get_route_hotspots():
    data = request.get_json()
    start_lat = float(data['start_lat'])
    start_lon = float(data['start_lon'])
    end_lat = float(data['end_lat'])
    end_lon = float(data['end_lon'])
    
    try:
        hotspots = get_accident_hotspots(start_lat, start_lon, end_lat, end_lon)
        return jsonify({'hotspots': hotspots})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.before_request
def check_login():
    if request.endpoint and request.endpoint != 'login' and 'static' not in request.endpoint:
        if 'logged_in' not in session:
            return redirect(url_for('login'))

# Emergency and Community Routes
@app.route('/report-emergency', methods=['POST'])
def report_emergency():
    data = request.get_json()
    emergency_alert = {
        'id': len(alert_list) + 1,
        'type': 'user_emergency',
        'location': {
            'lat': data.get('lat'),
            'lng': data.get('lng')
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'active'
    }
    alert_list.append(emergency_alert)
    return jsonify({'status': 'success', 'alert': emergency_alert})

@app.route('/submit-report', methods=['POST'])
def submit_report():
    data = request.get_json()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    report = {
        'id': len(community_reports) + 1,
        'type': data.get('type'),
        'location': {
            'lat': data.get('lat'),
            'lng': data.get('lng')
        },
        'description': data.get('description'),
        'reporter': data.get('reporter', 'Anonymous'),
        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'active',
        'address': data.get('location', 'Location not specified'),
        'image_url': data.get('image_url', ''),
        'severity': data.get('severity', 'medium'),
        'verified': False
    }
    community_reports.append(report)
    print("New community report added:", report)  # Debug logging
    return jsonify({'status': 'success', 'report': report})

# Dashboard Routes
@app.route('/authority-dashboard')
def authority_dashboard():
    if session.get('user_type') != 'authority':
        return redirect(url_for('login'))
    return render_template('authority_dashboard.html')

@app.route('/insurance-dashboard')
def insurance_dashboard():
    if session.get('user_type') != 'insurance':
        return redirect(url_for('login'))
    return render_template('insurance_dashboard.html')

@app.route('/parking-dashboard')
def parking_dashboard():
    if session.get('user_type') != 'parking':
        return redirect(url_for('login'))
    print("Number of bookings:", len(parking_bookings))  # Add logging
    return render_template('parking_dashboard.html', bookings=parking_bookings)

# API Routes
@app.route('/get-alerts')
def get_alerts():
    return jsonify({
        'emergency_alerts': alert_list,
        'school_zones': school_zones,
        'emergency_vehicles': emergency_vehicles,
        'community_reports': community_reports
    })

@app.route('/get-nearby-schools')
def get_nearby_schools():
    user_lat = float(request.args.get('lat'))
    user_lng = float(request.args.get('lng'))
    nearby_schools = []
    for school in school_zones:
        distance = calculate_distance(
            user_lat, user_lng,
            school['pos'][0], school['pos'][1]
        )
        if distance <= 2:  # Schools within 2km
            school_info = school.copy()
            school_info['distance'] = round(distance, 2)
            nearby_schools.append(school_info)
    nearby_schools.sort(key=lambda x: x['distance'])
    return jsonify(nearby_schools)

# Chatbot Routes
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    try:
        system_prompt = """You are RoadSafe AI, a specialized road navigation assistant for Chennai. 
Keep responses concise (2-3 sentences per point) but informative. Focus on practical details like location, timing, and directions.
When handling location queries:
- Distinguish between different beaches (Marina, Elliot's, OMR, ECR beaches)
- Provide accurate area-specific information
- Consider the user's current location when suggesting routes or places"""

        # Extract location from message if it contains coordinates
        location_context = ""
        if "current location is:" in user_message.lower():
            try:
                coords = user_message.split("location is:")[-1].strip().split(",")
                lat, lon = float(coords[0]), float(coords[1])
                location_context = f"\nUser's current location: {lat}, {lon}"
            except:
                pass

        user_prompt = f"""Provide brief but specific information about Chennai with:
- Key locations and landmarks
- Basic distance/time estimates
- Current traffic status if relevant
- Essential safety tips{location_context}

User question: {user_message}"""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-8b-8192",
        )
        response = chat_completion.choices[0].message.content
        
        # Format response for HTML
        response = response.replace('\n', '<br>')
        response = response.replace('• ', '<br>• ')
        response = response.replace('- ', '<br>• ')
        
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error getting chatbot response: {e}")
        return jsonify({
            "response": "I apologize, but I'm having trouble processing your request right now. Please try again."
        }), 500

@app.route('/api/parking/book', methods=['POST'])
def book_parking():
    data = request.get_json()
    print("Received booking request:", data)
    
    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    booking = {
        'id': len(parking_bookings) + 1,
        'spot_name': data.get('spotName'),
        'vehicle_number': data.get('vehicleNumber'),
        'entry_time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'active',
        'spot_number': f"A-{len(parking_bookings) + 1}",
        'amount': 50,
        'location': data.get('spotDetails', {}).get('location', []),
        'duration': '2 hours',  # Default duration
        'payment_status': 'Paid',
        'booking_time': current_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    parking_bookings.append(booking)
    print("Current parking bookings:", parking_bookings)
    return jsonify({'status': 'success', 'booking': booking})

@app.route('/api/parking/bookings')
def get_parking_bookings():
    return jsonify(parking_bookings)

if __name__ == '__main__':
    # Train and save the model on first run
    try:
        X, y, label_encoders, scaler = load_and_preprocess_data('chennai_hotspot_dataset.csv')
        model = train_model(X, y)
        save_model_and_objects(model, label_encoders, scaler)
        print("Model trained and saved successfully!")
    except Exception as e:
        print("Error training model:", e)
        print("Using existing model if available...")
    
    app.run(debug=True,port=8080) 