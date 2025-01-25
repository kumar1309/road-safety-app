from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from datetime import datetime
import json
from math import radians, sin, cos, sqrt, atan2
from groq import Groq

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Rename the global variable to avoid conflict with route name
alert_list = []

# Add this with other global variables
emergency_vehicles = []
community_reports = []

# Initialize Groq client
client = Groq(api_key="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")  # Replace with your actual API key

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

# Update the school zones with more data
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

@app.before_request
def check_login():
    if request.endpoint and request.endpoint != 'login' and 'static' not in request.endpoint:
        if 'logged_in' not in session:
            return redirect(url_for('login'))

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
    report = {
        'id': len(community_reports) + 1,
        'type': data.get('type'),
        'location': {
            'lat': data.get('lat'),
            'lng': data.get('lng')
        },
        'description': data.get('description'),
        'reporter': data.get('reporter', 'Anonymous'),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'active'
    }
    community_reports.append(report)
    return jsonify({'status': 'success', 'report': report})

@app.route('/get-alerts')
def get_alerts():
    return jsonify({
        'emergency_alerts': alert_list,
        'school_zones': school_zones,
        'emergency_vehicles': emergency_vehicles,
        'community_reports': community_reports
    })

@app.route('/authority-dashboard')
def authority_dashboard():
    if session.get('user_type') != 'authority':
        return redirect(url_for('login'))
    return render_template('authority_dashboard.html')

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

@app.route('/add-emergency-vehicle', methods=['POST'])
def add_emergency_vehicle():
    data = request.get_json()
    vehicle = {
        'id': len(emergency_vehicles) + 1,
        'type': data.get('type'),  # 'police', 'ambulance', or 'fire'
        'location': {
            'lat': data.get('lat'),
            'lng': data.get('lng')
        },
        'destination': {
            'lat': data.get('dest_lat'),
            'lng': data.get('dest_lng')
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'active'
    }
    emergency_vehicles.append(vehicle)
    return jsonify({'status': 'success', 'vehicle': vehicle})

@app.route('/get-emergency-vehicles')
def get_emergency_vehicles():
    return jsonify(emergency_vehicles)

@app.route('/get-community-reports')
def get_community_reports():
    return jsonify(community_reports)

@app.route('/insurance-dashboard')
def insurance_dashboard():
    if session.get('user_type') != 'insurance':
        return redirect(url_for('login'))
    return render_template('insurance_dashboard.html')

@app.route('/parking-dashboard')
def parking_dashboard():
    if session.get('user_type') != 'parking':
        return redirect(url_for('login'))
    return render_template('parking_dashboard.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    try:
        # Add instruction for structured response
        prompt = f"""Please provide a well-structured response with:
- Clear paragraphs separated by line breaks
- Bullet points where appropriate
- Short, clear sentences

User question: {user_message}"""

        # Get response from Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful health assistant. Format your responses with clear paragraphs, line breaks between sections, and bullet points where appropriate."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )
        response = chat_completion.choices[0].message.content
        
        # Ensure proper formatting in HTML
        response = response.replace('\n', '<br>')  # Convert newlines to HTML breaks
        response = response.replace('• ', '<br>• ')  # Add breaks before bullet points
        response = response.replace('- ', '<br>• ')  # Convert dashes to bullet points
        
        return jsonify({
            "response": response
        })
    except Exception as e:
        app.logger.error(f"Error getting chatbot response: {e}")
        return jsonify({
            "response": "I apologize, but I'm having trouble processing your request right now. Please try again."
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 