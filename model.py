import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import osmnx as ox
import networkx as nx

# Step 1: Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Encode categorical variables
    categorical_cols = ['Zone', 'Vehicle_Type', 'Cause']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalize numerical features
    scaler = StandardScaler()
    df[['Latitude', 'Longitude']] = scaler.fit_transform(df[['Latitude', 'Longitude']])

    # Define features and target
    X = df.drop(['Accident_Count', 'Fatalities', 'Grievous_Injuries', 'Minor_Injuries'], axis=1)
    y = df['Accident_Count']  # Use Accident_Count as the target variable

    return X, y, label_encoders, scaler

# Step 2: Train the model
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    return model

# Step 3: Save the model and preprocessing objects
def save_model_and_objects(model, label_encoders, scaler):
    joblib.dump(model, 'accident_hotspot_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# Step 4: Get road network between two points
def get_route(start_lat, start_lon, end_lat, end_lon):
    # Create a graph of the road network
    G = ox.graph_from_point((start_lat, start_lon), dist=5000, network_type='drive')

    # Get the nearest nodes to the start and end points
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

    # Get the shortest path
    route = nx.shortest_path(G, start_node, end_node, weight='length')

    # Get the coordinates of the route
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

    return route_coords

# Step 5: Predict accident hotspots along a route
def predict_accident_hotspots_along_route(route_coords, model, label_encoders, scaler):
    hotspots = []
    for lat, lon in route_coords:
        # Create a sample input
        sample = {
            'Zone': 'Thiruvanmiyur',  # Use a default zone (will be encoded)
            'Latitude': lat,
            'Longitude': lon,
            'Vehicle_Type': 'Two Wheeler',  # Use a default vehicle type
            'Cause': 'Speeding'  # Use a default cause
        }
        sample_df = pd.DataFrame([sample])

        # Encode categorical variables
        for col in label_encoders:
            sample_df[col] = label_encoders[col].transform(sample_df[col])

        # Normalize numerical features
        sample_df[['Latitude', 'Longitude']] = scaler.transform(sample_df[['Latitude', 'Longitude']])

        # Predict accident count
        accident_count = model.predict(sample_df)
        hotspots.append({"latitude": lat, "longitude": lon, "accident_count": float(accident_count[0])})

    return hotspots

# Step 6: Main function to get accident hotspots
def get_accident_hotspots(start_lat, start_lon, end_lat, end_lon):
    # Load the model and preprocessing objects
    model = joblib.load('accident_hotspot_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')

    # Get the road network route
    route_coords = get_route(start_lat, start_lon, end_lat, end_lon)

    # Predict accident hotspots along the route
    hotspots = predict_accident_hotspots_along_route(route_coords, model, label_encoders, scaler)

    return hotspots

# Example usage
if __name__ == "__main__":
    # Train and save the model (only needs to be done once)
    X, y, label_encoders, scaler = load_and_preprocess_data('chennai_hotspot_dataset.csv')
    model = train_model(X, y)
    save_model_and_objects(model, label_encoders, scaler)

    # Get accident hotspots for a given start and destination
    start_lat, start_lon = 13.0067, 80.2206  # Example: Guindy
    end_lat, end_lon = 13.0827, 80.2707  # Example: Chennai Central
    hotspots = get_accident_hotspots(start_lat, start_lon, end_lat, end_lon)
    print(hotspots)