import pandas as pd
import folium
from folium.plugins import HeatMap
import branca.colormap as cm

def visualize_accident_hotspots():
    # Read the dataset
    df = pd.read_csv('chennai_hotspot_dataset.csv')
    
    # Calculate the weight based on accident severity
    df['weight'] = df['Fatalities']*3 + df['Grievous_Injuries']*2 + df['Minor_Injuries']
    
    # Create a base map centered on Chennai
    chennai_map = folium.Map(location=[13.0827, 80.2707], zoom_start=11)
    
    # Create a color map for different accident causes
    causes = df['Cause'].unique()
    color_map = cm.LinearColormap(
        colors=['green', 'yellow', 'red'],
        vmin=df['weight'].min(),
        vmax=df['weight'].max()
    )
    
    # Add heatmap layer
    locations = df[['Latitude', 'Longitude', 'weight']].values.tolist()
    HeatMap(locations).add_to(chennai_map)
    
    # Add markers for each accident location
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"Zone: {row['Zone']}<br>"
                  f"Cause: {row['Cause']}<br>"
                  f"Vehicle: {row['Vehicle_Type']}<br>"
                  f"Accidents: {row['Accident_Count']}",
            color='red',
            fill=True
        ).add_to(chennai_map)
    
    # Add color map to the map
    color_map.add_to(chennai_map)
    
    # Save the map
    chennai_map.save('chennai_accident_hotspots.html')

if __name__ == "__main__":
    visualize_accident_hotspots() 