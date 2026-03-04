
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

# Configuration
INPUT_CSV = Path("00_Ancillary_Data/EAD_Hourly_2022-2024_AQ_Points_AQI_dedup.csv")
OUTPUT_CSV = Path("data/processed/station_geography.csv")

# UAE Coastline approximation (Simplified points along the Arabian Gulf and Gulf of Oman)
# These points are used to find the "Distance to nearest Coast"
COASTLINE_POINTS = [
    (24.4667, 54.3667), # Abu Dhabi
    (25.2048, 55.2708), # Dubai
    (25.3463, 55.4209), # Sharjah
    (25.4052, 55.4424), # Ajman
    (25.5647, 55.8500), # Umm Al Quwain
    (25.7895, 55.9432), # Ras Al Khaimah
    (25.1288, 56.3265), # Fujairah
    (24.2123, 51.5273), # Near Sila
    (24.1206, 52.6074), # Near Ruwais
    (24.0620, 53.6499), # Near Mirfa
    (24.7820, 54.6738), # Northeast of AD
    (26.0463, 56.0903), # Tip of RAK
    (24.9602, 56.3475), # Kalba
]

# Abu Dhabi center (Corniche/Khalidiya area) - proxy for dense urban emissions
AD_CENTER = (24.48, 54.34)

# E11 Highway approximation (Main logistics artery)
# Using key nodes along the E11 from Sila to Ras Al Khaimah
E11_POINTS = [
    (24.05, 51.75), # Ghuwaifat/Sila
    (23.95, 52.55), # Near Ruwais
    (24.02, 53.48), # Near Mirfa
    (24.15, 54.45), # Mussafah/MAFZ
    (24.45, 54.65), # Yas/Airport area
    (24.85, 54.95), # Ghantoot
    (25.05, 55.15), # Jebel Ali
    (25.25, 55.35), # Dubai central
    (25.45, 55.55), # Ajman/Sharjah E11
    (25.65, 55.80), # UAQ
    (25.85, 56.05), # RAK
]

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

def get_elevation(lat, lon):
    """Query open-elevation API for a single point."""
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['results'][0]['elevation']
    except Exception as e:
        print(f"Error fetching elevation for {lat}, {lon}: {e}")
    return np.nan

def main():
    print(f"Loading stations from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Get unique stations and their coordinates
    stations = df[['StationName', 'y', 'x']].drop_duplicates().reset_index(drop=True)
    stations = stations.rename(columns={'y': 'Latitude', 'x': 'Longitude'})
    
    print(f"Found {len(stations)} unique stations.")
    
    elevations = []
    coast_distances = []
    corniche_distances = []
    e11_distances = []
    
    for idx, row in stations.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        name = row['StationName']
        
        # 1. Get Elevation
        print(f"[{idx+1}/{len(stations)}] Fetching elevation for {name}...")
        elev = get_elevation(lat, lon)
        elevations.append(elev)
        
        # 2. Distance to nearest coast point
        min_coast = min([haversine(lat, lon, clat, clon) for clat, clon in COASTLINE_POINTS])
        coast_distances.append(min_coast)
        
        # 3. Distance to Corniche (AD Center)
        dist_ad = haversine(lat, lon, AD_CENTER[0], AD_CENTER[1])
        corniche_distances.append(dist_ad)
        
        # 4. Distance to E11 Highway (nearest point)
        min_e11 = min([haversine(lat, lon, relat, relon) for relat, relon in E11_POINTS])
        e11_distances.append(min_e11)
        
        # Rate limit respect for free API
        time.sleep(1)
        
    stations['Elevation_m'] = elevations
    stations['Dist_Coast_km'] = coast_distances
    stations['Dist_Corniche_km'] = corniche_distances
    stations['Dist_E11_km'] = e11_distances
    
    # Fill NaNs if any failures occurred (fallback to mean)
    stations['Elevation_m'] = stations['Elevation_m'].fillna(stations['Elevation_m'].mean())
    
    print(f"Saving geography features to {OUTPUT_CSV}...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(OUTPUT_CSV, index=False)
    
    print("\nExtraction Complete!")
    print(stations[['StationName', 'Elevation_m', 'Dist_Coast_km', 'Dist_Corniche_km', 'Dist_E11_km']])

if __name__ == "__main__":
    main()
