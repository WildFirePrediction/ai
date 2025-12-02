"""Convert KMA station coordinates from WGS84 to EPSG:5179"""

import pandas as pd
from pyproj import Transformer

# Read original file
df = pd.read_csv('/home/chaseungjoon/code/WildfirePrediction/embedded_data/kma_aws_stations.csv')

# WGS84 to EPSG:5179 transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

# Convert coordinates
x_coords, y_coords = transformer.transform(df['LON'].values, df['LAT'].values)

# Add to dataframe
df['X'] = x_coords
df['Y'] = y_coords

# Save
output_file = '/home/chaseungjoon/code/WildfirePrediction/data/KMA/kma_weather_station_5179.csv'
df.to_csv(output_file, index=False)

print(f"Converted {len(df)} stations to EPSG:5179")
print(f"Saved to: {output_file}")
print(f"\nSample:")
print(df.head())
