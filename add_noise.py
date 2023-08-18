import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
base = "/home/parallels/Desktop/haha/data/"
input_folder = base + 'Apartment/local_frame'   # Replace with your input folder path
output_folder = base + 'Apartment/lf_sensor/003' # Replace with your desired output folder path

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each CSV file in the input folder
k = 0
while True:
    # Construct full file path
    filename = "Hokuyo_" + str(k) + ".csv"
    filepath = os.path.join(input_folder, filename)
    if not os.path.exists(filepath):
        break
    # Read CSV data
    df = pd.read_csv(filepath)
    
    # Add Gaussian noise to x, y, z columns
    noise_stddev = 0.03  # Adjust this value if needed
    df['x'] += np.random.normal(0, noise_stddev, df.shape[0])
    df['y'] += np.random.normal(0, noise_stddev, df.shape[0])
    df['z'] += np.random.normal(0, noise_stddev, df.shape[0])
    
    # Write noisy data to the output folder
    df.to_csv(os.path.join(output_folder, filename), index=False)
    k += 1

print("Processing complete!")
