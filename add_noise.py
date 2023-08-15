import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
input_folder = 'Apartment/local_frame'   # Replace with your input folder path
output_folder = 'Apartment/lf_sensor' # Replace with your desired output folder path

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each CSV file in the input folder
# for filename in os.listdir(input_folder):
    # if filename.endswith('.csv'):
        
for k in range(45):

    # Construct full file path
    filename = "Hokuyo_" + str(k) + ".csv"
    filepath = os.path.join(input_folder, filename)
    
    # Read CSV data
    df = pd.read_csv(filepath)
    
    # Add Gaussian noise to x, y, z columns
    noise_stddev = 0.01  # Adjust this value if needed
    df['x'] += np.random.normal(0, noise_stddev, df.shape[0])
    df['y'] += np.random.normal(0, noise_stddev, df.shape[0])
    df['z'] += np.random.normal(0, noise_stddev, df.shape[0])
    
    # Write noisy data to the output folder
    df.to_csv(os.path.join(output_folder, filename), index=False)

print("Processing complete!")
