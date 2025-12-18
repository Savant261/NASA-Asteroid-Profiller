# Import necessary libraries for data handling
import pandas as pd
import numpy as np

# Define a function to create the final dataset
def create_final_dataset(measured_path, unmeasured_path, target_unmeasured_count=2500):
    """
    This function loads, cleans, combines, and preprocesses two datasets of asteroids:
    1. Measured: Asteroids with known diameters.
    2. Unmeasured: Asteroids with unknown diameters.
    """
    print("🚀 Building Hybrid Project Dataset...")

    # 1. Column Name Mapping
    # Define a dictionary to rename columns to a consistent and clean format.
    col_map = {
        'spkid': 'id', 
        'full_name': 'name', 
        'pha': 'pha_flag', 
        'H': 'magnitude_h',
        'diameter': 'diameter', 
        'albedo': 'albedo', 
        'e': 'eccentricity', 
        'a': 'semi_major_axis', 
        'q': 'perihelion', 
        'i': 'inclination', 
        'moid': 'moid_au',
        'class': 'class_name'
    }

    # 2. Load the Datasets
    try:
        # Load the dataset with measured diameters (our training data)
        df_train = pd.read_csv(measured_path, comment='#', low_memory=False)
        print(f"   - Loaded Measured Data: {len(df_train)} rows")
        
        # Load the dataset with unmeasured diameters (data we want to make predictions on)
        df_predict_raw = pd.read_csv(unmeasured_path, comment='#', low_memory=False)
        print(f"   - Loaded Unmeasured Data: {len(df_predict_raw)} rows")
        
    except FileNotFoundError:
        print("❌ Error: Files not found. Check filenames.")
        return

    # 3. Clean Column Names
    # Remove any leading/trailing whitespace from column names in both dataframes.
    df_train.columns = df_train.columns.str.strip()
    df_predict_raw.columns = df_predict_raw.columns.str.strip()
    
    # Debugging check for a specific column name issue
    if 'Earth MOID (au)' not in df_train.columns:
        print("⚠️ Warning: 'Earth MOID (au)' not found. Available columns:")
        print(df_train.columns.tolist())

    # 4. Rename Columns
    # Apply the column name mapping to both dataframes.
    for df in [df_train, df_predict_raw]:
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # 5. Smart Sampling
    # If the unmeasured dataset is too large, take a random sample to keep processing times manageable.
    if len(df_predict_raw) > target_unmeasured_count:
        df_predict = df_predict_raw.sample(n=target_unmeasured_count, random_state=42)
        print(f"   - ✂️ Downsampled unmeasured data to {target_unmeasured_count} random rows.")
    else:
        df_predict = df_predict_raw

    # 6. Add 'Type' Labels
    # Add a column to distinguish between the original measured data and the unmeasured data.
    df_train['dataset_type'] = 'measured'
    df_predict['dataset_type'] = 'unmeasured'

    # 7. Combine Datasets
    # Concatenate the two dataframes into a single, unified dataset.
    full_df = pd.concat([df_train, df_predict], ignore_index=True)
    
    # 8. Feature Engineering: Standardize PHA Flag
    # Create a function to convert the 'pha_flag' (Potentially Hazardous Asteroid) into a binary format (1 for Yes, 0 for No).
    # It also calculates the flag based on NASA's definition if the flag is missing.
    def calculate_pha(row):
        if str(row.get('pha_flag')) == 'Y': return 1
        if str(row.get('pha_flag')) == 'N': return 0
        try:
            # NASA's definition of a PHA: MOID <= 0.05 AU and Absolute Magnitude (H) <= 22.0
            if row.get('moid_au') is not None and row.get('magnitude_h') is not None:
                if row['moid_au'] <= 0.05 and row['magnitude_h'] <= 22.0:
                    return 1
            return 0
        except:
            return 0

    # Apply the function to the 'pha_flag' column.
    full_df['pha_flag'] = full_df.apply(calculate_pha, axis=1)

    # 9. Final Cleaning and Saving
    # Define the essential columns needed for our analysis.
    essential_cols = ['semi_major_axis', 'eccentricity', 'inclination', 'magnitude_h', 'moid_au']
    
    # Check if any essential columns are missing.
    missing_cols = [c for c in essential_cols if c not in full_df.columns]
    if missing_cols:
        print(f"❌ Critical Error: Missing columns {missing_cols}. Check your CSV headers.")
        return

    # Drop any rows that have missing values in the essential columns.
    full_df.dropna(subset=essential_cols, inplace=True)
    
    # Save the final, cleaned dataset to a new CSV file.
    full_df.to_csv('nasa_asteroid_final.csv', index=False)
    print("---")
    print(f"✅ SUCCESS! Final Dataset: {len(full_df)} rows saved to 'nasa_asteroid_final.csv'")
    print("   You are ready for Day 2.")

# Execute the function with the paths to the raw data files.
create_final_dataset(r'E:\Python\Datasets\nasa_measured.csv', r'E:\Python\Datasets\nasa_unmeasured.csv')

