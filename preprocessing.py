import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file="raw_data.csv", output_file="processed_data.csv"):
    """Preprocesses raw survival data and saves it as processed_data.csv."""
    # Load raw data
    df = pd.read_csv(input_file)
    
    # Extract feature columns
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    
    # Normalize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Ensure no negative survival times
    df['left_interval'] = df['left_interval'].clip(lower=0)
    df['right_interval'] = df['right_interval'].clip(lower=df['left_interval'])
    
    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Processed data saved as {output_file}")

if __name__ == "__main__":
    preprocess_data()
