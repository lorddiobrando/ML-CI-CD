import pandas as pd
from sklearn.datasets import load_digits
import os

def save_digits_dataset(output_path="data/digits.csv"):
    """Download the digits dataset and save it to a CSV file."""
    print("Loading digits dataset...")
    digits = load_digits()
    
    # Create a DataFrame
    df = pd.DataFrame(digits.data)
    df['target'] = digits.target
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    print(f"Saving dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Dataset saved successfully.")

if __name__ == "__main__":
    save_digits_dataset()
