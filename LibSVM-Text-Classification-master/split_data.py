# split_data.py

import pandas as pd
import os

# Path to your CSV file
csv_file_path = r'song_lyrics.csv'  # Update with your actual file path

# Output path
output_path = 'data/lyrics_top3.csv'  # Combined dataset with 15,000 samples

# Parameters
total_samples_per_category = 200  # Number of samples per category

# List of categories (update based on your actual category names)
categories = ['Pop', 'Rap', 'Misc',]  # Update with your actual categories

# Initialize a dictionary to store data for each category
category_data = {category: [] for category in categories}

# Define the chunk size (number of rows per chunk)
chunk_size = 100000  # Adjust based on your system's memory

# Function to check if we have enough samples for all categories
def enough_samples(category_counts):
    return all(count >= total_samples_per_category for count in category_counts.values())

# Initialize counts for each category
category_counts = {category: 0 for category in categories}

# Read and process the CSV in chunks
for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, usecols=['tag', 'lyrics']):
    # Drop rows with missing values in 'tag' or 'lyrics'
    chunk.dropna(subset=['tag', 'lyrics'], inplace=True)
    # Convert 'tag' to lowercase for consistency
    chunk['tag'] = chunk['tag'].str.lower()

    # Iterate over each category
    for category in categories:
        if category_counts[category] < total_samples_per_category:
            # Filter rows belonging to the current category
            category_chunk = chunk[chunk['tag'] == category.lower()]

            # Calculate how many samples are needed for this category
            samples_needed = total_samples_per_category - category_counts[category]

            if len(category_chunk) > 0:
                # If there's more data than needed, sample only the required amount
                if len(category_chunk) > samples_needed:
                    category_chunk = category_chunk.sample(n=samples_needed, random_state=42)
                else:
                    # Shuffle the chunk
                    category_chunk = category_chunk.sample(frac=1, random_state=42)

                # Append the data
                category_data[category].append(category_chunk)
                category_counts[category] += len(category_chunk)

    # Check if we have enough samples for all categories
    if enough_samples(category_counts):
        break

# Concatenate data from all categories
data_df = pd.concat([pd.concat(category_data[category]) for category in categories], ignore_index=True)

# Shuffle the data
data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Save the combined data to a CSV file
data_df.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")