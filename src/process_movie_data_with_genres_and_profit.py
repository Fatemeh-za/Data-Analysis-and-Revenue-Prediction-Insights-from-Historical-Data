
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Convert 'release_date' to datetime in movies_df
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
movies_df.columns = movies_df.columns.str.strip()

# Extract year from release date in movies_df
movies_df['release_year'] = movies_df['release_date'].dt.year

# Convert 'budget' and 'revenue' to numeric in movies_df
movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')
movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')

# Calculate profit in movies_df
movies_df['profit'] = movies_df['revenue'] - movies_df['budget']

def extract_genres(genre_str):
    if not genre_str or genre_str == "[]":
        return ["Unknown"]
    
    try:
        genre_list = ast.literal_eval(genre_str)  # Convert string to list of dictionaries
        return [genre['name'] for genre in genre_list]  # Extract genre names
    except (ValueError, SyntaxError):
        return ["Unknown"]

movies_df["cleaned_genres"] = movies_df["genres"].apply(extract_genres)

# One-hot encoding genres
all_genres = set(genre for sublist in movies_df["cleaned_genres"] for genre in sublist)  # Get unique genres

for genre in all_genres:
    movies_df[genre] = movies_df["cleaned_genres"].apply(lambda x: 1 if genre in x else 0)



