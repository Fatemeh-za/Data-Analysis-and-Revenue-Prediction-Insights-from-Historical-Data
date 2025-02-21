
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure your DataFrame 'movies_df' is defined before this code

# Convert 'release_date' to datetime format
movies_df["release_date"] = pd.to_datetime(movies_df["release_date"], errors='coerce')

# Extract year from 'release_date'
movies_df["release_year"] = movies_df["release_date"].dt.year

# Calculate the number of genres each movie belongs to
movies_df["num_genres"] = movies_df[genre_columns].sum(axis=1)

# Initialize lists to store data for each genre
genre_revenue_data = []
genre_num_genres_data = []

# Loop through each genre and calculate revenue and number of genres over years
for genre in genre_columns:
    # Group by release year and calculate total and average revenue for each genre
    revenue_data = movies_df[movies_df[genre] == 1].groupby("release_year")["revenue"].mean().reset_index()
    revenue_data["genre"] = genre
    genre_revenue_data.append(revenue_data)
    
    # Group by release year and calculate average number of genres for each genre
    num_genres_data = movies_df[movies_df[genre] == 1].groupby("release_year")["num_genres"].mean().reset_index()
    num_genres_data["genre"] = genre
    genre_num_genres_data.append(num_genres_data)

# Concatenate data for each genre
genre_revenue_df = pd.concat(genre_revenue_data, ignore_index=True)
genre_num_genres_df = pd.concat(genre_num_genres_data, ignore_index=True)

# Visualize the data
plt.figure(figsize=(18, 18))

# Revenue Over Years for Each Genre
plt.subplot(2, 1, 1)
sns.lineplot(data=genre_revenue_df, x="release_year", y="revenue", hue="genre", palette="viridis")
plt.title("Average Revenue Over Years for Each Genre")
plt.xlabel("Release Year")
plt.ylabel("Average Revenue")
plt.legend(title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')

# Number of Genres per Movie Over Years for Each Genre
plt.subplot(2, 1, 2)
sns.lineplot(data=genre_num_genres_df, x="release_year", y="num_genres", hue="genre", palette="viridis")
plt.title("Average Number of Genres per Movie Over Years for Each Genre")
plt.xlabel("Release Year")
plt.ylabel("Average Number of Genres")
plt.legend(title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the figure with both subplots
plt.tight_layout()
plt.show()




    
#![png](output_13_0.png)
    
