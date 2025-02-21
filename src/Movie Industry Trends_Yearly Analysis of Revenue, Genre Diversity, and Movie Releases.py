
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

# Group by release year and calculate the average number of genres per movie
year_genre_summary = movies_df.groupby("release_year")["num_genres"].mean().reset_index()

# Group by release year and genre count, then calculate the average revenue
genre_year_revenue_summary = movies_df.groupby(["release_year", "num_genres"])["revenue"].mean().reset_index()

# Additional analysis by release year
yearly_summary = movies_df.groupby("release_year").agg({
    "revenue": ["sum", "mean"],
    "id": "count"
}).reset_index()
yearly_summary.columns = ["release_year", "total_revenue", "average_revenue", "number_of_movies"]

# Visualize the data
plt.figure(figsize=(18, 18))

# Number of Movies Released Each Year
plt.subplot(3, 2, 1)
sns.lineplot(data=yearly_summary, x="release_year", y="number_of_movies", marker='o', color='purple')
plt.title("Number of Movies Released Each Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Movies")

# Average Revenue Each Year
plt.subplot(3, 2, 2)
sns.lineplot(data=yearly_summary, x="release_year", y="average_revenue", marker='o', color='gold')
plt.title("Average Revenue Each Year")
plt.xlabel("Release Year")
plt.ylabel("Average Revenue")

# Average Number of Genres per Movie Over Years
plt.subplot(3, 2, 3)
sns.lineplot(data=year_genre_summary, x="release_year", y="num_genres", marker='o', color='blue')
plt.title("Average Number of Genres per Movie Over Years")
plt.xlabel("Release Year")
plt.ylabel("Average Number of Genres")

# Average Revenue per Genre Count Over Years
plt.subplot(3, 2, 4)
sns.lineplot(data=genre_year_revenue_summary, x="release_year", y="revenue", hue="num_genres", palette="viridis", marker='o')
plt.title("Average Revenue per Genre Count Over Years")
plt.xlabel("Release Year")
plt.ylabel("Average Revenue")
plt.legend(title="Number of Genres")

# Show the figure with all subplots
plt.tight_layout()
plt.show()



    
#![png](output_12_0.png)
    

