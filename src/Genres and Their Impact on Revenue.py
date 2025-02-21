
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



# Select only the binary genre columns for analysis
genre_columns = ["Romance", "Crime", "Mystery", "Drama", "Unknown", "Science Fiction", 
                 "Thriller", "Western", "Action", "War"]

# 1. Genre Distribution Bar Chart
plt.figure(figsize=(20, 18))
plt.subplot(3, 2, 1)
genre_counts = movies_df[genre_columns].sum().sort_values(ascending=False)  # Count occurrences of each genre (sum of 1s for each genre column)
genre_counts.plot(kind="bar", color="skyblue")
plt.title("Genre Distribution")
plt.xlabel("Genres")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)

# 2. Genre Correlation Heatmap
plt.subplot(3, 2, 2)
sns.heatmap(movies_df[genre_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Genres")

# 3. Number of Genres per Movie Bar Chart
plt.subplot(3, 2, 3)
movies_df["num_genres"] = movies_df[genre_columns].sum(axis=1)  # Count how many genres each movie has (sum across genre columns)
single_vs_multi = movies_df["num_genres"].value_counts().sort_index()  # Count occurrences of single vs multi-genres
single_vs_multi.plot(kind="bar", color=["lightcoral", "royalblue"])
plt.title("Number of Genres per Movie")
plt.xlabel("Number of Genres")
plt.ylabel("Number of Movies")
plt.xticks(rotation=0)

# 4. Average Revenue by Genre Bar Chart
plt.subplot(3, 2, 4)
genre_revenue = movies_df[genre_columns].mul(movies_df["revenue"], axis=0)  # Multiply genres with revenue for each movie
avg_revenue = genre_revenue.sum() / movies_df[genre_columns].sum()  # Average revenue per genre
avg_revenue.sort_values(ascending=False).plot(kind="bar", color="gold")
plt.title("Average Revenue by Genre")
plt.xlabel("Genres")
plt.ylabel("Average Revenue")
plt.xticks(rotation=45)

# 5. Top 5 Movie Genres Pie Chart
plt.subplot(3, 2, 5)
top_genres = genre_counts.head(5)
top_genres.plot(kind="pie", autopct="%1.1f%%", cmap="Pastel1")
plt.title("Top 5 Movie Genres")
plt.ylabel("")

# 6. Revenue Distribution by Genre
plt.subplot(3, 2, 6)
genre_cols = [col for col in movies_df.columns if col not in ["revenue", "num_genres"]]  # Exclude non-genre columns
melted_df = movies_df.melt(id_vars=["revenue"], value_vars=genre_cols, var_name="Genre", value_name="Has_Genre")

# Filter to only rows where the movie has that genre
melted_df = melted_df[melted_df["Has_Genre"] == 1]

sns.boxplot(data=melted_df, x="Genre", y="revenue")
plt.xticks(rotation=45)
plt.title("Revenue Distribution by Genre")

plt.tight_layout()
plt.show()



    
#![png](output_10_0.png)
    

