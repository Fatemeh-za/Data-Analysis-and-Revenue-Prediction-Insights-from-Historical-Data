
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure your DataFrame 'movies_df' is defined before this code

# Calculate the number of genres each movie belongs to
movies_df["num_genres"] = movies_df[genre_columns].sum(axis=1)

# Group by the number of genres and calculate the total and average revenue for each group
genre_revenue_summary = movies_df.groupby("num_genres").agg({
    "revenue": ["sum", "mean"],
    "id": "count"  # Count number of movies for each genre combination
}).reset_index()
genre_revenue_summary.columns = ["num_genres", "total_revenue", "average_revenue", "number_of_movies"]

# Visualize the data
plt.figure(figsize=(18, 12))

# Number of Movies per Genre Count
plt.subplot(2, 2, 1)
sns.barplot(data=genre_revenue_summary, x="num_genres", y="number_of_movies", palette="viridis")
plt.title("Number of Movies per Genre Count")
plt.xlabel("Number of Genres")
plt.ylabel("Number of Movies")

# Total Revenue per Genre Count
plt.subplot(2, 2, 2)
sns.barplot(data=genre_revenue_summary, x="num_genres", y="total_revenue", palette="viridis")
plt.title("Total Revenue per Genre Count")
plt.xlabel("Number of Genres")
plt.ylabel("Total Revenue")

# Average Revenue per Genre Count
plt.subplot(2, 2, 3)
sns.barplot(data=genre_revenue_summary, x="num_genres", y="average_revenue", palette="viridis")
plt.title("Average Revenue per Genre Count")
plt.xlabel("Number of Genres")
plt.ylabel("Average Revenue")

# Boxplot of Revenue by Genre Count
plt.subplot(2, 2, 4)
sns.boxplot(data=movies_df, x="num_genres", y="revenue", palette="viridis")
plt.title("Revenue Distribution by Genre Count")
plt.xlabel("Number of Genres")
plt.ylabel("Revenue")

# Show the figure with all subplots
plt.tight_layout()
plt.show()



    
#![png](output_11_0.png)
    


