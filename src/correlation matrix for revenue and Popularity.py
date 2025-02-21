
# Correlation Heatmap for Revenue and Popularity (number of movies)
plt.figure(figsize=(12, 10))

# Calculate correlation matrix for revenue
revenue_corr = movies_df[genre_columns].mul(movies_df["revenue"], axis=0).corr()

# Calculate correlation matrix for popularity
popularity_corr = movies_df[genre_columns].corr()

# Plot heatmaps
plt.subplot(2, 1, 1)
sns.heatmap(revenue_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Genres in Terms of Revenue")

plt.subplot(2, 1, 2)
sns.heatmap(popularity_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Genres in Terms of Popularity")

plt.tight_layout()
plt.show()




    
#![png](output_14_0.png)
    

