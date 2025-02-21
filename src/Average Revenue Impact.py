
# Genre Impact on Revenue
plt.figure(figsize=(12, 6))

# Calculate revenue impact for each genre
genre_revenue_impact = movies_df[genre_columns].mul(movies_df["revenue"], axis=0).mean()

# Plot the revenue impact
genre_revenue_impact.sort_values(ascending=False).plot(kind="bar", color="lightblue")
plt.title("Genre Impact on Revenue")
plt.xlabel("Genres")
plt.ylabel("Average Revenue Impact")
plt.xticks(rotation=45)
plt.show()



    
#![png](output_16_0.png)
    
