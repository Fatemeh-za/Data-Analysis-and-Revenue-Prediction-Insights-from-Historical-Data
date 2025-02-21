# 📊 Trends and Revenue Analysis  

This project analyzes movie genre trends, revenue impact, and time-series forecasting using machine learning models.

## 📁 Repository Structure  


    │── 📂 data # Raw datasets
    │── 📂 notebooks # Jupyter notebooks (.ipynb) with exploratory analysis
    │── 📂 graphs # PNG images of visualizations and analysis results
    │── 📂 src # Python scripts for analysis
    │   │── Analysis_of_Movie_Genre_Trends_Revenue_and_Genre_Diversity_Over_Time.py
    │   │── Average_Revenue_Impact.py
    │   │── Correlation_Matrix_for_Revenue_and_Popularity.py
    │   │── Genre_Trends_and_Revenue.py
    │   │── Genres_and_Their_Impact_on_Revenue.py
    │   │── Movie_Industry_Trends_Yearly_Analysis_of_Revenue_Genre_Diversity_and_Movie_Releases.py
    │   │── Process_Movie_Data_with_Genres_and_Profit.py
    │   │── Revenue_and_Movie_Distribution_by_Genre_Count.py
    │── 📂 time_series_prediction # LSTM & SARIMA models for revenue forecasting
    │   │── LSTM.py
    │   │── SARIMA.py
    │── README.md # Project documentation



## 📌 Project Overview  

This project explores trends in the **movie industry**, focusing on revenue, genre diversity, and popularity. The analysis includes:

✅ **Movie Revenue Trends** - How revenue has changed over time.  
✅ **Genre Popularity & Impact** - Which genres generate the most revenue?  
✅ **Correlation Analysis** - Relationship between revenue and other factors.  
✅ **Time-Series Forecasting** - Predicting future movie revenue using LSTM and SARIMA.  

## 📂 Description of Folders  

### 📂 `data/`  
Contains raw datasets used for analysis.  

### 📂 `notebooks/`  
Jupyter Notebook files (`.ipynb`) with full analysis and code.  

### 📂 `graphs/`  
Stores all the generated visualizations and figures in PNG format.  

### 📂 `src/`  
Python scripts (`.py`) for different parts of the analysis:

- **`Analysis_of_Movie_Genre_Trends_Revenue_and_Genre_Diversity_Over_Time.py`** → Examines how genre diversity and revenue trends have changed over time.  
- **`Average_Revenue_Impact.py`** → Evaluates the average impact of different genres on revenue.  
- **`Correlation_Matrix_for_Revenue_and_Popularity.py`** → Analyzes the correlation between revenue and popularity.  
- **`Genre_Trends_and_Revenue.py`** → Studies evolving genre trends and their effect on revenue.  
- **`Genres_and_Their_Impact_on_Revenue.py`** → Investigates how different genres contribute to revenue.  
- **`Movie_Industry_Trends_Yearly_Analysis_of_Revenue_Genre_Diversity_and_Movie_Releases.py`** → Analyzes yearly trends in revenue, genre diversity, and movie releases.  
- **`Process_Movie_Data_with_Genres_and_Profit.py`** → Prepares and cleans the movie dataset for analysis.  
- **`Revenue_and_Movie_Distribution_by_Genre_Count.py`** → Examines how the number of movies in each genre impacts revenue.  

### 📂 `time_series_prediction/`  
Contains forecasting models:  

- **`LSTM.py`** → Deep learning model for revenue forecasting.  
- **`SARIMA.py`** → Time series model for revenue prediction.  

## 📊 Key Findings  

🔹 **Action and Adventure movies** consistently generate the highest revenue.  
🔹 **Genre diversity has increased over the years,** with new hybrid genres emerging.  
🔹 **Revenue and popularity are positively correlated,** but not all popular movies are profitable.  
🔹 **LSTM outperformed SARIMA** in predicting future revenue trends, capturing nonlinear patterns better.  

