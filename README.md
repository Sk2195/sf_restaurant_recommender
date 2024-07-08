# Yelp Data Restaurant Recommendation System

## Objective
The objective of this project is to create a comprehensive recommendation system for restaurants using Yelp data. The system includes:
1. **Data Extraction**: Extracting restaurant data from Yelp.
2. **Exploratory Data Analysis (EDA)**: Analyzing the data to understand key insights.
3. **Recommendation Systems**: 
   - **Location-Based Recommendation**: Recommending restaurants based on location.
   - **Hybrid Recommendation**: Recommending restaurants based on a combination of rating, review count, categories, and price type.
4. **App Deployment**: Deploying the recommendation system using Streamlit for an interactive user experience.

## Project Structure
- **task1**: Contains the API code used to extract data from Yelp.
- **task2**: Includes the EDA and two recommendation systems:
  - **Location-Based Recommendation**: Recommends restaurants based on their location.
  - **Hybrid Recommendation**: Uses content-based filtering combined with numeric feature similarity to recommend restaurants.
- **task3**: Contains the code for deploying the application using Streamlit.

## Installation
To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yelp-restaurant-recommendation.git
   cd yelp-restaurant-recommendation

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt

streamlit run app.py
