# San Francisco Restaurant Recommender System

## Objective
The objective of this project is to create a comprehensive recommendation system specifically for restaurants in San Francisco using Yelp data. The system includes:
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
   git clone https://github.com/yourusername/san-francisco-recommender-system.git
   cd san-francisco-recommender-system

## Folder Structure

The project is structured as follows:

```plaintext

yelp-restaurant-recommendation/
│
├── task1/
│   ├── extract_yelp_data.py         # API code to extract Yelp data
│   ├── README.md                    # Documentation for data extraction
│
├── task2/
│   ├── eda_and_recommendation.py    # EDA and recommendation system script
│   ├── README.md                    # Documentation for EDA and recommendation systems
│
├── task3/
│   ├── app.py                       # Streamlit app deployment code
│   ├── README.md                    # Documentation for app deployment
│
├── data/
│   ├── raw/                         # Raw data extracted from Yelp
│   ├── processed/                   # Processed data for analysis and modeling
│
├── requirements.txt                 # Project dependencies
├── LICENSE                          # License for the project
├── README.md                        # Main project README
└── .gitignore                       # Git ignore file
