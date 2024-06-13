import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Set the page configuration
st.set_page_config(page_title='SF Restaurant Recommender', layout='wide')

# Function to load and clean the dataset
def load_data():
    data = pd.read_csv('dataset/sfres_cleaned.csv')
    return data

# Load the dataset
data = load_data()

# Sidebar navigation with a dropdown menu using st.selectbox
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["Home", "Collaborative Filtering Recommendation", "Hybrid Recommendation", "Dataset"]
)

# Embed CSS for styling
st.markdown("""
<style>
    body {
        margin: 0;
        font-family: Arial, sans-serif;
        color: #333;
    }
    header {
        background: rgba(255, 87, 34, 0.8); /* Deep orange */
        padding: 20px;
        text-align: center;
        color: white;
    }
    .container {
        padding: 20px;
        position: relative;
        z-index: 1;
    }
    .card {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .card button {
        background: #FF5722; /* Deep orange */
        color: white;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 10px;
    }
    .card button:hover {
        background: #FF8A65; /* Light deep orange */
    }
</style>
""", unsafe_allow_html=True)

# Embed HTML for the header
st.markdown("""
<header>
    <h1>San Francisco Restaurant Recommender</h1>
</header>
""", unsafe_allow_html=True)

# Collaborative Filtering Model
def collaborative_filtering(data):
    # Create a matrix of features
    features = data[['rating', 'review_count', 'price_type']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform SVD
    svd = TruncatedSVD(n_components=min(2, scaled_features.shape[1] - 1))
    latent_matrix = svd.fit_transform(scaled_features)
    
    # Compute similarities
    item_similarity = cosine_similarity(latent_matrix)
    
    return item_similarity

# Hybrid Recommendation Model
def hybrid_recommendation(data, rating_pref, review_count_pref, price_type_pref, category_pref):
    # Filter data based on user preferences
    filtered_data = data[(data['rating'] >= rating_pref) & 
                         (data['review_count'] >= review_count_pref) &
                         (data['price_type'] == price_type_pref) &
                         (data['categories'].str.contains(category_pref, case=False, na=False))]
    
    if filtered_data.shape[0] == 0:
        return None, filtered_data

    # Collaborative filtering
    item_similarity_cf = collaborative_filtering(filtered_data)
    
    # Content-based filtering (using categories, etc.)
    content_df = filtered_data[['name', 'rating', 'review_count', 'categories', 'price_type']]
    content_df['Content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])
    
    # Compute cosine similarity between the content of different restaurants
    content_similarity = linear_kernel(content_matrix, content_matrix)
    
    # Combine collaborative and content-based similarities
    combined_similarity = (item_similarity_cf + content_similarity) / 2
    
    return combined_similarity, filtered_data

# Generate recommendations
def get_recommendations(similarity_matrix, filtered_data, top_n=10):
    # Calculate average similarity scores for each restaurant
    avg_similarity_scores = similarity_matrix.mean(axis=0)
    similar_restaurants = sorted(list(enumerate(avg_similarity_scores)),
    key=lambda x: x[1], reverse=True)
    recommended_restaurants = [filtered_data.iloc[i] for i, score in               similar_restaurants[:top_n]]
    
    return recommended_restaurants

# Main content based on dropdown selection
if page == "Home":
    st.markdown("""
    <div class="container">
        <div class="card">
            <h2>Home</h2>
            <p>Welcome to the San Francisco Restaurant Recommender app! We hope you enjoy our recommendations.</p>
            <p>San Francisco is a culinary haven known for its diverse food scene, offering everything from Michelin-starred restaurants to vibrant street food. The city's rich cultural tapestry is reflected in its wide variety of cuisines, making it a true paradise for food lovers.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    image_url = 'C:/users/chimi/Desktop/Python Data Science Projects/sf_restaurant_recommender/dataset/A_warm_and_inviting_background_image_suitable_for_.jpg'  
    st.image(image_url, use_column_width=True)

elif page == "Collaborative Filtering Recommendation":
    st.markdown("""
    <div class="container">
        <div class="card">
            <h2>Collaborative Filtering Recommendation</h2>
    """, unsafe_allow_html=True)
    
    # Form for user input
    st.markdown("### Enter your preferences")
    rating_pref = st.slider("Minimum Rating", 1, 5, 3)
    review_count_pref = st.number_input("Minimum Number of Reviews", min_value=1, value=50)
    price_type_pref = st.selectbox("Price Type", options=[2.0, 3.0, 4.0, 1.0])
    
    if st.button("Get Collaborative Filtering Recommendations"):
        # Get collaborative filtering recommendations
        filtered_data = data[(data['rating'] >= rating_pref) & 
                             (data['review_count'] >= review_count_pref) &
                             (data['price_type'] == price_type_pref)]
        if filtered_data.shape[0] > 0:
            similarity_matrix = collaborative_filtering(filtered_data)
            recommendations = get_recommendations(similarity_matrix, filtered_data)
            st.write("Recommended Restaurants:")
            for rec in recommendations:
                st.write(f"Name: {rec['name']}, Address: {rec['address']}, Phone: {rec['phone']}")
        else:
            st.write("No restaurants match your preferences.")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

elif page == "Hybrid Recommendation":
    st.markdown("""
    <div class="container">
        <div class="card">
            <h2>Hybrid Recommendation</h2>
    """, unsafe_allow_html=True)
    
    # Form for user input
    st.markdown("### Enter your preferences")
    rating_pref = st.slider("Minimum Rating", 1, 5, 3)
    review_count_pref = st.number_input("Minimum Number of Reviews", min_value=1, value=50)
    price_type_pref = st.selectbox("Price Type", options=[2.0, 3.0, 4.0, 1.0])
    category_pref = st.text_input("Category (e.g., Italian, Chinese, etc.)")
    
    if st.button("Get Hybrid Recommendations"):
        # Get hybrid recommendations
        similarity_matrix, filtered_data = hybrid_recommendation(data, rating_pref, review_count_pref, price_type_pref, category_pref)
        if similarity_matrix is not None:
            recommendations = get_recommendations(similarity_matrix, filtered_data)
            st.write("Recommended Restaurants:")
            for rec in recommendations:
                st.write(f"Name: {rec['name']}, Address: {rec['address']}, Phone: {rec['phone']}")
        else:
            st.write("No restaurants match your preferences.")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

elif page == "Dataset":
    st.markdown("""
    <div class="container">
        <div class="card">
            <h2>Dataset</h2>
            <p>Here is the dataset used for these recommendations.</p>
    """, unsafe_allow_html=True)
    st.dataframe(data)
    st.markdown("</div></div>", unsafe_allow_html=True)

