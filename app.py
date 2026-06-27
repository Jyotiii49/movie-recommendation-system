import streamlit as st
from model import recommend, new_df

st.title("🎬 Movie Recommendation System")

movie_list = new_df['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(movie)