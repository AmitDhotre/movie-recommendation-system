# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Recommendation-green)

## 📌 Project Overview

This project is a Content-Based Movie Recommendation System that suggests movies similar to a user's selected movie using Machine Learning techniques.

The system analyzes movie metadata such as genres, keywords, cast, crew, and overview to find similar movies and provide personalized recommendations.

## 🚀 Features

- 🎥 Movie Recommendation Engine
- 📊 Similarity Score Calculation
- 🖼️ Movie Poster Display
- ⭐ Rating-Based Filtering
- 📅 Release Year Filtering
- 🎭 Genre-Based Filtering
- 🔍 Smart Movie Search
- 🌐 Interactive Streamlit Web Interface

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- TMDB 5000 Dataset

## 📂 Project Structure

```text
Movie-Recommendation-System
│
├── amit.py
├── movies_with_posters.csv
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Working Process

### 1️⃣ Data Collection
- TMDB 5000 Movies Dataset
- Movie Posters Dataset

### 2️⃣ Data Preprocessing
- Handle missing values
- Merge movie and credit datasets
- Extract genres, cast, crew, and keywords

### 3️⃣ Feature Engineering
- Create movie tags using:
  - Genres
  - Keywords
  - Overview
  - Cast
  - Director

### 4️⃣ Text Vectorization
- TF-IDF Vectorizer
- Convert movie information into numerical vectors

### 5️⃣ Similarity Calculation
- Cosine Similarity
- Find movies with similar content

### 6️⃣ Recommendation Generation
- Return top similar movies
- Display posters and similarity scores

## 📊 Machine Learning Concepts Used

- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity
- Content-Based Filtering
- Feature Engineering

## 📸 Application Preview

### Home Page
- Movie Search
- Filters
- Top Rated Movies

### Recommendation Page
- Similar Movies
- Similarity Percentage
- Movie Posters
- Movie Description

## ▶️ Run Locally

### Clone Repository

```bash
git clone https://github.com/AmitDhotre/movie-recommendation-system.git
```

### Navigate to Project

```bash
cd movie-recommendation-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run amit.py
```

## 🎯 Future Enhancements

- Deep Learning-Based Recommendations
- User Authentication
- Watchlist Feature
- TMDB API Integration
- Hybrid Recommendation System
- Personalized User Recommendations

## 👨‍💻 Author

**Amit Dhotre**

🎓 Computer Engineering Student  
📊 Data Science & AI/ML Enthusiast  
🐍 Python Developer  
☁️ Cloud Computing Learner

### Connect With Me

- LinkedIn: https://www.linkedin.com/in/amit-dhotre
- GitHub: https://github.com/AmitDhotre

---

⭐ If you like this project, don't forget to star the repository!
