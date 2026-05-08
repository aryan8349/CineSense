# CineSense — Content-Based Movie Recommendation System

Recommendation engine built on Netflix movie metadata 
using FAISS vector similarity search and a live 
Streamlit interface.

**Stack:** Python · Pandas · Scikit-learn · FAISS · 
           Plotly · Streamlit · TMDb API

---

## Problem Statement

Generic movie browsing is inefficient. Most users 
abandon search without finding relevant content.
CineSense solves this by recommending movies based 
on content similarity — genre, cast, director, 
and thematic metadata — not collaborative filters 
that require user history.

---

## Why Content-Based + FAISS

- **Content-based**: works without user history — 
  cold start problem doesn't apply
- **FAISS over brute-force cosine search**: 
  scales to large catalogs with sub-millisecond 
  retrieval vs O(n) linear scan
- **TF-IDF + metadata embeddings**: captures 
  semantic similarity across genre, cast, and 
  description fields

---

## System Architecture
Raw Netflix Data
↓
Data Cleaning & Feature Engineering
↓
Metadata Embedding (TF-IDF vectorization)
↓
FAISS Index Construction
↓
Query Input (movie title)
↓
Similarity Search → Top-N Results
↓
Streamlit UI + TMDb Poster Integration

---

## Features

- Real-time recommendations with similarity 
  scores (%)
- TMDb API integration for poster display
- Interactive EDA with Plotly (genre trends, 
  rating distributions)
- Modular codebase: preprocessing, indexing, 
  and query logic separated

---

## Performance

- Retrieval latency: <50ms for top-10 results
- Vector matrices precomputed at index build time 
  — no repeated feature extraction at query time
- Similarity threshold tuned to reduce 
  irrelevant matches

---

## Repository Structure
CineSense/
├── app/            # Streamlit application  
├── data/           # Raw and processed datasets  
├── model/          # FAISS index + saved vectors  
├── notebook/       # EDA and model building  
├── screenshots/    # UI previews  
├── requirements.txt  
└── README.md  

---

## How to Run

1. Clone the repository
2. Install dependencies:
  pip install -r requirements.txt
3. Add your TMDb API key in `app/config.py`
4. Run the app:
   streamlit run app/app.py

---

## Screenshots

### Recommendation Output
<img width="1919" height="930" alt="Screenshot 2026-01-11 222051" src="https://github.com/user-attachments/assets/d360943f-b20e-4567-a81d-531acf0e6232" />

### EDA — Top Genre Trends
<img width="1916" height="925" alt="Screenshot 2026-01-11 222121" src="https://github.com/user-attachments/assets/9e53f446-c8ff-45a1-84f5-fa39e0817e9a" />
