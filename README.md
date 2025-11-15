**CineSense – Movie Recommendation System**

**Overview**



**CineSense** is a content-based movie recommendation system built using a Netflix movie dataset.

The system creates vector embeddings from movie metadata, builds a FAISS index for fast similarity search, and provides real-time recommendations through a Streamlit interface.



**Features**



* Cleaned and enriched movie dataset



* Poster integration using TMDb API



* Fast similarity search using FAISS



* Interactive EDA with Plotly



* Streamlit UI for real-time recommendations



* Similarity scores displayed as percentages



Tech Stack



Python, pandas, numpy, scikit-learn, FAISS, Plotly, Streamlit, TMDb API



**Project Structure**

**CineSense/**

│

├── app/

│   ├── app.py

│   └── style.css

│

├── model/

│   ├── embeddings.npy

│   └── faiss\_index.pkl

│

├── data/

│   └── netflix\_data\_with\_links.csv

│

├── notebooks/

│   ├── EDA.ipynb

│   └── faiss\_index.ipynb

│

├── screenshots/

│   ├── demo.png

│   └── eda\_plot.png

│

├── README.md

├── requirements.txt

└── .gitignore



**How to Run**



Install dependencies:



pip install -r requirements.txt





**Run the Streamlit app:**



streamlit run app/app.py



**Future Improvements**



* Hybrid recommendation system



* Online deployment



* Enhanced embedding generation (transformer-based)
## Screenshots

### Recommendation Output
![Demo](screenshots/demo.png)

### EDA Visualization
![EDA](screenshots/Top 5 genre trend.png)
