# Recommendation System Prototype

## Overview
This project is a prototype of a recommendation system for anime using a machine learning model trained with CatBoost. The system is based on user ratings of 7000 anime given by 5000 users. The API is implemented using FastAPI.

## Features
- Uses **CatBoostClassifier** for recommendations
- Processes anime data with **TF-IDF vectorization**
- Provides **Top-N recommendations** for a given user
- Computes **Hitrate@5** as an evaluation metric
- Exposes API endpoints for fetching recommendations

## Project Structure
```
📂 project_root
├── 📂 data              # Dataset and preprocessed data
├── 📂 models            # Trained models
├── 📂 database          # Database-related scripts
├── 📂 notebooks         # Jupyter notebooks with EDA
├── 📄 RecSys.py         # Core recommendation system functions
├── 📄 app.py            # FastAPI application
├── 📄 db_init.py        # Database initialization
├── 📄 tables.py         # ORM models for the database
└── 📄 README.md         # Project documentation
```

## Installation
### Requirements
- Python 3.8+
- Required libraries:
  ```sh
  pip install -r requirements.txt
  ```

## Usage
### Running the API
```sh
uvicorn app:app --reload
```

### API Endpoints
- `GET /recomendations/all` – Get recommendations for all users and compute average Hitrate@5.
- `GET /recomendations/{user_id}` – Get top 5 anime recommendations for a specific user.

## Core Components
### `RecSys.py`
Contains functions for data processing and recommendation generation:
- **`fetch_and_process_anime_data()`** – Loads and prepares anime data
- **`prepare_data_for_user(df_anime, user_id)`** – Formats data for prediction
- **`get_top_n_recommendations(model, user_df, n=5)`** – Predicts top-N anime for a user
- **`hitrate_at_5(model, df_anime, df_watch_history, user_ids)`** – Computes Hitrate@5

### `app.py`
Implements the API with FastAPI and serves recommendations:
- Loads the trained CatBoost model
- Provides API endpoints for retrieving recommendations

## Model Training
The recommendation model is trained on a dataset containing user ratings for anime. The anime data is preprocessed, including:
- Extracting **TF-IDF** features from genres
- Transforming numerical features
- Filtering users with sufficient rating history

## Evaluation
The model is evaluated using **Hitrate@5**, which measures the percentage of users who have at least one relevant recommendation among the top 5.

## Exploratory Data Analysis (EDA)
Jupyter notebooks containing exploratory data analysis (EDA) can be found in the `notebooks/` directory. These notebooks include:
- Data cleaning and preprocessing
- Initial insights into user behavior and anime ratings
- Feature selection and engineering strategies

## TODO
- Add requirements
- Improve feature engineering for better model accuracy
- Implement additional evaluation metrics

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss the proposal.

## License
MIT License

