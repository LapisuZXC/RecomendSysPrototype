from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from db_init import engine, SessionLocal
from tables import AnimeInfo, AnimeRatings


def batch_load_sql(query) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


session = SessionLocal()

query_anime_info = session.query(AnimeInfo)
query_anime_ratings = session.query(AnimeRatings)

df_anime_info = batch_load_sql(query_anime_info)
df_anime_ratings = batch_load_sql(query_anime_ratings)


df_merged = pd.merge(
    df_anime_ratings,
    df_anime_info,
    left_on="anime_id",
    right_on="anime_ids",
    how="left",
)
del df_anime_info, df_anime_ratings
X = df_merged.drop(columns="feedback")
y = df_merged["feedback"]

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
cb = CatBoostRegressor(100, 0.01, max_depth=100)
