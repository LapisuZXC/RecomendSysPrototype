import random
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from fastapi import FastAPI
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from db_init import SessionLocal
from tables import AnimeInfo
from RecSys import (
    prepare_data_for_user,
    fetch_and_process_anime_data,
    get_top_n_recommendations,
    hitrate_at_5,
)

session = SessionLocal()
app = FastAPI()
model = CatBoostClassifier()
model.load_model("models/catboost_recommender.cbm")

# TODO вынести функции куда-нибудь
# TODO сделать эндпоинты: /recomendations/all - все рекомендации, средний hitrate@5????
#                        /recomendations/{user_id} рекомендации для hitrate@5


@app.get("/recomendations/all")
def get_recomendations_for_all_users():
    df_anime, df_watch_history = fetch_and_process_anime_data()
    user_ids = random.sample(
        df_watch_history["User_ID"].unique().tolist(),
        df_watch_history["User_ID"].nunique(),
    )
    avg_hit, recomendations = hitrate_at_5(model, df_anime, df_watch_history, user_ids)
    if avg_hit >= 0.4:
        print("🎉 Модель достигла целевого уровня Hitrate@5!")
    else:
        print("⚠️ Нужно улучшить модель или фичи.")
    return json.dumps(recomendations)


@app.get("/recomendations/{user_id}")
async def get_user(user_id: int):
    df_anime, _ = fetch_and_process_anime_data()
    user_df = prepare_data_for_user(df_anime, user_id)
    recommendations = get_top_n_recommendations(model, user_df, n=5)
    return json.dumps({"user_id": user_id, "recomendations": recommendations.to_dict()})
