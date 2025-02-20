import time
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from db_init import SessionLocal, engine
from tables import AnimeInfo

session = SessionLocal()


def batch_load_sql(query) -> pd.DataFrame:
    """
    Optimaly loads data from sql to pd.DataFrame

    Args:
        query: sqlalchemy session.query for pd.read_sql()
    Returns:
        df: resulted DataFrame
    """
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query.statement, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def fetch_and_process_anime_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch data from table anime_info in database and adjust it to the format model was trained

    Returns:
        Tuple[df_merged,df_watch_history]
        df_merged - transfomed df to fit the format
        df_watch_history - filtered DataFrame, consists of only users with 6+ anime watched (and rated)
    """
    query_anime_info = session.query(AnimeInfo)
    df_anime = batch_load_sql(query_anime_info)
    df_anime["genre"] = df_anime["genre"].str.split(", ")
    df_anime = df_anime.rename(columns={"anime_ids": "Anime_ID"})
    df_anime["genre_str"] = df_anime["genre"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    )
    vectorizer = TfidfVectorizer()
    genre_tfidf = vectorizer.fit_transform(df_anime["genre_str"])
    genre_tfidf_df = pd.DataFrame(
        genre_tfidf.toarray(), columns=vectorizer.get_feature_names_out()
    )
    # Добавляем к основному датасету
    df_merged = pd.concat([df_anime, genre_tfidf_df], axis=1).drop(
        columns=["genre", "genre_str"]
    )
    df_merged["members_log"] = np.log1p(df_merged["members"])
    df_merged = df_merged.drop(columns=["members", "name"])
    df_watch_history = pd.read_csv("data/df_filtered.csv")
    return df_merged, df_watch_history


def prepare_data_for_user(df_anime: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Creates DataFrame that matches model prediction format. Basicaly this function is needed to add column User_ID consisting of only 1 user_id
    and fit the format to match prediciton format.

    Args:
        df_anime: pd.DataFrame describing every anime (type, ratings mean value,amount of episodes etc.)
        user_id: int user_id

    Returns:
        user_df: pd.DataFrame matching format of the model prediction
    """
    user_df = df_anime.copy()
    user_df["User_ID"] = user_id  # Добавляем user_id ко всем аниме

    # Приводим к виду, как в обучении
    # TODO добавить параметр features, чтобы не хардкодить внутри
    binary_genres = list(df_anime.columns[7:-2])
    features = [
        "User_ID",
        "Anime_ID",
        "type",
        "rating",
        "episodes",
        "members_log",
    ] + binary_genres

    return user_df[features]


def get_top_n_recommendations(model, user_df: pd.DataFrame, n=5) -> pd.DataFrame:
    """
    Gets top n (highest in probability) recommendations. Default n is 5.

    Args:
        model: catboost.CatboostClassifier pretrained model for this task
        user_df: pd.DataFrame processed with prepare_data_for_user() DataFrame

    Returns:
        recommendations: pd.DataFrame with 2 columns : Anime_ID and score(probability
    """
    user_df = user_df.copy()
    user_df["score"] = model.predict_proba(user_df)[:, 1]
    recommendations = user_df.sort_values("score", ascending=False).head(n)
    return recommendations[["Anime_ID", "score"]]


def hitrate_at_5(
    model, df_anime, df_watch_history, user_ids
) -> Tuple[int | float, Dict]:
    """
    Metric that measures if at least 1 out of 5 recommendations was valid for the user. If there are more than 1 user returns average hit_rate

    Args:
    model: catboost.CatboostClassifier pretrained model for this task
    df_anime: pd.DataFrame describing every anime (type, ratings mean value,amount of episodes etc.)
    df_watch_history pd.DataFrame, consists of only users with 6+ anime watched (and rated)
    user_ids: #TODO describe wth is this T_T

    Returns:
        Tuple[avg_hit, rec_dict]
        avg_hit: int | float hitrate score
        rec_dict: Dict of recommendations for users
    """
    start_time = time.time()
    hits = []
    rec_dict = {}
    with tqdm(
        total=len(user_ids),
        desc="Processing users",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} users processed",
    ) as pbar:
        for user_id in user_ids:
            user_df = prepare_data_for_user(df_anime, user_id)
            recommendations = get_top_n_recommendations(model, user_df, n=5)

            watched_anime = df_watch_history[df_watch_history["User_ID"] == user_id][
                "Anime_ID"
            ].tolist()
            hit = int(
                any(
                    anime_id in watched_anime
                    for anime_id in recommendations["Anime_ID"]
                )
            )
            rec_dict[user_id] = recommendations.to_dict()
            hits.append(hit)

            pbar.update(1)  # Обновляем прогресс

    # Подсчитаем средний hitrate
    avg_hitrate = sum(hits) / len(hits)

    # Логирование квантилей
    quantiles = [0.1, 0.25, 0.5, 0.75, 1.0]
    hit_quantiles = np.percentile(hits, [q * 100 for q in quantiles])

    print(f"Avg Hitrate@5: {avg_hitrate:.3f}")
    print(
        f"Hitrate percentiles: 10th={hit_quantiles[0]:.3f}, 25th={hit_quantiles[1]:.3f}, "
        f"50th={hit_quantiles[2]:.3f}, 75th={hit_quantiles[3]:.3f}, 100th={hit_quantiles[4]:.3f}"
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    return avg_hitrate, rec_dict
