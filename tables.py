from sqlalchemy import REAL, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from db_init import Base


class AnimeInfo(Base):
    __tablename__ = "anime_info"
    __table_args__ = {'extend_existing':True}
    anime_ids = Column(Integer, primary_key=True)
    name = Column(String)
    genre = Column(String)
    anime_type = Column(String, name="type")
    episodes = Column(String)
    rating = Column(REAL)
    members = Column(Integer)

    # Добавляем связь
    ratings = relationship("AnimeRatings", back_populates="anime")


class AnimeRatings(Base):
    __tablename__ = "anime_ratings"
    __table_args__ = {'extend_existing':True}
    User_ID = Column(Integer, name="User_ID", primary_key=True)  # Добавляем primary_key
    Anime_ID = Column(Integer, ForeignKey("anime_info.anime_ids"))
    Feedback = Column(Integer, name="Feedback")

    anime = relationship("AnimeInfo", back_populates="ratings")
