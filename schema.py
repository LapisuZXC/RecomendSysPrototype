from pydantic import BaseModel
from typing import Optional


class AnimeInfoBase(BaseModel):
    name: str
    genre: Optional[str] = None
    anime_type: Optional[str] = None
    episodes: Optional[str] = None
    rating: Optional[float] = None
    members: Optional[int] = None

    class Config:
        orm_mode = True
        from_attributes = True


class AnimeRatingsBase(BaseModel):
    user_id: int
    anime_id: int
    feedback: Optional[int] = None

    class Config:
        orm_mode = True
        from_attributes = True
