from fastapi import FastAPI

from db_init import SessionLocal

app = FastAPI()
session = SessionLocal()
