import databases
import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.sql import text


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./my_database.db")

database = databases.Database(DATABASE_URL)
metadata = MetaData()

engine = create_engine(DATABASE_URL)
metadata.create_all(engine)