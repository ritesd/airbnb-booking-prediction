import sqlite3
from fastapi import Depends, HTTPException, APIRouter, Request
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from src.app.db_setup import create_db
from src.models.database import database, text
from src.models.app_models import password_context
from src.user_apis.api_schema import Token, UserInDB
from src.user_apis.create_token import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token

router = APIRouter(prefix="/user")

@router.post("/token", response_model=Token)
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    logger = request.app.logger
    logger.info("API call for user login and aceess token")
    query = text("SELECT username, password FROM users WHERE username = :username")
    query = query.bindparams(username=form_data.username)
    try:
        user = await database.fetch_one(query)
    except sqlite3.OperationalError:
        create_db()
        user = await database.fetch_one(query)
    logger.info("Fetch user from database")
    if user is None or not password_context.verify(form_data.password, user['password']):
        logger.info("Incorrect username or password")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register")
async def register_user(user: UserInDB):
    hashed_password = password_context.hash(user.password)
    query = text("INSERT INTO users (username, password) VALUES (:username, :password)")
    query = query.bindparams(username=user.username, password=hashed_password)
    try:
        await database.execute(query)
    except sqlite3.OperationalError:
        create_db()
        await database.execute(query)
    return {"message": "User registered successfully"}
