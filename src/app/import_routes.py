from fastapi import FastAPI

from src.predict import route
from src.user_apis import user_auth_api

def import_routes(app: FastAPI) -> None:
    # user auth apis
    app.include_router(user_auth_api.router)

    # sample apis
    app.include_router(route.router)