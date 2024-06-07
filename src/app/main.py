import os
from dotenv import load_dotenv
load_dotenv('.env')
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.app.import_routes import import_routes

app = FastAPI()

LOG_FILE = os.getenv('LOG_FILE')

@app.on_event("startup")
async def startup() -> None:
    """
    call startup functions
    """
    app.logger = logging.getLogger(name="FASTAPI-DOCKER-VSCODE")
    __file_handler = logging.FileHandler(filename="logs/app_logs.log")
    app.logger.addFilter(__file_handler)
    app.logger.info("Starting Up the application")

    import_routes(app)
    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)