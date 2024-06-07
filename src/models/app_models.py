"""
Here we defined and store all the database models for prequin app
"""
from passlib.context import CryptContext

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")