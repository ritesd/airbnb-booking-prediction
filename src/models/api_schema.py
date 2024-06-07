"""
Here we definded all core level schema like error
"""
from pydantic import BaseModel


class HTTPError(BaseModel):
    """
    JSON Schema for rendering HTTP Errors
    """
    status_code: int
    detail: str
