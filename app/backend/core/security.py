import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_security = HTTPBasic()
_USERNAME = os.getenv("API_USERNAME", "tobio")
_PASSWORD = os.getenv("API_PASSWORD", "tobio")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(_security)) -> str:
    valid_user = secrets.compare_digest(credentials.username, _USERNAME)
    valid_pass = secrets.compare_digest(credentials.password, _PASSWORD)
    if not (valid_user and valid_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
