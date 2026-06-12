import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # App Settings
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    upload_dir: str = "data/uploads"
    report_dir: str = "data/reports"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
