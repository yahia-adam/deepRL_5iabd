from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Configuration de Pydantic pour lire le fichier .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Instanciation unique
settings = Settings()