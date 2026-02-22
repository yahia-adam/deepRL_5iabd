from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

class Settings(BaseSettings):
    # Configuration de Pydantic pour lire le fichier .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Instanciation unique
settings = Settings()