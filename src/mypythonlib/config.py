import os
from pathlib import Path
from typing import ClassVar
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from mypythonlib.helper import get_default_device

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    debug_mode: str
    device: str = Field(default_factory=get_default_device)
    seeds: ClassVar[list[int]] = [13,17,42,55, 88]
    
    @computed_field
    @property
    def projets_path(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @computed_field
    @property
    def models_path(self) -> Path:
        models_path = self.projets_path / "experimentation" / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        return models_path

    @computed_field
    @property
    def training_logs_path(self) -> Path:
        training_logs_path = self.projets_path / "experimentation" / "train_logs"
        training_logs_path.mkdir(parents=True, exist_ok=True)
        return training_logs_path

    @computed_field
    @property
    def games_assets_path(self) -> Path:
        return self.projets_path / "game_assets"

settings = Settings()
