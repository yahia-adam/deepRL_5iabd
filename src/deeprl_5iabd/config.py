import os
from pathlib import Path
from typing import ClassVar
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from deeprl_5iabd.helper import get_default_device

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    debug_mode: bool = False
    device: str = Field(default_factory=get_default_device)

    models_dir: str = "experimentation_logs/models"
    training_logs_dir: str = "experimentation_logs/train_logs"

    line_world_assets_dir: str = "game_assets/line_world/v1"
    tictactoe_assets_dir: str = "game_assets/tictactoe/v1"
    quarto_assets_dir: str = "game_assets/quarto/v1"

    @computed_field
    @property
    def project_path(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @computed_field
    @property
    def models_path(self) -> Path:
        return self.project_path / self.models_dir

    @computed_field
    @property
    def training_logs_path(self) -> Path:
        return self.project_path / self.training_logs_dir

    @computed_field
    @property
    def line_world_assets_path(self) -> Path:
        return self.project_path / self.line_world_assets_dir

    @computed_field
    @property
    def tictactoe_assets_path(self) -> Path:
        return self.project_path / self.tictactoe_assets_dir

    @computed_field
    @property
    def quarto_assets_path(self) -> Path:
        return self.project_path / self.quarto_assets_dir

    def setup_directories(self) -> None:
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.training_logs_path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.setup_directories()