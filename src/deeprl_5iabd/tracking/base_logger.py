from abc import ABC, abstractmethod
from typing import Any

class BaseLogger(ABC):
    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Enregistre une valeur numérique pour un graphe (ex: Loss, WinRate)."""
        pass

    @abstractmethod
    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        """Enregistre plusieurs valeurs en une seule fois."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Ferme la connexion au logger."""
        pass
