from abc import ABC, abstractmethod
from typing import Any


class InferenceClient(ABC):
    @abstractmethod
    def __init__(self, args) -> None:
        """
        Initializes the client.
        """
        pass

    @abstractmethod
    def infer(self, obs, instruction) -> dict:
        """
        Does inference on observation and returns the final processed
        dictionary used to do inference.
        """

        pass

    @abstractmethod
    def reset(self):
        """
        Resets the client to start a new episode.
        """
        pass

    def episode_metrics(self) -> dict[str, Any]:
        """Return JSON-serializable metrics for the current episode."""
        return {}

    def close(self) -> None:
        """Release client-owned resources."""
