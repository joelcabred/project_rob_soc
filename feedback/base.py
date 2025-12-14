
from abc import ABC, abstractmethod

class FeedbackBackend(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def poll(self):
        """
        Returns: -1, 0, +1 or None
        """
        pass

    def close(self):
        pass
