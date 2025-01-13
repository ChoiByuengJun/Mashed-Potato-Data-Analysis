from abc import ABC, abstractmethod

class PCAService(ABC):
    @abstractmethod
    async def performPCA(self):
        pass
