from abc import ABC, abstractmethod
from typing import Dict

from twang.track import BaseTrack

SourceSeparationDict = Dict[str, BaseTrack]


class SourceSeparation(ABC):
    @abstractmethod
    def run(self, track: BaseTrack) -> SourceSeparationDict:
        ...
