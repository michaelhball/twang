from abc import ABC, abstractmethod
from typing import Dict

from twang.track import BaseTrack, LibrosaTrack
from twang.types import AudioFormat

SourceSeparationDict = Dict[str, LibrosaTrack]


def save_sources(sources: SourceSeparationDict, save_dir: str, audio_format: AudioFormat = AudioFormat.WAV):
    ...


class SourceSeparation(ABC):
    """A generic base class defining the API for all source-separation implementations."""

    @abstractmethod
    def run(self, track: BaseTrack) -> SourceSeparationDict:
        ...
