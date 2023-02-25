from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from twang.types import AudioFormat

_BaseTrack = TypeVar("_BaseTrack", bound="BaseTrack")


class BaseTrack(ABC):
    """Defines a standard API for working with audio tracks."""

    # the audio signal; its format depends on the `Track` implementation.
    y: Any

    # sampling rate (TODO: should this be optional)
    sr: Optional[int]

    # TODO: consider whether this should be an attribute
    # whether or not to use track onsets in feature extraction
    use_onset: bool

    # TODO: make this non-optional by forcing the Librosa track to also do it
    audio_format: Optional[AudioFormat]

    # TODO: add ability to slice (using milliseconds) to return a shortened version of the track

    @classmethod
    @abstractmethod
    def from_file(cls: Type[_BaseTrack], file_path: str) -> _BaseTrack:
        """Loads the track from disk"""
        ...

    @classmethod
    @abstractmethod
    def from_file_snippet(cls: Type[_BaseTrack], file_path: str, start_time: float, end_time: float) -> _BaseTrack:
        ...

    @abstractmethod
    def save(self, save_path: str, audio_format: Optional[AudioFormat]):
        """Serializes the track to disk"""
        ...

    @abstractmethod
    def display(self):
        """Displays the Track in a Jupyter environment"""
        ...

    @property
    @abstractmethod
    def duration(self) -> float:
        ...

    @property
    @abstractmethod
    def bpm(self) -> float:
        ...
