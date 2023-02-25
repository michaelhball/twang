from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar, Union

from twang.types import AudioFormat

_BaseTrack = TypeVar("_BaseTrack", bound="BaseTrack")


class BaseTrack(ABC):
    """Defines a standard API for working with audio tracks."""

    # the audio signal; its format depends on the `Track` implementation.
    y: Any

    # sampling rate (TODO: should this be non-optional ?)
    sr: Optional[int]

    @classmethod
    @abstractmethod
    def from_file(cls: Type[_BaseTrack], file_path: str) -> _BaseTrack:
        """Load an audio file from disk."""
        ...

    @classmethod
    @abstractmethod
    def from_file_snippet(cls: Type[_BaseTrack], file_path: str, start_time: float, end_time: float) -> _BaseTrack:
        ...

    @abstractmethod
    def _repr_html_(self) -> str:
        ...

    @abstractmethod
    def __getitem__(self, ms_or_slice: Union[int, float, slice]):
        """A standardised API for slicing the audio into a shorter `BaseTrack`,  or accessing a specific timestamp."""
        ...

    @abstractmethod
    def save(self, save_path: str, audio_format: AudioFormat):
        """Serialize the track to disk."""
        ...

    @property
    @abstractmethod
    def duration(self) -> float:
        ...

    @property
    @abstractmethod
    def bpm(self) -> float:
        ...
