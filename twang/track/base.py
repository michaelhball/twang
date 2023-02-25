from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar, Union

import librosa
import numpy as np
import pydub
import soundfile as sf
from IPython import display as ipd

from twang.types import AudioFormat
from twang.util import time as time_util

_BaseTrack = TypeVar("_BaseTrack", bound="BaseTrack")


class BaseTrack(ABC):
    """Defines a standard API for working with audio tracks."""

    # the audio signal; its format depends on the `Track` implementation.
    y: Any

    # sampling rate
    sr: int

    @classmethod
    @abstractmethod
    def from_file(cls: Type[_BaseTrack], file_path: str) -> _BaseTrack:
        """Load an audio file from disk."""
        ...

    @abstractmethod
    def _repr_html_(self) -> str:
        """Display the BaseTrack in a Jupyter notebook"""
        ...

    @abstractmethod
    def __getitem__(self, ms_or_slice: Union[int, slice]):
        """A standardised API for slicing the audio into a shorter `BaseTrack`,  or accessing a specific timestamp."""
        ...

    @abstractmethod
    def save(self, save_path: str, audio_format: AudioFormat = AudioFormat.WAV):
        """Serialize the track to disk.

        Args:
            save_path: path at which to save the audio file
            audio_format: format in which to save the track
        """
        ...

    @property
    @abstractmethod
    def duration(self) -> float:
        ...

    @property
    @abstractmethod
    def bpm(self) -> float:
        ...


class PyDubTrack(BaseTrack):
    """# TODO: docstring"""

    y: pydub.AudioSegment

    def __init__(self, y: pydub.AudioSegment):
        self.y = y
        self.sr = y.frame_rate

    @classmethod
    def from_file(cls, file_path: str, audio_format: AudioFormat = AudioFormat.NONE):
        audio_segment = pydub.AudioSegment.from_file(file_path, format=audio_format.value)
        return cls(y=audio_segment)

    @classmethod
    def from_file_snippet(
        cls,
        file_path: str,
        start_time: float,
        end_time: float,
        audio_format: AudioFormat = AudioFormat.NONE,
    ):
        track = cls.from_file(file_path, audio_format=audio_format)
        return track[int(time_util.s_to_ms(start_time)) : int(time_util.s_to_ms(end_time))]

    @classmethod
    def from_librosa_track(cls, librosa_track: "LibrosaTrack"):
        y = y_new = librosa_track.y

        # interleave left & right channels into a 1D array
        if len(y.shape) == 2:
            y_new = np.zeros((y.shape[1] * 2), dtype=y.dtype)
            y_new[::2] = y[0]
            y_new[1::2] = y[1]

        return cls(
            pydub.AudioSegment(
                y_new.tobytes(),
                frame_rate=librosa_track.sr,
                sample_width=y_new.dtype.itemsize,
                channels=len(y.shape),
            )
        )

    def _repr_html_(self) -> str:
        return self.y._repr_html_()

    def __getitem__(self, ms_or_slice: Union[int, slice]):
        """TODO: test & document"""
        self.y = self.y[ms_or_slice]
        return self

    def save(self, save_path: str, audio_format: AudioFormat = AudioFormat.WAV):
        self.y.export(save_path, format=audio_format.value)

    @property
    def duration(self) -> float:
        raise NotImplementedError

    @property
    def bpm(self) -> float:
        raise NotImplementedError


class LibrosaTrack(BaseTrack):
    """TODO: docstring"""

    y: np.ndarray

    def __init__(self, y: np.ndarray, sr: int, use_onset: bool = False):
        self.y = y
        self.sr = sr
        self.use_onset = use_onset

    @classmethod
    def from_file(cls, file_path: str, use_onset: bool = False):
        return cls(*librosa.load(file_path), use_onset=use_onset)

    @classmethod
    def from_pydub_track(cls, pydub_track: PyDubTrack, use_onset: bool = False):
        samples = pydub_track.y.get_array_of_samples()

        # convert from int16 —> float32
        samples = librosa.util.buf_to_float(samples, n_bytes=2, dtype=np.float32)

        # pydub interleaves the left & right channels, => undo that to reconstruct a 2D array
        if pydub_track.y.channels == 2:
            samples = np.array([samples[::2], samples[1::2]])

        return cls(y=samples, sr=pydub_track.sr, use_onset=use_onset)

    def _repr_html_(self) -> str:
        return ipd.Audio(self.y, rate=self.sr)._repr_html_()

    def __getitem__(self, ms_or_slice: Union[int, slice]):
        raise NotImplementedError

    def save(self, save_path: str, audio_format: AudioFormat = AudioFormat.WAV):
        sf.write(
            save_path,
            data=self.y,
            samplerate=self.sr,
            format=audio_format.name,
            subtype="vorbis" if audio_format == AudioFormat.OGG else "PCM_24",
        )

    @property
    def duration(self) -> float:
        # TODO: add different formatting options here
        return librosa.get_duration(y=self.y)

    @property
    def bpm(self) -> float:
        raise NotImplementedError
