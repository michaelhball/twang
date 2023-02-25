from typing import Optional, Union

import pydub

from twang.track.base import BaseTrack
from twang.types import AudioFormat
from twang.util import time as time_util

# TODO: add a function that converts to / from a Librosa track


class PyDubTrack(BaseTrack):
    y: pydub.AudioSegment

    def __init__(self, y: pydub.AudioSegment, sr: Optional[int] = None):
        self.y = y
        self.sr = y.frame_rate

    @classmethod
    def from_file(cls, file_path: str, audio_format: AudioFormat = AudioFormat.NONE):
        audio_segment = pydub.AudioSegment.from_file(file_path, format=audio_format.value)
        # TODO: can I also extract the sample rate? (is it 'sample_width')
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
        track.y = track.y[int(time_util.s_to_ms(start_time)) : int(time_util.s_to_ms(end_time))]
        return track

    def _repr_html_(self) -> str:
        return self.y._repr_html_()

    def __getitem__(self, ms_or_slice: Union[int, float, slice]):
        """TODO: test & document"""
        self.y = self.y[ms_or_slice]
        return self

    def save(self, save_path: str, audio_format: AudioFormat = AudioFormat.WAV):
        self.y.export(save_path, format=audio_format.value)

    @property
    def duration(self):
        pass

    @property
    def bpm(self):
        pass
