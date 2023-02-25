from typing import Optional

import pydub

from twang.track.base import BaseTrack
from twang.types import AudioFormat
from twang.util import time as time_util

# TODO: add a function that converts to / from a Librosa track


class PyDubTrack(BaseTrack):
    y: pydub.AudioSegment

    def __init__(self, y: pydub.AudioSegment, sr: Optional[int] = None, audio_format: Optional[AudioFormat] = None):
        self.y = y
        self.sr = sr
        self.audio_format = audio_format

    @classmethod
    def from_file(cls, file_path: str, audio_format: Optional[AudioFormat] = None):
        audio_format_str = None if audio_format is None else audio_format.name.lower()
        audio_segment = pydub.AudioSegment.from_file(file_path, format=audio_format_str)
        # TODO: can I also extract the sample rate? (is it 'sample_width')
        return cls(y=audio_segment, audio_format=audio_format)

    @classmethod
    def from_file_snippet(
        cls,
        file_path: str,
        start_time: float,
        end_time: float,
        audio_format: Optional[AudioFormat] = None,
    ):
        track = cls.from_file(file_path, audio_format=audio_format)
        track.y = track.y[int(time_util.s_to_ms(start_time)) : int(time_util.s_to_ms(end_time))]
        return track

    def save(self, save_path: str):
        self.y.export(save_path, format=self.audio_format or "wav")

    def display(self) -> pydub.AudioSegment:
        return self.y

    @property
    def duration(self):
        pass

    @property
    def bpm(self):
        pass
