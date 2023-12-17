import os
from abc import ABC, abstractmethod
from typing import Dict

from twang.track import BaseTrack, LibrosaTrack
from twang.types import AudioFormat
from twang.util import file_util


class SourceSeparationDict(Dict[str, LibrosaTrack]):
    def save(self, save_dir: str, audio_format: AudioFormat = AudioFormat.WAV):
        file_util.mkdir(save_dir)
        for source_name, source_track in self.items():
            source_track.save(os.path.join(save_dir, f"{source_name}.{audio_format.value}"), audio_format=audio_format)


class SourceSeparation(ABC):
    """A generic base class defining the API for all source-separation implementations."""

    @abstractmethod
    def run(self, track: BaseTrack) -> SourceSeparationDict:
        """
        Runs a source separation algorithm, separating the `track` audio into different components. Note that the API
        supports basing any BaseTrack subclass (e.g. PyDubTrack, LibrosaTrack). It's up to the SourceSeparation
        subclass implementation to ensure that this is possible.
        """
        raise NotImplementedError
