import enum
from typing import List, Mapping, Union


class AudioFormat(enum.Enum):
    NONE = None
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    MP3 = "mp3"


JSON = Union[str, int, float, bool, None, Mapping[str, "JSON"], List["JSON"]]
