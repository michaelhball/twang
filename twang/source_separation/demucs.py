import enum
import os
import tempfile

import torch
from demucs.apply import apply_model
from demucs.pretrained import DEFAULT_MODEL, get_model
from demucs.repo import AnyModel
from demucs.separate import load_track

from twang.source_separation.base import SourceSeparation, SourceSeparationDict
from twang.track.base import BaseTrack, LibrosaTrack


class DemucsModelType(enum.Enum):
    DEFAULT_MODEL = DEFAULT_MODEL

    # A fine-tuned version of the default that takes 4 times longer to run but is a bit better
    HTDEMUCS_FT = "htdemucs_ft"

    # Adds "piano" and "guitar" sources to the default four ["drums", "bass", "other", "vocals"].
    HTDEMUCS_6S = "htdemucs_6s"


def _track_to_demucs_audio_file(track: BaseTrack, model: AnyModel) -> torch.Tensor:
    """Converts any kind of BaseTrack to a Demucs audio file (i.e. a torch Tensor representation)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        track_path = track.file_path or os.path.join(tmp_dir, "track.wav")
        track.save(track_path)
        demucs_audio_file = load_track(track_path, model.audio_channels, model.samplerate)
    return demucs_audio_file


class DemucsSourceSeparation(SourceSeparation):
    """https://github.com/facebookresearch/demucs."""

    def __init__(self, model: DemucsModelType = DemucsModelType.DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: allow GPU usage (i.e. obey self.device)
        self.model: AnyModel = get_model(model.value)  # .cpu().eval()

    def run(
        self,
        track: BaseTrack,
        shifts: int = 1,
        split: bool = True,
        overlap: float = 0.25,
        jobs: int = 0,
    ) -> SourceSeparationDict:
        """"""

        # Convert our track representation to a demucs `AudioFile`
        # TODO: work out what is being "mean"ed here ...
        # TODO: we're normalizing here ?
        audio_file = _track_to_demucs_audio_file(track, self.model)
        ref = audio_file.mean(0)
        audio_file = (audio_file - ref.mean()) / ref.std()

        sources = apply_model(
            self.model,
            audio_file[None],
            device=self.device,
            shifts=shifts,
            split=split,
            overlap=overlap,
            progress=True,
            num_workers=jobs,
        )[0]
        sources = sources * ref.std() + ref.mean()

        return SourceSeparationDict(
            {
                name: LibrosaTrack(y=source.numpy(), sr=self.model.samplerate)
                for source, name in zip(sources, self.model.sources)
            }
        )
