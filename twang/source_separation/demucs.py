import os
import tempfile

import torch as th
from demucs.apply import apply_model
from demucs.pretrained import DEFAULT_MODEL, get_model
from demucs.repo import AnyModel
from demucs.separate import load_track

from twang.source_separation.base import SourceSeparation, SourceSeparationDict
from twang.track.base import BaseTrack, LibrosaTrack


def _track_to_demucs_audio_file(track: BaseTrack, model: AnyModel):
    track_path = track.file_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        if track_path is None:
            track_path = os.path.join(tmp_dir, "track.wav")
            track.save(track_path)
        return load_track(track_path, model.audio_channels, model.samplerate)


class Demucs(SourceSeparation):
    """https://github.com/facebookresearch/demucs."""

    def run(
        self,
        track: BaseTrack,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
        shifts: int = 1,
        split: bool = True,
        overlap: float = 0.25,
        jobs: int = 0,
    ) -> SourceSeparationDict:

        model = get_model(model_name).cpu().eval()

        audio_file = _track_to_demucs_audio_file(track, model)

        ref = audio_file.mean(0)
        audio_file = (audio_file - ref.mean()) / ref.std()
        sources = apply_model(
            model,
            audio_file[None],
            device=device,
            shifts=shifts,
            split=split,
            overlap=overlap,
            progress=True,
            num_workers=jobs,
        )[0]
        sources = sources * ref.std() + ref.mean()

        return {
            name: LibrosaTrack(y=source.numpy(), sr=model.samplerate) for source, name in zip(sources, model.sources)
        }
