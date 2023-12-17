import librosa

from twang.source_separation.base import SourceSeparation, SourceSeparationDict
from twang.track import BaseTrack, LibrosaTrack


class HarmonicPercussiveSourceSeparation(SourceSeparation):
    """Computes the short-time Fourier transform of y and decomposes it into its harmonic & percussive components."""

    def run(self, track: BaseTrack) -> SourceSeparationDict:
        D_harmonic, D_percussive = librosa.decompose.hpss(librosa.stft(track.to_librosa_track().y))
        return SourceSeparationDict(
            {
                "harmonic": LibrosaTrack(y=librosa.istft(D_harmonic), sr=track.sr),
                "percussive": LibrosaTrack(y=librosa.istft(D_percussive), sr=track.sr),
            }
        )
