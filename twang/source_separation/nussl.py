import nussl

from twang.source_separation.base import SourceSeparation, SourceSeparationDict
from twang.track import BaseTrack, LibrosaTrack


def _track_to_nussl_audio_signal(track: BaseTrack) -> nussl.AudioSignal:
    librosa_track = track.to_librosa_track()
    return nussl.AudioSignal(audio_data_array=librosa_track.y, sample_rate=librosa_track.sr)


def _audio_signal_to_librosa_track(audio_signal: nussl.AudioSignal) -> LibrosaTrack:
    return LibrosaTrack(audio_signal.audio_data, sr=audio_signal.sample_rate)


class Repet(SourceSeparation):
    """https://nussl.github.io/docs/separation.html#foreground-background-via-repet"""

    repet_cls = nussl.separation.primitive.Repet

    def run(self, track: BaseTrack) -> SourceSeparationDict:
        audio_signal = _track_to_nussl_audio_signal(track)
        repet = self.repet_cls(audio_signal)
        estimates = repet()
        return {
            "background": _audio_signal_to_librosa_track(estimates[0]),
            "foreground": _audio_signal_to_librosa_track(estimates[1]),
        }


class RepetSim(Repet):
    """https://nussl.github.io/docs/separation.html#foreground-background-via-repet-sim"""

    repet_cls = nussl.separation.primitive.RepetSim
