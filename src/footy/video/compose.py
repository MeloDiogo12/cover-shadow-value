"""Composição de vídeos verticais (9:16) a partir de animações matplotlib e áudio.

Fluxo: o áudio (voiceover) manda no tempo. Cada cena recebe a duração da frase
que a acompanha, e o vídeo final junta cenas + voz + legendas.
"""

from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
from matplotlib import animation

# Usa o ffmpeg que vem com o imageio-ffmpeg (não depende de instalação no sistema)
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def save_animation(anim: animation.FuncAnimation, path: Path, fps: int = 30) -> Path:
    """Grava uma animação matplotlib em MP4 (H.264, yuv420p para tocar em todos os telemóveis)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(path, writer=writer)
    return path


def compose(scenes: list[Path], out: Path, audio: Path | None = None,
            fps: int = 30) -> Path:
    """Junta cenas MP4 em sequência e, se houver, acrescenta o voiceover."""
    from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips

    clips = [VideoFileClip(str(p)) for p in scenes]
    video = concatenate_videoclips(clips, method="compose")
    if audio is not None:
        voice = AudioFileClip(str(audio))
        video = video.with_audio(voice).with_duration(max(video.duration, voice.duration))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(str(out), fps=fps, codec="libx264", audio_codec="aac",
                          ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None)
    for c in clips:
        c.close()
    return out
