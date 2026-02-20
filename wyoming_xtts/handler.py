import logging
import time
from pathlib import Path
from typing import Any

from wyoming.audio import AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .engine import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, XTTSEngine
from .segmenter import BufferedSegmenter
from .streaming import StreamingHandler
from .voice import resolve_language, resolve_voice

_LOGGER = logging.getLogger(__name__)


def _stabilize_short_text(text: str) -> str:
    """
    XTTS kan hallusinere på veldig korte strenger som 'hallo'.
    Dette gjør prompten litt mer "setningsaktig" uten å endre meningen.
    """
    t = " ".join(text.replace("\n", " ").strip().split())

    # Hvis det er veldig kort og slutter på alfanumerisk tegn: legg på punktum.
    # (Dette hjelper overraskende ofte på "hallo" / "ja" / "nei" osv.)
    if 0 < len(t) < 10 and t[-1].isalnum():
        t = t + "."

    return t


class XTTSEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        engine: XTTSEngine,
        voices_path: Path,
        language_fallback: str | None,
        no_detect_language: bool,
        min_segment_chars: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.engine = engine
        self.voices_path = voices_path
        self.language_fallback = language_fallback
        self.no_detect_language = no_detect_language
        self.min_segment_chars = min_segment_chars
        self._streaming = StreamingHandler(
            self, engine, voices_path, language_fallback, no_detect_language, min_segment_chars
        )

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if SynthesizeStart.is_type(event.type):
            try:
                await self._streaming.handle_start(SynthesizeStart.from_event(event))
            except Exception as err:
                await self._streaming.handle_error(err)
            return True

        if SynthesizeChunk.is_type(event.type):
            try:
                await self._streaming.handle_chunk(SynthesizeChunk.from_event(event))
            except Exception as err:
                await self._streaming.handle_error(err)
            return True

        if SynthesizeStop.is_type(event.type):
            try:
                await self._streaming.handle_stop()
            except Exception as err:
                await self._streaming.handle_error(err)
            return True

        if Synthesize.is_type(event.type):
            if self._streaming.has_active_session:
                return True
            try:
                await self._handle_synthesize(Synthesize.from_event(event))
            except Exception as err:
                await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
                await self.write_event(SynthesizeStopped().event())
            return True

        return False

    async def _handle_synthesize(self, synthesize: Synthesize) -> None:
        text = _stabilize_short_text(synthesize.text)
        if not text:
            await self.write_event(SynthesizeStopped().event())
            return

        voice_name = synthesize.voice.name if synthesize.voice else None
        voice_path = resolve_voice(self.voices_path, voice_name)

        # resolve_language tar hensyn til voice.language + fallback + detect_language (hvis aktiv)
        language = resolve_language(
            synthesize.voice, text, self.language_fallback, self.no_detect_language
        )

        await self.write_event(
            AudioStart(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event()
        )

        first_audio: float | None = None
        segmenter = BufferedSegmenter(min_chars=self.min_segment_chars)

        for segment in segmenter.add_chunk(text):
            result = await self.engine.stream_to_handler(self, segment, voice_path, language)
            if first_audio is None:
                first_audio = result

        remaining = segmenter.finish()
        if remaining:
            result = await self.engine.stream_to_handler(self, remaining, voice_path, language)
            if first_audio is None:
                first_audio = result

        await self.write_event(AudioStop().event())
