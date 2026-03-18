"""Speaker identification via voice embeddings.

Uses resemblyzer to create a voice fingerprint during calibration,
then compares every utterance against it to classify as user or other.
Does NOT rely on Deepgram's diarization labels — uses actual voice similarity.
"""

import numpy as np
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # int16
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE


class AudioRingBuffer:
    """Time-indexed audio buffer that maps Deepgram timestamps back to audio.
    
    Stores the last N seconds of raw PCM audio so we can extract
    segments corresponding to Deepgram word timestamps.
    """

    def __init__(self, max_seconds: int = 30):
        self.max_bytes = max_seconds * BYTES_PER_SECOND
        self._buffer = bytearray()
        self._stream_start_time: float = 0
        self._bytes_written: int = 0
        self._started = False

    def start(self):
        """Mark the start of audio streaming."""
        self._stream_start_time = time.time()
        self._started = True
        self._bytes_written = 0
        self._buffer = bytearray()

    def add_audio(self, chunk: bytes):
        """Add an audio chunk to the buffer."""
        if not self._started:
            self.start()
        self._buffer.extend(chunk)
        self._bytes_written += len(chunk)
        # Trim to max size
        if len(self._buffer) > self.max_bytes:
            excess = len(self._buffer) - self.max_bytes
            self._buffer = self._buffer[excess:]


    def extract_segment(self, start_sec: float, end_sec: float) -> Optional[np.ndarray]:
        """Extract audio segment by Deepgram stream-relative timestamps.
        
        Args:
            start_sec: Start time in seconds from stream start (from Deepgram word.start)
            end_sec: End time in seconds from stream start (from Deepgram word.end)
        
        Returns:
            Float32 numpy array of audio samples, or None if not available.
        """
        if not self._started or len(self._buffer) == 0:
            return None

        start_byte = int(start_sec * BYTES_PER_SECOND)
        end_byte = int(end_sec * BYTES_PER_SECOND)

        # Where does our buffer start in the stream?
        buffer_start_byte = self._bytes_written - len(self._buffer)

        # Convert to buffer-relative positions
        rel_start = start_byte - buffer_start_byte
        rel_end = end_byte - buffer_start_byte

        if rel_start < 0:
            rel_start = 0
        if rel_end > len(self._buffer):
            rel_end = len(self._buffer)
        if rel_end <= rel_start or (rel_end - rel_start) < BYTES_PER_SECOND // 4:
            return None  # Too short (< 250ms)

        raw = bytes(self._buffer[rel_start:rel_end])
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio


class SpeakerManager:
    """Voice-embedding speaker identification.
    
    Flow:
    1. Calibration: user speaks for ~5s, we compute their voice embedding
    2. Runtime: for each Deepgram transcript, extract audio segment,
       compute embedding, compare against user embedding
    3. Classify as 'user' (similarity > threshold) or 'other'
    """

    def __init__(self, similarity_threshold: float = 0.72):
        self._encoder = None
        self._encoder_loaded = False
        self.user_embedding: Optional[np.ndarray] = None
        self.calibrated: bool = False
        self.similarity_threshold = similarity_threshold
        self.audio_buffer = AudioRingBuffer(max_seconds=30)
        self._calibration_audio_chunks: list = []
        self._is_calibrating: bool = False
        # Cache: map (start_sec, end_sec) → label to avoid re-computing
        self._classification_cache: dict = {}
        # Keep speaker_map for backward compat (Deepgram speaker_id → label)
        self.speaker_map: dict = {}

    def _get_encoder(self):
        """Lazy-load the voice encoder (takes a few seconds first time)."""
        if self._encoder is None and not self._encoder_loaded:
            self._encoder_loaded = True
            try:
                from resemblyzer import VoiceEncoder
                self._encoder = VoiceEncoder()
                logger.info("VoiceEncoder loaded successfully")
            except ImportError:
                logger.warning("resemblyzer not installed — using heuristic fallback")
            except Exception as e:
                logger.error(f"VoiceEncoder load failed: {e}")
        return self._encoder


    def start_calibration(self):
        """Begin calibration — user should speak."""
        self._is_calibrating = True
        self._calibration_audio_chunks = []
        # Pre-load encoder during calibration so it's ready
        self._get_encoder()
        logger.info("Speaker calibration started")

    def add_calibration_audio(self, audio_bytes: bytes):
        """Buffer audio during calibration phase."""
        if self._is_calibrating:
            self._calibration_audio_chunks.append(audio_bytes)

    def add_audio(self, audio_bytes: bytes):
        """Add audio to the ring buffer (call on every audio frame)."""
        self.audio_buffer.add_audio(audio_bytes)

    def finish_calibration(self) -> bool:
        """Process calibration audio and create user voice embedding."""
        self._is_calibrating = False
        encoder = self._get_encoder()

        if not self._calibration_audio_chunks:
            logger.warning("No calibration audio captured")
            self.calibrated = True  # Fall back to heuristic
            return True

        if encoder is None:
            logger.info("No encoder — using first-speaker heuristic")
            self.calibrated = True
            return True

        try:
            raw = b"".join(self._calibration_audio_chunks)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio) < SAMPLE_RATE:  # < 1 second
                logger.warning("Calibration audio too short (< 1s)")
                self.calibrated = True
                return True

            self.user_embedding = encoder.embed_utterance(audio)
            self.calibrated = True
            similarity_self = float(np.dot(self.user_embedding, self.user_embedding))
            logger.info(f"Calibration complete — embedding shape: {self.user_embedding.shape}, "
                       f"self-similarity: {similarity_self:.3f}, "
                       f"threshold: {self.similarity_threshold}")
            return True

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            self.calibrated = True
            return True

    def classify_utterance(self, words: list) -> str:
        """Classify an utterance as 'user' or 'other' using voice embeddings.
        
        Args:
            words: List of Deepgram word objects with 'start' and 'end' timestamps.
        
        Returns:
            'user' if the voice matches the calibrated user, 'other' otherwise.
        """
        encoder = self._get_encoder()

        # If no encoder or no calibration, fall back to Deepgram diarization
        if encoder is None or self.user_embedding is None:
            return self._heuristic_classify(words)

        # Get time range for this utterance
        if not words:
            return "unknown"

        start_sec = words[0].get("start", 0)
        end_sec = words[-1].get("end", 0)

        if end_sec <= start_sec:
            return "unknown"

        # Check cache
        cache_key = (round(start_sec, 2), round(end_sec, 2))
        if cache_key in self._classification_cache:
            return self._classification_cache[cache_key]

        # Extract audio segment
        audio_segment = self.audio_buffer.extract_segment(start_sec, end_sec)
        if audio_segment is None or len(audio_segment) < SAMPLE_RATE // 4:
            # Too short to classify reliably — use Deepgram's label
            return self._heuristic_classify(words)

        try:
            segment_embedding = encoder.embed_utterance(audio_segment)
            similarity = float(np.dot(self.user_embedding, segment_embedding))
            label = "user" if similarity > self.similarity_threshold else "other"
            self._classification_cache[cache_key] = label
            logger.info(f"Voice ID: {label} (similarity={similarity:.3f}, "
                       f"threshold={self.similarity_threshold}, "
                       f"segment={end_sec-start_sec:.1f}s)")
            return label
        except Exception as e:
            logger.warning(f"Embedding comparison failed: {e}")
            return self._heuristic_classify(words)

    def _heuristic_classify(self, words: list) -> str:
        """Fallback: use Deepgram's speaker labels with first-speaker heuristic."""
        if not words:
            return "unknown"
        speaker_ids = [w.get("speaker", 0) for w in words if "speaker" in w]
        if not speaker_ids:
            return "unknown"
        speaker_id = max(set(speaker_ids), key=speaker_ids.count)
        return self.identify_speaker(speaker_id)

    def identify_speaker(self, speaker_id: int, audio_segment=None) -> str:
        """Map Deepgram speaker ID → user/other (backward compat, heuristic only)."""
        if speaker_id in self.speaker_map:
            return self.speaker_map[speaker_id]
        if len(self.speaker_map) == 0:
            self.speaker_map[speaker_id] = "user"
            logger.info(f"Speaker {speaker_id} → user (first-speaker heuristic)")
            return "user"
        else:
            self.speaker_map[speaker_id] = "other"
            logger.info(f"Speaker {speaker_id} → other (heuristic)")
            return "other"

    def get_label(self, speaker_id: int) -> str:
        return self.speaker_map.get(speaker_id, f"speaker_{speaker_id}")
