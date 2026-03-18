"""LLM engine for real-time question answering.

Detects questions in live conversation audio and generates
immediate, accurate answers via Claude API streaming.
"""

import asyncio
import anthropic
import logging
import time
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field
from config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a real-time knowledge assistant listening to a live conversation.
When you receive a question or statement from the conversation, provide an immediate,
accurate, and concise answer.

Rules:
- Answer the question DIRECTLY. No preamble like "Great question" or "That's interesting."
- If it's a factual question (math, science, history, etc.), give the precise answer first, then a 1-sentence explanation.
- If it's math: show the calculation and the answer prominently.
- For definitions: give a crisp 1-2 sentence definition.
- For complex topics: give a short structured answer with key points.
- Keep answers concise but complete. Aim for 2-6 sentences unless the topic demands more.
- Use the conversation context to give better answers when relevant.
- If a statement isn't a question but contains an interesting claim, briefly fact-check or add useful context.
- Format for easy scanning: bold key terms, use short lines."""



@dataclass
class TranscriptEntry:
    """Single transcript entry with speaker and timing."""
    speaker: str  # 'user' or 'other'
    text: str
    timestamp: float
    is_final: bool = True


@dataclass
class ConversationBuffer:
    """Rolling buffer of transcript entries."""
    entries: list = field(default_factory=list)
    max_age_seconds: int = 180  # 3 minutes

    def add(self, entry: TranscriptEntry):
        self.entries.append(entry)
        self._prune()

    def _prune(self):
        cutoff = time.time() - self.max_age_seconds
        self.entries = [e for e in self.entries if e.timestamp > cutoff]

    def get_formatted(self) -> str:
        if not self.entries:
            return "[No conversation yet]"
        lines = []
        for e in self.entries:
            label = "SPEAKER-A" if e.speaker == "user" else "SPEAKER-B"
            lines.append(f"[{label}]: {e.text}")
        return "\n".join(lines)

    def get_last_utterance(self) -> Optional[str]:
        """Get the most recent utterance from ANY speaker."""
        for e in reversed(self.entries):
            if e.is_final:
                return e.text
        return None

    def get_last_other_utterance(self) -> Optional[str]:
        """Get the most recent 'other' speaker utterance (kept for compat)."""
        for e in reversed(self.entries):
            if e.speaker == "other" and e.is_final:
                return e.text
        return None

    def word_count_since_last_trigger(self, last_trigger_time: float) -> int:
        """Count words from OTHER speakers only since last trigger."""
        count = 0
        for e in self.entries:
            if e.timestamp > last_trigger_time and e.speaker == "other":
                count += len(e.text.split())
        return count


class LLMEngine:
    """Real-time Q&A engine powered by Claude."""

    def __init__(self, domain: str = "general"):
        self.client = anthropic.AsyncAnthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.domain = domain
        self.system_prompt = SYSTEM_PROMPT
        self.buffer = ConversationBuffer(
            max_age_seconds=Config.TRANSCRIPT_BUFFER_MINUTES * 60
        )
        self.last_trigger_time: float = 0
        self.last_trigger_text: str = ""
        self._generating = False
        self._generation_lock = asyncio.Lock()
        self.custom_context: str = ""

    def set_domain(self, domain: str):
        self.domain = domain
        logger.info(f"Domain switched to: {domain}")

    def set_context(self, context: str):
        self.custom_context = context
        logger.info(f"Context set: {context[:80]}")

    def add_transcript(self, entry: TranscriptEntry):
        self.buffer.add(entry)

    def should_trigger(self) -> bool:
        """Trigger only when a non-user voice says something.
        
        Ignores the calibrated user's voice entirely.
        Low word threshold (3 words) to catch short questions.
        """
        if self._generating:
            return False

        last = self.buffer.get_last_other_utterance()
        if not last:
            return False

        # Don't re-trigger on same utterance
        if last == self.last_trigger_text:
            return False

        # Low threshold — "what is 3+5" is only 4 words
        word_count = self.buffer.word_count_since_last_trigger(self.last_trigger_time)
        if word_count < 3:
            return False

        return True

    async def generate_talking_points(self, force: bool = False) -> AsyncGenerator[str, None]:
        """Stream an answer to whatever was just said/asked.
        
        Method name kept for backward compat with main.py.
        """
        async with self._generation_lock:
            if self._generating:
                return
            self._generating = True

        try:
            conversation = self.buffer.get_formatted()
            last = (self.buffer.get_last_utterance() if force else self.buffer.get_last_other_utterance()) or ""

            self.last_trigger_time = time.time()
            self.last_trigger_text = last

            context_block = ""
            if self.custom_context:
                context_block = f"\nContext: {self.custom_context}\n"

            user_message = f"""{context_block}
=== RECENT CONVERSATION ===
{conversation}

=== LATEST UTTERANCE ===
{last}

Answer or respond to what was just said. If it's a question, answer it directly.
If it's a statement, add useful context or fact-check if relevant."""

            logger.info(f"Q&A trigger: '{last[:80]}'")

            async with self.client.messages.stream(
                model=Config.LLM_MODEL,
                max_tokens=Config.LLM_MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except anthropic.APIError as e:
            logger.error(f"API error: {e}")
            yield f"[API Error: {e.message}]"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"[Error: {e}]"
        finally:
            self._generating = False

    async def generate_summary(self) -> str:
        """Generate a post-call summary."""
        conversation = self.buffer.get_formatted()
        if not conversation or conversation == "[No conversation yet]":
            return "No conversation to summarize."

        try:
            response = await self.client.messages.create(
                model=Config.LLM_MODEL,
                max_tokens=1024,
                system="Summarize this conversation. List: topics discussed, "
                       "questions asked and answers given, key facts mentioned, "
                       "and any action items. Use bullet points.",
                messages=[{
                    "role": "user",
                    "content": f"Summarize:\n\n{conversation}"
                }],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return f"Error: {e}"
