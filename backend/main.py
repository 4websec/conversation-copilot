"""Conversation Copilot â€” Main Server

FastAPI WebSocket server that:
1. Receives audio from browser mic
2. Streams to Deepgram for real-time STT + diarization
3. Identifies speakers (user vs other)
4. Triggers Claude for talking points when other party finishes speaking
5. Pushes talking points to the UI via WebSocket
"""

import asyncio
import json
import logging
import time
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import Config
from speaker_id import SpeakerManager
from llm_engine import LLMEngine, TranscriptEntry

# â”€â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("copilot")

# Quiet noisy libs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.INFO)


# â”€â”€â”€ Session State â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Session:
    """Per-session state for a single conversation."""

    def __init__(self, session_id: str, domain: str = "general"):
        self.session_id = session_id
        self.speaker_mgr = SpeakerManager()
        self.llm_engine = LLMEngine(domain=domain)
        self.ui_ws: WebSocket | None = None
        self.audio_ws: WebSocket | None = None
        self.deepgram_ws = None
        self.is_active = False
        self.silence_timer: asyncio.Task | None = None
        self.created_at = time.time()
        self._audio_frames_sent = 0
        self._dg_messages_received = 0

    async def send_to_ui(self, msg_type: str, data: dict):
        """Send a message to the UI overlay."""
        if self.ui_ws:
            try:
                await self.ui_ws.send_json({"type": msg_type, **data})
            except Exception as e:
                logger.warning(f"UI send failed: {e}")
        else:
            logger.debug(f"UI WS not connected, dropping: {msg_type}")


sessions: dict[str, Session] = {}


# â”€â”€â”€ Websockets v13+ compatibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _ws_is_open(ws) -> bool:
    """Check if a websockets connection is still open.
    Works with websockets v10-v16+.
    """
    if ws is None:
        return False
    # v13+: close_code is None while connection is open
    try:
        return ws.close_code is None
    except AttributeError:
        pass
    # v10-v12 fallback
    try:
        return ws.open
    except AttributeError:
        return False


# â”€â”€â”€ App Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@asynccontextmanager
async def lifespan(app: FastAPI):
    errors = Config.validate()
    if errors:
        for e in errors:
            logger.error(f"CONFIG ERROR: {e}")
        logger.error("Fix .env file and restart!")
    else:
        logger.info("âœ“ Config validated â€” API keys present")
    logger.info(f"Conversation Copilot starting on http://{Config.HOST}:{Config.PORT}")
    yield
    for sid, session in sessions.items():
        if session.deepgram_ws:
            try:
                await session.deepgram_ws.close()
            except Exception:
                pass
    logger.info("Server shut down")


app = FastAPI(title="Conversation Copilot", lifespan=lifespan)

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def root():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(sessions),
        "config_valid": len(Config.validate()) == 0,
    }


@app.get("/debug")
async def debug():
    session_info = []
    for sid, s in sessions.items():
        session_info.append({
            "id": sid,
            "is_active": s.is_active,
            "audio_frames_sent": s._audio_frames_sent,
            "dg_messages_received": s._dg_messages_received,
            "has_ui_ws": s.ui_ws is not None,
            "has_audio_ws": s.audio_ws is not None,
            "has_deepgram_ws": s.deepgram_ws is not None,
            "dg_ws_open": _ws_is_open(s.deepgram_ws),
            "speaker_map": s.speaker_mgr.speaker_map,
            "calibrated": s.speaker_mgr.calibrated,
            "transcript_buffer": len(s.llm_engine.buffer.entries),
            "age_seconds": round(time.time() - s.created_at),
        })
    return {"sessions": session_info}


# â”€â”€â”€ Deepgram Connection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"

DEEPGRAM_PARAMS = {
    "model": "nova-2",
    "language": "en-US",
    "smart_format": "true",
    "diarize": "true",
    "interim_results": "true",
    "utterance_end_ms": "1500",
    "vad_events": "true",
    "encoding": "linear16",
    "sample_rate": "16000",
    "channels": "1",
}


async def connect_deepgram(session: Session):
    """Establish WebSocket connection to Deepgram."""
    import websockets

    params = "&".join(f"{k}={v}" for k, v in DEEPGRAM_PARAMS.items())
    url = f"{DEEPGRAM_WS_URL}?{params}"
    headers = {"Authorization": f"Token {Config.DEEPGRAM_API_KEY}"}

    logger.info(f"[{session.session_id}] Connecting to Deepgram...")
    logger.debug(f"[{session.session_id}] Key length: {len(Config.DEEPGRAM_API_KEY)}, "
                 f"starts with: {Config.DEEPGRAM_API_KEY[:8]}...")

    try:
        session.deepgram_ws = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        )
        logger.info(f"[{session.session_id}] âœ“ Connected to Deepgram (close_code={session.deepgram_ws.close_code})")
        return True
    except Exception as e:
        logger.error(f"[{session.session_id}] âœ— Deepgram connection failed: {type(e).__name__}: {e}")
        await session.send_to_ui("error", {"message": f"Deepgram connection failed: {e}"})
        return False


async def handle_deepgram_messages(session: Session):
    """Process incoming Deepgram messages."""
    try:
        async for message in session.deepgram_ws:
            session._dg_messages_received += 1
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "Results":
                    await process_transcript_result(session, data)
                elif msg_type == "UtteranceEnd":
                    logger.debug(f"[{session.session_id}] UtteranceEnd")
                    await maybe_trigger_llm(session)
                elif msg_type == "SpeechStarted":
                    logger.debug(f"[{session.session_id}] VAD: speech started")
                elif msg_type == "Metadata":
                    req_id = data.get("request_id", "n/a")
                    logger.info(f"[{session.session_id}] Deepgram ready (request_id: {req_id})")
                    await session.send_to_ui("status", {"message": "Deepgram connected and ready"})
                elif msg_type == "Error":
                    err_msg = data.get("description", data.get("message", str(data)))
                    logger.error(f"[{session.session_id}] Deepgram error: {err_msg}")
                    await session.send_to_ui("error", {"message": f"Deepgram: {err_msg}"})
                else:
                    logger.debug(f"[{session.session_id}] DG msg: {msg_type}")
            except Exception as e:
                logger.error(f"[{session.session_id}] Error processing DG message: {type(e).__name__}: {e}")

    except Exception as e:
        logger.error(f"[{session.session_id}] Deepgram listener error: {type(e).__name__}: {e}")
    finally:
        logger.info(f"[{session.session_id}] Deepgram listener stopped "
                    f"({session._dg_messages_received} msgs received)")


async def process_transcript_result(session: Session, data: dict):
    """Process a Deepgram transcript result with diarization."""
    channel = data.get("channel", {})
    alternatives = channel.get("alternatives", [])
    if not alternatives:
        return

    alt = alternatives[0]
    transcript = alt.get("transcript", "").strip()
    if not transcript:
        return

    is_final = data.get("is_final", False)
    speech_final = data.get("speech_final", False)

    # Classify speaker using voice embeddings (falls back to Deepgram diarization)
    words = alt.get("words", [])
    if words and session.speaker_mgr.calibrated:
        speaker_label = session.speaker_mgr.classify_utterance(words)
    else:
        # Pre-calibration or no words: use Deepgram diarization
        speaker_id = None
        if words:
            speaker_ids = [w.get("speaker", 0) for w in words if "speaker" in w]
            if speaker_ids:
                speaker_id = max(set(speaker_ids), key=speaker_ids.count)
        if speaker_id is not None:
            speaker_label = session.speaker_mgr.identify_speaker(speaker_id)
        else:
            speaker_label = "unknown"

    tag = "FINAL" if is_final else "interim"
    logger.info(f"[{session.session_id}] [{speaker_label}] ({tag}): {transcript[:100]}")

    # Send to UI
    await session.send_to_ui("transcript", {
        "speaker": speaker_label,
        "text": transcript,
        "is_final": is_final,
        "speech_final": speech_final,
    })

    # Buffer final results for LLM
    if is_final and transcript:
        entry = TranscriptEntry(
            speaker=speaker_label,
            text=transcript,
            timestamp=time.time(),
            is_final=True,
        )
        session.llm_engine.add_transcript(entry)

    # Trigger LLM only when someone OTHER than the user speaks
    if speech_final and speaker_label == "other":
        if session.silence_timer:
            session.silence_timer.cancel()
        session.silence_timer = asyncio.create_task(
            delayed_llm_trigger(session, delay=0.5)
        )


async def delayed_llm_trigger(session: Session, delay: float = 0.5):
    await asyncio.sleep(delay)
    await maybe_trigger_llm(session)


async def maybe_trigger_llm(session: Session, force: bool = False):
    if not force and not session.llm_engine.should_trigger():
        return

    logger.info(f"[{session.session_id}] LLM trigger fired")
    await session.send_to_ui("thinking", {"message": "Finding answer..."})

    full_response = ""
    async for chunk in session.llm_engine.generate_talking_points(force=force):
        full_response += chunk
        await session.send_to_ui("talking_point_chunk", {"text": chunk})

    if full_response:
        await session.send_to_ui("talking_points", {"text": full_response})
    logger.info(f"[{session.session_id}] Talking points delivered ({len(full_response)} chars)")


# â”€â”€â”€ WebSocket: Audio Input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.websocket("/ws/audio/{session_id}")
async def audio_websocket(ws: WebSocket, session_id: str):
    """Receives raw audio from browser and forwards to Deepgram."""
    await ws.accept()
    logger.info(f"[{session_id}] Audio WebSocket connected")

    if session_id not in sessions:
        sessions[session_id] = Session(session_id)
    session = sessions[session_id]
    session.audio_ws = ws

    # Connect to Deepgram
    if not await connect_deepgram(session):
        await ws.close(code=1011, reason="Deepgram connection failed")
        return

    session.is_active = True
    dg_task = asyncio.create_task(handle_deepgram_messages(session))

    try:
        while True:
            audio_data = await ws.receive_bytes()

            # Always feed the ring buffer for voice-embedding classification
            session.speaker_mgr.add_audio(audio_data)

            # Buffer during calibration for voice enrollment
            if session.speaker_mgr._is_calibrating:
                session.speaker_mgr.add_calibration_audio(audio_data)

            # Forward to Deepgram (no .open check â€” use try/except)
            if session.deepgram_ws and _ws_is_open(session.deepgram_ws):
                try:
                    await session.deepgram_ws.send(audio_data)
                    session._audio_frames_sent += 1
                    if session._audio_frames_sent == 1:
                        logger.info(f"[{session_id}] âœ“ First audio frame sent to Deepgram "
                                   f"({len(audio_data)} bytes)")
                    elif session._audio_frames_sent % 200 == 0:
                        logger.debug(f"[{session_id}] Audio: {session._audio_frames_sent} frames sent, "
                                    f"{session._dg_messages_received} DG msgs received")
                except Exception as e:
                    logger.error(f"[{session_id}] Deepgram send error: {type(e).__name__}: {e}")
                    # Attempt reconnect
                    dg_task.cancel()
                    if await connect_deepgram(session):
                        dg_task = asyncio.create_task(handle_deepgram_messages(session))
                    else:
                        break
            else:
                logger.warning(f"[{session_id}] Deepgram WS closed, reconnecting...")
                dg_task.cancel()
                if await connect_deepgram(session):
                    dg_task = asyncio.create_task(handle_deepgram_messages(session))
                else:
                    logger.error(f"[{session_id}] Reconnect failed")
                    break

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Audio WebSocket disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] Audio loop error: {type(e).__name__}: {e}")
    finally:
        session.is_active = False
        logger.info(f"[{session_id}] Audio pipeline stopped ({session._audio_frames_sent} frames sent)")
        if session.deepgram_ws:
            try:
                await session.deepgram_ws.send(json.dumps({"type": "CloseStream"}))
                await session.deepgram_ws.close()
            except Exception:
                pass
        dg_task.cancel()


# â”€â”€â”€ WebSocket: UI Control â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.websocket("/ws/ui/{session_id}")
async def ui_websocket(ws: WebSocket, session_id: str):
    """UI control channel."""
    await ws.accept()
    logger.info(f"[{session_id}] UI WebSocket connected")

    if session_id not in sessions:
        sessions[session_id] = Session(session_id)
    session = sessions[session_id]
    session.ui_ws = ws

    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("command", "")
            logger.debug(f"[{session_id}] UI cmd: {cmd}")

            try:
                if cmd == "set_domain":
                    domain = msg.get("domain", "general")
                    session.llm_engine.set_domain(domain)
                    await session.send_to_ui("status", {"message": f"Domain set to: {domain}"})

                elif cmd == "set_context":
                    context = msg.get("context", "")
                    session.llm_engine.set_context(context)
                    await session.send_to_ui("status", {"message": "Meeting context updated"})

                elif cmd == "start_calibration":
                    session.speaker_mgr.start_calibration()
                    await session.send_to_ui("calibration", {"state": "recording"})

                elif cmd == "stop_calibration":
                    success = session.speaker_mgr.finish_calibration()
                    state = "complete" if success else "failed"
                    logger.info(f"[{session_id}] Calibration: {state}")
                    await session.send_to_ui("calibration", {"state": state})

                elif cmd == "force_trigger":
                    logger.info(f"[{session_id}] Manual trigger (force)")
                    await maybe_trigger_llm(session, force=True)

                elif cmd == "generate_summary":
                    logger.info(f"[{session_id}] Summary requested")
                    summary = await session.llm_engine.generate_summary()
                    await session.send_to_ui("summary", {"text": summary})

                elif cmd == "ping":
                    await session.send_to_ui("pong", {})

            except Exception as e:
                logger.error(f"[{session_id}] Error handling command '{cmd}': {type(e).__name__}: {e}")

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] UI WebSocket disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] UI error: {type(e).__name__}: {e}")
    finally:
        session.ui_ws = None


# â”€â”€â”€ Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,
        log_level="info",
    )
