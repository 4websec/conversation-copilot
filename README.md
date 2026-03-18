# Conversation Copilot

Real-time AI conversation advisor. Listens to live conversations, identifies speakers, and provides domain-specific talking points via Claude.

## Architecture

```
Browser Mic → WebSocket → FastAPI → Deepgram STT (streaming + diarization)
                                   → Speaker ID (calibration-based)
                                   → Claude API (talking points)
                                   → WebSocket → React UI
```

## Quick Start (Windows)

### 1. Prerequisites

- Python 3.11+
- Deepgram API key (console.deepgram.com)
- Anthropic API key (console.anthropic.com)

### 2. Install

```powershell
cd conversation-copilot

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r backend\requirements.txt
```

### 3. Configure

```powershell
# Copy example env file
copy .env.example .env

# Edit .env and add your API keys
notepad .env
```

### 4. Run

```powershell
cd backend
python main.py
```

### 5. Open

Navigate to `http://127.0.0.1:8765` in Chrome.

## Usage

1. **Setup**: Select domain (Sales / Legal / Security / General) and optional meeting context
2. **Calibrate**: Speak for 5 seconds so the system learns your voice
3. **Go Live**: Start your call — transcript appears on the left, talking points on the right
4. **Hotkeys**:
   - `Ctrl+Space` — Force generate talking points
   - `Ctrl+S` — Generate post-call summary

## Domain Modes

| Mode | Focus |
|------|-------|
| **Sales** | Objection handling, value articulation, buying signals, next steps |
| **Legal** | Key facts, risk flags, documentation needs, procedural considerations |
| **Security** | Control gaps, compliance alignment, evidence requests, risk severity |
| **General** | Key themes, questions to ask, commitments, follow-ups |

## Project Structure

```
conversation-copilot/
├── backend/
│   ├── main.py            # FastAPI server, WebSocket handlers, Deepgram integration
│   ├── config.py          # Environment config management
│   ├── speaker_id.py      # Speaker calibration and identification
│   ├── llm_engine.py      # Claude API, prompt management, trigger logic
│   └── requirements.txt   # Python dependencies
├── frontend/
│   └── index.html         # Single-file React UI (teleprompter overlay)
├── .env.example           # Configuration template
└── README.md
```

## Audio Pipeline

- **Capture**: Browser WebAudio API → 16kHz mono PCM (Int16)
- **Transport**: WebSocket binary frames to FastAPI
- **STT**: Deepgram Nova-2 streaming with diarization
- **Speaker ID**: First-speaker heuristic (v1), voice embedding calibration (optional via resemblyzer)
- **LLM Trigger**: Fires when other party completes substantive utterance (≥8 words + 800ms silence)

## Notes

- **One-party consent**: Texas allows recording without other party's consent. If calls cross state lines, verify consent requirements.
- **Latency budget**: Target <3s from utterance → talking point. Deepgram ~300ms, Claude streaming ~1-2s.
- Audio never stored to disk in v1. Transcripts held in memory only during session.

## Phase 2 Roadmap

- [ ] Voice embedding enrollment (resemblyzer) for robust speaker ID
- [ ] Electron always-on-top overlay
- [ ] Conversation state tracking (topics covered, commitments)
- [ ] Sentiment/tone analysis on other party
- [ ] Auto post-call summary email
- [ ] Multi-language support
- [ ] Local Whisper fallback for air-gapped environments
- [ ] Consent management workflow
