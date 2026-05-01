# Vera Message Engine (Deterministic)

This is a deterministic, rule-based implementation of the Vera message engine with the required /v1 endpoints.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python bot.py
```

The server listens on `http://localhost:8080`.

## Quick local check

- Set `BOT_URL` in `judge_simulator.py` to `http://localhost:8080`
- Run:

```bash
python judge_simulator.py
```

## Endpoints

- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`
- `POST /v1/tick`
- `POST /v1/reply`
