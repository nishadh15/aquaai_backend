# AquaAI FastAPI Backend (Render)

This API accepts calibrated ESP32 sensor readings, provides a synthetic-data Random Forest water recommendation, applies a rain-aware postponement rule, and exposes the latest readings for the frontend dashboard.

## Endpoints

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/api/health` | Render health check |
| `GET` | `/api/dashboard` | Latest reading and recent history |
| `GET` | `/api/recent-readings` | Recent readings only |
| `POST` | `/api/sensor-data` | Save ESP32/dashboard reading and return recommendation |
| `POST` | `/api/recommendation` | Get recommendation without saving history |
| `GET` | `/api/model-info` | Synthetic-model disclosure |
| `GET` | `/api/synthetic-data?limit=10` | Small sample of synthetic training data |
| `GET` | `/docs` | Interactive FastAPI documentation |

## Run locally

Use Python 3.11+.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visit `http://127.0.0.1:8000/docs`. The dashboard endpoint is `http://127.0.0.1:8000/api/dashboard`.

## Example ESP32-compatible payload

```json
{
  "moisture": 38.5,
  "temperature": 30.2,
  "humidity": 57.8,
  "rain": false,
  "crop": "Tomato",
  "growth_stage": "flowering",
  "previous_water": 0.3
}
```

POST it to `/api/sensor-data` with header `Content-Type: application/json`. `moisture` must be the **calibrated 0–100% value**, not the raw ESP32 ADC number.

## Deploy to Render

1. Put the contents of this `backend` folder in a GitHub repository, then in Render choose **New → Web Service** and connect it.
2. Use runtime **Python**, build command `pip install -r requirements.txt`, and start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Set environment variable `PYTHON_VERSION` to `3.11.9`.
4. After deploying, copy the service URL. It should return `{ "status": "ok", ... }` at `/api/health`.
5. Set `ALLOWED_ORIGINS` to your Vercel URL after that deployment, for example `https://aquaai-dashboard.vercel.app`. For a local and deployed frontend, comma-separate values: `http://127.0.0.1:5500,https://aquaai-dashboard.vercel.app`.

`render.yaml` includes the same service settings. Its example CORS variable is deliberately local-only; override it in the Render dashboard before using the live frontend.

## Recommendation rules

The Random Forest learns from a generated, reproducible sample with `moisture`, `temperature`, `humidity`, `rain`, `crop`, `growth_stage`, `previous_water`, and `required_water`. Output is capped to 0–2.5 L/plant/day. Two readable rules take priority:

- Rain reported: recommendation is 0 and pump status is `POSTPONED`.
- Moisture at or above 62%: recommendation is 0 and pump status is `OFF`.

The data and model are **synthetic prototype material**, not field validation. Calibrate both soil moisture and pump flow, then compare against a conventional schedule using measured water volumes before making water-saving claims.

## Hardware safety

The backend provides advice only; it does not directly control a relay. Keep a maximum run-time and final threshold safety check in ESP32 firmware. Never connect a pump to ESP32 GPIO, and do not use 230 V mains for this prototype. Use an isolated relay/MOSFET driver and a separate low-voltage DC pump supply.

## Storage note

Recent readings are stored only in memory so the prototype stays simple. Render restarts clear the history. Add a managed database only if durable data is part of the next project phase.
