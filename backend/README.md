# Backend (FastAPI)

## What it does

- `POST /predict`: upload a video, returns:
  - `risk`: e.g. `"CRITICAL RISK"`
  - `status`: e.g. `"INTERSECTION IMMINENT"`
  - `impact_in`: e.g. `"4.0s"`
  - `confidence`: e.g. `"99.9%"`

## Setup

From repo root:

```bash
python -m pip install -r backend/requirements.txt
```

## Run

You must point the server at a saved XGBoost model:

```bash
export XGB_MODEL_PATH="/home/asus/new/cosmos-predict2.5/xgb_nexar.json"
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Optional env vars:

- `PRED_SECONDS` (default `5.0`)
- `PRED_FPS_SAMPLE` (default `1.0`)
- `PRED_POOL` (default `mean_hw_keep_t`)
- `PRED_RESOLUTION` (default `360,640`)
- `PRED_DEVICE` (default `cuda`)
- `COSMOS_MODEL_SIZE` (default `2B`)
- `COSMOS_EXPERIMENT` (optional override)
- `COSMOS_CKPT` (optional override)
- `CORS_ALLOW_ORIGINS` (default `*`)

