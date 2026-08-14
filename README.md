# Assignment II — Implementation, Code Quality and QA

Group 173 · AIMLCZG546 Software Engineering for Machine Learning

This document covers the Assignment II work. The problem statement, GR4ML views
and architecture from Assignment I are in the main `README.md`.

---

## Group details

Group No: 173

| Sl. No | BITS ID | Name | Contribution (Qualitative) | % |
|---|---|---|---|---|
| 1 | 2025AA05957 | Abhishek | Modular OOP / functional design | 100 |
| 2 | 2025AA05729 | Ashmit Bhandari | Two or more test types/Formatting and linting | 100 |
| 3 | 2025AA05478 | Rishabh Jain | Error handling and logging | 100 |
| 4 | 2025AB05319 | Udit Sharma | REST API / Research vs production code | 100 |

---

## Quick start

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # runtime + QA tooling

# 2. Train (writes models/ and metrics.json — required before tests)
python -m src.train

# 3. Verify
pytest -v                             # 44 tests
flake8 src api tests streamlit scripts
python -m scripts.logging_demo        # INFO / WARNING / ERROR evidence

# 4. Run
uvicorn api.main:app --reload         # http://127.0.0.1:8000/docs
streamlit run streamlit/Home.py
```

`python -m src.train` must run before `pytest` — several tests load the
persisted model and will fail with `ModelNotFoundError` if the artefacts are
missing.

**macOS:** XGBoost needs the OpenMP runtime. If you see
`libxgboost.dylib could not be loaded`, run `brew install libomp`.

---

## Repository layout

```
seml2_group173/
├── api/                     # FastAPI service
│   ├── main.py              # app + exception→status-code handlers
│   ├── routes.py            # endpoints
│   └── schemas.py           # pydantic request/response contracts
├── src/                     # ML pipeline
│   ├── config.py            # paths, schema contract, decision threshold
│   ├── logging_config.py    # central logging setup
│   ├── exceptions.py        # typed exception hierarchy
│   ├── preprocessing.py     # loading, validation, encoding
│   ├── feature_engineering.py
│   ├── train.py             # ModelTrainer class + pipeline
│   ├── predict.py           # Predictor class
│   └── data_quality.py      # missing values, PSI drift
├── tests/                   # 44 pytest tests
├── scripts/logging_demo.py  # demonstrates all three log levels
├── streamlit/               # dashboard (calls the deployed API)
├── models/                  # persisted artefacts
├── reports/                 # lint reports, before/after code, write-ups
├── Notebooks/model.ipynb    # research artefact (deliberately unrefactored)
├── pyproject.toml           # black + isort + pytest config
└── .flake8
```

---

## Objective 1 — Implementation and code sharing

### 1. Modular design

Each stage of the pipeline is a separate module with one responsibility.
`ModelTrainer` and `Predictor` are classes because both hold state — fitted
estimators, feature order, the decision threshold — that is expensive to
rebuild per call. The feature engineering stage stays functional: it is a
stateless transformation, and a class there would add ceremony without
adding anything.

### 2. Research code vs production code

`Notebooks/model.ipynb` is the exploratory artefact and is deliberately left
as it is. `src/` is the production path. A side-by-side comparison of the same
function in both styles is in `reports/research_vs_production.md`, generated
from the actual before/after of `map_features()` (originals preserved in
`reports/before_refactor/`).

### 3. Logging and error handling

`src/logging_config.py` configures the root logger once: INFO to console,
DEBUG to a rotating file at `logs/application.log`. Modules obtain loggers via
`get_logger(__name__)` so every record identifies its source.

Failures raise typed exceptions from `src/exceptions.py` — `DataValidationError`,
`ModelNotFoundError`, `InferenceError` — which the API maps to status codes in
one place. Logging and error handling are implemented across five modules:
`preprocessing`, `train`, `predict`, `data_quality`, and the API layer.

Run `python -m scripts.logging_demo` to see all three levels in one pass.

### 4. Formatting and linting

| | flake8 issues |
|---|---|
| Before | 107 |
| After | 0 |

Full reports in `reports/lint_before.txt` and `reports/lint_after.txt`.
Configuration is committed (`.flake8`, `pyproject.toml`) so every group member
gets identical results.

### 5. REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service banner |
| GET | `/health` | Liveness probe; reports `model_loaded` |
| GET | `/metrics` | Metrics recorded at training time |
| POST | `/predict` | Score one customer record |
| POST | `/batch-predict` | Score an uploaded CSV |

Every route declares a `response_model`, so the OpenAPI schema is generated
from the code. Status codes: 200 success, 422 invalid input, 503 model
unavailable.

---

## Objective 2 — Quality assurance

### 6 & 7. Tests

```
tests/test_data_validation.py   10   schema, missing values, PSI drift
tests/test_model_training.py     8   overfit-small-batch, threshold semantics
tests/test_inference.py         11   shape/range, directional, invariance
tests/test_api.py                8   end-to-end integration
                                --
                                44
```

Three test types are represented: unit (individual functions), integration
(full stack via `TestClient`), and data validation (schema and drift).

The ML-specific tests assert on *properties* rather than fixed numbers, so
they survive a retrain:

- **Overfitting a small batch** — an unregularised model given 50 rows should
  nearly memorise them. Failure indicates a structural fault (features not
  reaching the model, labels misaligned) rather than a tuning problem.
- **Loss decreases** — a 200-estimator model must beat a 2-estimator one on
  the same split.
- **Directional** — prior vehicle damage must not lower predicted interest;
  already being insured must not raise it.
- **Invariance** — changing `id` must not change the prediction, which would
  indicate an identifier leaking into the feature matrix.

### 8. Metrics

**Model quality** (`src/train.py::evaluate`), at threshold 0.40:

| Metric | Value |
|---|---|
| ROC-AUC | 0.8593 |
| Recall | 0.8827 |
| Precision | 0.3009 |
| F1 | 0.4488 |
| Accuracy | 0.7343 |

**Data quality**:

- **Schema validation** (`preprocessing.validate_schema`) — required columns
  present, category values recognised. Hard pass/fail.
- **Missing-value rate** (`data_quality.missing_value_report`) — per-column,
  against a 5% tolerance.
- **PSI drift** (`data_quality.population_stability_index`) — quantile-binned
  population stability index against the training distribution.
  < 0.10 stable, 0.10–0.25 moderate, > 0.25 significant and a retraining
  trigger.

### 9. Production testing and security

See `reports/production_and_security.md` — shadow deployment, canary release
and A/B testing compared with a staged recommendation, plus input validation
as the primary security control.

---

## Defects found and fixed

Four defects were found in the Assignment I codebase during this work. Each
has a regression test.

**1. `Age = 18` crashed prediction.** `pd.cut` excludes the left bin edge, so
an age of 18 fell outside every bin, became `NaN`, and `OrdinalEncoder.transform`
raised `ValueError`. The API schema explicitly accepts `Age >= 18`, so this
was reachable from a valid request. Fixed with `include_lowest=True`.
Test: `test_minimum_age_is_binned_not_dropped`.

**2. Invalid categories were scored silently.** `map_features` used `.map()`,
which returns `NaN` for unrecognised keys. XGBoost treats `NaN` as a missing
value, so `Gender="Other"` returned a confident 0.72 "Interested" rather than
an error. Fixed with `Literal` types in the API schema plus schema validation
inside the preprocessing chain.
Test: `test_unrecognised_category_is_rejected`.

**3. Reported metrics described an undeployed decision rule.** `train.py`
evaluated at a 0.40 threshold while `predict.py` called `model.predict()`,
which hard-codes 0.50. Published recall was 0.8827; the API actually delivered
0.7874. Both paths now read `DECISION_THRESHOLD` from `config.py`.
Test: `test_evaluate_uses_the_shared_decision_threshold`.

**4. `src/featureEngineering.py` was dead code** — a divergent copy of
`feature_engineering.py` that nothing imported, and which collided with it on
case-insensitive filesystems. Deleted.

A fifth issue was introduced and caught during refactoring: removing
`train_samples`/`test_samples` from `metrics.json` broke the Streamlit metrics
dashboard with a `KeyError`. `metrics.json` is a published contract between
two deployed components, so it now has a regression test asserting every key
the dashboard reads.
Test: `test_metrics_payload_satisfies_dashboard_contract`.

---

## Deployment

| Component | Platform | URL |
|---|---|---|
| FastAPI | Render | https://seml2-group173.onrender.com/docs#/ |
| Streamlit | Streamlit Cloud | https://health-insurance-cross-sell-prediction-2.streamlit.app/ |

The Streamlit app calls the deployed API rather than importing `src/` directly,
so the API must be redeployed before the dashboard reflects code changes.
Render's free tier spins down after inactivity — the first request after a
quiet period may take 30–60 seconds.

`requirements.txt` holds runtime dependencies only; QA tooling is in
`requirements-dev.txt` so deployments stay lean.

---