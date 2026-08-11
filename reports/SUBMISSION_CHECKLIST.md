# Assignment II — submission checklist (Group 173)

## Done in the repo

| # | Requirement | Where | Evidence |
|---|---|---|---|
| 1 | OOP / modular design | `src/` — `ModelTrainer`, `Predictor` classes; functional feature pipeline | Module tree + class docstrings |
| 2 | Research vs production | `Notebooks/model.ipynb` vs `src/preprocessing.py` | `reports/research_vs_production.md` |
| 3 | Logging & error handling | `src/logging_config.py`, `src/exceptions.py`, used in 5 modules | Run `python -m scripts.logging_demo` |
| 4 | Formatting & linting | `pyproject.toml`, `.flake8` | `reports/lint_before.txt` (107 issues) → `reports/lint_after.txt` (0) |
| 5 | REST API | `api/` — `/health`, `/predict`, `/batch-predict`, `/metrics` | Swagger at `/docs` |
| 6 | 2+ test types | `tests/` — unit, integration, data validation | `reports/pytest_output.txt` |
| 7a | Training tests | `tests/test_model_training.py` | overfit-small-batch, more-rounds-improves-AUC |
| 7b | Inference tests | `tests/test_inference.py` | shape/range, directional, invariance |
| 8a | Model quality metrics | `src/train.py::evaluate` | accuracy, precision, recall, F1, ROC-AUC |
| 8b | Data quality metrics | `src/preprocessing.py::validate_schema`, `src/data_quality.py` | schema gate, missing-value rate, PSI drift |
| 9 | Production testing + security | — | `reports/production_and_security.md` |

**Status:** 43 tests passing, flake8 clean, ROC-AUC 0.8593, recall 0.8827 at threshold 0.40.

## Commands to run (screenshot each)

```bash
pip install -r requirements.txt

python -m src.train                    # training pipeline with logging
python -m scripts.logging_demo         # INFO / WARNING / ERROR evidence
pytest -v                              # test suite
flake8 src api tests streamlit         # clean lint
uvicorn api.main:app --reload          # then open http://127.0.0.1:8000/docs
streamlit run streamlit/Home.py
```

## Still to do — only the group can supply these

- [ ] Group number and the contribution table (names, BITS IDs, qualitative + % split)
- [ ] Screenshots from your own machines: `pytest -v`, flake8 before/after, Swagger `/docs`, Streamlit app
- [ ] Rename the notebook to match the `<Group no>.ipynb` convention
- [ ] Redeploy the Render API so the live URL matches the fixed code
- [ ] Assemble `173.pdf` / `173.docx` from the three reports/ markdown files plus screenshots

## Four defects found and fixed — be ready to explain these

1. **`Age == 18` crashed prediction.** `pd.cut` excludes the left edge, so 18 became `NaN` and `OrdinalEncoder` raised. The API schema explicitly accepts `Age >= 18`. Fixed with `include_lowest=True`.
2. **Invalid categories scored silently.** `.map()` produced `NaN`, XGBoost read it as missing, `Gender="Other"` returned 0.72 "Interested". Fixed with `Literal` types plus schema validation.
3. **Reported metrics described an undeployed rule.** `train.py` evaluated at 0.40, `predict.py` used `model.predict()` (0.50). Recall claimed 0.8827, delivered 0.7874. Fixed with a shared `DECISION_THRESHOLD`.
4. **`src/featureEngineering.py` was dead code** and collided with `feature_engineering.py` on case-insensitive filesystems. Deleted.
