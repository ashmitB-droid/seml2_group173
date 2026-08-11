# Objective 1, Requirement 2 — Research code vs production code

The same component, `map_features()`, before and after refactoring. This is a
real diff from this repository, not an illustrative example: the "before" is
the version submitted for Assignment I.

## Before — research code (`src/preprocessing.py`, Assignment I)

```python
def map_features(df: pd.DataFrame):
    gender_map = {
        "Male": 1,
        "Female": 0
    }
    vehicle_age_map = {
        "< 1 Year": 0,
        "1-2 Year": 1,
        "> 2 Years": 2
    }
    damage_map = {
        "No": 0,
        "Yes": 1
    }
    df["Gender"] = df["Gender"].map(gender_map)
    df["Vehicle_Age"] = df["Vehicle_Age"].map(vehicle_age_map)
    df["Vehicle_Damage"] = df["Vehicle_Damage"].map(damage_map)
    return df
```

## After — production code

```python
def map_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map the three categorical columns to integer codes.

    Previously this used a bare .map(), which turns any unrecognised value
    into NaN. XGBoost treats NaN as a legitimate missing value, so a record
    with Gender="Other" was scored and returned a confident probability
    instead of being rejected. We now validate first and fail loudly.
    """
    df = df.copy()
    validate_schema(df)

    df["Gender"] = df["Gender"].map(GENDER_MAP)
    df["Vehicle_Age"] = df["Vehicle_Age"].map(VEHICLE_AGE_MAP)
    df["Vehicle_Damage"] = df["Vehicle_Damage"].map(DAMAGE_MAP)
    return df
```

## What changed, and why each change matters

| Aspect | Research version | Production version | Consequence |
|---|---|---|---|
| Unknown categories | `.map()` returns `NaN` silently | `validate_schema()` raises `DataValidationError` | `Gender="Other"` previously returned a confident 0.72 "Interested" |
| Mutation | Edits the caller's dataframe in place | Operates on `.copy()` | Callers no longer see their input silently modified |
| Failure signal | None — the pipeline continues | Typed exception the API maps to HTTP 422 | Failures are visible instead of silent |
| Observability | No output at all | Logs the offending column and allowed values at ERROR | An operator can diagnose from the log alone |
| Type hints | Argument only | Argument and return | Tooling can check callers |
| Testability | Only observable via the final prediction | Failure mode directly assertable | See `tests/test_data_validation.py` |

## The general principle

Research code optimises for iteration speed and is read by its author, who
holds the context in their head. Production code optimises for being operated
by someone else, months later, when it breaks at 2am.

The concrete difference is what happens on the unhappy path. The research
version has no unhappy path: every input produces an output, including inputs
that are meaningless. That is a reasonable trade in a notebook where you can
see the dataframe. It is not reasonable in a service, where the caller has no
way to distinguish a good prediction from a prediction on garbage.

The notebook in `Notebooks/model.ipynb` remains the exploratory artefact and
is deliberately not refactored — it documents how the model was arrived at.
The production path in `src/` is what gets deployed.
