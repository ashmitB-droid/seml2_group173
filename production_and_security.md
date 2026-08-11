## Part A — Testing and experimentation in production

Offline metrics are computed on a held-out slice of historical data. They tell
us how the model would have performed on last year's customers, under last
year's campaign. They cannot tell us whether the marketing team will actually
convert more policies. The gap between "ROC-AUC 0.859" and "revenue went up"
is what production experimentation exists to close.

Three standard approaches, and why we would choose the one we do.

### Shadow deployment

The new model runs alongside the current one and scores every live request,
but its output is logged rather than returned. Nobody acts on it.

*What it validates:* that the new model can handle real traffic — real
latency, real payload shapes, real missing fields — and how far its scores
diverge from the incumbent's. It surfaces train/serve skew, which is exactly
the failure class our regression tests target.

*What it cannot validate:* whether the predictions are any good. No customer
is ever contacted on the shadow model's recommendation, so no outcome label
is ever generated for it.

### Canary release

A small share of traffic (say 5%) is routed to the new model; the rest stays
on the incumbent. The share increases if metrics hold.

*What it validates:* operational health under partial exposure, with a small
blast radius. If the canary errors or latency spikes, you roll back having
affected 5% of requests.

*What it cannot validate:* anything statistically subtle. 5% of traffic on a
campaign with a 12.3% positive rate takes a long time to produce a
conclusive conversion signal.

### A/B test

Customers are randomly assigned to model A or model B, both models' outputs
are acted on, and conversion is compared between arms.

*What it validates:* the business outcome, causally. Random assignment means
the difference in conversion rate is attributable to the model rather than
to the two groups differing.

*Cost:* it requires exposing real customers to the challenger, needs enough
sample size to detect a realistic effect, and takes as long as the sales
cycle.

### Our choice: shadow first, then canary, then A/B

These are complementary stages, not alternatives, and the order matters
because each stage rules out a class of problem more cheaply than the next.

1. **Shadow (1 week).** Deploy the challenger behind `/predict`, log both
   scores per request, and compare distributions. Gate: no errors, p99
   latency within the 500 ms NFR from Assignment I, and PSI between the two
   score distributions below 0.25. This catches the "model works on my
   laptop, breaks on live payloads" failure at zero customer cost — the
   category our `Age == 18` crash belonged to.

2. **Canary (1 week).** Route 5% of scoring traffic to the challenger, using
   the `/health` endpoint as the readiness gate so a revision that cannot
   load its artefacts never receives traffic. Roll back automatically on
   elevated 5xx rate.

3. **A/B test (one full campaign cycle).** Split the contact list randomly
   between arms and compare realised conversion. This is the only stage that
   answers the question the business actually asked: does this model make us
   more money than the last one?

A practical note specific to this problem. Our decision threshold is 0.40,
chosen to favour recall over precision because a missed interested customer
costs more than a wasted call. That trade-off is a business assumption, and
an A/B test is the right instrument for checking it. Running the same model
at 0.40 against 0.50 as two arms would measure the assumption directly —
and this is worth doing because the threshold, not the model, is where most
of the campaign economics live.

**Monitoring after rollout.** The PSI drift check in `src/data_quality.py`
runs on incoming batches against the training distribution. PSI above 0.25
on any feature raises an ERROR and triggers retraining review. This matters
more than it might seem: the model does not degrade because the code changed,
it degrades because the customer population changed, and nothing in the test
suite can detect that.

---

## Part B — Security consideration: input validation at the API boundary

**The vulnerability, as it existed.** `api/schemas.py` typed the categorical
fields as plain `str`. A request carrying `Gender: "Other"` passed pydantic
validation, reached `map_features()`, and hit `.map(GENDER_MAP)` — which
returns `NaN` for any key not in the mapping. XGBoost treats `NaN` as a
legitimate missing value and scored it happily. The caller received:

```json
{"prediction": 1, "probability": 0.7201, "label": "Interested"}
```

A confident-looking prediction, on input the model was never trained to
handle. Nothing was logged, because nothing had failed as far as the code
was concerned.

**Why this is a security issue and not just a bug.** It is a silent failure
in a system that drives spending decisions. An attacker — or, far more
likely, a buggy upstream integration — can submit values outside the
training distribution and receive authoritative-looking scores. Because the
`/batch-predict` endpoint accepted arbitrary uploaded CSVs, a single
malformed file could poison an entire campaign's targeting list, and the
only symptom would be a disappointing conversion rate a month later. Silent
wrong answers are worse than loud failures precisely because they are acted
upon.

**The controls implemented.**

| Control | Where | Effect |
|---|---|---|
| `Literal` types on all categorical fields | `api/schemas.py` | Rejects unknown categories at the edge with HTTP 422 |
| Numeric bounds (`ge`/`le`) on every numeric field | `api/schemas.py` | Rejects `Age=5`, `Annual_Premium=-100` |
| Schema re-validation inside `map_features()` | `src/preprocessing.py` | Defence in depth: the model is protected even when called from Streamlit or a batch job, not just through the API |
| 5 MB upload cap and 10,000-row batch limit | `api/routes.py` | Bounds memory use; an unbounded read is a denial-of-service vector |
| File extension check on upload | `api/routes.py` | Rejects non-CSV payloads before parsing |
| Typed exceptions mapped to status codes | `api/main.py` | Client errors return 422, server-side unavailability returns 503; internal detail is not leaked in the response body |

The layering is deliberate. Pydantic guards the HTTP boundary, but the
Streamlit app and any future batch job call `Predictor` directly and would
bypass it entirely. Validating again inside the preprocessing chain means
the model itself has a contract, independent of who is calling.

**What we would add next, given more time.** Authentication is the obvious
gap: `/predict` is currently unauthenticated, so anyone who can reach the
service can score records against it. For a model trained on customer data,
repeated querying is itself an information-disclosure risk — an attacker can
probe the decision boundary to infer what the training population looked
like. An API key per consumer, plus per-key rate limiting, would address
both the disclosure risk and the cost of unbounded usage.
