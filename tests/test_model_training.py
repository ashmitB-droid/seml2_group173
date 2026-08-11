"""
Model-training tests (Objective 2, Req. 7a).

Testing an ML training loop is not like testing a pure function: the output
is stochastic and there is no single correct answer to assert on. The
strategy here is to assert on *properties* a healthy training loop must
have, rather than on specific numbers that would make the test brittle.
"""

import pytest

from src.config import DECISION_THRESHOLD
from src.train import ModelTrainer


@pytest.fixture(scope="module")
def trained(engineered_features):
    """Fit once per module - training is the slow part of this suite."""
    X, y = engineered_features
    split = int(len(X) * 0.8)
    trainer = ModelTrainer(n_estimators=120)
    trainer.fit(X.iloc[:split], y.iloc[:split])
    return trainer, X.iloc[split:], y.iloc[split:]


def test_model_can_overfit_a_small_batch(engineered_features):
    """The canonical ML smoke test.

    An unregularised model given 50 rows should very nearly memorise them.
    Failure here means the fault is structural - features not reaching the
    model, labels misaligned with rows - not a tuning problem. It is the
    fastest way to distinguish "the model is bad" from "the pipeline is
    broken".
    """
    X, y = engineered_features
    accuracy = ModelTrainer().overfit_small_batch(X, y, n=50)
    assert accuracy > 0.95


def test_training_improves_over_more_boosting_rounds(engineered_features):
    """Loss must decrease as training proceeds.

    XGBoost has no epoch loop to inspect, so we compare a deliberately
    under-trained model against a fully trained one on the same split. If
    more boosting rounds do not help, the training signal is not reaching
    the model.
    """
    X, y = engineered_features
    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    weak = ModelTrainer(n_estimators=2, max_depth=1).fit(X_train, y_train)
    strong = ModelTrainer(n_estimators=200).fit(X_train, y_train)

    assert strong.evaluate(X_test, y_test)["roc_auc"] > weak.evaluate(X_test, y_test)["roc_auc"]


def test_metrics_are_in_valid_ranges(trained):
    trainer, X_test, y_test = trained
    metrics = trainer.evaluate(X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0


def test_model_beats_random_guessing(trained):
    """ROC-AUC of 0.5 is a coin flip. Anything near it means no signal."""
    trainer, X_test, y_test = trained
    assert trainer.evaluate(X_test, y_test)["roc_auc"] > 0.75


def test_evaluate_uses_the_shared_decision_threshold(trained):
    """Regression test.

    train.py previously reported metrics at 0.40 while predict.py used
    model.predict(), which hard-codes 0.50 - so published recall (0.8827)
    described a rule the API never applied (0.7874). Both now read the same
    constant.
    """
    trainer, X_test, y_test = trained
    assert trainer.evaluate(X_test, y_test)["threshold"] == DECISION_THRESHOLD


def test_lower_threshold_increases_recall(trained):
    """Sanity check on the threshold semantics: a lower bar catches more
    positives, at the cost of precision."""
    trainer, X_test, y_test = trained
    low = trainer.evaluate(X_test, y_test, threshold=0.30)
    high = trainer.evaluate(X_test, y_test, threshold=0.70)
    assert low["recall"] >= high["recall"]
    assert low["precision"] <= high["precision"]


def test_fit_rejects_empty_training_set(engineered_features):
    X, y = engineered_features
    with pytest.raises(ValueError, match="empty dataset"):
        ModelTrainer().fit(X.iloc[0:0], y.iloc[0:0])


def test_fit_rejects_single_class_target(engineered_features):
    """A target with one class produces a model that cannot discriminate."""
    X, y = engineered_features
    single_class = y.copy()
    single_class[:] = 0
    with pytest.raises(ValueError, match="both classes"):
        ModelTrainer().fit(X.head(100), single_class.head(100))
