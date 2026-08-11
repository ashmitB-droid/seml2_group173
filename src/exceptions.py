"""
exceptions.py

A small exception hierarchy for the cross-sell system.

Catching bare `Exception` loses information: the caller cannot tell a
malformed customer record (the client's fault, HTTP 422) from a missing
model file (the server's fault, HTTP 503). Typed exceptions let the API
layer map each failure mode to the correct status code in one place, and
let tests assert on the specific failure rather than "something raised".
"""


class CrossSellError(Exception):
    """Base class for every application-specific error."""


class DataValidationError(CrossSellError):
    """Input data failed a schema or quality check."""


class ModelNotFoundError(CrossSellError):
    """A model artefact was expected on disk but is missing or unreadable."""


class InferenceError(CrossSellError):
    """A structurally valid record could not be scored."""
