"""Version the meaning of sequence features independently of their dimensions.

Unversioned artifacts predate the nine-residue PTS2 correction. Feature extraction
outside inference uses the current schema; inference explicitly scopes extraction
to the artifact, including nested TargetP feature helpers.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature


LEGACY_FEATURE_SCHEMA = "localize-pts2-8-v1"
CURRENT_FEATURE_SCHEMA = "localize-pts2-9-v2"
FEATURE_SCHEMAS = (LEGACY_FEATURE_SCHEMA, CURRENT_FEATURE_SCHEMA)
_SCHEMA: ContextVar[str | None] = ContextVar("localize_feature_schema", default=None)
_INPUT_SCHEMA: ContextVar[str | None] = ContextVar(
    "localize_input_feature_schema", default=None
)


def validate_feature_schema(value):
    if value not in FEATURE_SCHEMAS:
        raise ValueError("Unsupported localization feature schema: {}".format(value))
    return value


def extraction_schema(value=None):
    if value is not None:
        return validate_feature_schema(value)
    return validate_feature_schema(_SCHEMA.get() or CURRENT_FEATURE_SCHEMA)


def feature_schema_changed():
    """Whether an enclosing predictor supplied features under a different schema."""
    return _INPUT_SCHEMA.get() is not None and _INPUT_SCHEMA.get() != _SCHEMA.get()


def model_feature_schema(model):
    head = model.get("localization_model", {})
    if (
        "feature_schema" in model
        and head.get("feature_schema", model["feature_schema"])
        != model["feature_schema"]
    ):
        raise ValueError("Localization head and model feature schemas differ.")
    return validate_feature_schema(
        model.get("feature_schema", head.get("feature_schema", LEGACY_FEATURE_SCHEMA))
    )


def scoped_localization_head(model):
    """Preserve a nested wrapper's schema when passing only its head to a backend."""
    original = model.get("localization_model", {})
    schema = validate_feature_schema(
        model.get("feature_schema", original.get("feature_schema", extraction_schema()))
    )
    if original.get("feature_schema", schema) != schema:
        raise ValueError("Localization head and model feature schemas differ.")
    if original.get("feature_schema") == schema:
        return original
    head = dict(original, feature_schema=schema)
    # Runtime models belong to the persistent head, not a temporary schema view.
    head["_runtime_model_cache"] = original.setdefault("_runtime_model_cache", {})
    return head


@contextmanager
def feature_schema_scope(value):
    token = _SCHEMA.set(validate_feature_schema(value))
    try:
        yield
    finally:
        _SCHEMA.reset(token)


def with_model_feature_schema(argument):
    """Scope nested feature extraction without changing legacy API signatures."""

    def decorate(function):
        spec = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            model = spec.bind(*args, **kwargs).arguments[argument]
            # A versioned subhead may intentionally differ from its parent, e.g.
            # a newly attached perox head on a published targeting5 checkpoint.
            schema = (
                model_feature_schema(model)
                if "localization_model" in model
                else model.get("feature_schema")
            )
            if schema is None:
                schema = _SCHEMA.get() or model_feature_schema(model)
            token = _INPUT_SCHEMA.set(_SCHEMA.get())
            try:
                with feature_schema_scope(schema):
                    return function(*args, **kwargs)
            finally:
                _INPUT_SCHEMA.reset(token)

        return wrapped

    return decorate


def version_model(model, default=LEGACY_FEATURE_SCHEMA):
    """Copy schema-bearing containers; never relabel or mutate supplied weights."""
    result = dict(model)
    head = dict(result["localization_model"])
    if head.get("decision_policy", "legacy") not in ("legacy", "safe-v1"):
        raise ValueError("Unsupported localization decision policy.")
    schema = validate_feature_schema(
        result.get("feature_schema", head.get("feature_schema", default))
    )
    if head.get("feature_schema", schema) != schema:
        raise ValueError("Localization head and model feature schemas differ.")
    result["feature_schema"] = head["feature_schema"] = schema
    if "specialist_head" in head:
        specialist = dict(head["specialist_head"])
        specialist_schema = validate_feature_schema(
            specialist.get("feature_schema", schema)
        )
        specialist["feature_schema"] = specialist_schema
        head["specialist_head"] = specialist
    result["localization_model"] = head
    if "base_models" in head:
        bases = []
        for base in head["base_models"]:
            child = dict(base)
            child.setdefault("perox_model", {})
            child = version_model(child, default=schema)
            if "perox_model" not in base:
                child.pop("perox_model")
            bases.append(child)
        head["base_models"] = bases
    perox = dict(result["perox_model"])
    perox["feature_schema"] = validate_feature_schema(
        perox.get("feature_schema", schema)
    )
    result["perox_model"] = perox
    return result
