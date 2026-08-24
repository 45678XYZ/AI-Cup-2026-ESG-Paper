"""Frozen corpus/model matrix for the post-English multilingual replication."""

LANGUAGES = ("mlpromise_fr", "mlpromise_ja", "mlpromise_ko")
SEEDS = (42, 123, 456)
LAMBDAS = (0.0, 0.3)

# Every checkpoint explicitly lists French, Japanese and Korean (or documents
# 100-language coverage that includes them). Using one checkpoint set across
# corpora prevents a language effect from being a model-selection effect.
MODELS = (
    {
        "name": "FacebookAI/xlm-roberta-large",
        "revision": "c23d21b0620b635a76227c604d44e43a9f0ee389",
        "protocols": ("pdf_group", "row_strat"),
        "worker": "large",
        "amp_dtype": "float16",
    },
    {
        "name": "google/rembert",
        "revision": "65da5133da36e29dfca67d4f0dd9f7f9db21b563",
        "protocols": ("pdf_group",),
        "worker": "large",
        "amp_dtype": "bfloat16",
    },
    {
        "name": "FacebookAI/xlm-roberta-base",
        "revision": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        "protocols": ("pdf_group",),
        "worker": "base",
        "amp_dtype": "float16",
    },
    {
        "name": "google-bert/bert-base-multilingual-cased",
        "revision": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
        "protocols": ("pdf_group",),
        "worker": "base",
        "amp_dtype": "float16",
    },
)


def models_for(worker: str) -> tuple[dict, ...]:
    if worker not in {"large", "base"}:
        raise ValueError(f"unknown worker {worker!r}")
    return tuple(model for model in MODELS if model["worker"] == worker)


def expected_bundle_count(corpus: str | None = None) -> int:
    """150 fits per language: 60 main-model plus 30 for each other model."""
    per_language = sum(
        len(model["protocols"]) * len(LAMBDAS) * len(SEEDS) * 5
        for model in MODELS
    )
    return per_language if corpus else per_language * len(LANGUAGES)
