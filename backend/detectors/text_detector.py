import pickle
import sys
from pathlib import Path

_TEXT_MODULE = Path(__file__).resolve().parents[2] / "modules" / "text_detection"
_CKPT_DIR = _TEXT_MODULE / "artifacts" / "cspf_transformer" / "checkpoints" / "cspf_feature_transformer"
# Prefer the full transformer pipeline; fall back to the lighter post-fit checkpoint
_MODEL_PATH = (
    _CKPT_DIR / "pipeline.pkl"
    if (_CKPT_DIR / "pipeline.pkl").exists()
    else _CKPT_DIR / "pipeline_after_fit.pkl"
)

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Text model not found at {_MODEL_PATH}. "
            "Train and save it first:\n"
            "  from cspf_text.pipeline import CSPFTextPipeline\n"
            "  pipeline = CSPFTextPipeline()\n"
            "  pipeline.fit(texts, labels)\n"
            "  import pickle; pickle.dump(pipeline, open(path, 'wb'))"
        )
    if str(_TEXT_MODULE) not in sys.path:
        sys.path.insert(0, str(_TEXT_MODULE))
    import torch
    # The pipeline was saved on a CUDA machine; remap tensors to CPU on load
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": "cpu"})
    try:
        with open(_MODEL_PATH, "rb") as f:
            _pipeline = pickle.load(f)
    finally:
        torch.load = _orig_load

    # The pipeline was saved on a CUDA machine — remap every component to CPU
    def _force_cpu(obj):
        if obj is None:
            return
        if hasattr(obj, "device"):
            obj.device = "cpu"
        if hasattr(obj, "_resolve_device"):
            obj._resolve_device = lambda _torch: "cpu"
        if hasattr(obj, "feature_cache_dir"):
            obj.feature_cache_dir = None
        if hasattr(obj, "_model") and obj._model is not None:
            obj._model.to("cpu")
        # Recurse into known sub-components
        for attr in ("probabilistic_extractor", "style_extractor", "cohesion_extractor", "model"):
            _force_cpu(getattr(obj, attr, None))

    _force_cpu(_pipeline)

    return _pipeline


def predict(text: str) -> dict:
    pipeline = _load_pipeline()
    doc = pipeline.predict_document(text)
    label = "AI" if doc.document_probability >= 0.5 else "HUMAN"
    return {
        "label": label,
        "ai_probability": round(doc.document_probability, 4),
        "ai_sentence_ratio": round(doc.ai_contribution_ratio, 4),
        "sentences": doc.sentences,
        "sentence_probabilities": [round(p, 4) for p in doc.sentence_probabilities],
    }
