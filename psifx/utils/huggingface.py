"""Compatibility helpers for Hugging Face integrations."""

from __future__ import annotations

import importlib
import inspect
from functools import wraps


def patch_hf_hub_download_use_auth_token() -> None:
    """
    Patch ``hf_hub_download`` so callers using ``use_auth_token=...`` still work.

    Some dependencies still call ``hf_hub_download(..., use_auth_token=...)`` while
    newer ``huggingface_hub`` versions expect ``token=...``.
    """
    try:
        import huggingface_hub
    except Exception:
        return

    hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
    if hf_hub_download is None:
        return

    try:
        signature = inspect.signature(hf_hub_download)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "use_auth_token" in signature.parameters:
        return

    @wraps(hf_hub_download)
    def compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = compat_hf_hub_download

    # pyannote imports hf_hub_download directly in these modules.
    for module_name in (
        "pyannote.audio.core.pipeline",
        "pyannote.audio.core.model",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "hf_hub_download"):
            setattr(module, "hf_hub_download", compat_hf_hub_download)


def patch_speechbrain_for_hf_hub_1() -> None:
    """
    Let speechbrain 1.0 load models with ``huggingface_hub`` 1.x and pyannote.audio 4.

    Two mismatches stop the loading of ``speechbrain/spkrec-ecapa-voxceleb``:
    - speechbrain tells a file missing from the Hub apart by catching ``requests.HTTPError``; since
      ``huggingface_hub`` 1.0 (httpx), it is an ``EntryNotFoundError``, so the optional ``custom.py``
      it looks for in every model repository becomes fatal. Only speechbrain's ``except`` is widened;
    - pyannote.audio 4 passes ``token=`` to ``from_hparams``, which only knows ``use_auth_token=``
      and hands the unknown keyword to the model constructor.
    """
    try:
        from huggingface_hub.errors import EntryNotFoundError
        from speechbrain.inference.interfaces import Pretrained
        from speechbrain.utils import fetching
    except Exception:
        return

    caught = getattr(fetching, "HTTPError", None)
    if caught is not None and not isinstance(caught, tuple):
        # ``except HTTPError as e`` in speechbrain.utils.fetching.fetch accepts a tuple of classes.
        fetching.HTTPError = (caught, EntryNotFoundError)

    from_hparams = Pretrained.from_hparams.__func__
    parameters = inspect.signature(from_hparams).parameters
    if getattr(from_hparams, "_psifx_token_compat", False) or "token" in parameters \
            or "use_auth_token" not in parameters:
        return

    @wraps(from_hparams)
    def compat_from_hparams(cls, *args, token=None, **kwargs):
        if token is not None and "use_auth_token" not in kwargs:
            kwargs["use_auth_token"] = token
        return from_hparams(cls, *args, **kwargs)

    compat_from_hparams._psifx_token_compat = True
    Pretrained.from_hparams = classmethod(compat_from_hparams)


def pyannote_token_kwargs(token: str | None) -> dict:
    """
    Keyword argument carrying the Hugging Face token for pyannote loaders.

    pyannote.audio 4 renamed ``use_auth_token`` to ``token``.
    """
    from pyannote.audio import Pipeline

    parameters = inspect.signature(Pipeline.from_pretrained).parameters
    return {"token": token} if "token" in parameters else {"use_auth_token": token}
