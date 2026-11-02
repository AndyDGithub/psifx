"""speechbrain 1.0 with huggingface_hub 1.x: a file missing from the Hub must read as a 404."""

import pytest

fetching = pytest.importorskip("speechbrain.utils.fetching")
from huggingface_hub.errors import EntryNotFoundError

from psifx.utils.huggingface import patch_speechbrain_for_hf_hub_1


@pytest.mark.unit
def test_missing_optional_file_becomes_value_error(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise EntryNotFoundError("404 Client Error. Entry Not Found for url: .../custom.py.")

    monkeypatch.setattr(fetching.huggingface_hub, "hf_hub_download", missing)
    monkeypatch.setattr(fetching, "HTTPError", fetching.HTTPError)  # restored after the test
    patch_speechbrain_for_hf_hub_1()
    patch_speechbrain_for_hf_hub_1()  # a second call must not nest the tuple

    assert isinstance(fetching.HTTPError, tuple) and len(fetching.HTTPError) == 2
    # speechbrain's from_hparams ignores this ValueError for the optional custom.py.
    with pytest.raises(ValueError, match="File not found on HF hub"):
        fetching.fetch(filename="custom.py", source="speechbrain/spkrec-ecapa-voxceleb", savedir=tmp_path)


@pytest.mark.unit
def test_token_keyword_reaches_from_hparams_as_use_auth_token(monkeypatch):
    from speechbrain.inference.interfaces import Pretrained

    seen = {}

    def fake_from_hparams(cls, source, use_auth_token=False, **kwargs):
        seen.update(use_auth_token=use_auth_token, kwargs=kwargs)
        return "model"

    monkeypatch.setattr(Pretrained, "from_hparams", classmethod(fake_from_hparams))
    patch_speechbrain_for_hf_hub_1()
    patch_speechbrain_for_hf_hub_1()  # idempotent

    assert Pretrained.from_hparams(source="x", token="hf_abc") == "model"
    assert seen == {"use_auth_token": "hf_abc", "kwargs": {}}
