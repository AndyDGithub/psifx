"""Unit tests for the ONNX fp16 helpers, on a tiny hand-built Conv+Relu model."""

from types import SimpleNamespace

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from psifx.video.pose.onnx_optimization import (  # noqa: E402
    check_session_device,
    convert_fp16,
    optimize_rtmlib_models,
    preload_cuda_libraries,
)

SHAPE = (1, 3, 8, 8)


@pytest.fixture
def conv_model(tmp_path):
    weight = numpy_helper.from_array(np.random.default_rng(0).normal(size=(4, 3, 3, 3)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), helper.make_node("Relu", ["c"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, (1, 4, 8, 8))],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


class FakeRtmlibModel:
    """Same surface as an rtmlib model: onnx_model and session."""

    def __init__(self, path):
        self.onnx_model = str(path)
        self.session = ort.InferenceSession(self.onnx_model, providers=["CPUExecutionProvider"])


@pytest.mark.unit
def test_fp16_converts_weights_but_keeps_float32_io(conv_model, tmp_path):
    out = convert_fp16(conv_model, tmp_path / "tiny.fp16.onnx")
    model = onnx.load(str(out))
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert all(init.data_type == TensorProto.FLOAT16 for init in model.graph.initializer)


@pytest.mark.unit
def test_fp32_leaves_session_untouched(conv_model, tmp_path):
    model = FakeRtmlibModel(conv_model)
    session = model.session
    optimize_rtmlib_models({"m": model}, "fp32", "cpu", tmp_path)
    assert model.session is session
    assert not list(tmp_path.glob("*.fp16.onnx"))


@pytest.mark.unit
def test_rejects_fp16_on_cpu_and_unknown_precision(conv_model, tmp_path):
    model = FakeRtmlibModel(conv_model)
    with pytest.raises(ValueError, match="CUDA"):
        optimize_rtmlib_models({"m": model}, "fp16", "cpu", tmp_path)
    # INT8 was evaluated and dropped (slower and less accurate): it must not be accepted.
    with pytest.raises(ValueError, match="precision"):
        optimize_rtmlib_models({"m": model}, "int8", "cpu", tmp_path)


@pytest.mark.unit
def test_check_session_device():
    cpu_session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    check_session_device(cpu_session, "cpu", "m")
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        check_session_device(cpu_session, "cuda", "m")


@pytest.mark.unit
def test_preload_skips_missing_nvidia_packages(monkeypatch):
    import sys

    # None in sys.modules makes the import fail, as on a machine without CUDA PyTorch wheels.
    monkeypatch.setitem(sys.modules, "nvidia.cublas", None)
    monkeypatch.setitem(sys.modules, "nvidia.cudnn", None)
    assert preload_cuda_libraries() == []
