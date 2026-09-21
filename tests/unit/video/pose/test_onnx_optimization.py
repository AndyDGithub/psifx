"""Unit tests for the ONNX fp16 / INT8 helpers, on a tiny hand-built Conv+Relu model."""

from types import SimpleNamespace

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from psifx.io.video import VideoWriter  # noqa: E402
from psifx.video.pose.onnx_optimization import (  # noqa: E402
    check_session_device,
    convert_fp16,
    optimize_rtmlib_models,
    quantize_int8,
    record_inputs,
    sample_frames,
)

SHAPE = (1, 3, 8, 8)


def build_conv_model(tmp_path, opset):
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.normal(size=(4, 3, 3, 3)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), helper.make_node("Relu", ["c"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, (1, 4, 8, 8))],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    path = tmp_path / f"tiny_opset{opset}.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def conv_model(tmp_path):
    return build_conv_model(tmp_path, 17)


def run(path, x):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


@pytest.mark.unit
def test_fp16_converts_weights_but_keeps_float32_io(conv_model, tmp_path):
    out = convert_fp16(conv_model, tmp_path / "tiny.fp16.onnx")
    model = onnx.load(str(out))
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert all(init.data_type == TensorProto.FLOAT16 for init in model.graph.initializer)


@pytest.mark.unit
def test_int8_stays_close_to_fp32(conv_model, tmp_path):
    rng = np.random.default_rng(1)
    calibration = [rng.uniform(0, 1, SHAPE).astype(np.float32) for _ in range(8)]
    out = quantize_int8(conv_model, tmp_path / "tiny.int8.onnx", calibration)

    assert any(node.op_type == "QuantizeLinear" for node in onnx.load(str(out)).graph.node)
    x = calibration[0]
    reference, quantized = run(conv_model, x), run(out, x)
    # INT8 error should be a small fraction of the output range.
    assert np.abs(quantized - reference).max() < 0.05 * np.abs(reference).max()


@pytest.mark.unit
def test_int8_upgrades_old_opset(tmp_path):
    """rtmlib's YOLOX is opset 11: per-channel QDQ must still produce a loadable model."""
    old = build_conv_model(tmp_path, 11)
    calibration = [np.random.default_rng(i).uniform(0, 1, SHAPE).astype(np.float32) for i in range(4)]
    out = quantize_int8(old, tmp_path / "old.int8.onnx", calibration)
    assert run(out, calibration[0]).shape == (1, 4, 8, 8)


@pytest.mark.unit
def test_int8_needs_calibration(conv_model, tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        quantize_int8(conv_model, tmp_path / "x.onnx", [])


class FakeRtmlibModel:
    """Same surface as an rtmlib model: onnx_model, session, inference(img [H, W, 3])."""

    def __init__(self, path):
        self.onnx_model = str(path)
        self.session = ort.InferenceSession(self.onnx_model, providers=["CPUExecutionProvider"])

    def inference(self, img):
        return self.session.run(None, {"x": np.ascontiguousarray(img.transpose(2, 0, 1), np.float32)[None]})


@pytest.mark.unit
def test_record_inputs_captures_onnx_layout_and_restores(conv_model):
    model = FakeRtmlibModel(conv_model)
    original = model.inference
    img = np.random.default_rng(2).uniform(0, 1, (8, 8, 3)).astype(np.float32)
    with record_inputs({"m": model}) as recorded:
        model.inference(img)
    assert recorded["m"][0].shape == SHAPE
    np.testing.assert_allclose(recorded["m"][0][0], img.transpose(2, 0, 1))
    assert model.inference == original


@pytest.mark.unit
def test_optimize_int8_swaps_session(conv_model, tmp_path):
    model = FakeRtmlibModel(conv_model)
    frames = [np.random.default_rng(i).uniform(0, 1, (8, 8, 3)).astype(np.float32) for i in range(4)]
    # Like rtmlib, look inference up at call time so the recording wrapper is used.
    optimize_rtmlib_models({"m": model}, "int8", lambda f: model.inference(f), frames, "cpu", tmp_path)
    assert (tmp_path / "m.int8.onnx").exists()
    assert model.inference(frames[0])[0].shape == (1, 4, 8, 8)


@pytest.mark.unit
def test_optimize_rejects_fp16_on_cpu_and_unknown_precision(conv_model, tmp_path):
    model = FakeRtmlibModel(conv_model)
    with pytest.raises(ValueError, match="CUDA"):
        optimize_rtmlib_models({"m": model}, "fp16", model.inference, [], "cpu", tmp_path)
    with pytest.raises(ValueError, match="precision"):
        optimize_rtmlib_models({"m": model}, "int4", model.inference, [], "cpu", tmp_path)


@pytest.mark.unit
def test_check_session_device():
    cpu_session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    check_session_device(cpu_session, "cpu", "m")
    with pytest.raises(RuntimeError, match="LD_LIBRARY_PATH"):
        check_session_device(cpu_session, "cuda", "m")


@pytest.mark.unit
def test_sample_frames(tmp_path):
    path = tmp_path / "v.mp4"
    with VideoWriter(path, input_dict={"-r": "5"}, output_dict={"-pix_fmt": "yuv420p"}) as writer:
        for _ in range(10):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    assert len(sample_frames(path, 4)) == 4
    assert len(sample_frames(path, 50)) == 10
