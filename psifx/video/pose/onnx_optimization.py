"""Inference optimizations for ONNX pose models (fp16 conversion, INT8 static quantization).

Used by the rtmlib-based tools (RTMPose, RTMO). Following the ONNX Runtime
documentation: static QDQ S8S8 quantization is the recommended INT8 scheme for
CNNs, and float16 conversion keeps float32 inputs/outputs so the callers'
pre/post-processing is unchanged.
"""

from typing import Callable, Dict, List, Sequence, Union

import contextlib
import tempfile
from pathlib import Path

import numpy as np

from psifx.io import video

PRECISIONS = ("fp32", "fp16", "int8")


def convert_fp16(model_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    """
    Converts an ONNX model to float16, keeping float32 inputs and outputs.

    :param model_path: Path to the float32 ONNX model.
    :param output_path: Path to the float16 ONNX model.
    :return: The output path.
    """
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = convert_float_to_float16(onnx.load(str(model_path)), keep_io_types=True)
    onnx.save(model, str(output_path))
    return Path(output_path)


def quantize_int8(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    calibration_inputs: Sequence[np.ndarray],
) -> Path:
    """
    Quantizes an ONNX model to INT8 (static, QDQ, signed weights and activations, per-channel weights).

    :param model_path: Path to the float32 ONNX model.
    :param output_path: Path to the INT8 ONNX model.
    :param calibration_inputs: Representative model inputs, each with the model's input shape.
    :return: The output path.
    """
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if len(calibration_inputs) == 0:
        raise ValueError("INT8 quantization needs at least one calibration input")
    model = onnx.load(str(model_path))
    input_name = model.graph.input[0].name

    # Per-channel QDQ needs the ``axis`` attribute of (De)QuantizeLinear, added in opset 13.
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    per_channel = True
    if opset < 13:
        try:
            from onnx import version_converter

            model = version_converter.convert_version(model, 13)
        except Exception as error:  # the converter does not support every op/opset pair
            print(f"Could not upgrade {model_path} from opset {opset} to 13 ({error}): per-tensor INT8 instead.")
            per_channel = False

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.inputs = iter(calibration_inputs)

        def get_next(self):
            value = next(self.inputs, None)
            return None if value is None else {input_name: value}

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Shape inference + graph cleanup first, as recommended before quantize_static.
        preprocessed = Path(tmp_dir) / "preprocessed.onnx"
        quant_pre_process(model, str(preprocessed))
        quantize_static(
            str(preprocessed),
            str(output_path),
            Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
        )
    return Path(output_path)


@contextlib.contextmanager
def record_inputs(tools: Dict[str, object]):
    """
    Records what each rtmlib model receives, to reuse it as calibration data.

    rtmlib models expose ``inference(img)`` with an [H, W, 3] preprocessed image,
    which they transpose to a [1, 3, H, W] float32 ONNX input: the same transform is applied here.

    :param tools: rtmlib models (e.g. ``{"det": body.det_model, "pose": body.pose_model}``).
    :return: A dict mapping each name to the list of recorded ONNX inputs, filled while the context is active.
    """
    recorded = {name: [] for name in tools}
    originals = {}
    for name, tool in tools.items():
        originals[name] = tool.inference

        def wrapper(img, _name=name, _original=originals[name]):
            onnx_input = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None]
            recorded[_name].append(onnx_input)
            return _original(img)

        tool.inference = wrapper
    try:
        yield recorded
    finally:
        for name, tool in tools.items():
            tool.inference = originals[name]


def sample_frames(video_path: Union[str, Path], n_frames: int) -> List[np.ndarray]:
    """
    Reads ``n_frames`` frames evenly spread over a video.

    :param video_path: Path to the video file.
    :param n_frames: Number of frames to keep.
    :return: [H, W, 3] RGB uint8 frames.
    """
    with video.VideoReader(path=video_path) as reader:
        total = len(reader)
        keep = set(np.linspace(0, max(total - 1, 0), num=min(n_frames, max(total, 1)), dtype=int).tolist())
        frames = [frame for i, frame in enumerate(reader) if i in keep]
    if not frames:
        raise ValueError(f"no frame could be read from {video_path}")
    return frames


def check_session_device(session, device: str, name: str):
    """
    Fails loudly if an ONNX Runtime session meant for CUDA silently fell back to CPU.

    This happens when ``onnxruntime-gpu`` cannot load cuDNN/cuBLAS, e.g. because
    their pip-installed ``nvidia/*/lib`` folders are not on ``LD_LIBRARY_PATH``.

    :param session: The ``onnxruntime.InferenceSession``.
    :param device: The requested device.
    :param name: Model name, for the error message.
    :return:
    """
    if str(device).startswith("cuda") and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"{name} runs on {session.get_providers()[0]} instead of CUDA: onnxruntime-gpu could not load "
            "its CUDA libraries (check that the nvidia cudnn/cublas lib folders are on LD_LIBRARY_PATH)."
        )


def optimize_rtmlib_models(
    models: Dict[str, object],
    precision: str,
    run_fp32: Callable[[np.ndarray], object],
    calibration_frames: Sequence[np.ndarray],
    device: str,
    work_dir: Union[str, Path],
):
    """
    Swaps the ONNX session of each rtmlib model for an fp16 or INT8 version.

    :param models: rtmlib models to optimize, by name (each has ``session``, ``onnx_model``, ``inference``).
    :param precision: One of ``PRECISIONS``.
    :param run_fp32: Runs the full float32 pipeline on one [H, W, 3] BGR frame (used to record calibration inputs).
    :param calibration_frames: BGR frames for INT8 calibration (unused otherwise).
    :param device: 'cpu' or 'cuda'.
    :param work_dir: Directory where the optimized models are written.
    :return:
    """
    import onnxruntime as ort

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision}")
    if precision == "fp32":
        return
    if precision == "fp16" and not str(device).startswith("cuda"):
        raise ValueError(f"fp16 needs a CUDA device (ONNX Runtime has few fp16 CPU kernels), got {device}")

    work_dir = Path(work_dir)
    calibration = {}
    if precision == "int8":
        if len(calibration_frames) == 0:
            raise ValueError("INT8 quantization needs calibration frames")
        with record_inputs(models) as calibration:
            for frame in calibration_frames:
                run_fp32(frame)

    provider = "CUDAExecutionProvider" if str(device).startswith("cuda") else "CPUExecutionProvider"
    for name, model in models.items():
        output_path = work_dir / f"{name}.{precision}.onnx"
        if precision == "fp16":
            convert_fp16(model.onnx_model, output_path)
        else:
            if not calibration[name]:
                raise ValueError(f"no calibration input recorded for {name} (no person detected?)")
            quantize_int8(model.onnx_model, output_path, calibration[name])
        model.session = ort.InferenceSession(str(output_path), providers=[provider])
        check_session_device(model.session, device, name)
