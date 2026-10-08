"""Inference helpers for ONNX pose models run with rtmlib (fp16 conversion, device checks).

float16 conversion keeps float32 inputs/outputs, so rtmlib's pre/post-processing is
unchanged. INT8 quantization was evaluated and dropped: slower and much less accurate
on these models (see research/05_optimisations_synthese.md on the dev branch).
"""

from typing import Dict, List, Union

import ctypes
import glob
import importlib
import os
from pathlib import Path

PRECISIONS = ("fp32", "fp16")

RTM_INSTALL_HINT = (
    "RTMO needs the optional rtmlib dependencies: pip install 'psifx[rtm]' "
    "(or 'psifx[rtm-gpu]' for GPU inference, see the documentation)."
)


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


def preload_cuda_libraries() -> List[str]:
    """
    Loads the cuDNN and cuBLAS libraries installed with PyTorch (``nvidia-cudnn-cu12``,
    ``nvidia-cublas-cu12``) so that ``onnxruntime-gpu`` finds them without ``LD_LIBRARY_PATH``.

    onnxruntime opens these libraries by name; once a library with the same name is loaded
    in the process, the dynamic loader reuses it. Missing packages are skipped: the session
    check (``check_session_device``) then reports the problem.

    :return: The names of the loaded libraries.
    """
    loaded = []
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            lib_dir = os.path.join(importlib.import_module(package).__path__[0], "lib")
        except ImportError:
            continue
        pending = sorted(glob.glob(os.path.join(lib_dir, "lib*.so*")))
        # Some libraries depend on others of the same folder: retry until no progress.
        while pending:
            failed = []
            for path in pending:
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(path))
                except OSError:
                    failed.append(path)
            if len(failed) == len(pending):
                break
            pending = failed
    return loaded


def check_session_device(session, device: str, name: str):
    """
    Fails loudly if an ONNX Runtime session meant for CUDA silently fell back to CPU.

    This happens when ``onnxruntime-gpu`` cannot load cuDNN/cuBLAS (their pip-installed
    ``nvidia/*/lib`` folders are not on ``LD_LIBRARY_PATH``), or when the CPU-only
    ``onnxruntime`` package was installed after ``onnxruntime-gpu`` and overwrote it.

    :param session: The ``onnxruntime.InferenceSession``.
    :param device: The requested device.
    :param name: Model name, for the error message.
    :return:
    """
    if str(device).startswith("cuda") and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"{name} runs on {session.get_providers()[0]} instead of CUDA: onnxruntime-gpu could not load "
            "its CUDA libraries (check that only onnxruntime-gpu is installed, not onnxruntime, and that the "
            "nvidia cudnn/cublas lib folders are on LD_LIBRARY_PATH)."
        )


def optimize_rtmlib_models(
    models: Dict[str, object],
    precision: str,
    device: str,
    work_dir: Union[str, Path],
):
    """
    Swaps the ONNX session of each rtmlib model for a float16 version.

    :param models: rtmlib models to optimize, by name (each has ``session`` and ``onnx_model``).
    :param precision: One of ``PRECISIONS``; 'fp32' leaves the models untouched.
    :param device: 'cpu' or 'cuda'.
    :param work_dir: Directory where the converted models are written.
    :return:
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision}")
    if precision == "fp32":
        return
    # Checked before importing onnxruntime: the error is about the setting, not the install.
    if not str(device).startswith("cuda"):
        raise ValueError(f"fp16 needs a CUDA device (ONNX Runtime has few fp16 CPU kernels), got {device}")

    import onnxruntime as ort

    for name, model in models.items():
        output_path = Path(work_dir) / f"{name}.{precision}.onnx"
        convert_fp16(model.onnx_model, output_path)
        model.session = ort.InferenceSession(str(output_path), providers=["CUDAExecutionProvider"])
        check_session_device(model.session, device, name)
