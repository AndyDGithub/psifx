"""Pure PyTorch/SciPy stand-in for the optional SAM3 post-processing kernels.

Without the ``kernels`` package, transformers' SAM3 video model skips three post-processing steps:
non-maximum suppression (NMS) of duplicate detections, hole filling and sprinkle removal. Skipping
NMS keeps every overlapping detection of the same person, which then becomes an extra track.
``kernels`` would download compiled code from the Hugging Face Hub at run time; these two functions
reproduce the operations SAM3 needs with the libraries psifx already depends on.
"""

import numpy as np
import torch
from scipy import ndimage


class TorchCvUtils:
    """Same interface as ``kernels-community/cv-utils``: ``generic_nms`` and ``cc_2d``."""

    @staticmethod
    def generic_nms(
        ious: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_iou_matrix: bool = True,
    ) -> torch.Tensor:
        """
        Greedy NMS on a precomputed IoU matrix.

        :param ious: (N, N) IoU between the detections.
        :param scores: (N,) detection scores.
        :param iou_threshold: A detection overlapping a better one above this IoU is dropped.
        :param use_iou_matrix: Kept for interface compatibility, ``ious`` is always a matrix here.
        :return: Indices (into ``scores``) of the kept detections, best first.
        """
        ious = ious.detach().float().cpu()
        order = torch.argsort(scores.detach().float().cpu(), descending=True).tolist()
        suppressed = torch.zeros(len(order), dtype=torch.bool)
        keep = []
        for idx in order:
            if suppressed[idx]:
                continue
            keep.append(idx)
            suppressed |= ious[idx] > iou_threshold
        return torch.tensor(keep, dtype=torch.int64, device=scores.device)

    @staticmethod
    def cc_2d(mask: torch.Tensor, get_counts: bool = True):
        """
        8-connected components of binary masks.

        :param mask: (B, 1, H, W) masks, nonzero is foreground.
        :param get_counts: Kept for interface compatibility, counts are always returned.
        :return: ``labels`` (component id per pixel, 0 for background) and ``counts`` (size of the
            pixel's component, 0 for background), both int32 and shaped like ``mask``.
        """
        array = mask.detach().cpu().numpy() != 0
        labels = np.zeros(array.shape, dtype=np.int32)
        counts = np.zeros(array.shape, dtype=np.int32)
        structure = np.ones((3, 3), dtype=bool)
        for b in range(array.shape[0]):
            for c in range(array.shape[1]):
                lab, _ = ndimage.label(array[b, c], structure=structure)
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                labels[b, c] = lab
                counts[b, c] = sizes[lab]
        return (
            torch.from_numpy(labels).to(mask.device),
            torch.from_numpy(counts).to(mask.device),
        )


def install_cv_utils_fallback(verbose: bool = False) -> bool:
    """
    Gives transformers' SAM3 video model the stand-in above when the ``kernels`` package is missing.

    :return: Whether the stand-in was installed.
    """
    try:
        from transformers.models.sam3_video import modeling_sam3_video
        from transformers.utils import is_kernels_available
    except ImportError:
        return False
    # None: not loaded yet, False: loading failed. A loaded hub kernel is left in place.
    if is_kernels_available() or getattr(modeling_sam3_video, "cv_utils_kernel", True) not in (None, False):
        return False
    modeling_sam3_video.cv_utils_kernel = TorchCvUtils()
    if verbose:
        print("Using the built-in NMS and hole filling for SAM3 (the kernels package is not installed).")
    return True
