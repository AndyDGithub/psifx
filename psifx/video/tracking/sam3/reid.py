"""Shot-change detection and appearance re-identification for SAM3 tracking.

In edited videos (talk shows, interviews, films) the camera cuts between shots every few seconds.
SAM3 and the chunk stitching both rely on continuity: across a cut, a person either gets a new
track id, or, when two close-ups alternate at the same place in the image, inherits the id of the
other person. The tracking tool therefore:

1. starts a new SAM3 session at each detected cut, so no identity is carried over a cut;
2. links each new track to a known person by appearance: a DINOv2 embedding of the masked person,
   compared to the embeddings already seen for each person (cosine similarity, best match).
"""

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image


class ShotDetector:
    """
    Detects hard cuts from the mean absolute difference between consecutive small thumbnails.

    A frame starts a new shot when its difference to the previous frame is above ``threshold``
    and well above the recent motion level, so that fast but continuous motion (dance, camera
    pan) is not taken for a cut.

    :param threshold: Minimum mean absolute difference (0 to 1) between consecutive thumbnails.
    :param history: Number of recent differences that define the motion level.
    """

    def __init__(self, threshold: float = 0.15, history: int = 12):
        self.threshold = threshold
        self.history = history
        self._previous: Optional[np.ndarray] = None
        self._recent: List[float] = []

    def is_cut(self, frame: np.ndarray) -> bool:
        """
        :param frame: RGB frame, (H, W, 3) uint8.
        :return: Whether this frame starts a new shot.
        """
        thumbnail = np.asarray(Image.fromarray(frame).resize((64, 36), Image.BILINEAR), dtype=np.float32)
        previous, self._previous = self._previous, thumbnail
        if previous is None:
            return False
        difference = float(np.abs(thumbnail - previous).mean() / 255.0)
        motion = float(np.median(self._recent)) if self._recent else 0.0
        self._recent = (self._recent + [difference])[-self.history:]
        return difference > self.threshold and difference > 3.0 * motion + 0.02


class AppearanceGallery:
    """
    Appearance embeddings of the people seen so far, to re-identify them after a cut.

    :param model_path: Hugging Face id or local path of a DINOv2-like image encoder.
    :param threshold: Minimum cosine similarity to link a new track to a known person.
    :param device: Device of the encoder.
    :param max_samples: Embeddings kept per person (the most recent ones).
    :param api_token: Hugging Face token, if the model needs one.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float,
        device: str = "cpu",
        max_samples: int = 32,
        api_token: Optional[str] = None,
    ):
        from transformers import AutoImageProcessor, AutoModel

        self.threshold = threshold
        self.device = device
        self.max_samples = max_samples
        self.processor = AutoImageProcessor.from_pretrained(model_path, token=api_token)
        self.model = AutoModel.from_pretrained(model_path, token=api_token).to(device).eval()
        self.samples: Dict[int, List[np.ndarray]] = {}

    @staticmethod
    def crop(image: Image.Image, mask: np.ndarray) -> Optional[Image.Image]:
        """Bounding box of the mask, background set to gray so that only the person is described."""
        ys, xs = np.nonzero(mask)
        if len(ys) < 64:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pixels = np.asarray(image)[y0:y1, x0:x1].copy()
        pixels[~mask[y0:y1, x0:x1]] = 127
        return Image.fromarray(pixels)

    @torch.no_grad()
    def embed(self, crops: Sequence[Image.Image]) -> Optional[np.ndarray]:
        """Mean of the normalized CLS embeddings of the crops, normalized again."""
        if not crops:
            return None
        inputs = self.processor(images=list(crops), return_tensors="pt").to(self.device)
        cls = self.model(**inputs).last_hidden_state[:, 0].float()
        mean = torch.nn.functional.normalize(cls, dim=-1).mean(dim=0)
        return torch.nn.functional.normalize(mean, dim=0).cpu().numpy()

    def similarity(self, embedding: np.ndarray, global_id: int) -> float:
        """Best cosine similarity to the person's stored embeddings."""
        return max(float(embedding @ sample) for sample in self.samples[global_id])

    def match(self, embedding: np.ndarray, excluded: Set[int]) -> Tuple[Optional[int], float]:
        """Most similar known person outside ``excluded``, and its similarity."""
        best_id, best_similarity = None, -1.0
        for global_id in self.samples:
            if global_id in excluded:
                continue
            similarity = self.similarity(embedding, global_id)
            if similarity > best_similarity:
                best_id, best_similarity = global_id, similarity
        return best_id, best_similarity

    def add(self, global_id: int, embedding: np.ndarray):
        samples = self.samples.setdefault(global_id, [])
        samples.append(embedding)
        del samples[:-self.max_samples]


def sample_frames(frame_indices: Iterable[int], count: int) -> List[int]:
    """Up to ``count`` frame indices spread evenly over ``frame_indices``."""
    frame_indices = sorted(frame_indices)
    if len(frame_indices) <= count:
        return frame_indices
    picks = np.linspace(0, len(frame_indices) - 1, count).round().astype(int)
    return [frame_indices[i] for i in picks]
