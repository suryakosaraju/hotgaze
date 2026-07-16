"""Face detection layer — Gaussian attention blobs over detected faces.

Uses YuNet via cv2.FaceDetectorYN (OpenCV >= 4.5.4).  The ONNX model is
fetched through ``weights.download_weight("yunet")`` on first use and
cached in ``~/.cache/hotgaze/``.  No torch needed — core-package only.
"""

from __future__ import annotations

import cv2
import numpy as np

from .._imageops import gaussian_blur
from ..weights import download_weight
from .base import SignalLayer

# YuNet input size (model was trained at 320×320).
_YUNET_INPUT_SIZE = (320, 320)

# Detection parameters.
_SCORE_THRESHOLD = 0.6
_NMS_THRESHOLD = 0.3


class Faces(SignalLayer):
    """Attention layer that places Gaussian blobs over detected faces.

    Each detected face generates a Gaussian blob scaled by its detection
    confidence.  Multiple faces are combined additively.
    """

    def __init__(self) -> None:
        self._detector: cv2.FaceDetectorYN | None = None

    def _get_detector(self, w: int, h: int) -> cv2.FaceDetectorYN:
        """Lazy-load the YuNet detector, resized to match input dimensions."""
        if self._detector is None or self._detector.getInputSize() != (w, h):
            weight_path = str(download_weight("yunet"))
            detector = cv2.FaceDetectorYN.create(
                model=weight_path,
                config="",
                input_size=(w, h),
                score_threshold=_SCORE_THRESHOLD,
                nms_threshold=_NMS_THRESHOLD,
                top_k=50,
            )
            self._detector = detector
        return self._detector

    def compute(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        # Detect faces
        det = self._get_detector(w, h)
        _, faces = det.detect(img)

        # Build attention map
        attn = np.zeros((h, w), dtype=np.float32)
        if faces is None or len(faces) == 0:
            return attn

        ygrid, xgrid = np.mgrid[0:h, 0:w].astype(np.float32)

        for face in faces:
            fx, fy, fw, fh = face[0:4].astype(np.int32)
            confidence = float(face[14]) if face.shape[0] > 14 else 1.0

            cx = fx + fw / 2.0
            cy = fy + fh / 2.0
            sigma = max(fw, fh) / 3.0

            gauss = np.exp(-((xgrid - cx) ** 2 + (ygrid - cy) ** 2) / (2 * sigma**2))
            attn += gauss * confidence

        # Soften edges
        attn = gaussian_blur(attn, sigma=max(w, h) / 50.0)

        # Normalize
        mn, mx = attn.min(), attn.max()
        if mx - mn > 1e-10:
            attn = (attn - mn) / (mx - mn)

        return attn.astype(np.float32)
