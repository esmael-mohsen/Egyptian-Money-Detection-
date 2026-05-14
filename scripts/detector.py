from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from config import DetectorConfig


CLASS_ID_TO_NAME = {
    
    10: "5 Pounds",
    1: "5 Pounds",

    0: "10 Pounds",
    3: "10 Pounds",

    9: "20 Pounds",
    5: "20 Pounds",

    8: "50 Pounds",
    2: "50 Pounds",

    4: "100 Pounds",
    11: "100 Pounds",

    6: "200 Pounds",
    7: "200 Pounds"
}


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class Detector:
    """Single-responsibility YOLO detector wrapper for runtime usage."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._model = YOLO(str(config.model_path))

    def infer(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self._config.conf_threshold,
            iou=self._config.iou_threshold,
            max_det=self._config.max_det,
            half=self._config.use_half,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                class_name = CLASS_ID_TO_NAME.get(cls_id)
                if not class_name:
                    continue

                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=float(boxes.conf[i]),
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections
