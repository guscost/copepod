"""YOLO-based logo/ad detection + blanking pipeline (alternative backend).

Instead of regenerating every frame with the SANA diffusion model, this
pipeline detects logos and advertising with YOLO-World (zero-shot: detection
classes are plain text) or a fine-tuned YOLO checkpoint, tracks the boxes
across frames with ByteTrack, and composites a fill (black square by default)
over the detected regions. Untouched pixels pass through unmodified, so the
output matches the input everywhere except the blanked areas, and per-frame
latency is a few milliseconds on a modern GPU.

Selected with BACKEND_MODE=detect in server.py. Env knobs:

  YOLO_MODEL      Checkpoint path/name. A model whose name contains "world"
                  (default: yolov8s-worldv2.pt, auto-downloaded on first run)
                  supports zero-shot classes via set_classes(); a fine-tuned
                  YOLO checkpoint uses the classes it was trained on.
  DETECT_CLASSES  Comma-separated zero-shot classes (YOLO-World only).
  DETECT_CONF     Confidence threshold (default 0.15 — zero-shot models score
                  conservatively; raise it if false positives appear).
  DETECT_EVERY    Run the detector every Nth frame, reusing the previous
                  boxes in between (default 1; raise on weaker GPUs).
  FILL_MODE       black (default) | white | blur | debug
                  ("debug" draws boxes + labels and fills nothing — use it
                  while tuning classes and confidence).
  BOX_DILATE      Grow each box by this fraction of its width/height before
                  filling, so anti-aliased logo edges are covered (0.1).
"""

import io
import os
import time

import cv2
import numpy as np
from PIL import Image

YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8s-worldv2.pt")
DEFAULT_CLASSES = "corporate logo,brand logo,advertisement,advertising banner,sponsor logo"
DETECT_CLASSES = [
    c.strip() for c in os.environ.get("DETECT_CLASSES", DEFAULT_CLASSES).split(",") if c.strip()
]
DETECT_CONF = float(os.environ.get("DETECT_CONF", "0.15"))
DETECT_EVERY = max(1, int(os.environ.get("DETECT_EVERY", "1")))
FILL_MODE = os.environ.get("FILL_MODE", "black")  # black|white|blur|debug
BOX_DILATE = float(os.environ.get("BOX_DILATE", "0.1"))
JPEG_QUALITY = 80


class DetectorPipeline:
    """Stateless-per-frame detector with persistent ByteTrack box state.

    All methods must be called from a single thread (the server's GPU worker).
    """

    def __init__(self, device):
        from ultralytics import YOLO, YOLOWorld

        self.device = device
        if device.type == "cuda":
            self.ul_device = device.index if device.index is not None else 0
        else:
            self.ul_device = "cpu"

        self.is_world = "world" in os.path.basename(YOLO_MODEL).lower()
        self.classes = list(DETECT_CLASSES)
        print(f"[detect] loading {YOLO_MODEL}...", flush=True)
        if self.is_world:
            self.model = YOLOWorld(YOLO_MODEL)
            self.model.set_classes(self.classes)
            print(f"[detect] zero-shot classes: {self.classes}", flush=True)
        else:
            self.model = YOLO(YOLO_MODEL)
            print(f"[detect] fine-tuned classes: {list(self.model.names.values())}", flush=True)

        self._detections = []  # [(x1, y1, x2, y2, conf, label), ...]
        self._frames = 0
        self._detects = 0
        self._detect_ms = 0.0

        # Warm up weights download + CUDA kernels so the first real frame is fast.
        self._detect(np.zeros((512, 512, 3), dtype=np.uint8))

    def set_prompt(self, text: str):
        """In detect mode the "prompt" is a comma-separated class list."""
        classes = [c.strip() for c in text.split(",") if c.strip()]
        if not classes:
            return
        if not self.is_world:
            print("[detect] fine-tuned model has fixed classes; ignoring prompt", flush=True)
            return
        self.classes = classes
        self.model.set_classes(classes)
        self.reset()
        print(f"[detect] classes updated: {classes}", flush=True)

    def reset(self):
        """Clear tracker + box state (called when the producer disconnects)."""
        self._detections = []
        self._frames = 0
        # Drop the predictor so ByteTrack starts from a clean slate.
        self.model.predictor = None

    def _detect(self, frame: np.ndarray):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=DETECT_CONF,
            device=self.ul_device,
            verbose=False,
        )
        boxes = results[0].boxes
        detections = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            names = results[0].names
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                detections.append((x1, y1, x2, y2, float(conf), names[cls]))
        return detections

    def process_frame(self, data: bytes) -> bytes:
        """Detect logos/ads in one JPEG frame, return the blanked JPEG."""
        frame = np.array(Image.open(io.BytesIO(data)).convert("RGB"))  # writable copy
        self._frames += 1

        if self._frames % DETECT_EVERY == 1 or not self._detections:
            t0 = time.perf_counter()
            self._detections = self._detect(frame)
            self._detects += 1
            self._detect_ms += (time.perf_counter() - t0) * 1000
            if self._frames % 60 == 1 and self._frames > 1:
                print(
                    f"[detect] frame {self._frames}: "
                    f"avg detect {self._detect_ms / self._detects:.1f} ms, "
                    f"{len(self._detections)} box(es)",
                    flush=True,
                )

        out = self._composite(frame)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()

    def _composite(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for x1, y1, x2, y2, conf, label in self._detections:
            bw, bh = x2 - x1, y2 - y1
            x1 = int(max(0, x1 - bw * BOX_DILATE))
            y1 = int(max(0, y1 - bh * BOX_DILATE))
            x2 = int(min(w, x2 + bw * BOX_DILATE))
            y2 = int(min(h, y2 + bh * BOX_DILATE))
            if x2 <= x1 or y2 <= y1:
                continue
            if FILL_MODE == "debug":
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    out,
                    f"{label} {conf:.2f}",
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
            elif FILL_MODE == "white":
                out[y1:y2, x1:x2] = 255
            elif FILL_MODE == "blur":
                region = out[y1:y2, x1:x2]
                k = max(3, (min(region.shape[:2]) // 8) | 1)
                out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
            else:  # black
                out[y1:y2, x1:x2] = 0
        return out
