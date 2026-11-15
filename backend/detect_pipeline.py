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
  MAX_BOX_AREA    Reject detections covering more than this fraction of the
                  frame (0.6). Zero-shot models occasionally "detect" the
                  whole screen as an advertisement; without this cap the
                  entire frame flickers black for a frame.
  MIN_HITS        A track must be detected this many times before its box is
                  rendered (2). Suppresses one-frame false positives.
  TRACK_TTL       Keep rendering a track's box for this many frames after the
                  detector last saw it (5). Suppresses flicker when a
                  detection briefly drops out.
  TRACK_SMOOTH    EMA weight for new box coordinates (0.6). Lower values
                  smooth jitter more but lag behind fast-moving ads.
"""

import io
import os
import time

import cv2
import numpy as np
from PIL import Image

YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8s-worldv2.pt")
DEFAULT_CLASSES = "logo,brand,corporate logo,brand logo,advertisement,billboard,sponsor,consumer product"
DETECT_CLASSES = [
    c.strip() for c in os.environ.get("DETECT_CLASSES", DEFAULT_CLASSES).split(",") if c.strip()
]
DETECT_CONF = float(os.environ.get("DETECT_CONF", "0.15"))
DETECT_EVERY = max(1, int(os.environ.get("DETECT_EVERY", "1")))
FILL_MODE = os.environ.get("FILL_MODE", "black")  # black|white|blur|debug
BOX_DILATE = float(os.environ.get("BOX_DILATE", "0.1"))
MAX_BOX_AREA = float(os.environ.get("MAX_BOX_AREA", "0.6"))
MIN_HITS = max(1, int(os.environ.get("MIN_HITS", "2")))
TRACK_TTL = max(0, int(os.environ.get("TRACK_TTL", "5")))
TRACK_SMOOTH = float(os.environ.get("TRACK_SMOOTH", "0.6"))
JPEG_QUALITY = 80


class DetectorPipeline:
    """Per-frame detector with persistent, temporally-smoothed box state.

    Raw per-frame detections are fused into tracks keyed by ByteTrack id:
    coordinates are EMA-smoothed, a track must be seen MIN_HITS times before
    its box is rendered, and a lost track's box is kept for TRACK_TTL frames
    so brief detection dropouts don't flicker.

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

        # track_id -> {"box": np.array([x1,y1,x2,y2]), "conf", "label",
        #              "last_seen", "hits"}
        self._tracks = {}
        self._synthetic_id = -1  # fallback ids if the tracker returns none
        self._frames = 0
        self._detects = 0
        self._detect_ms = 0.0

        # Warm up weights download + CUDA kernels so the first real frame is fast.
        self._detect(np.zeros((512, 512, 3), dtype=np.uint8))

    @property
    def prompt(self) -> str:
        """Server interface parity with SanaBackend: the active class list."""
        return ", ".join(self.classes)

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
        self._tracks = {}
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
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            names = results[0].names
            for i, ((x1, y1, x2, y2), conf, cls) in enumerate(zip(xyxy, confs, clss)):
                track_id = int(ids[i]) if ids is not None else None
                detections.append((track_id, x1, y1, x2, y2, float(conf), names[cls]))
        return detections

    def _update_tracks(self, detections, frame_shape):
        """Fuse one frame's raw detections into the persistent track state."""
        h, w = frame_shape[:2]
        for track_id, x1, y1, x2, y2, conf, label in detections:
            if (x2 - x1) * (y2 - y1) > MAX_BOX_AREA * w * h:
                continue  # whole-frame false positive; would black out the screen
            if track_id is None:
                track_id = self._synthetic_id
                self._synthetic_id -= 1
            track = self._tracks.get(track_id)
            if track is None:
                self._tracks[track_id] = {
                    "box": np.array([x1, y1, x2, y2]),
                    "conf": conf,
                    "label": label,
                    "last_seen": self._frames,
                    "hits": 1,
                }
            else:
                track["box"] = (1 - TRACK_SMOOTH) * track["box"] + TRACK_SMOOTH * np.array(
                    [x1, y1, x2, y2]
                )
                track["conf"] = conf
                track["label"] = label
                track["last_seen"] = self._frames
                track["hits"] += 1
        # Prune tracks that have been missing longer than the grace period.
        stale = [t for t, tr in self._tracks.items() if self._frames - tr["last_seen"] > TRACK_TTL]
        for t in stale:
            del self._tracks[t]

    def _renderable_tracks(self):
        return [
            tr
            for tr in self._tracks.values()
            if tr["hits"] >= MIN_HITS and self._frames - tr["last_seen"] <= TRACK_TTL
        ]

    def process_frame(self, data: bytes) -> bytes:
        """Detect logos/ads in one JPEG frame, return the blanked JPEG."""
        frame = np.array(Image.open(io.BytesIO(data)).convert("RGB"))  # writable copy
        self._frames += 1

        # _frames is 1-based, so detect on frames 1, 1+N, 1+2N, ... (with
        # DETECT_EVERY=1 that is every frame). Also re-detect immediately
        # whenever no tracks are known, so new ads are boxed without waiting
        # for the next scheduled detection frame.
        if (self._frames - 1) % DETECT_EVERY == 0 or not self._tracks:
            t0 = time.perf_counter()
            self._update_tracks(self._detect(frame), frame.shape)
            self._detects += 1
            self._detect_ms += (time.perf_counter() - t0) * 1000
            if self._frames % 60 == 1 and self._frames > 1:
                print(
                    f"[detect] frame {self._frames}: "
                    f"avg detect {self._detect_ms / self._detects:.1f} ms, "
                    f"{len(self._renderable_tracks())} box(es)",
                    flush=True,
                )

        out = self._composite(frame)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()

    def _composite(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for track in self._renderable_tracks():
            x1, y1, x2, y2 = track["box"]
            conf, label = track["conf"], track["label"]
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
