"""YOLO-based logo/ad detection + blanking pipeline (alternative backend).

Instead of regenerating every frame with the SANA diffusion model, this
pipeline detects logos and advertising with YOLO-World (zero-shot: detection
classes are plain text) or a fine-tuned YOLO checkpoint, tracks the boxes
across frames with ByteTrack, and composites a fill (black square by default)
over the detected regions. Untouched pixels pass through unmodified, so the
output matches the input everywhere except the blanked areas, and per-frame
latency is a few milliseconds on a modern GPU.

Selected with BACKEND_MODE=detect in server.py. Env knobs:

  YOLO_MODEL      Checkpoint path/name. Three flavors are supported:
                  * contains "world" (default: yolov8s-worldv2.pt,
                    auto-downloaded on first run): YOLO-World, zero-shot
                    classes via set_classes(). Fastest, weakest.
                  * contains "grounding" (e.g. IDEA-Research/grounding-dino-tiny):
                    Grounding DINO via Hugging Face transformers — a heavier
                    zero-shot transformer detector with much better accuracy
                    on ambiguous classes like "logo" (~20-30 ms/frame on a
                    high-end GPU in fp16). Needs `pip install transformers`.
                    Track ids come from a lightweight IoU matcher (Grounding
                    DINO has no built-in tracker).
                  * anything else: a fine-tuned YOLO checkpoint, using the
                    classes it was trained on.
  DETECT_CLASSES  Comma-separated zero-shot classes (YOLO-World / Grounding
                  DINO only).
  DETECT_CONF     Confidence threshold (default 0.15 — zero-shot models score
                  conservatively; raise it if false positives appear). Doubles
                  as Grounding DINO's box threshold.
  GDINO_TEXT_THRESHOLD  Grounding DINO token-to-phrase threshold (0.25).
  GDINO_FP16      Run Grounding DINO in fp16 on CUDA (default 1; set 0 to
                  force fp32).
  GDINO_TRT       Compile Grounding DINO with Torch-TensorRT (default 0;
                  experimental — roughly 2x throughput over eager fp16 once
                  compiled). Requires a torch-tensorrt build matching the
                  installed torch/CUDA (see requirements.txt). Compilation is
                  lazy and happens on the warmup frame, so set FRAME_WIDTH /
                  FRAME_HEIGHT (default 1280x720) to the actual incoming frame
                  size, otherwise the first real frame pays a recompile. A
                  call that fails under TensorRT permanently falls back to
                  eager for the rest of the session.
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

import importlib.util
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
GDINO_TEXT_THRESHOLD = float(os.environ.get("GDINO_TEXT_THRESHOLD", "0.25"))
GDINO_FP16 = os.environ.get("GDINO_FP16", "1") != "0"
GDINO_TRT = os.environ.get("GDINO_TRT", "0") != "0"
DETECT_EVERY = max(1, int(os.environ.get("DETECT_EVERY", "1")))
FILL_MODE = os.environ.get("FILL_MODE", "black")  # black|white|blur|debug
BOX_DILATE = float(os.environ.get("BOX_DILATE", "0.1"))
MAX_BOX_AREA = float(os.environ.get("MAX_BOX_AREA", "0.6"))
MIN_HITS = max(1, int(os.environ.get("MIN_HITS", "2")))
TRACK_TTL = max(0, int(os.environ.get("TRACK_TTL", "5")))
TRACK_SMOOTH = float(os.environ.get("TRACK_SMOOTH", "0.6"))
JPEG_QUALITY = 80


class _IoUTracker:
    """Greedy IoU matcher that gives per-frame detections persistent ids.

    ByteTrack ships with ultralytics; Grounding DINO has no tracker, so this
    stands in: each new box is matched to the previous frame's box with the
    highest IoU above the threshold, otherwise it starts a new track. That is
    sufficient here because detection runs at video framerate and logo boxes
    move slowly relative to their size.
    """

    def __init__(self, iou_thresh=0.3):
        self.iou_thresh = iou_thresh
        self._next_id = 0
        self._boxes = {}  # id -> np.array([x1, y1, x2, y2])

    def reset(self):
        self._boxes = {}

    def update(self, boxes: np.ndarray) -> list:
        """Match (N, 4) xyxy boxes against the previous frame; return N ids."""
        ids = [-1] * len(boxes)
        if len(boxes) and self._boxes:
            prev_ids = list(self._boxes)
            prev = np.stack([self._boxes[i] for i in prev_ids])
            inter_x1 = np.maximum(boxes[:, None, 0], prev[None, :, 0])
            inter_y1 = np.maximum(boxes[:, None, 1], prev[None, :, 1])
            inter_x2 = np.minimum(boxes[:, None, 2], prev[None, :, 2])
            inter_y2 = np.minimum(boxes[:, None, 3], prev[None, :, 3])
            inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(
                inter_y2 - inter_y1, 0, None
            )
            area_d = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            area_p = (prev[:, 2] - prev[:, 0]) * (prev[:, 3] - prev[:, 1])
            iou = inter / np.maximum(area_d[:, None] + area_p[None, :] - inter, 1e-9)
            used_d, used_p = set(), set()
            for d, p in sorted(
                np.argwhere(iou >= self.iou_thresh), key=lambda dp: -iou[tuple(dp)]
            ):
                d, p = int(d), int(p)
                if d not in used_d and p not in used_p:
                    ids[d] = prev_ids[p]
                    used_d.add(d)
                    used_p.add(p)
        for d in range(len(boxes)):
            if ids[d] == -1:
                ids[d] = self._next_id
                self._next_id += 1
        self._boxes = {ids[d]: boxes[d] for d in range(len(boxes))}
        return ids


class DetectorPipeline:
    """Per-frame detector with persistent, temporally-smoothed box state.

    Raw per-frame detections are fused into tracks keyed by ByteTrack id:
    coordinates are EMA-smoothed, a track must be seen MIN_HITS times before
    its box is rendered, and a lost track's box is kept for TRACK_TTL frames
    so brief detection dropouts don't flicker.

    All methods must be called from a single thread (the server's GPU worker).
    """

    def __init__(self, device):
        self.device = device
        if device.type == "cuda":
            self.ul_device = device.index if device.index is not None else 0
        else:
            self.ul_device = "cpu"

        model_name = os.path.basename(YOLO_MODEL).lower()
        self.is_gdino = "grounding" in model_name
        self.is_world = not self.is_gdino and "world" in model_name
        self.classes = list(DETECT_CLASSES)
        print(f"[detect] loading {YOLO_MODEL}...", flush=True)
        if self.is_gdino:
            self._init_gdino(device)
        elif self.is_world:
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(YOLO_MODEL)
            self.model.set_classes(self.classes)
            print(f"[detect] zero-shot classes: {self.classes}", flush=True)
        else:
            from ultralytics import YOLO

            self.model = YOLO(YOLO_MODEL)
            print(f"[detect] fine-tuned classes: {list(self.model.names.values())}", flush=True)

        # track_id -> {"box": np.array([x1,y1,x2,y2]), "conf", "label",
        #              "last_seen", "hits"}
        self._tracks = {}
        self._synthetic_id = -1  # fallback ids if the tracker returns none
        self._frames = 0
        self._detects = 0
        self._detect_ms = 0.0

        # Warm up weights download + CUDA kernels so the first real frame is
        # fast. With GDINO_TRT the warmup is also when TensorRT compilation
        # happens, so it runs at the configured frame size (dynamic=False
        # means a differently-sized first real frame would recompile).
        if self.is_gdino and GDINO_TRT:
            warm_h = int(os.environ.get("FRAME_HEIGHT", "720"))
            warm_w = int(os.environ.get("FRAME_WIDTH", "1280"))
        else:
            warm_h, warm_w = 512, 512
        self._detect(np.zeros((warm_h, warm_w, 3), dtype=np.uint8))

    def _init_gdino(self, device):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(YOLO_MODEL)
        dtype = torch.float16 if (device.type == "cuda" and GDINO_FP16) else torch.float32
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(YOLO_MODEL).to(
            device=device, dtype=dtype
        )
        self.gdino.eval()
        self._gdino_forward = (
            self._trt_wrap(self.gdino)
            if (GDINO_TRT and device.type == "cuda")
            else self.gdino
        )
        self._iou_tracker = _IoUTracker()
        print(
            f"[detect] Grounding DINO zero-shot ({dtype}) classes: {self.classes}",
            flush=True,
        )

    @staticmethod
    def _trt_wrap(model):
        """Wrap the Grounding DINO forward in torch.compile(backend="tensorrt").

        Compilation is lazy (first call), so the wrapper catches a failure
        from any compiled call and permanently falls back to the eager model —
        a bad graph can never kill the stream. Same policy as sana_pipeline's
        TENSORRT path.
        """
        import torch

        try:
            trt_installed = importlib.util.find_spec("torch_tensorrt") is not None
        except ValueError:  # present in sys.modules without a spec — treat as installed
            trt_installed = True
        if not trt_installed:
            print(
                "[tensorrt] gdino: torch-tensorrt not installed; staying eager "
                "(install a torch-tensorrt build matching your torch/CUDA to enable)",
                flush=True,
            )
            return model
        compiled = torch.compile(model, backend="tensorrt", dynamic=False)
        failed = False

        def wrapper(*args, **kwargs):
            nonlocal failed
            if failed:
                return model(*args, **kwargs)
            try:
                return compiled(*args, **kwargs)
            except Exception as e:
                failed = True
                print(
                    f"[tensorrt] gdino failed under TensorRT ({e}); "
                    "eager for the rest of the session",
                    flush=True,
                )
                return model(*args, **kwargs)

        print(
            "[tensorrt] gdino: Torch-TensorRT enabled (compiles on warmup frame)",
            flush=True,
        )
        return wrapper

    @property
    def prompt(self) -> str:
        """Server interface parity with SanaBackend: the active class list."""
        return ", ".join(self.classes)

    def set_prompt(self, text: str):
        """In detect mode the "prompt" is a comma-separated class list."""
        classes = [c.strip() for c in text.split(",") if c.strip()]
        if not classes:
            return
        if not self.is_world and not self.is_gdino:
            print("[detect] fine-tuned model has fixed classes; ignoring prompt", flush=True)
            return
        self.classes = classes
        if self.is_world:
            self.model.set_classes(classes)
        self.reset()
        print(f"[detect] classes updated: {classes}", flush=True)

    def reset(self):
        """Clear tracker + box state (called when the producer disconnects)."""
        self._tracks = {}
        self._frames = 0
        if self.is_gdino:
            self._iou_tracker.reset()
        else:
            # Drop the predictor so ByteTrack starts from a clean slate.
            self.model.predictor = None

    def _detect(self, frame: np.ndarray):
        if self.is_gdino:
            return self._detect_gdino(frame)
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

    def _detect_gdino(self, frame: np.ndarray):
        import torch

        image = Image.fromarray(frame)
        # Grounding DINO prompt format: lowercase phrases, period-separated.
        text = ". ".join(c.lower() for c in self.classes) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.gdino.dtype)
        with torch.inference_mode():
            outputs = self._gdino_forward(**inputs)
        kwargs = dict(
            input_ids=inputs["input_ids"],
            text_threshold=GDINO_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, box_threshold=DETECT_CONF, **kwargs
            )
        except TypeError:
            # transformers < 4.55 names the box threshold just `threshold`.
            results = self.processor.post_process_grounded_object_detection(
                outputs, threshold=DETECT_CONF, **kwargs
            )
        res = results[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        # transformers >= 4.51 deprecates string "labels" in favor of "text_labels".
        labels = res["text_labels"] if "text_labels" in res else res["labels"]
        ids = self._iou_tracker.update(boxes)
        return [
            (track_id, x1, y1, x2, y2, float(conf), label)
            for track_id, (x1, y1, x2, y2), conf, label in zip(ids, boxes, scores, labels)
        ]

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
