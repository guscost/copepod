"""Detector-based logo/ad detection + blanking pipeline (alternative backend).

Instead of regenerating every frame with the SANA diffusion model, this
pipeline detects logos and advertising with a zero-shot detector (Grounding
DINO by default; detection classes are plain text) or a fine-tuned YOLO
checkpoint, tracks the boxes across frames (ByteTrack for YOLO, a lightweight
IoU matcher for Grounding DINO), and composites a fill (black square by
default) over the detected regions. Untouched pixels pass through unmodified,
so the output matches the input everywhere except the blanked areas, and
per-frame latency is a few milliseconds on a modern GPU.

Selected with BACKEND_MODE=detect in server.py. Env knobs:

  YOLO_MODEL      Single-detector override. Three flavors are supported:
                  * contains "grounding" (e.g.
                    IDEA-Research/grounding-dino-base, auto-downloaded from
                    Hugging Face on first run): Grounding DINO via
                    transformers — a heavier zero-shot transformer detector
                    with much better accuracy on ambiguous classes like
                    "logo" (~30+ fps on a high-end GPU in fp16). Needs
                    `pip install transformers`. Track ids come from a
                    lightweight IoU matcher (Grounding DINO has no built-in
                    tracker).
                  * contains "world" (e.g. yolov8s-worldv2.pt,
                    auto-downloaded on first run): YOLO-World, zero-shot
                    classes via set_classes(). Fastest, weakest — the
                    lightweight option for weaker GPUs.
                  * anything else: a fine-tuned YOLO checkpoint, using the
                    classes it was trained on.
                  Unset (default): the two fine-tuned checkpoints below run
                  together and their detections are combined.
  BILLBOARD_YOLO  Billboard detector spec (default
                  maco018/billboard-detection-Yolo11:yolo11s.pt — swap the
                  filename for the yolo11n/m/l/x variants). Set to "off" to
                  disable.
  BRAND_EYE_YOLO  Brand/logo detector spec (default
                  haydarkadioglu/brand-eye:brandeye.pt). Set to "off" to
                  disable.
                  Fine-tuned specs are owner/repo[:file] on Hugging Face
                  (weights auto-download on first run; the file may be
                  omitted when the repo contains exactly one .pt) or a local
                  .pt path. Zero-shot detectors (grounding/world) are
                  single-model only — select them with YOLO_MODEL.
  DETECT_CLASSES  Comma-separated zero-shot classes (YOLO-World / Grounding
                  DINO only).
  DETECT_CONF     Confidence threshold (default 0.25; zero-shot models score
                  conservatively — lower towards 0.15 for those, raise it if
                  false positives appear). Doubles as Grounding DINO's box
                  threshold.
  GDINO_TEXT_THRESHOLD  Grounding DINO token-to-phrase threshold (0.25).
  GDINO_FP16      Run Grounding DINO in fp16 on CUDA (default 1; set 0 to
                  force fp32). Eager runs under fp16 autocast, which keeps
                  fp32-only ops (LayerNorm, deformable-attention sampling,
                  contrastive heads) in fp32 instead of casting the whole
                  model with .half(), avoiding float/half mismatches.
                  Doubles as the default for GDINO_TRT_DTYPE.
  GDINO_TRT_DTYPE Precision the TensorRT engine is compiled at: fp16
                  (default when GDINO_FP16=1) | bf16 | fp32. The graph is
                  always traced in fp32 (autocast disabled for the compiled
                  call — feeding an autocast-traced graph to the compiler
                  builds an engine that emits NaN logits) and the low
                  precision is applied by Torch-TensorRT itself: rule-based
                  autocast (enable_autocast + fp32 accumulation) on
                  torch_tensorrt >= 2.15, enabled_precisions on older
                  versions. Attention is also decomposed (matmul/softmax
                  instead of TRT's fused kernel, which returns NaN on
                  fully-masked rows in fp16 — pytorch/TensorRT#5011). bf16
                  has fp32's exponent range — use it if fp16 logits come
                  out NaN (fp16 overflows past 65504).
  GDINO_TRT_TORCH_OPS
                  Comma-separated aten ops to keep in eager PyTorch inside
                  the compiled model (torch_executed_ops) — a bisection knob
                  for localizing which op produces NaN/wrong output, e.g.
                  "aten._softmax.default,aten.grid_sampler_2d.default".
  GDINO_TRT       Compile Grounding DINO with Torch-TensorRT (default 0;
                  experimental — roughly 2x throughput over eager fp16 once
                  compiled). Requires a torch-tensorrt build matching the
                  installed torch/CUDA (see requirements.txt). Two dynamo
                  mis-trace workarounds are applied before compiling: HF
                  Swin's window ops and the view() calls in Swin and
                  Grounding DINO module forwards are patched
                  (view->reshape), and fake-tensor real-tensor propagation
                  is enabled so contiguous() isn't wrongly elided after
                  permutes.
                  Compilation is lazy and happens on the warmup frame, so set
                  FRAME_WIDTH / FRAME_HEIGHT (default 1280x720) to the actual
                  incoming frame size, otherwise the first real frame pays a
                  recompile. A call that fails under TensorRT permanently
                  falls back to eager for the rest of the session. The
                  workarounds also rewrite .view as .reshape in every Swin
                  and Grounding DINO module forward for the same mis-trace.
  GDINO_TRT_COMPARE  With GDINO_TRT, also run the eager model on the first N
                  frames (default 8) and compare raw logit scores against the
                  compiled output. A compiled engine that silently degrades
                  (scores collapsed/NaN under TensorRT while eager still
                  detects) is logged and permanently swapped back to eager.
                  The comparison doubles detect cost for those N frames; set
                  0 to disable.
  GDINO_DEBUG   Log raw per-frame detection count and top scores before
                  tracking/filtering (default 0).
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
  MIN_BOX_AREA    Reject detections covering less than this fraction of the frame
                  (0.0). May be used to suppress detector noise and specks.
  MIN_BOX_WIDTH   Minimum detected box width in pixels (0, disabled).
  MIN_BOX_HEIGHT  Minimum detected box height in pixels (0, disabled).
  JPEG_QUALITY    Encoded output JPEG/WebP quality (80).
  OUTPUT_FORMAT   Output encoding: "jpeg" (default) or "webp" (smaller at the
                  same quality, but may encode more slowly).
  MIN_HITS        A track must be detected this many times before its box is
                  rendered (2). Suppresses one-frame false positives.
  TRACK_TTL       Keep rendering a track's box for this many frames after the
                  detector last saw it (5). Suppresses flicker when a
                  detection briefly drops out.
  TRACK_SMOOTH    EMA weight for new box coordinates (0.6). Lower values
                  smooth jitter more but lag behind fast-moving ads.
  BOTTOM_COVERAGE_THRESHOLD
                  If the union of retrieved boxes covers this fraction or
                  more of the bottom half, treat it as a false positive and
                  do not block (0.8).
  FULLSCREEN_AD_CLASSIFIER
                  Hugging Face CLIP model used for binary full-screen-ad
                  classification (default openai/clip-vit-base-patch32).
                  Set to off/0/none to disable.
  FULLSCREEN_AD_CONF
                  Probability required to classify a frame as a full-screen ad
                  (0.70).
  FULLSCREEN_AD_EVERY
                  Run full-screen classification every Nth frame (1).
"""

import io
import os
import time

import cv2
import numpy as np
from PIL import Image

# Single-detector override (Grounding DINO / YOLO-World / one fine-tuned
# checkpoint). Unset: the BILLBOARD_YOLO + BRAND_EYE_YOLO pair below runs.
YOLO_MODEL = os.environ.get("YOLO_MODEL", "")
# Default detector: two fine-tuned YOLO checkpoints combined — billboards
# plus brand logos. Each spec is owner/repo[:file] on Hugging Face (weights
# auto-download on first run) or a local .pt path; "off"/"0"/"" disables it.
BILLBOARD_YOLO = os.environ.get(
    "BILLBOARD_YOLO", "maco018/billboard-detection-Yolo11:yolo11s.pt"
)
BRAND_EYE_YOLO = os.environ.get(
    "BRAND_EYE_YOLO", "haydarkadioglu/brand-eye:brandeye.pt"
)
DEFAULT_CLASSES = "corporate logo,brand logo,advertisement,billboard,sponsor"
DETECT_CLASSES = [
    c.strip() for c in os.environ.get("DETECT_CLASSES", DEFAULT_CLASSES).split(",") if c.strip()
]
DETECT_CONF = float(os.environ.get("DETECT_CONF", "0.25"))
GDINO_TEXT_THRESHOLD = float(os.environ.get("GDINO_TEXT_THRESHOLD", "0.25"))
GDINO_FP16 = os.environ.get("GDINO_FP16", "1") != "0"
GDINO_TRT = os.environ.get("GDINO_TRT", "0") != "0"
# Precision the TensorRT engine is compiled at: fp16 (default when
# GDINO_FP16=1), bf16, or fp32. bf16 has fp32's exponent range, so if fp16
# logits come out NaN the engine is overflowing fp16's 65504 max and bf16
# keeps the speed without the overflow.
GDINO_TRT_DTYPE = os.environ.get(
    "GDINO_TRT_DTYPE", "fp16" if GDINO_FP16 else "fp32"
).lower()
GDINO_TRT_COMPARE = max(0, int(os.environ.get("GDINO_TRT_COMPARE", "8")))
# Comma-separated aten ops to keep in eager PyTorch instead of TensorRT
# (bisection knob for localizing a NaN/wrong-output op, e.g.
# "aten._softmax.default,aten.grid_sampler_2d.default").
GDINO_TRT_TORCH_OPS = {
    op.strip() for op in os.environ.get("GDINO_TRT_TORCH_OPS", "").split(",") if op.strip()
}
GDINO_DEBUG = os.environ.get("GDINO_DEBUG", "0") != "0"
DETECT_EVERY = max(1, int(os.environ.get("DETECT_EVERY", "1")))
FILL_MODE = os.environ.get("FILL_MODE", "black")  # black|white|blur|debug
BOX_DILATE = float(os.environ.get("BOX_DILATE", "0.1"))
MAX_BOX_AREA = float(os.environ.get("MAX_BOX_AREA", "0.6"))
MIN_BOX_AREA = float(os.environ.get("MIN_BOX_AREA", "0.0"))
MIN_BOX_WIDTH = max(0.0, float(os.environ.get("MIN_BOX_WIDTH", "0")))
MIN_BOX_HEIGHT = max(0.0, float(os.environ.get("MIN_BOX_HEIGHT", "0")))
MIN_HITS = max(1, int(os.environ.get("MIN_HITS", "2")))
TRACK_TTL = max(0, int(os.environ.get("TRACK_TTL", "5")))
TRACK_SMOOTH = float(os.environ.get("TRACK_SMOOTH", "0.6"))
BOTTOM_COVERAGE_THRESHOLD = float(os.environ.get("BOTTOM_COVERAGE_THRESHOLD", "0.8"))
FULLSCREEN_AD_CLASSIFIER = os.environ.get(
    "FULLSCREEN_AD_CLASSIFIER", "openai/clip-vit-base-patch32"
).strip()
FULLSCREEN_AD_CONF = float(os.environ.get("FULLSCREEN_AD_CONF", "0.70"))
FULLSCREEN_AD_EVERY = max(1, int(os.environ.get("FULLSCREEN_AD_EVERY", "1")))
JPEG_QUALITY = max(1, min(100, int(os.environ.get("JPEG_QUALITY", "80"))))
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "jpeg").strip().lower()
if OUTPUT_FORMAT not in ("jpeg", "webp"):
    raise SystemExit("Unknown OUTPUT_FORMAT (expected 'jpeg' or 'webp')")
# ByteTrack ids are per-model; offset each model's ids by this stride so
# tracks from different fine-tuned checkpoints can't collide in the shared
# track state.
_TRACK_ID_STRIDE = 1_000_000


def _yolo_specs() -> list:
    """Active detector spec list.

    YOLO_MODEL selects a single detector (Grounding DINO, YOLO-World, or one
    fine-tuned checkpoint), as before. Unset, the default is the billboard +
    brand-eye fine-tuned pair; each is disabled by setting its knob to
    "off"/"0"/"".
    """
    if YOLO_MODEL:
        return [YOLO_MODEL]
    specs = [
        s
        for s in (BILLBOARD_YOLO, BRAND_EYE_YOLO)
        if s.strip().lower() not in ("", "0", "off", "none")
    ]
    if not specs:
        raise SystemExit(
            "No detector enabled: set YOLO_MODEL, BILLBOARD_YOLO, or BRAND_EYE_YOLO"
        )
    return specs


def _resolve_yolo_spec(spec: str) -> str:
    """Resolve a model spec to something ultralytics.YOLO() can load.

    Local paths and plain ultralytics model names (e.g. yolov8s-worldv2.pt,
    auto-downloaded by ultralytics) pass through; owner/repo[:file] is a
    Hugging Face hub checkpoint downloaded via huggingface_hub.
    """
    if os.path.exists(spec) or "/" not in spec:
        return spec
    from huggingface_hub import hf_hub_download, list_repo_files

    repo, _, filename = spec.partition(":")
    if not filename:
        candidates = [f for f in list_repo_files(repo) if f.endswith(".pt")]
        if len(candidates) != 1:
            raise SystemExit(
                f"{repo} has multiple checkpoints; specify one as "
                f"'{repo}:<file>' — one of: {', '.join(candidates)}"
            )
        filename = candidates[0]
    return hf_hub_download(repo, filename)


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


def _patch_views_in_module(module, tag):
    """Rewrite .view( as .reshape( in every module forward defined in `module`.

    The elided-contiguous mis-trace hits views inside several HF module
    forwards (SwinSelfAttention's head merge, SwinLayer's un-pad slice +
    flatten, GroundingDinoTextCrossAttention's head merge, ...). Those views
    sit mid-forward, so rewrite each forward's source with .view( ->
    .reshape( and rebind it. A reshape of already-contiguous data is free,
    so eager is numerically unaffected; doing it by source rewrite keeps it
    working across transformers versions.
    """
    import inspect
    import textwrap

    for cls in vars(module).values():
        if not (isinstance(cls, type) and cls.__module__ == module.__name__):
            continue
        forward = cls.__dict__.get("forward")
        if forward is None:
            continue
        try:
            src = textwrap.dedent(inspect.getsource(forward))
            if ".view(" not in src:
                continue
            # Drop decorator lines (e.g. transformers' @auto_docstring):
            # re-executing them against a plain function can fail, and the
            # class already carries the decorated metadata.
            src = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith("@")
            )
            ns = {}
            exec(
                compile(src.replace(".view(", ".reshape("), f"<{tag}-trt-patch>", "exec"),
                module.__dict__,
                ns,
            )
            cls.forward = ns["forward"]
        except Exception:
            pass  # source unavailable/unpatchable; TRT falls back to eager if it fails


def _patch_swin_for_trt():
    """Work around dynamo/Torch-TensorRT mis-traces of HF Swin's reshapes.

    window_partition/window_reverse and several module forwards
    (SwinSelfAttention's head merge, SwinLayer's un-pad slice + flatten, ...)
    do ``permute/slice(...).contiguous().view(...)``; under torch.compile the
    tracer can drop the contiguous() and then fail the view on the
    still-strided tensor ("view size is not compatible with input tensor's
    size and stride"). Rewriting the .view calls as .reshape forces a copy
    when the strides require one. Numerically identical in eager (a reshape
    of contiguous data is free), so the patch is safe to leave applied.
    """
    import transformers.models.swin.modeling_swin as swin

    def window_partition(input_feature, window_size):
        batch_size, height, width, num_channels = input_feature.shape
        input_feature = input_feature.view(
            batch_size,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            num_channels,
        )
        return input_feature.permute(0, 1, 3, 2, 4, 5).reshape(
            -1, window_size, window_size, num_channels
        )

    def window_reverse(windows, window_size, height, width):
        num_channels = windows.shape[-1]
        windows = windows.view(
            -1, height // window_size, width // window_size, window_size, window_size, num_channels
        )
        return windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, height, width, num_channels)

    swin.window_partition = window_partition
    swin.window_reverse = window_reverse

    _patch_views_in_module(swin, "swin")


def _patch_grounding_dino_for_trt():
    """Same mis-trace in Grounding DINO's own attention modules.

    GroundingDinoTextCrossAttention (and the other attention classes in the
    model file) do ``context_layer.permute(0, 2, 1, 3).contiguous().view(...)``
    to merge attention heads; under AOT export the contiguous() can be elided
    and the view then fails on the still-strided tensor, aborting the whole
    TRT compilation.
    """
    import transformers.models.grounding_dino.modeling_grounding_dino as gdino

    _patch_views_in_module(gdino, "gdino")


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

        specs = _yolo_specs()
        # Zero-shot detectors are only reachable as a single-model override.
        single = specs[0] if len(specs) == 1 else ""
        model_name = os.path.basename(single).lower()
        self.is_gdino = "grounding" in model_name
        self.is_world = not self.is_gdino and "world" in model_name
        if len(specs) > 1 and any(
            "grounding" in os.path.basename(s).lower() or "world" in os.path.basename(s).lower()
            for s in specs
        ):
            raise SystemExit(
                "Grounding DINO / YOLO-World are zero-shot single detectors; "
                "select one with YOLO_MODEL=... instead of the multi-model knobs"
            )
        self.classes = list(DETECT_CLASSES)
        self.models = []  # [(tag, YOLO)] for world/fine-tuned; unused by gdino
        if self.is_gdino:
            print(f"[detect] loading {single}...", flush=True)
            self._init_gdino(device, single)
        elif self.is_world:
            from ultralytics import YOLOWorld

            print(f"[detect] loading {single}...", flush=True)
            model = YOLOWorld(_resolve_yolo_spec(single))
            model.set_classes(self.classes)
            self.models.append(("world", model))
            print(f"[detect] zero-shot classes: {self.classes}", flush=True)
        else:
            from ultralytics import YOLO

            for spec in specs:
                print(f"[detect] loading {spec}...", flush=True)
                model = YOLO(_resolve_yolo_spec(spec))
                tag = os.path.splitext(os.path.basename(spec.partition(":")[2] or spec))[0]
                self.models.append((tag, model))
                print(f"[detect] {tag} classes: {list(model.names.values())}", flush=True)

        # track_id -> {"box": np.array([x1,y1,x2,y2]), "conf", "label",
        #              "last_seen", "hits"}
        self._tracks = {}
        self._synthetic_id = -1  # fallback ids if the tracker returns none
        self._fullscreen_active = False
        self._fullscreen_classifier = None
        self._fullscreen_classifier_failed = False
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
        self._warm_shape = (warm_h, warm_w)
        self._shape_logged = True  # suppress the frame-size log for the warmup
        self._detect(np.zeros((warm_h, warm_w, 3), dtype=np.uint8))
        self._shape_logged = False  # ... then log it for the first real frame

    def _init_gdino(self, device, model_id):
        import warnings

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        # transformers builds Grounding DINO's deformable-attention op with
        # torch.jit.script, which now emits a FutureWarning on every load.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*torch.jit.script is deprecated.*", category=FutureWarning
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            # Weights stay in fp32; fp16 happens via autocast at inference
            # time so fp32-only ops don't hit float/half mismatches.
            self._gdino_amp = device.type == "cuda" and GDINO_FP16
            self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.gdino.eval()
        if GDINO_TRT and device.type == "cuda":
            _patch_swin_for_trt()
            _patch_grounding_dino_for_trt()
            self._gdino_forward = self._trt_wrap(self.gdino)
            self._trt_checks = GDINO_TRT_COMPARE
        else:
            self._gdino_forward = self.gdino
            self._trt_checks = 0
        self._iou_tracker = _IoUTracker()
        mode = "fp16 autocast" if self._gdino_amp else "fp32"
        print(
            f"[detect] Grounding DINO zero-shot ({mode}) classes: {self.classes}",
            flush=True,
        )

    @staticmethod
    def _trt_wrap(model):
        """Wrap the Grounding DINO forward in torch.compile(backend="torch_tensorrt").

        Compilation is lazy (first call), so the wrapper catches a failure
        from any compiled call and permanently falls back to the eager model —
        a bad graph can never kill the stream. Same policy as sana_pipeline's
        TENSORRT path.
        """
        import contextlib
        import warnings

        import torch

        try:
            # The import registers "torch_tensorrt" as a torch.compile
            # backend; without it torch.compile raises InvalidBackend. It
            # also prints "Unable to import quantiz(e/ation) op... modelopt"
            # noise that only matters when compiling quantized models.
            with contextlib.redirect_stdout(io.StringIO()):
                import torch_tensorrt  # noqa: F401
        except ImportError:
            print(
                "[tensorrt] gdino: torch-tensorrt not installed; staying eager "
                "(install a torch-tensorrt build matching your torch/CUDA to enable)",
                flush=True,
            )
            return model
        # HF Swin and Grounding DINO attention do
        # permute(...).contiguous().view(...) in several places
        # (window_partition/reverse, the self-attention output merge, ...).
        # Dynamo's fake-tensor stride propagation can wrongly conclude the
        # permuted tensor is already contiguous, elide contiguous() from the
        # traced graph, and then fail the view downstream ("Cannot view a
        # tensor with shape ... and strides ..."). Propagating real tensors
        # during tracing computes correct strides so contiguous() survives.
        # Tracing gets slower, but it happens once, on the warmup frame.
        try:
            from torch._functorch import config as functorch_config

            if hasattr(functorch_config, "fake_tensor_propagate_real_tensors"):
                functorch_config.fake_tensor_propagate_real_tensors = True
            if hasattr(functorch_config, "generate_fake_kernels_from_real_mismatches"):
                functorch_config.generate_fake_kernels_from_real_mismatches = True
        except ImportError:
            pass
        # Reduced precision is done by the compiler, not by tracing under
        # autocast: an autocast-traced graph arrives with explicit fp16 casts
        # and compiles to an engine that emits NaN logits (scores collapse vs
        # eager). Tracing a clean fp32 graph and letting Torch-TensorRT pick
        # the low-precision kernels keeps fp32 accumulation where it matters.
        # The option names changed at torch_tensorrt 2.15: enabled_precisions
        # is silently ignored there (pytorch/TensorRT#5030) in favor of
        # rule-based autocast, so detect which API this install speaks.
        options = {}
        low_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(GDINO_TRT_DTYPE)
        if low_dtype is not None:
            try:
                import inspect

                from torch_tensorrt.dynamo import compile as trt_dynamo_compile

                params = inspect.signature(trt_dynamo_compile).parameters
            except Exception:
                params = {}
            if "enable_autocast" in params:
                options["enable_autocast"] = True
                options["autocast_low_precision_type"] = low_dtype
                if "use_fp32_acc" in params:
                    # fp32 accumulation inside fp16 matmuls; cheap insurance
                    # against accumulation-error NaNs.
                    options["use_fp32_acc"] = True
                if "decompose_attention" in params:
                    # TRT's fused fp16 attention kernel returns NaN on
                    # fully-masked rows (pytorch/TensorRT#5011) — Grounding
                    # DINO's text padding mask produces exactly those.
                    # Decomposing attention into matmul/softmax ops routes
                    # around the buggy kernel and, with use_fp32_acc, runs
                    # the softmax in fp32.
                    options["decompose_attention"] = True
            else:
                options["enabled_precisions"] = {low_dtype}
        if GDINO_TRT_TORCH_OPS:
            options["torch_executed_ops"] = set(GDINO_TRT_TORCH_OPS)
        compiled = torch.compile(
            model, backend="torch_tensorrt", dynamic=False, options=options
        )

        def wrapper(*args, **kwargs):
            # `failed` is an attribute (not a closure flag) so the eager/TRT
            # score cross-check in _detect_gdino can also force the fallback.
            if wrapper.failed:
                return model(*args, **kwargs)
            try:
                # Compilation happens lazily inside this call. torch_tensorrt's
                # dynamo backend builds a RuntimeSettings object that only does
                # anything on TRT-RTX builds and warns about it otherwise.
                # Autocast is disabled so the traced graph is pure fp32; the
                # engine's own reduced precision (the options above) handles
                # the speedup. The eager fallback below still runs under
                # autocast.
                with warnings.catch_warnings(), torch.autocast("cuda", enabled=False):
                    warnings.filterwarnings(
                        "ignore", message="RuntimeSettings is only honored.*"
                    )
                    return compiled(*args, **kwargs)
            except Exception as e:
                wrapper.failed = True
                print(
                    f"[tensorrt] gdino failed under TensorRT ({e}); "
                    "eager for the rest of the session",
                    flush=True,
                )
                return model(*args, **kwargs)

        wrapper.failed = False

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
            self.models[0][1].set_classes(classes)
        self.reset()
        print(f"[detect] classes updated: {classes}", flush=True)

    def reset(self):
        """Clear tracker + box state (called when the producer disconnects)."""
        self._tracks = {}
        self._fullscreen_active = False
        self._frames = 0
        if self.is_gdino:
            self._iou_tracker.reset()
        else:
            # Drop the predictors so ByteTrack starts from a clean slate.
            for _, model in self.models:
                model.predictor = None

    def _init_fullscreen_classifier(self):
        if FULLSCREEN_AD_CLASSIFIER.lower() in ("", "0", "off", "none"):
            self._fullscreen_classifier_failed = True
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._fullscreen_processor = CLIPProcessor.from_pretrained(
                FULLSCREEN_AD_CLASSIFIER
            )
            self._fullscreen_classifier = CLIPModel.from_pretrained(
                FULLSCREEN_AD_CLASSIFIER
            ).to(self.device)
            self._fullscreen_classifier.eval()
            print(
                f"[detect] full-screen ad classifier: {FULLSCREEN_AD_CLASSIFIER}",
                flush=True,
            )
        except Exception as exc:
            self._fullscreen_classifier_failed = True
            print(
                f"[detect] full-screen ad classifier unavailable; continuing: {exc}",
                flush=True,
            )

    def _classify_fullscreen_ad(self, frame: np.ndarray):
        if self._fullscreen_classifier_failed:
            return False, 0.0
        if self._fullscreen_classifier is None:
            self._init_fullscreen_classifier()
        if self._fullscreen_classifier is None:
            return False, 0.0

        import torch

        try:
            image = Image.fromarray(frame)
            prompts = [
                "a full-screen advertisement or sponsored video",
                "normal video content without a full-screen advertisement",
            ]
            inputs = self._fullscreen_processor(
                text=prompts,
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.inference_mode():
                logits = self._fullscreen_classifier(**inputs).logits_per_image
                probability = logits.softmax(dim=1)[0, 0].item()
            return probability >= FULLSCREEN_AD_CONF, probability
        except Exception as exc:
            self._fullscreen_classifier_failed = True
            print(
                f"[detect] full-screen ad classification failed; disabling: {exc}",
                flush=True,
            )
            return False, 0.0

    def _set_fullscreen_ad(self, frame_shape, conf):
        h, w = frame_shape[:2]
        self._tracks = {
            -1: {
                "box": np.array([0.0, 0.0, float(w), float(h)]),
                "conf": conf,
                "label": "fullscreen advertisement",
                "last_seen": self._frames,
                "hits": max(MIN_HITS, 1),
            }
        }
        self._fullscreen_active = True

    def _detect(self, frame: np.ndarray):
        if self.is_gdino:
            return self._detect_gdino(frame)
        detections = []
        multi = len(self.models) > 1
        for m_idx, (tag, model) in enumerate(self.models):
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=DETECT_CONF,
                device=self.ul_device,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is None or not len(boxes):
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            names = results[0].names
            for i, ((x1, y1, x2, y2), conf, cls) in enumerate(zip(xyxy, confs, clss)):
                # Offset ByteTrack ids per model so tracks from different
                # checkpoints can't collide in the shared track state.
                track_id = int(ids[i]) + m_idx * _TRACK_ID_STRIDE if ids is not None else None
                # With several models, tag the label so FILL_MODE=debug shows
                # which checkpoint fired.
                label = f"{tag}:{names[cls]}" if multi else names[cls]
                detections.append((track_id, x1, y1, x2, y2, float(conf), label))
        return detections

    def _detect_gdino(self, frame: np.ndarray):
        import torch

        if not self._shape_logged:
            self._shape_logged = True
            h, w = frame.shape[:2]
            note = ""
            if GDINO_TRT and (h, w) != self._warm_shape:
                wh, ww = self._warm_shape
                note = (
                    f" — differs from the {ww}x{wh} warmup, so the first frame "
                    f"recompiles; set FRAME_WIDTH={w} FRAME_HEIGHT={h} to compile "
                    "at the right size"
                )
            print(f"[detect] incoming frame: {w}x{h}{note}", flush=True)

        image = Image.fromarray(frame)
        # Grounding DINO prompt format: lowercase phrases, period-separated.
        text = ". ".join(c.lower() for c in self.classes) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.inference_mode(), torch.autocast(
            amp_device, dtype=torch.float16, enabled=self._gdino_amp
        ):
            outputs = self._gdino_forward(**inputs)
            outputs = self._trt_cross_check(inputs, outputs)
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
        if GDINO_DEBUG:
            top = sorted(zip(scores, labels), reverse=True)[:3]
            top_str = ", ".join(f"{label} {float(score):.3f}" for score, label in top)
            print(
                f"[detect] raw: {len(boxes)} box(es)"
                + (f" (top: {top_str})" if top else ""),
                flush=True,
            )
        ids = self._iou_tracker.update(boxes)
        return [
            (track_id, x1, y1, x2, y2, float(conf), label)
            for track_id, (x1, y1, x2, y2), conf, label in zip(ids, boxes, scores, labels)
        ]

    def _trt_cross_check(self, inputs, outputs):
        """Compare TRT-compiled logits against an eager run on the same frame.

        Runs for the first GDINO_TRT_COMPARE frames while the compiled path
        is active. A silently-degraded engine (top score collapsed or NaN
        while eager still clears the threshold) is logged and swapped back to
        eager for the rest of the session.
        """
        if not self._trt_checks or getattr(self._gdino_forward, "failed", True):
            return outputs
        import torch

        self._trt_checks -= 1
        eager_outputs = self.gdino(**inputs)
        if not (hasattr(outputs, "logits") and hasattr(eager_outputs, "logits")):
            self._trt_checks = 0  # unexpected output structure; stop checking
            return outputs
        trt_top = outputs.logits.sigmoid().max().item()
        eager_top = eager_outputs.logits.sigmoid().max().item()
        logit_diff = (outputs.logits - eager_outputs.logits).abs().max().item()
        print(
            f"[tensorrt] gdino cross-check: top score trt={trt_top:.3f} "
            f"eager={eager_top:.3f}, max logit diff {logit_diff:.3f}",
            flush=True,
        )
        # When the engine diverges, say WHERE: scan every tensor field of the
        # model output and name the ones where the compiled output has NaNs
        # the eager output doesn't. That names the failing stage — e.g.
        # encoder_last_hidden_state_vision = Swin backbone,
        # encoder_last_hidden_state_text = BERT, last_hidden_state /
        # intermediate_hidden_states = decoder (deformable attention),
        # logits only = the contrastive class head.
        nan_fields = [
            k
            for k in eager_outputs
            if torch.is_tensor(eager_outputs[k])
            and torch.is_tensor(outputs[k])
            and eager_outputs[k].is_floating_point()
            and outputs[k].shape == eager_outputs[k].shape
            and outputs[k].isnan().sum().item() > eager_outputs[k].isnan().sum().item()
        ]
        if nan_fields:
            print(
                f"[tensorrt] gdino: NaN first appears in: {', '.join(nan_fields)}",
                flush=True,
            )
        # `not trt_top >= ...` (rather than `trt_top < ...`) so a NaN top
        # score from the compiled engine also counts as divergence.
        if eager_top >= DETECT_CONF and not trt_top >= eager_top * 0.5:
            print(
                "[tensorrt] gdino: TensorRT output diverges from eager "
                f"(trt top {trt_top:.3f} vs eager {eager_top:.3f}) — eager for "
                "the rest of the session. To keep TensorRT, try "
                "GDINO_TRT_DTYPE=bf16 (same speed, fp32 exponent range) or "
                "GDINO_TRT_DTYPE=fp32.",
                flush=True,
            )
            self._gdino_forward.failed = True
            self._trt_checks = 0
            return eager_outputs
        if not self._trt_checks:
            print(
                f"[tensorrt] gdino cross-check: {GDINO_TRT_COMPARE} frames "
                "matched eager; running TensorRT only from here on",
                flush=True,
            )
        return outputs

    def _update_tracks(self, detections, frame_shape):
        """Fuse one frame's raw detections into the persistent track state."""
        h, w = frame_shape[:2]

        # A detector can mistake a dense lower-screen caption/control area for
        # a cluster of ads. Use the union of raw boxes, not their sum, so
        # overlapping detections do not inflate the false-positive score.
        bottom_mask = np.zeros((h, w), dtype=np.uint8)
        for _, x1, y1, x2, y2, _, _ in detections:
            ix1 = int(max(0, min(w, x1)))
            iy1 = int(max(h // 2, min(h, y1)))
            ix2 = int(max(0, min(w, x2)))
            iy2 = int(max(h // 2, min(h, y2)))
            if ix2 > ix1 and iy2 > iy1:
                bottom_mask[iy1:iy2, ix1:ix2] = 1
        bottom_area = w * (h - h // 2)
        bottom_coverage = float(bottom_mask.sum()) / max(bottom_area, 1)
        if bottom_coverage >= BOTTOM_COVERAGE_THRESHOLD:
            self._tracks = {}
            if self.is_gdino:
                self._iou_tracker.reset()
            return

        for track_id, x1, y1, x2, y2, conf, label in detections:
            box_w, box_h = x2 - x1, y2 - y1
            box_area = box_w * box_h
            if box_area > MAX_BOX_AREA * w * h:
                continue  # whole-frame false positive; would black out the screen
            if box_area < MIN_BOX_AREA * w * h:
                continue
            if box_w < MIN_BOX_WIDTH or box_h < MIN_BOX_HEIGHT:
                continue
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
        """Detect logos/ads in one JPEG/WebP frame and return an encoded frame."""
        frame = np.array(Image.open(io.BytesIO(data)).convert("RGB"))  # writable copy
        self._frames += 1

        # Binary full-screen-ad classification runs before the object detector.
        # A positive result becomes an immediate whole-screen box and skips the
        # rest of this pipeline until the classifier clears it.
        classify_fullscreen = (
            (self._frames - 1) % FULLSCREEN_AD_EVERY == 0
            and FULLSCREEN_AD_CLASSIFIER.lower() not in ("", "0", "off", "none")
        )
        if classify_fullscreen:
            is_fullscreen, fullscreen_conf = self._classify_fullscreen_ad(frame)
            if is_fullscreen:
                self._set_fullscreen_ad(frame.shape, fullscreen_conf)
            elif self._fullscreen_active:
                self._fullscreen_active = False
                self._tracks = {}

        # _frames is 1-based, so detect on frames 1, 1+N, 1+2N, ... (with
        # DETECT_EVERY=1 that is every frame). Also re-detect immediately
        # whenever no tracks are known, so new ads are boxed without waiting
        # for the next scheduled detection frame. An active full-screen ad
        # bypasses object detection and tracking entirely.
        if not self._fullscreen_active and (
            (self._frames - 1) % DETECT_EVERY == 0 or not self._tracks
        ):
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
        if OUTPUT_FORMAT == "webp":
            Image.fromarray(out).save(buf, format="WEBP", quality=JPEG_QUALITY, method=4)
        else:
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
