# Model Notes — YOLOv8n Detector

## Summary

| Property | Value |
|----------|-------|
| Model name | **YOLOv8n** (`yolov8n.pt`) — Ultralytics YOLOv8 "nano" |
| Task | Object detection (single-stage, anchor-free) |
| Pretraining | **COCO** (80 classes), pretrained weights — no training required |
| Weights file | `assets/yolov8n.pt` |
| Weights size | **6,549,796 bytes (≈ 6.25 MB)** |
| Input size | **640 px** (`imgsz=640`, square; configurable in `config.VisionConfig.inference_size`) |
| Number of classes | **80** (COCO) |
| Confidence threshold | **0.45** (default, `config.VisionConfig.confidence`) |
| IoU / NMS threshold | **0.45** (default, `config.VisionConfig.iou`) |
| Device | **CPU** by default; **CUDA GPU** auto-selected when `torch.cuda.is_available()` |
| Runtime | Fully **offline** after the one-time weight download |

Relevant COCO class ids used by the app: `person = 0`, `bottle = 39`, `chair = 56`
(the app's default *indoor-navigation* whitelist keeps person/car/dog **plus**
chair, bottle, couch, table, tv, laptop, phone, etc. — see
`config.DEFAULT_CLASS_WHITELIST`; pass `--all-classes` for all 80 COCO classes).

## The 80 COCO classes

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow,
elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,
skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard,
surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana,
apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell
phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush

## Why YOLOv8n?

* **Small and fast on CPU.** The "n" (nano) variant is the smallest YOLOv8 model
  (~6 MB, fewest parameters). It is designed to run in real time on CPU-only and
  low-power hardware — exactly the constraint for a portable assistive device.
* **No training needed.** The published weights are already trained on COCO, so the
  detector works out of the box with no dataset or GPU cluster.
* **Good accuracy for its size.** Nano trades some mAP for speed versus the
  `s`/`m`/`l` variants, but is more than accurate enough for naming everyday objects
  (person, chair, bottle, car, dog, …) in a live video stream.
* **Ultralytics ecosystem.** One-line load/inference, active maintenance, MIT-licensed
  tooling, and easy to swap to a larger model later by only changing the weights path.

Trade-off: nano has the lowest accuracy of the YOLOv8 family and can miss small /
heavily occluded objects or occasionally mislabel cluttered scenes. If more accuracy
is needed (at the cost of speed), point `config.model_path` at `yolov8s.pt` /
`yolov8m.pt` — no code changes required.

## How it runs in BlindAssist

`blindassist/detector.py` loads the checkpoint **once** (`Detector.load()`), picks
the device, then `Detector.detect(bgr_frame)` runs `model.predict(...)` with the
configured `conf`, `iou`, `imgsz` and `device`, and converts each result into a plain
`Detection(label, confidence, x1, y1, x2, y2)`.

## Validation (real run)

`scripts/validate_detector.py` downloaded three free sample images, ran the detector,
saved annotated copies to `Evidence/model_validation/`, and printed the detections.
Result on this machine (CPU):

| Sample | Source | Expected | Detected (conf) | Result |
|--------|--------|----------|-----------------|--------|
| person | ultralytics `bus.jpg` | person | bus 0.87, **person 0.87**, **person 0.85**, **person 0.83** | PASS |
| chair  | COCO `000000000139.jpg` | chair | tv 0.93, **chair 0.85**, person 0.71, **chair 0.52** | PASS |
| bottle | COCO `000000000632.jpg` | bottle | bed 0.91, potted plant 0.88/0.77, **bottle 0.60**, chair 0.57 | PASS |

**3/3 expected classes detected.** Annotated evidence:
`Evidence/model_validation/person.jpg`, `chair.jpg`, `bottle.jpg`.
