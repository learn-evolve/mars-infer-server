#!/usr/bin/env python3
"""Mars inference server — runs on Ubuntu 4090.

Endpoints:
  POST /infer/gdino_sam2 — Grounding DINO detection + SAM2 segmentation (one round-trip)
  GET  /health           — liveness check

All image inputs are multipart/form-data (binary, no base64 overhead).
Models are loaded once at startup and kept resident in GPU memory.
"""
from __future__ import annotations
import json
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

GDINO_MODEL_ID = os.environ.get("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
SAM2_MODEL_ID  = os.environ.get("SAM2_MODEL_ID",  "facebook/sam2-hiera-small")
DEVICE         = os.environ.get("INFER_DEVICE",   "cuda:0")
PORT           = int(os.environ.get("INFER_PORT", "8765"))

_gdino_model = None
_gdino_proc  = None
_sam2_model  = None


def _load_models():
    global _gdino_model, _gdino_proc, _sam2_model

    print(f"[server] loading models on {DEVICE} ...")

    t = time.time()
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    _gdino_proc  = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL_ID).to(DEVICE)
    _gdino_model.eval()
    print(f"[server] GDINO loaded in {time.time()-t:.1f}s")

    t = time.time()
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _sam2_model = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID, device=DEVICE)
    print(f"[server] SAM2 loaded in {time.time()-t:.1f}s")

    print("[server] all models ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="Mars Inference Server", lifespan=lifespan)


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    return img


def _mask_to_polygon(mask_u8: np.ndarray, max_points: int = 32) -> tuple[list, int]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], int(mask_u8.sum())
    cnt = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(cnt))
    eps = max(0.5, 0.005 * cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) > max_points:
        idx = np.linspace(0, len(approx) - 1, max_points).astype(int)
        approx = approx[idx]
    return [[int(x), int(y)] for x, y in approx], area


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models": {
            "gdino": _gdino_model is not None,
            "sam2":  _sam2_model  is not None,
        },
    }


@app.post("/infer/gdino_sam2")
async def infer_gdino_sam2(
    image:          UploadFile = File(...),
    text_prompt:    str   = Form(...),
    box_threshold:  float = Form(0.30),
    text_threshold: float = Form(0.25),
    max_detections: int   = Form(10),
):
    if _gdino_model is None or _sam2_model is None:
        raise HTTPException(503, "Models not loaded")

    img_bgr = _decode_image(await image.read())
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_rgb)

    # 1) Grounding DINO
    inputs = _gdino_proc(images=pil_img, text=text_prompt, return_tensors="pt").to(DEVICE)
    t = time.time()
    with torch.no_grad():
        outputs = _gdino_model(**inputs)
    gdino_ms = round((time.time() - t) * 1000, 1)

    results = _gdino_proc.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        box_threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    boxes  = results["boxes"].cpu().numpy().astype(int)

    detections = []
    for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
        if i >= max_detections:
            break
        detections.append({
            "id":         i,
            "label":      label,
            "confidence": round(float(score), 4),
            "bbox_xyxy":  box.tolist(),
        })

    # 2) SAM2
    t = time.time()
    with torch.inference_mode():
        _sam2_model.set_image(img_rgb)
        for det in detections:
            box_np = np.array(det["bbox_xyxy"], dtype=np.float32)
            masks, seg_scores, _ = _sam2_model.predict(box=box_np, multimask_output=False)
            mask = masks[0].astype(np.uint8)
            poly, area = _mask_to_polygon(mask)
            det["mask_polygon"] = poly
            det["mask_area_px"] = area
            det["iou_score"]    = float(seg_scores[0])
    sam2_ms = round((time.time() - t) * 1000, 1)

    return JSONResponse({
        "ok":           True,
        "gdino_ms":     gdino_ms,
        "sam2_ms":      sam2_ms,
        "inference_ms": gdino_ms + sam2_ms,
        "image_shape":  [h, w],
        "n_detections": len(detections),
        "detections":   detections,
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
