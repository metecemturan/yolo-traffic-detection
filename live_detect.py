#!/usr/bin/env python3
"""
live_detect_auto_save.py

- Default placeholders (no required CLI params):
  models: v5.pt v8.pt v11.pt
  source: video.mp4

- Behavior:
  * Shows scaled window that fits the screen (press 'f' to toggle fullscreen, ESC to exit).
  * Draws detections with white & black readable labels.
  * Puts FPS counter on the top-left (readable).
  * Automatically saves:
      - annotated video -> runs/auto/YYYYMMDD_HHMMSS/annotated_output.mp4
      - detections jsonl -> runs/auto/.../detections.jsonl
      - stats json -> runs/auto/.../stats.json (contains avg fps, models, device, per-model avg times, runtime, frames)
  * Also prints out the saved paths when finished.
  * Uses a capture thread + optional pre-resize + optional half precision when on CUDA.

Usage examples:
  - Defaults (will try placeholders):
      python live_detect_auto_save.py
  - With explicit models/source:
      python live_detect_auto_save.py -m v8.pt -s video.mp4 -d cuda:0 --half --resize 640

Notes:
  - For realtime target, prefer single model & use --half on CUDA.
  - If models are not present, script exits with an error message.
"""
from __future__ import annotations
import argparse
import os
import time
import json
import platform
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import threading
import queue
from datetime import datetime

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

# ---------------- Utilities ----------------
def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def get_screen_size() -> Tuple[int, int]:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1366, 768

def compute_display_size(frame_w: int, frame_h: int, screen_w: int, screen_h: int, margin: int = 120) -> Tuple[int,int,float]:
    max_w = max(100, screen_w - margin)
    max_h = max(100, screen_h - margin)
    scale = min(max_w / frame_w, max_h / frame_h, 1.0)
    dw = max(1, int(frame_w * scale))
    dh = max(1, int(frame_h * scale))
    return dw, dh, scale

# ---------------- Capture thread ----------------
class VideoCaptureThread(threading.Thread):
    def __init__(self, src, qmax=4):
        super().__init__(daemon=True)
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=qmax)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                try:
                    self.q.put_nowait(None)
                except Exception:
                    pass
                break
            try:
                self.q.put(frame, timeout=0.05)
            except queue.Full:
                try:
                    _ = self.q.get_nowait()  # drop oldest
                    self.q.put_nowait(frame)
                except Exception:
                    pass

    def read(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

# ---------------- Model wrapper ----------------
class ModelWrapper:
    def __init__(self, path: str, device: Optional[str] = None, conf: float = 0.25, imgsz: int = 640, half: bool = False):
        self.path = path
        self.basename = os.path.basename(path)
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.conf = conf
        self.imgsz = int(imgsz)
        self.half = half and self.device.startswith("cuda")
        self.backend = None
        self.model = None
        self.names: Dict[int, str] = {}
        self._load()

    def _load(self):
        # Try ultralytics first
        try:
            from ultralytics import YOLO  # type: ignore
            self.backend = "ultralytics"
            self.model = YOLO(self.path)
            try:
                self.model.to(self.device)
            except Exception:
                pass
            if self.half:
                try:
                    if hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                        self.model.model.half()
                except Exception:
                    pass
            names = getattr(self.model, "names", None)
            if isinstance(names, (list, tuple)):
                self.names = {i: n for i, n in enumerate(names)}
            elif isinstance(names, dict):
                self.names = names
            print(f"[INFO] Loaded {self.basename} via ultralytics on {self.device} half={self.half}")
            return
        except Exception as e:
            # fallback
            print(f"[WARN] ultralytics load failed for {self.basename}: {e}")

        # Fallback: yolov5 hub
        if torch is not None:
            try:
                self.backend = "yolov5_hub"
                self.model = torch.hub.load("ultralytics/yolov5", "custom", path=self.path, trust_repo=True)
                if self.device.startswith("cuda"):
                    try:
                        self.model.to("cuda")
                    except Exception:
                        pass
                if self.half and hasattr(self.model, "half"):
                    try:
                        self.model.half()
                    except Exception:
                        pass
                self.names = getattr(self.model, "names", {}) or {}
                print(f"[INFO] Loaded {self.basename} via yolov5_hub on {self.device} half={self.half}")
                return
            except Exception as e2:
                print(f"[WARN] yolov5 hub load failed for {self.basename}: {e2}")

        raise RuntimeError(f"Model yüklenemedi: {self.path}.")

    def predict(self, frame: np.ndarray, pre_resize: Optional[int] = None) -> List[Dict[str, Any]]:
        src_h, src_w = frame.shape[:2]
        frame_in = frame
        if pre_resize:
            frame_in = cv2.resize(frame, (pre_resize, pre_resize))
        dets: List[Dict[str, Any]] = []
        if self.backend == "ultralytics":
            try:
                if torch is not None:
                    with torch.inference_mode():
                        results = self.model(frame_in, conf=self.conf, device=self.device, imgsz=self.imgsz)
                else:
                    results = self.model(frame_in, conf=self.conf, imgsz=self.imgsz)
            except Exception:
                results = self.model(frame_in, conf=self.conf)
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                return []
            try:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            except Exception:
                xyxy = np.array(boxes.xyxy)
                cls = np.array(boxes.cls)
                confs = np.array(boxes.conf)
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
                dets.append({"class_id": int(c), "class_name": self.names.get(int(c), str(int(c))), "conf": float(cf), "xyxy": [float(x1), float(y1), float(x2), float(y2)]})
            if pre_resize:
                sx = src_w / pre_resize
                sy = src_h / pre_resize
                for d in dets:
                    x1,y1,x2,y2 = d["xyxy"]
                    d["xyxy"] = [x1*sx, y1*sy, x2*sx, y2*sy]
            return dets

        elif self.backend == "yolov5_hub":
            try:
                with torch.inference_mode():
                    results = self.model(frame_in, size=self.imgsz)
            except Exception:
                results = self.model(frame_in)
            try:
                xy = results.xyxy[0].cpu().numpy()
            except Exception:
                xy = np.array(results.xyxy[0])
            for *box, conf, cls in xy:
                x1,y1,x2,y2 = box
                dets.append({"class_id": int(cls), "class_name": self.names.get(int(cls), str(int(cls))), "conf": float(conf), "xyxy": [float(x1), float(y1), float(x2), float(y2)]})
            if pre_resize:
                sx = src_w / pre_resize
                sy = src_h / pre_resize
                for d in dets:
                    x1,y1,x2,y2 = d["xyxy"]
                    d["xyxy"] = [x1*sx, y1*sy, x2*sx, y2*sy]
            return dets

        else:
            return []

# ---------------- Drawing utilities ----------------
def draw_box_and_label(frame, box, label_text: str, box_color=(0,255,0)):
    x1,y1,x2,y2 = map(int, box)
    # bounding box
    cv2.rectangle(frame, (x1,y1),(x2,y2), box_color, 2)

    # text background: white filled rect with box_color border; black text for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    ft = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, fs, ft)
    pad_x = 6
    pad_y = 4

    bx1 = x1
    by2 = y1
    by1 = by2 - (th + pad_y*2)
    bx2 = x1 + tw + pad_x*2

    if by1 < 0:
        by1 = y2
        by2 = by1 + (th + pad_y*2)

    fh, fw = frame.shape[:2]
    if bx2 > fw:
        bx2 = fw - 1
        bx1 = max(0, bx2 - (tw + pad_x*2))

    # white background
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255,255,255), -1)
    # border with box color
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1)
    # black text
    text_x = bx1 + pad_x
    text_y = by2 - pad_y - baseline
    cv2.putText(frame, label_text, (text_x, text_y), font, fs, (0,0,0), ft, cv2.LINE_AA)

def draw_fps_top_left(frame, fps_value: float):
    """
    Draw small FPS counter on top-left with white background and black text for readability.
    """
    text = f"FPS: {fps_value:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    ft = 2
    (tw, th), baseline = cv2.getTextSize(text, font, fs, ft)
    pad_x = 8
    pad_y = 6
    bx1, by1 = 8, 8
    bx2 = bx1 + tw + pad_x*2
    by2 = by1 + th + pad_y*2
    # white filled rectangle
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255,255,255), -1)
    # black text
    tx = bx1 + pad_x
    ty = by2 - pad_y - baseline
    cv2.putText(frame, text, (tx, ty), font, fs, (0,0,0), ft, cv2.LINE_AA)

# ---------------- Argument parsing ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-m","--models", nargs="+", required=False, help="Model .pt files (default placeholders used if omitted).")
    p.add_argument("-s","--source", default="video.mp4", help="Video source (default: video.mp4).")
    p.add_argument("-d","--device", default=None, help="device e.g. cuda:0 or cpu (auto-detect if omitted).")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="imgsz passed to model")
    p.add_argument("--half", action="store_true", help="use fp16 on CUDA")
    p.add_argument("--resize", type=int, default=0, help="pre-resize frames to this square size for faster inference (0 = none)")
    p.add_argument("--max-frames", type=int, default=0, help="process max frames (0=all)")
    return p.parse_args()

# ---------------- Main ----------------
def main():
    args = parse_args()

    # defaults if user didn't pass
    if not args.models:
        args.models = ["v5.pt", "v8.pt", "v11.pt"]

    # prepare output dir
    out_root = ensure_dir(os.path.join("runs", "auto", now_tag()))
    vid_out_path = os.path.join(out_root, "annotated_output.mp4")
    jsonl_out_path = os.path.join(out_root, "detections.jsonl")
    stats_out_path = os.path.join(out_root, "stats.json")

    # load models
    wrappers: List[ModelWrapper] = []
    for mp in args.models:
        if not os.path.exists(mp):
            print(f"[ERROR] Model not found: {mp}")
            return
        try:
            wrappers.append(ModelWrapper(mp, device=args.device, conf=args.conf, imgsz=args.imgsz, half=args.half))
        except Exception as e:
            print(f"[ERROR] Could not load model {mp}: {e}")
            return

    n_models = len(wrappers)

    # capture thread
    cap_thread = VideoCaptureThread(args.source, qmax=4)
    cap_thread.start()
    time.sleep(0.2)

    # get initial frame
    init_frame = None
    while init_frame is None:
        init_frame = cap_thread.read(timeout=1.0)
        if init_frame is None and not cap_thread.running:
            print("[ERROR] Could not read initial frame.")
            return

    frame_h, frame_w = init_frame.shape[:2]
    screen_w, screen_h = get_screen_size()
    disp_w, disp_h, scale = compute_display_size(frame_w, frame_h, screen_w, screen_h, margin=120)

    window_name = "auto_detect"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)

    # video writer (always save annotated video automatically)
    fps_guess = 25.0
    try:
        # try to read fps from capture if available
        cap = cv2.VideoCapture(args.source if not isinstance(args.source, str) or not args.source.isdigit() else int(args.source))
        fps_guess = cap.get(cv2.CAP_PROP_FPS) or fps_guess
        cap.release()
    except Exception:
        pass

    writer = cv2.VideoWriter(vid_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_guess if fps_guess>0 else 25.0, (frame_w, frame_h))
    jsonl_f = open(jsonl_out_path, "w", encoding="utf-8")

    # statistics accumulators
    total_frames = 0
    start_time = time.time()
    times = deque(maxlen=60)  # per-frame processing times
    per_model_times = defaultdict(float)   # total inference time per model
    per_model_counts = defaultdict(int)    # count per model

    frame = init_frame
    frame_idx = 0
    fullscreen = False

    try:
        while True:
            if frame is None:
                frame = cap_thread.read(timeout=1.0)
            if frame is None:
                if not cap_thread.running:
                    break
                continue

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            t0 = time.time()
            all_entry_models = []
            # run each model sequentially (note: multiple models slows down fps)
            for mi, w in enumerate(wrappers):
                t_m0 = time.time()
                dets = w.predict(frame, pre_resize=(args.resize if args.resize>0 else None))
                t_m1 = time.time()
                per_model_times[w.basename] += (t_m1 - t_m0)
                per_model_counts[w.basename] += 1

                # draw detections for this model (colored by model index)
                color = tuple(int(c) for c in np.array([0,200,0]) if isinstance(c,(int,float)))  # fallback green
                # better color pick
                palette = [(0,200,0),(0,150,255),(200,100,0),(200,0,200)]
                color = palette[mi % len(palette)]
                for d in dets:
                    label = f"{w.basename}:{d.get('class_name',d.get('class_id'))} {d.get('conf',0):.2f}"
                    draw_box_and_label(frame, d["xyxy"], label_text=label, box_color=color)

                # build jsonl entry part
                entry_model = {"model": w.basename, "detections": dets}
                all_entry_models.append(entry_model)

            t1 = time.time()
            proc_time = t1 - t0
            times.append(proc_time)
            total_frames += 1

            avg_proc = sum(times)/len(times) if times else proc_time
            fps_proc = 1.0/avg_proc if avg_proc>0 else 0.0

            # draw FPS at top-left
            draw_fps_top_left(frame, fps_proc)

            # show scaled display copy
            disp_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, disp_frame)

            # save original-size annotated frame to video
            writer.write(frame)

            # save jsonl line for frame
            base_entry = {"frame_idx": frame_idx, "timestamp": time.time(), "width": frame_w, "height": frame_h, "models": all_entry_models}
            jsonl_f.write(json.dumps(base_entry, ensure_ascii=False) + "\n")

            frame = None

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, disp_w, disp_h)

    finally:
        cap_thread.stop()
        writer.release()
        jsonl_f.close()
        cv2.destroyAllWindows()
        end_time = time.time()

        # compile stats
        total_time = end_time - start_time
        avg_fps_overall = total_frames / total_time if total_time > 0 else 0.0
        per_model_avg = {}
        for m in per_model_counts:
            cnt = per_model_counts[m]
            per_model_avg[m] = {
                "calls": cnt,
                "total_time_s": per_model_times[m],
                "avg_time_s": (per_model_times[m] / cnt) if cnt>0 else None,
                "approx_fps": (1.0 / (per_model_times[m] / cnt)) if cnt>0 and per_model_times[m]>0 else None
            }

        stats = {
            "timestamp": now_tag(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": None,
            "device": args.device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"),
            "models": [w.basename for w in wrappers],
            "imgsz": args.imgsz,
            "half": args.half,
            "resize": args.resize,
            "total_frames": total_frames,
            "total_time_s": total_time,
            "avg_fps_overall": avg_fps_overall,
            "running_avg_proc_fps": fps_proc,
            "per_model": per_model_avg
        }
        try:
            if torch is not None:
                stats["torch"] = {
                    "version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count()>0 else None
                }
        except Exception:
            pass

        # write stats file
        with open(stats_out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print("\n[INFO] Finished.")
        print(f"[INFO] Annotated video saved to: {vid_out_path}")
        print(f"[INFO] Detections saved to: {jsonl_out_path}")
        print(f"[INFO] Stats saved to: {stats_out_path}")
        print(f"[INFO] Summary: frames={total_frames}, total_time_s={total_time:.2f}, avg_fps_overall={avg_fps_overall:.2f}")
        # pretty print per-model
        for m, v in stats["per_model"].items():
            print(f" - {m}: calls={v['calls']}, avg_time_s={v['avg_time_s']:.4f}, approx_fps={v['approx_fps']}")

if __name__ == "__main__":
    main()