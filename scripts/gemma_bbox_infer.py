#!/usr/bin/env python3
"""Detect, per keyframe, the bounding box(es) of the object(s) that
`dataset/gemma_result.json` says are held in that video, using Gemma-4 E4B.

Output mirrors the conventions of `dataset/qwen3vl_keyframe_bbox.json`:

  { "meta": {...},
    "results": {
       "<video_base>": {
          "gemma_key": "...mp4",
          "has_object_in_hand": bool,
          "objects": ["brush", ...],
          "keyframe_dir": "<base>___<slug>",
          "frames": {
             "frame_000000.jpg": {
                "path": "dataset/.../frame_000000.jpg",
                "width": 1920, "height": 1080,
                "detections": [
                   {"label": "brush",
                    "bbox_2d_norm1000": [ymin,xmin,ymax,xmax],   # raw Gemma grid (0-1000)
                    "bbox_2d": [x1,y1,x2,y2]},                    # absolute pixels
                   ...
                ]
             }, ...
          }
       }, ...
    }
  }

Frames with no object present -> detections: [] (with raw_output kept for audit).
The run is resumable: an existing output file is loaded and completed frames skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYFRAMES_DIR = os.path.join(REPO, "dataset",
                             "datatang_session2_videos_fps24_012_keyframes")
GEMMA_JSON = os.path.join(REPO, "dataset", "gemma_result.json")
OUT_JSON = os.path.join(REPO, "dataset", "gemma_keyframe_bbox.json")
DEFAULT_CKPT = os.path.join(REPO, ".cache", "checkpoints", "gemma4-e4b-it")

NUM_RE = re.compile(r"-?\d+\.?\d*")


def parse_boxes(text: str):
    """Extract every group of 4 numbers from the model output.

    Gemma emits detection boxes as `[ymin, xmin, ymax, xmax]` normalized to
    0-1000.  Returns a list of 4-number lists in that raw order.
    """
    nums = [float(x) for x in NUM_RE.findall(text)]
    boxes = []
    for i in range(0, len(nums) - 3, 4):
        boxes.append(nums[i:i + 4])
    return boxes


def norm1000_to_abs(box, W, H):
    """[ymin,xmin,ymax,xmax] in 0-1000 -> [x1,y1,x2,y2] absolute pixels."""
    ymin, xmin, ymax, xmax = box
    x1 = xmin / 1000.0 * W
    y1 = ymin / 1000.0 * H
    x2 = xmax / 1000.0 * W
    y2 = ymax / 1000.0 * H
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, W)); x2 = max(0.0, min(x2, W))
    y1 = max(0.0, min(y1, H)); y2 = max(0.0, min(y2, H))
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def box_looks_valid(box):
    """Sanity: all coords within 0-1000 and non-degenerate."""
    if not all(0.0 <= c <= 1000.0 for c in box):
        return False
    ymin, xmin, ymax, xmax = box
    return (ymax - ymin) > 1 and (xmax - xmin) > 1


def build_worklist():
    """Return list of (video_base, gemma_key, has_obj, objects, kf_dir)."""
    g = json.load(open(GEMMA_JSON))
    disk_dirs = set(os.listdir(KEYFRAMES_DIR))
    work = []
    for key, v in g.items():
        objs = [o["name"] for o in v.get("objects", [])]
        if not objs:
            continue
        base = os.path.basename(key).replace(".mp4", "")
        # keyframe dir is named after the FIRST object (matches on-disk layout)
        slug = objs[0].replace(" ", "_")
        kf_dir = f"{base}___{slug}"
        if kf_dir not in disk_dirs:
            # fall back: find any disk dir that starts with base
            cands = [d for d in disk_dirs if d.startswith(base + "___")]
            if not cands:
                print(f"  WARN no keyframe dir for {base}", flush=True)
                continue
            kf_dir = cands[0]
        work.append((base, key, bool(v.get("has_object_in_hand")), objs, kf_dir))
    return work


def make_prompt(obj: str) -> str:
    return (
        f"Detect the {obj} in this image.\n<|image|>\n"
        f"For every {obj} that is clearly visible, output its 2D bounding box "
        f"as [ymin, xmin, ymax, xmax] with integer coordinates normalized to "
        f"0-1000. If there are several, list one box per line. "
        f"If no {obj} is visible in the image, reply with exactly: none"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N frames (0 = all); for probing")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--probe", action="store_true",
                    help="print raw outputs instead of writing results")
    args = ap.parse_args()

    from gemma import gm  # heavy import; do it after arg parsing

    print(f"[{time.strftime('%H:%M:%S')}] loading model...", flush=True)
    model = gm.nn.Gemma4_E4B()
    params = gm.ckpts.load_params(args.ckpt)
    sampler = gm.text.ChatSampler(model=model, params=params, multi_turn=False)
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    work = build_worklist()
    total_frames = sum(
        len(glob.glob(os.path.join(KEYFRAMES_DIR, kf, "*.jpg")))
        for _, _, _, _, kf in work)
    print(f"{len(work)} videos, {total_frames} frames total", flush=True)

    # resume
    results = {}
    if os.path.exists(args.out) and not args.probe:
        try:
            results = json.load(open(args.out)).get("results", {})
            print(f"resuming: {len(results)} videos already in output", flush=True)
        except Exception:
            results = {}

    meta = {
        "checkpoint": args.ckpt,
        "keyframes_dir": KEYFRAMES_DIR,
        "gemma_json": GEMMA_JSON,
        "model": "gemma4-e4b-it (JAX/gm)",
        "prompt_mode": "detect-per-object",
        "coord_note": ("bbox_2d is absolute pixels [x1,y1,x2,y2]; "
                       "bbox_2d_norm1000 is the raw Gemma output "
                       "[ymin,xmin,ymax,xmax] on its 0-1000 grid."),
        "updated_frames": 0,
    }

    done_frames = 0
    t0 = time.time()
    for wi, (base, gkey, has_obj, objs, kf_dir) in enumerate(work):
        frame_paths = sorted(glob.glob(os.path.join(KEYFRAMES_DIR, kf_dir, "*.jpg")))
        entry = results.get(base) or {
            "gemma_key": gkey,
            "has_object_in_hand": has_obj,
            "objects": objs,
            "keyframe_dir": kf_dir,
            "frames": {},
        }
        for fp in frame_paths:
            fn = os.path.basename(fp)
            if fn in entry["frames"] and not args.probe:
                continue
            img = np.asarray(Image.open(fp).convert("RGB"))
            H, W = img.shape[:2]
            detections = []
            raw_all = {}
            for obj in objs:
                out = sampler.chat(make_prompt(obj), images=[img],
                                   max_new_tokens=args.max_new_tokens)
                raw_all[obj] = out.strip()
                if args.probe:
                    print(f"--- {kf_dir}/{fn} [{obj}] W={W} H={H} ---\n{out.strip()}\n",
                          flush=True)
                    continue
                low = out.strip().lower()
                if low.startswith("none") or "no " + obj in low or not low:
                    continue
                for box in parse_boxes(out):
                    if box_looks_valid(box):
                        detections.append({
                            "label": obj,
                            "bbox_2d_norm1000": [round(c, 1) for c in box],
                            "bbox_2d": norm1000_to_abs(box, W, H),
                        })
            if args.probe:
                done_frames += 1
                if args.limit and done_frames >= args.limit:
                    return
                continue
            rel = os.path.relpath(fp, REPO)
            frec = {"path": rel, "width": W, "height": H, "detections": detections}
            if not detections:
                frec["raw_output"] = raw_all
            entry["frames"][fn] = frec
            done_frames += 1
            if args.limit and done_frames >= args.limit:
                results[base] = entry
                _flush(args.out, meta, results, done_frames)
                print("hit --limit, stopping", flush=True)
                return
        results[base] = entry
        if (wi + 1) % 5 == 0 or wi == len(work) - 1:
            _flush(args.out, meta, results, done_frames)
            el = time.time() - t0
            rate = done_frames / el if el else 0
            print(f"[{time.strftime('%H:%M:%S')}] video {wi+1}/{len(work)} "
                  f"frames_done={done_frames} rate={rate:.2f}/s", flush=True)

    _flush(args.out, meta, results, done_frames)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE frames={done_frames}", flush=True)


def _flush(out, meta, results, done_frames):
    meta = dict(meta)
    meta["updated_frames"] = sum(len(v["frames"]) for v in results.values())
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2,
                  ensure_ascii=False)
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
