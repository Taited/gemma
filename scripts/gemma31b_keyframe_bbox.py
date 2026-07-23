#!/usr/bin/env python3
"""Detect, for every keyframe, the bounding box(es) of the object(s) that
`dataset/gemma_result.json` reports for that video, using the local
`gemma-4-31B-it` multimodal model via HuggingFace transformers.

Distributed 8-GPU data-parallel inference (launch with torchrun), modeled on the
project's own `detect_hand_object_from_prompt.py`.  The frame-level worklist is
split across ranks; each rank loads one copy of the model on its GPU.  Rank 0
merges the per-rank results into a nested JSON that mirrors the conventions of
`dataset/qwen3vl_keyframe_bbox.json`:

  results[<video_base>] = {
     gemma_key, has_object_in_hand, objects, keyframe_dir,
     frames: { <frame.jpg>: {
        path, width, height,
        detections: [ {label, box_2d_norm1000:[ymin,xmin,ymax,xmax],
                       bbox_2d:[x1,y1,x2,y2]} , ... ]
     } }
  }

Frames where nothing is detected -> detections: [] (raw_output kept for audit).
The run is resumable: existing per-rank tmp files are loaded and their frames
skipped.
"""
import argparse
import glob
import json
import os
import re
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYFRAMES_DIR = os.path.join(REPO, "dataset",
                             "datatang_session2_videos_fps24_012_keyframes")
GEMMA_JSON = os.path.join(REPO, "dataset", "gemma_result.json")
DEFAULT_MODEL = ("/mnt/data04/144632/public/ckpts/"
                 "models--google--gemma-4-31B-it/snapshots/"
                 "518276fb130dc81caf9a4f772e65e63ef2526493")


# --------------------------- worklist ---------------------------
def build_frame_worklist():
    """Flat list of frame-level jobs, one per (video, frame)."""
    g = json.load(open(GEMMA_JSON))
    disk_dirs = set(os.listdir(KEYFRAMES_DIR))
    jobs = []
    for key, v in g.items():
        objs = [o["name"] for o in v.get("objects", [])]
        if not objs:
            continue
        base = os.path.basename(key).replace(".mp4", "")
        slug = objs[0].replace(" ", "_")
        kf_dir = f"{base}___{slug}"
        if kf_dir not in disk_dirs:
            cands = [d for d in disk_dirs if d.startswith(base + "___")]
            if not cands:
                continue
            kf_dir = cands[0]
        has_obj = bool(v.get("has_object_in_hand"))
        for fp in sorted(glob.glob(os.path.join(KEYFRAMES_DIR, kf_dir, "*.jpg"))):
            jobs.append({
                "base": base, "gemma_key": key, "has_object_in_hand": has_obj,
                "objects": objs, "kf_dir": kf_dir, "frame_path": fp,
                "frame": os.path.basename(fp),
            })
    return jobs


# --------------------------- prompt / parse ---------------------------
def make_prompt(objs):
    names = ", ".join(objs)
    return (
        f"Detect the following object(s) in the image: {names}.\n"
        f"For every clearly visible instance, return a JSON list where each "
        f"element is {{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": "
        f"\"<object name>\", \"motion_blur\": <true|false>}} with integer "
        f"coordinates normalized to 0-1000. Set \"motion_blur\" to true if that "
        f"object instance looks motion-blurred (smeared or streaked from fast "
        f"motion during the exposure), or false if it looks sharp and clear. "
        f"Only use labels from this list: {names}. "
        f"If none of them are visible, return an empty list []."
    )


_JSON_OBJ = re.compile(r"\{[^{}]*\}")
_NUMS = re.compile(r"-?\d+\.?\d*")


def parse_detections(text, allowed, W, H):
    """Parse model output into a list of detections (abs pixel bbox + raw)."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    dets = []
    parsed_ok = False
    # Preferred: proper JSON list
    try:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(cleaned[start:end + 1])
            parsed_ok = True
            for el in data:
                if not isinstance(el, dict) or "box_2d" not in el:
                    continue
                box = el["box_2d"]
                label = str(el.get("label", "")).strip() or (
                    allowed[0] if len(allowed) == 1 else "")
                d = _to_det(box, label, W, H, _coerce_bool(el.get("motion_blur")))
                if d:
                    dets.append(d)
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed_ok = False
    # Fallback: regex per {...} block
    if not parsed_ok:
        for m in _JSON_OBJ.findall(cleaned):
            nums = [float(x) for x in _NUMS.findall(m)]
            if len(nums) < 4:
                continue
            lbl = ""
            for a in allowed:
                if a.lower() in m.lower():
                    lbl = a
                    break
            if not lbl and len(allowed) == 1:
                lbl = allowed[0]
            mb = None
            mm = re.search(r'"motion_blur"\s*:\s*(true|false)', m, re.I)
            if mm:
                mb = mm.group(1).lower() == "true"
            d = _to_det(nums[:4], lbl, W, H, mb)
            if d:
                dets.append(d)
    return dets


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def _to_det(box, label, W, H, motion_blur=None):
    try:
        ymin, xmin, ymax, xmax = [float(c) for c in box[:4]]
    except (TypeError, ValueError):
        return None
    if not all(0 <= c <= 1000 for c in (ymin, xmin, ymax, xmax)):
        return None
    if ymax - ymin < 1 or xmax - xmin < 1:
        return None
    x1 = max(0.0, min(xmin / 1000.0 * W, W))
    y1 = max(0.0, min(ymin / 1000.0 * H, H))
    x2 = max(0.0, min(xmax / 1000.0 * W, W))
    y2 = max(0.0, min(ymax / 1000.0 * H, H))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "label": label,
        "box_2d_norm1000": [round(ymin, 1), round(xmin, 1),
                            round(ymax, 1), round(xmax, 1)],
        "bbox_2d": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "motion_blur": bool(motion_blur) if motion_blur is not None else None,
    }


# --------------------------- distributed plumbing ---------------------------
def rank_tmp_path(out, rank):
    out = Path(out)
    return out.with_name(f"{out.name}.rank{rank}.tmp")


def save_rank(out, rank, data):
    p = rank_tmp_path(out, rank)
    p.parent.mkdir(parents=True, exist_ok=True)
    w = p.with_name(p.name + ".writing")
    with w.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(w, p)


def load_rank(out, rank):
    p = rank_tmp_path(out, rank)
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-json",
                    default=os.path.join(REPO, "dataset",
                                         "gemma_keyframe_bbox.json"))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-items", type=int, default=-1)
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun.")
    dist.init_process_group(backend="gloo", timeout=timedelta(hours=12))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    jobs = build_frame_worklist()
    if args.max_items > 0:
        jobs = jobs[:args.max_items]
    # deterministic split across ranks (no padding / duplication)
    my_jobs = jobs[rank::world]

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map={"": local_rank})
    model.eval()

    result = load_rank(args.output_json, rank)  # resume
    done_keys = set(result.keys())
    if rank == 0:
        print(f"total frames={len(jobs)} | this rank={len(my_jobs)} | "
              f"already done={len(done_keys)}", flush=True)

    processed = 0
    for job in tqdm(my_jobs, desc=f"rank{rank}", disable=(rank != 0)):
        fkey = f"{job['base']}||{job['frame']}"
        if fkey in done_keys:
            continue
        try:
            img = Image.open(job["frame_path"]).convert("RGB")
            W, H = img.size
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": make_prompt(job["objects"])},
            ]}]
            inputs = proc.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            in_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = model.generate(**inputs,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            text = proc.decode(out[0, in_len:], skip_special_tokens=True)
            dets = parse_detections(text, job["objects"], W, H)
            rec = {
                "base": job["base"],
                "gemma_key": job["gemma_key"],
                "has_object_in_hand": job["has_object_in_hand"],
                "objects": job["objects"],
                "kf_dir": job["kf_dir"],
                "frame": job["frame"],
                "path": os.path.relpath(job["frame_path"], REPO),
                "width": W, "height": H,
                "detections": dets,
            }
            if not dets:
                rec["raw_output"] = text.strip()[:500]
            result[fkey] = rec
        except Exception as exc:
            result[fkey] = {"base": job["base"], "frame": job["frame"],
                            "error": str(exc)[:300]}
            print(f"[rank{rank}] fail {fkey}: {exc}", flush=True)
        processed += 1
        if processed % args.save_every == 0:
            save_rank(args.output_json, rank, result)

    save_rank(args.output_json, rank, result)
    print(f"[rank{rank}] done={len(result)}", flush=True)
    dist.barrier()

    if rank == 0:
        merge(args.output_json, world)
    dist.barrier()
    dist.destroy_process_group()


def merge(out, world):
    flat = {}
    for r in range(world):
        flat.update(load_rank(out, r))
    # regroup into nested structure keyed by video base
    results = {}
    n_frames = 0
    n_det = 0
    for fkey, rec in flat.items():
        if "error" in rec and "path" not in rec:
            base = rec.get("base", "unknown")
            results.setdefault(base, {"frames": {}})
            results[base]["frames"][rec.get("frame", fkey)] = {
                "error": rec["error"]}
            continue
        base = rec["base"]
        entry = results.get(base)
        if entry is None:
            entry = {
                "gemma_key": rec["gemma_key"],
                "has_object_in_hand": rec["has_object_in_hand"],
                "objects": rec["objects"],
                "keyframe_dir": rec["kf_dir"],
                "frames": {},
            }
            results[base] = entry
        entry["frames"][rec["frame"]] = {
            "path": rec["path"], "width": rec["width"], "height": rec["height"],
            "detections": rec["detections"],
            **({"raw_output": rec["raw_output"]} if "raw_output" in rec else {}),
        }
        n_frames += 1
        n_det += len(rec["detections"])
    # sort frames within each video
    for entry in results.values():
        if "frames" in entry:
            entry["frames"] = dict(sorted(entry["frames"].items()))
    results = dict(sorted(results.items()))
    meta = {
        "model": "gemma-4-31B-it (HuggingFace transformers)",
        "checkpoint": DEFAULT_MODEL,
        "keyframes_dir": KEYFRAMES_DIR,
        "gemma_json": GEMMA_JSON,
        "prompt_mode": "detect-listed-objects",
        "coord_note": ("bbox_2d is absolute pixels [x1,y1,x2,y2]; "
                       "box_2d_norm1000 is the raw Gemma-4 output "
                       "[ymin,xmin,ymax,xmax] on its 0-1000 grid; "
                       "motion_blur is true/false per detection (whether that "
                       "object instance looks motion-blurred), null if unknown."),
        "num_videos": len(results),
        "updated_frames": n_frames,
        "total_detections": n_det,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"Saved {out}: videos={len(results)} frames={n_frames} "
          f"detections={n_det}", flush=True)


if __name__ == "__main__":
    main()
