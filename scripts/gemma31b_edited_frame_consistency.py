#!/usr/bin/env python3
"""Gemma-4 review of object-only edits across frames of the same video.

The script has two phases.  It first joins the anchor/extra-view source jobs to
the generation manifests and writes one stable, auditable JSONL worklist.  It
then reviews every edited frame using its reference crop and the other edited
views from the same (sample_id, entity_id) group.

Launch inference with torchrun, for example:
  torchrun --nproc_per_node=8 scripts/gemma31b_edited_frame_consistency.py

Progress is appended and fsynced after every image to a separate file per rank.
Valid records are skipped on restart.  Rank 0 atomically creates the merged
JSONL after all ranks finish.  Use --prepare-only to only create the worklist.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path



REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "dataset"
DEFAULT_MODEL = ("/mnt/data04/144632/public/ckpts/"
                 "models--google--gemma-4-31B-it/snapshots/"
                 "518276fb130dc81caf9a4f772e65e63ef2526493")
DEFAULT_MAIN_JOBS = DATASET / "s2v_keyframe_recontext/jobs.latest_all.jsonl"
DEFAULT_EXTRA_JOBS = DATASET / "s2v_keyframe_recontext/jobs.latest_all.extra_views.jsonl"
DEFAULT_MAIN_ROOT = DATASET / "s2v_object_only-hunyuan-distil/full_bbox_direct_v2"
DEFAULT_EXTRA_ROOT = DATASET / "s2v_object_only-hunyuan-distil/extra_views_latest_all"
DEFAULT_WORKLIST = DATASET / "gemma_edited_frame_consistency.jobs.jsonl"
DEFAULT_OUTPUT = DATASET / "gemma_edited_frame_consistency.jsonl"

SCHEMA = {
    "pass": "boolean",
    "object_identity_match": "boolean",
    "prompt_match": "boolean",
    "physically_plausible": "boolean",
    "consistent_with_other_views": "boolean",
    "defects": ["string enum"],
    "reason": "short string",
    "confidence": "number from 0 to 1",
}
DEFECTS = {
    "wrong_object", "identity_mismatch", "missing_part", "extra_part",
    "deformed", "broken_geometry", "wrong_material_or_color", "duplicate",
    "person_or_hand_remains", "other_object_remains", "bad_background",
    "cropped_or_incomplete", "cross_view_inconsistent", "unreadable",
    "none",
}


def read_jsonl(path, tolerate_bad_tail=False):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if tolerate_bad_tail:
                    print(f"warning: ignoring malformed line {lineno} in {path}")
                    continue
                raise
    return rows


def find_manifest_rows(root):
    paths = sorted(glob.glob(str(Path(root) / "**" / "manifest*.jsonl"),
                             recursive=True))
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def existing_path(raw, root, kind):
    """Resolve manifests moved together with this repository."""
    if raw and Path(raw).is_file():
        return str(Path(raw).resolve())
    if raw:
        name = Path(raw).name
        candidates = list(Path(root).glob(f"**/{kind}/{name}"))
        if len(candidates) == 1:
            return str(candidates[0].resolve())
    return None


def build_worklist(args):
    source_rows = [("anchor", r) for r in read_jsonl(args.main_jobs)]
    source_rows += [("extra", r) for r in read_jsonl(args.extra_jobs)]
    source = {}
    for view_type, row in source_rows:
        item = dict(row)
        item["view_type"] = view_type
        source[row["job_name"]] = item

    manifests = find_manifest_rows(args.main_root)
    manifests += find_manifest_rows(args.extra_root)
    generated = {}
    for row in manifests:
        key = row.get("source_job_name")
        if key and key in source and row.get("output_path"):
            # Prefer an actually present output if duplicate/resumed manifests exist.
            root = args.extra_root if source[key]["view_type"] == "extra" else args.main_root
            output = existing_path(row.get("output_path"), root, "edited")
            model_input = existing_path(row.get("model_input_path"), root,
                                        "reference_inputs")
            input_path = row.get("input_path") or source[key].get("frame_path")
            if output and (key not in generated or model_input):
                generated[key] = {**row, "resolved_output_path": output,
                                  "resolved_model_input_path": model_input,
                                  "resolved_input_path": input_path}

    groups = defaultdict(list)
    for key, src in source.items():
        gen = generated.get(key)
        if not gen:
            continue
        group_id = f"{src['sample_id']}||{src.get('entity_id', '')}"
        frame = {
            "review_id": key,
            "group_id": group_id,
            "sample_id": src["sample_id"],
            "original_id": src.get("original_id"),
            "entity_id": src.get("entity_id"),
            "object_name": src.get("object_name") or gen.get("object_name"),
            "frame_index": src.get("frame_index"),
            "frame_slot": src.get("frame_slot"),
            "view_type": src["view_type"],
            "input_path": gen.get("resolved_input_path"),
            "model_input_path": gen.get("resolved_model_input_path"),
            "edited_path": gen["resolved_output_path"],
            "generation_prompt": gen.get("prompt", ""),
        }
        groups[group_id].append(frame)

    jobs = []
    for frames in groups.values():
        frames.sort(key=lambda x: (x["frame_index"] is None,
                                   x["frame_index"] or -1, x["review_id"]))
        for target in frames:
            peers = [x["edited_path"] for x in frames
                     if x["review_id"] != target["review_id"]]
            jobs.append({**target, "peer_edited_paths": peers})
    jobs.sort(key=lambda x: x["review_id"])
    return jobs, len(source), len(generated), len(groups)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def prepare_worklist(args):
    jobs, source_n, generated_n, group_n = build_worklist(args)
    atomic_write_jsonl(args.worklist, jobs)
    print(f"worklist={args.worklist} jobs={len(jobs)} groups={group_n} "
          f"source_jobs={source_n} matched_outputs={generated_n}", flush=True)
    return jobs


def make_prompt(job, peer_count):
    prompt_text = job.get("generation_prompt") or (
        f"Create a clean product image of the {job['object_name']}.")
    return f"""You are a strict visual quality inspector for multi-view object edits.

Image 1 is the SOURCE/REFERENCE for the target object.
Image 2 is the TARGET EDITED FRAME that you must grade.
Images 3 through {peer_count + 2} (if present) are edited versions of the same physical object from other frames. Use them only to check cross-view identity and consistency. Viewpoint, pose, lighting, and partial visibility in the source may naturally differ.

Object that must be preserved: {job['object_name']}
Generation instruction:
{prompt_text}

Decide whether Image 2 is a usable edit. It passes only if all four checks pass:
1. It depicts the requested object and the same physical object as Image 1 (use distinctive shape, color, material, texture, markings and parts).
2. It follows the generation instruction; no person/hand or unrelated object remains, and the product is complete on an appropriate clean background.
3. The object is visually plausible: no abnormal missing/extra parts, melted or broken geometry, duplication, severe deformation, or nonsensical reconstruction. Do not reject normal articulation, flexible-object shape changes, or viewpoint changes.
4. It is consistent in identity with the peer edits. If peers disagree, grade Image 2 against the source first. If there are no peers, set consistent_with_other_views=true unless Image 2 contradicts the source.

Return exactly one JSON object and no markdown. Use this schema:
{json.dumps(SCHEMA)}
Allowed defects: {json.dumps(sorted(DEFECTS))}
Use defects=[\"none\"] only when no defect exists. Set pass=true exactly when all four boolean checks are true. Keep reason under 40 words and describe visible evidence."""


def parse_review(text):
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    obj = json.loads(cleaned[start:end + 1])
    bool_keys = ("pass", "object_identity_match", "prompt_match",
                 "physically_plausible", "consistent_with_other_views")
    if any(type(obj.get(k)) is not bool for k in bool_keys):
        raise ValueError("missing/non-boolean verdict field")
    defects = obj.get("defects")
    if not isinstance(defects, list) or not defects or any(
            x not in DEFECTS for x in defects):
        raise ValueError("invalid defects")
    expected = all(obj[k] for k in bool_keys[1:])
    obj["pass"] = expected  # enforce the documented invariant
    obj["reason"] = str(obj.get("reason", ""))[:500]
    try:
        obj["confidence"] = min(1.0, max(0.0, float(obj["confidence"])))
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid confidence")
    return {k: obj[k] for k in (*bool_keys, "defects", "reason", "confidence")}


def rank_path(output, rank):
    output = Path(output)
    return output.with_name(f"{output.name}.rank{rank}.jsonl")


def append_durable(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def completed_ids(path):
    if not Path(path).exists():
        return set()
    return {r["review_id"] for r in read_jsonl(path, tolerate_bad_tail=True)
            if r.get("status") == "ok" and r.get("review_id")}


def load_rgb(path, label):
    from PIL import Image

    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    with Image.open(path) as im:
        return im.convert("RGB")


def make_conversation(job, max_peers):
    source_path = job.get("model_input_path") or job.get("input_path")
    images = [load_rgb(source_path, "source"),
              load_rgb(job["edited_path"], "edited")]
    peer_paths = job.get("peer_edited_paths", [])[:max_peers]
    images += [load_rgb(p, "peer edited") for p in peer_paths]
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": make_prompt(job, len(peer_paths))})
    return [{"role": "user", "content": content}]


def review_batch(jobs, proc, model, max_new_tokens, max_peers):
    """Review a GPU batch; jobs in a batch have the same image count."""
    import torch

    conversations = [make_conversation(job, max_peers) for job in jobs]
    inputs = proc.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", processor_kwargs={"padding": True}).to(model.device)
    in_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False)
    raws = proc.batch_decode(output[:, in_len:], skip_special_tokens=True)
    return [(parse_review(raw), raw) for raw in raws]


def base_record(job):
    return {k: job.get(k) for k in (
        "review_id", "group_id", "sample_id", "original_id", "entity_id",
        "object_name", "frame_index", "frame_slot", "view_type",
        "input_path", "model_input_path", "edited_path")}


def chunks_by_image_count(jobs, batch_size, max_peers):
    """Keep image count homogeneous while retaining deterministic ordering."""
    buckets = defaultdict(list)
    for job in jobs:
        buckets[min(len(job.get("peer_edited_paths", [])), max_peers)].append(job)
    for peer_count in sorted(buckets):
        bucket = buckets[peer_count]
        for start in range(0, len(bucket), batch_size):
            yield bucket[start:start + batch_size]


def merge_outputs(output, world):
    by_id = {}
    for rank in range(world):
        path = rank_path(output, rank)
        if not path.exists():
            continue
        for row in read_jsonl(path, tolerate_bad_tail=True):
            key = row.get("review_id")
            if key and (key not in by_id or row.get("status") == "ok"):
                by_id[key] = row
    atomic_write_jsonl(output, [by_id[k] for k in sorted(by_id)])
    ok = sum(r.get("status") == "ok" for r in by_id.values())
    passed = sum(r.get("status") == "ok" and r.get("verdict", {}).get("pass")
                 for r in by_id.values())
    print(f"merged={output} records={len(by_id)} ok={ok} pass={passed}",
          flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--main-jobs", type=Path, default=DEFAULT_MAIN_JOBS)
    ap.add_argument("--extra-jobs", type=Path, default=DEFAULT_EXTRA_JOBS)
    ap.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    ap.add_argument("--extra-root", type=Path, default=DEFAULT_EXTRA_ROOT)
    ap.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    ap.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--reuse-worklist", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-peers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Per-GPU batch size; 1 is safest for 31B multi-image input")
    ap.add_argument("--max-items", type=int, default=-1)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.reuse_worklist:
        jobs = read_jsonl(args.worklist)
    else:
        jobs = prepare_worklist(args)
    if args.prepare_only:
        return
    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoProcessor

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch inference with torchrun (even for one GPU).")
    dist.init_process_group(backend="gloo", timeout=timedelta(hours=24))
    local_rank, rank, world = (int(os.environ["LOCAL_RANK"]), dist.get_rank(),
                               dist.get_world_size())
    torch.cuda.set_device(local_rank)
    if args.max_items > 0:
        jobs = jobs[:args.max_items]
    my_jobs = jobs[rank::world]
    progress = rank_path(args.output_jsonl, rank)
    done = completed_ids(progress)
    my_jobs = [job for job in my_jobs if job["review_id"] not in done]

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map={"": local_rank})
    model.eval()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    proc.tokenizer.padding_side = "left"
    print(f"rank={rank}/{world} pending={len(my_jobs)} resumed={len(done)} "
          f"batch_size={args.batch_size}", flush=True)
    bar = tqdm(total=len(my_jobs), desc=f"rank{rank}", disable=rank != 0)
    for batch in chunks_by_image_count(my_jobs, args.batch_size, args.max_peers):
        try:
            reviews = review_batch(batch, proc, model, args.max_new_tokens,
                                   args.max_peers)
            for job, (verdict, _) in zip(batch, reviews):
                append_durable(progress, {**base_record(job), "status": "ok",
                                          "verdict": verdict})
                bar.update(1)
        except Exception as batch_exc:
            if len(batch) > 1:
                print(f"[rank{rank}] batch failed, retrying individually: "
                      f"{batch_exc}", flush=True)
            for job in batch:
                raw = None
                try:
                    (verdict, raw), = review_batch(
                        [job], proc, model, args.max_new_tokens, args.max_peers)
                    record = {**base_record(job), "status": "ok",
                              "verdict": verdict}
                except Exception as exc:
                    record = {**base_record(job), "status": "error",
                              "error": str(exc)[:500]}
                    if raw is not None:
                        record["raw_output"] = raw[:1000]
                    print(f"[rank{rank}] {job['review_id']}: {exc}", flush=True)
                append_durable(progress, record)
                bar.update(1)
    bar.close()
    dist.barrier()
    if rank == 0:
        merge_outputs(args.output_jsonl, world)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
