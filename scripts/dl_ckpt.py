#!/usr/bin/env python3
"""Mirror a public gemma-data GCS checkpoint dir to a local folder (parallel curl).

Usage: python dl_ckpt.py <gcs_prefix> <local_dir>
  e.g. python dl_ckpt.py checkpoints/gemma4-e4b-it .cache/checkpoints/gemma4-e4b-it
"""
import json, os, sys, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "gemma-data"
prefix = sys.argv[1].rstrip("/") + "/"
outdir = sys.argv[2]

def list_objects(prefix):
    items, token = [], None
    while True:
        url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
               f"?prefix={prefix}&maxResults=1000")
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            break
    return items

def dl(obj):
    name = obj["name"]
    size = int(obj.get("size", 0))
    # strip the prefix's parent so local layout mirrors the checkpoint dir
    rel = name[len(prefix):] if name.startswith(prefix) else name
    if rel == "" or name.endswith("_$folder$"):
        return (name, 0, "skip-folder")
    dst = os.path.join(outdir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return (name, size, "cached")
    url = f"https://storage.googleapis.com/{BUCKET}/{name.replace(' ', '%20')}"
    for attempt in range(5):
        rc = subprocess.run(
            ["curl", "-sf", "--retry", "3", "--max-time", "1800",
             "-o", dst, url]).returncode
        if rc == 0 and (size == 0 or os.path.getsize(dst) == size):
            return (name, size, "ok")
    return (name, size, "FAIL")

items = list_objects(prefix)
total = sum(int(o.get("size", 0)) for o in items)
print(f"{len(items)} objects, {total/1e9:.2f} GB -> {outdir}", flush=True)
os.makedirs(outdir, exist_ok=True)
fails = []
with ThreadPoolExecutor(max_workers=6) as ex:
    for name, size, status in ex.map(dl, items):
        print(f"  [{status}] {size/1e6:8.1f}MB {name}", flush=True)
        if status == "FAIL":
            fails.append(name)
print("DONE" if not fails else f"FAILED: {fails}", flush=True)
sys.exit(1 if fails else 0)
