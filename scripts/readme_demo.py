#!/usr/bin/env python3
"""Run the README multi-turn, multi-modal demo with Gemma-4 E4B.

Mirrors the example in README.md but loads the checkpoint from a local path
(downloaded from gs://gemma-data) instead of streaming from GCS.
"""
import os
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, ".cache", "checkpoints", "gemma4-e4b-it")
KF = os.path.join(REPO, "dataset", "datatang_session2_videos_fps24_012_keyframes")

from gemma import gm

# Model and parameters (Gemma 4) -- as in README, with vision enabled
# (Gemma4_E4B defaults to text_only=True, which drops the vision encoder;
#  set text_only=False to use images.)
model = gm.nn.Gemma4_E4B(text_only=False)
params = gm.ckpts.load_params(CKPT)

# Multi-turn conversation, as in README
sampler = gm.text.ChatSampler(model=model, params=params, multi_turn=True)

# Two real keyframes to compare
d = sorted(os.listdir(KF))[0]
frames = sorted(f for f in os.listdir(os.path.join(KF, d)) if f.endswith(".jpg"))
image1 = np.asarray(Image.open(os.path.join(KF, d, frames[0])).convert("RGB"))
image2 = np.asarray(Image.open(os.path.join(KF, d, frames[len(frames)//2])).convert("RGB"))

prompt = """Which of the 2 images do you prefer ?

Image 1: <|image|>
Image 2: <|image|>

Write your answer as a poem."""

print("=" * 60)
print("TURN 0 prompt:\n", prompt)
out0 = sampler.chat(prompt, images=[image1, image2])
print("-" * 60)
print("TURN 0 answer:\n", out0)

print("=" * 60)
out1 = sampler.chat("What about the other image ?")
print("TURN 1 answer:\n", out1)
print("=" * 60)
print("README DEMO OK")
