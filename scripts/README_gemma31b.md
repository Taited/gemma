# Gemma 31B 视觉数据处理脚本

本文档说明以下两个离线推理脚本：

- [`gemma31b_keyframe_bbox.py`](./gemma31b_keyframe_bbox.py)：在视频关键帧中检测指定物体，输出边界框和运动模糊标记。
- [`gemma31b_edited_frame_consistency.py`](./gemma31b_edited_frame_consistency.py)：检查同一物体在多个编辑帧中的身份一致性、提示词符合度和视觉质量。

两个脚本都使用 Hugging Face `transformers` 加载本地 Gemma 31B 多模态模型，并通过 `torchrun` 做数据并行推理：每个进程占用一张 GPU，并各自加载一份完整模型。

## 环境准备

建议从仓库根目录运行命令。运行环境至少需要：

- 可用的 NVIDIA GPU 和 CUDA 版 PyTorch；
- `transformers`，且版本支持当前 Gemma 多模态 checkpoint；
- `Pillow` 和 `tqdm`；
- 足够的单卡显存来加载一份 31B 模型。

两个脚本中的默认模型路径是项目机器上的绝对路径：

```text
/mnt/data04/144632/public/ckpts/models--google--gemma-4-31B-it/snapshots/518276fb130dc81caf9a4f772e65e63ef2526493
```

如果 checkpoint 位于其他位置，请通过 `--model` 显式指定：

```bash
--model /path/to/gemma-4-31B-it
```

即使只使用一张 GPU，也必须通过 `torchrun --nproc_per_node=1` 启动推理。

## 1. 关键帧物体检测

脚本：`scripts/gemma31b_keyframe_bbox.py`

### 功能

脚本读取每个视频对应的目标物体名称，然后逐帧让 Gemma 返回目标物体的归一化边界框。每条检测还包含 `motion_blur`，用于表示该物体实例是否存在明显的运动模糊。

模型返回的坐标格式是：

```text
[ymin, xmin, ymax, xmax]，坐标范围为 0～1000
```

脚本同时将其转换为图像上的绝对像素坐标：

```text
[x1, y1, x2, y2]
```

### 默认输入

输入路径目前由脚本顶部的常量指定，不能通过命令行覆盖：

| 输入 | 默认路径 | 说明 |
| --- | --- | --- |
| 物体标注 | `dataset/gemma_result.json` | 记录各视频的 `objects` 和 `has_object_in_hand` |
| 关键帧目录 | `dataset/datatang_session2_videos_fps24_012_keyframes/` | 每个视频对应一个关键帧子目录 |

`gemma_result.json` 的核心结构如下：

```json
{
  "path/to/video.mp4": {
    "has_object_in_hand": true,
    "objects": [
      {"name": "brush", "confidence": 1.0}
    ]
  }
}
```

脚本优先查找名为 `<视频文件名去掉 .mp4>___<第一个物体名>` 的关键帧目录，其中物体名里的空格会替换为下划线。如果该目录不存在，则回退到第一个以 `<视频文件名>___` 开头的目录。目录中的所有 `*.jpg` 文件都会进入任务列表；没有目标物体的视频会被跳过。

### 运行

8 卡完整运行：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/gemma31b_keyframe_bbox.py
```

先用少量样本验证环境和输出：

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/gemma31b_keyframe_bbox.py \
  --max-items 10 \
  --output-json dataset/gemma_keyframe_bbox.smoke.json
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 脚本内置路径 | 本地模型或 Hugging Face checkpoint 路径 |
| `--output-json` | `dataset/gemma_keyframe_bbox.json` | 最终合并结果 |
| `--max-new-tokens` | `256` | 每帧最多生成的 token 数 |
| `--max-items` | `-1` | 全局最多处理的帧数；`-1` 表示不限制 |
| `--save-every` | `50` | 每个 rank 每处理多少帧保存一次临时结果 |

### 输出与断点续跑

每个 rank 的临时文件位于最终输出文件旁边：

```text
dataset/gemma_keyframe_bbox.json.rank0.tmp
dataset/gemma_keyframe_bbox.json.rank1.tmp
...
```

每次启动时，rank 会载入自己的临时文件并跳过已有的帧。所有 rank 完成后，rank 0 将结果合并到 `--output-json`。建议续跑时保持 GPU 数量和输出路径不变。

最终 JSON 的简化结构如下：

```json
{
  "meta": {
    "num_videos": 1,
    "updated_frames": 1,
    "total_detections": 1
  },
  "results": {
    "video_base": {
      "gemma_key": "path/to/video.mp4",
      "has_object_in_hand": true,
      "objects": ["brush"],
      "keyframe_dir": "video_base___brush",
      "frames": {
        "frame-0001.jpg": {
          "path": "dataset/.../frame-0001.jpg",
          "width": 1920,
          "height": 1080,
          "detections": [
            {
              "label": "brush",
              "box_2d_norm1000": [100.0, 200.0, 800.0, 900.0],
              "bbox_2d": [384.0, 108.0, 1728.0, 864.0],
              "motion_blur": false
            }
          ]
        }
      }
    }
  }
}
```

没有检测结果时，`detections` 是空列表，并保留最多 500 个字符的 `raw_output` 便于审计。处理异常会以 `error` 字段写入临时及合并结果。已有临时文件中的异常记录也会被视为已处理；若要重试它们，需要先备份并清理对应 rank 的临时文件，或者改用新的 `--output-json`。

## 2. 编辑帧一致性检查

脚本：`scripts/gemma31b_edited_frame_consistency.py`

### 功能

脚本分为两个阶段：

1. 将 anchor/extra-view 源任务与图像生成 manifest 关联，生成稳定、可审计的 JSONL worklist；
2. 对每张编辑图进行审核，并参考原始物体裁剪图和同一 `(sample_id, entity_id)` 下的其他编辑视图。

每张目标编辑图会检查：

- 是否是参考图中的同一个物体；
- 是否符合生成提示词；
- 几何、部件和整体外观是否合理；
- 是否与同组其他视图保持身份一致。

只有四项全部为 `true` 时，最终 `pass` 才为 `true`。

### 默认输入和输出

| 类型 | 默认路径 |
| --- | --- |
| anchor 源任务 | `dataset/s2v_keyframe_recontext/jobs.latest_all.jsonl` |
| extra-view 源任务 | `dataset/s2v_keyframe_recontext/jobs.latest_all.extra_views.jsonl` |
| anchor 生成根目录 | `dataset/s2v_object_only-hunyuan-distil/full_bbox_direct_v2` |
| extra-view 生成根目录 | `dataset/s2v_object_only-hunyuan-distil/extra_views_latest_all` |
| worklist | `dataset/gemma_edited_frame_consistency.jobs.jsonl` |
| 最终结果 | `dataset/gemma_edited_frame_consistency.jsonl` |

源任务至少需要可用于关联的 `job_name`，以及 `sample_id`、`entity_id`、`object_name` 等审核字段。生成根目录下会递归查找 `manifest*.jsonl`；manifest 通过 `source_job_name` 与源任务关联，并从 `output_path`、`model_input_path` 和 `input_path` 解析编辑图、参考裁剪图和原图。

如果 manifest 中保存的是移动前的绝对路径，脚本还会按文件名在当前生成根目录的 `edited/` 或 `reference_inputs/` 子目录中尝试唯一匹配。

### 推荐运行流程

先用单进程生成 worklist，并检查匹配数量：

```bash
python scripts/gemma31b_edited_frame_consistency.py --prepare-only
```

输出示例：

```text
worklist=... jobs=1200 groups=300 source_jobs=1300 matched_outputs=1200
```

然后复用该 worklist 启动多卡推理：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/gemma31b_edited_frame_consistency.py \
  --reuse-worklist
```

推荐拆成这两步执行，因为未指定 `--reuse-worklist` 时，每个 `torchrun` 进程都会在进入分布式初始化前尝试准备 worklist。

少量数据试跑：

```bash
python scripts/gemma31b_edited_frame_consistency.py \
  --prepare-only \
  --worklist dataset/gemma_edited_frame_consistency.smoke.jobs.jsonl

torchrun --standalone --nproc_per_node=1 \
  scripts/gemma31b_edited_frame_consistency.py \
  --reuse-worklist \
  --worklist dataset/gemma_edited_frame_consistency.smoke.jobs.jsonl \
  --output-jsonl dataset/gemma_edited_frame_consistency.smoke.jsonl \
  --max-items 10
```

自定义输入目录：

```bash
python scripts/gemma31b_edited_frame_consistency.py \
  --prepare-only \
  --main-jobs /path/to/anchor_jobs.jsonl \
  --extra-jobs /path/to/extra_jobs.jsonl \
  --main-root /path/to/anchor_generations \
  --extra-root /path/to/extra_generations \
  --worklist /path/to/review.jobs.jsonl

torchrun --standalone --nproc_per_node=8 \
  scripts/gemma31b_edited_frame_consistency.py \
  --reuse-worklist \
  --worklist /path/to/review.jobs.jsonl \
  --output-jsonl /path/to/review.jsonl \
  --model /path/to/gemma-4-31B-it
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--prepare-only` | 关闭 | 只生成 worklist，不加载模型 |
| `--reuse-worklist` | 关闭 | 直接读取已有 worklist |
| `--max-new-tokens` | `384` | 每条审核最多生成的 token 数 |
| `--max-peers` | `3` | 最多提供多少张同组编辑图作为交叉视图参考 |
| `--batch-size` | `1` | 单卡 batch size；31B 多图输入建议从 1 开始 |
| `--max-items` | `-1` | 全局最多审核多少条；`-1` 表示不限制 |

### 输出与断点续跑

worklist 每行对应一张待审核的目标编辑图，主要字段包括：

```json
{
  "review_id": "source-job-name",
  "group_id": "sample-id||entity-id",
  "object_name": "brush",
  "model_input_path": "/path/to/reference.jpg",
  "edited_path": "/path/to/edited.jpg",
  "peer_edited_paths": ["/path/to/peer-1.jpg"]
}
```

推理进度按 rank 追加写入，并在每条记录后执行 flush 和 fsync：

```text
dataset/gemma_edited_frame_consistency.jsonl.rank0.jsonl
dataset/gemma_edited_frame_consistency.jsonl.rank1.jsonl
...
```

重启时只跳过 `status == "ok"` 的记录，因此失败项会自动重试。建议续跑时保持 GPU 数量、worklist 和输出路径不变。全部完成后，rank 0 按 `review_id` 合并并排序，写出最终 JSONL。

成功记录的简化结构如下：

```json
{
  "review_id": "source-job-name",
  "group_id": "sample-id||entity-id",
  "object_name": "brush",
  "status": "ok",
  "verdict": {
    "pass": true,
    "object_identity_match": true,
    "prompt_match": true,
    "physically_plausible": true,
    "consistent_with_other_views": true,
    "defects": ["none"],
    "reason": "The edited object matches the reference and remains consistent across views.",
    "confidence": 0.96
  }
}
```

`defects` 可能包含 `wrong_object`、`identity_mismatch`、`missing_part`、`extra_part`、`deformed`、`broken_geometry`、`wrong_material_or_color`、`duplicate`、`person_or_hand_remains`、`other_object_remains`、`bad_background`、`cropped_or_incomplete`、`cross_view_inconsistent`、`unreadable` 或 `none`。

## 运行注意事项

- 两个脚本都是数据并行，而不是模型并行；增加 GPU 数量会增加吞吐量，但不会降低单卡加载模型所需的显存。
- `--max-items` 在任务分片前生效，表示所有 rank 合计的最大任务数，而不是每个 rank 的数量。
- 同一路径下的 rank 临时文件属于断点状态。想进行全新实验时，建议改用新的输出文件名，以免混入旧结果。
- 一致性检查的输入包含多张图片，显存占用会随 `--batch-size`、`--max-peers` 和图片尺寸上升；出现 OOM 时优先保持 `--batch-size 1`，再减小 `--max-peers`。
