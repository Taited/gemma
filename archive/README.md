# Historical experiment archive

Archived on 2026-10-12. These scripts are historical snapshots, not current pipeline entry points.
No datasets, images, checkpoints, credentials, virtual environments or logs are included.

## Restore before running

See manifest.json for every original workspace-relative path and SHA-256 checksum.
Back up any existing destination, then restore each file from its archived path to its original
path under the workspace root. Do not run scripts directly here: their relative paths and imports
expect the original layout. Restoring a group does not guarantee compatibility with current dependencies.

## Group

experiments/review-sampling/scripts/make_gemma_comparisons.py creates a fixed 30-case PASS/FAIL
review sample. It is independent of the current Gemma detection and consistency-review pipeline.
