# Data Layout

This folder is a lightweight standard entrypoint for the provided datasets.
It uses local dataset files or symlinks to avoid committing large images,
videos, and reconstruction outputs to git.

```text
data/
├── human1/
│   ├── images/   # rgb_*.png
│   └── masks/    # msk_*.png
├── human2/
│   ├── images/   # rgb_*.png
│   └── masks/    # msk_*.png
├── scene3/
│   ├── source_video.mp4
│   └── images/   # optional extracted frames
├── human1_mask_black/
│   └── images/   # one-time foreground-aware input
├── human1_mask_blur/
│   └── images/   # one-time foreground-aware input
├── scene3_fixed_interval_24/
├── scene3_uniform_24/
├── scene3_quality_24/
├── scene3_quality_48/
└── scene3_quality_96/
```

For VGGT, use each `images/` directory for human data. For scene3, the
`scene3_*` directories are the current keyframe-selection inputs and include
`view_scores.csv` plus `view_scores.png` for inspection.

When copying data to a server, include the directories listed above. Historical
`*_vggt_*`, `*_ns_*`, and `*_test` outputs are local experiment artifacts and
should be regenerated only when needed.
