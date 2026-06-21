# Data Layout

This folder is a lightweight standard entrypoint for the provided datasets.
It uses symlinks to avoid copying the original files in `../大作业数据`.

```text
data/
├── human1/
│   ├── images/   # rgb_*.png
│   └── masks/    # msk_*.png
├── human2/
│   ├── images/   # rgb_*.png
│   └── masks/    # msk_*.png
└── scene3/
    └── source_video.mp4
```

For VGGT, use each `images/` directory for human data. For scene3, extract
frames from `source_video.mp4` into `data/scene3/images/` in the next stage.
