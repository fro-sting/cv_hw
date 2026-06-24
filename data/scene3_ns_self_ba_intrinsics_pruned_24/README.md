# Nerfstudio / 3DGS Data

Source images: `/root/cv_hw/data/scene3_vggt_official_ba_24/images`
Source sparse: `/root/cv_hw/data/scene3_vggt_official_ba_24/sparse_self_ba_intrinsics_pruned`

Layout:

```text
images/      # images referenced by COLMAP images.bin
sparse/0/    # cameras.bin, images.bin, points3D.bin
```

Prepared images: 24
Prepared masks: 0

Train example:

```bash
bash scripts/07_train_splatfacto.sh \
  --data data/scene3_ns_self_ba_intrinsics_pruned_24 \
  --output outputs/3dgs
```
