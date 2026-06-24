# Nerfstudio / 3DGS Data

Source images: `/root/cv_hw/data/human2_vggt_official_ba/images`
Source sparse: `/root/cv_hw/data/human2_vggt_official_ba/sparse_self_ba_intrinsics_pruned`

Layout:

```text
images/      # images referenced by COLMAP images.bin
sparse/0/    # cameras.bin, images.bin, points3D.bin
```

Prepared images: 16
Prepared masks: 16

Train example:

```bash
bash scripts/07_train_splatfacto.sh \
  --data data/human2_ns_self_ba_intrinsics_pruned_masked_black \
  --output outputs/3dgs
```
