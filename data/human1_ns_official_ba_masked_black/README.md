# Nerfstudio / 3DGS Data

Source images: `/root/cv_hw/data/human1_vggt_official_ba/images`
Source sparse: `/root/cv_hw/data/human1_vggt_official_ba/sparse`

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
  --data data/human1_ns_official_ba_masked_black \
  --output outputs/3dgs
```
