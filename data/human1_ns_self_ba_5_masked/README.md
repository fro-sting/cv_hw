# Nerfstudio / 3DGS Data

Source images: `/root/cv_hw/data/human1_vggt_16_track_init_5/images`
Source sparse: `/root/cv_hw/data/human1_vggt_16_track_init_5/sparse_self_ba_softl1_12000`

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
  --data data/human1_ns_self_ba_5_masked \
  --output outputs/3dgs
```
