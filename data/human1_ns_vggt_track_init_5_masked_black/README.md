# Nerfstudio / 3DGS Data

Source images: `/home/frosting/26spring/大作业/data/human1_vggt_16_track_init_5/images`
Source sparse: `/home/frosting/26spring/大作业/data/human1_vggt_16_track_init_5/sparse`

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
  --data data/human1_ns_vggt_track_init_5_masked_black \
  --output outputs/3dgs
```
