# Real-time 3D reconstruction from smartphone camera

This code is mainly based on [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM). 

It first estimated camera poses by DROID-SLAM and then predicted dense depth maps by [CDS-MVSNet](https://github.com/TruongKhang/cds-mvsnet) 
by using all available pretrained models including `droid.pth` and `cds_mvsnet.pth`.

Sampled data can be download [here](https://drive.google.com/drive/folders/1OAuYLulHaD1ozt4fLEjrXewpqyD54xRg?usp=sharing)
## Demos

Run the demo on any of the samples (all demos can be run on a GPU with 8G of memory).

```Python
python demo.py --imagedir=data/nmail6 --calib=calib/nmail3.txt --stride=2 --buffer 384 --mvsnet_ckpt cds_mvsnet.pth
```

