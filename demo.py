import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]
    # image_list = ['image%s.jpg' % i for i in range(0, 2548)]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = 256 #int(h0 * np.sqrt((256 * 448) / (h0 * w0)))
        w1 = 448 #int(w0 * np.sqrt((256 * 448) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        # image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics, imfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--mvsnet_ckpt", default="None")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[256, 448])
    parser.add_argument("--depth_fusion_size", default=[512, 896])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    # for trajectory evaluation and write
    parser.add_argument("--datapath", type=str, default=None, help="tmu dataset folder which contains a sub folder: rgb")
    parser.add_argument("--dataset_home", type=str, default=None, help="you own dataset folder which contains a sub folder: rgb")

    args = parser.parse_args()

    import os
    os.system("mkdir " + args.dataset_home)
    os.system("cd " + args.dataset_home + " && mkdir rgb && mkdir depth")

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    tstamps = []
    for (t, image, intrinsics, imageName) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            # args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics, imageName=imageName)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))

    # 生成对应的rbg和depth数据
    ourOwnDataSetHome = args.dataset_home
    os.system("cp generateList.py " + ourOwnDataSetHome + 
        " && cd " + ourOwnDataSetHome + 
        " && python generateList.py rgb rgb.txt" )
    os.system("cp generateList.py " + ourOwnDataSetHome + 
        " && cd " + ourOwnDataSetHome + 
        " && python generateList.py depth depth.txt" )

    # 复制相机内参文件
    os.system("cp " + args.datapath + "/calibration.txt " + ourOwnDataSetHome)

    ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    image_path = os.path.join(args.datapath, 'rgb')
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    def write_tum_trajectory_file(file_path, traj: PoseTrajectory3D,
                              confirm_overwrite: bool = False) -> None:
        stamps = traj.timestamps
        xyz = traj.positions_xyz
        # shift -1 column -> w in back column
        np.set_printoptions(suppress=True, precision=6)
        quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
        mat = np.column_stack((stamps, xyz, quat))
        np.savetxt(file_path, mat, delimiter=" ", fmt='%.6f')
        if isinstance(file_path, str):
            print("Trajectory saved to: " + file_path)

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))

    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)

    # 写入slam以及和groundtruth的误差结果
    write_tum_trajectory_file(ourOwnDataSetHome + "/slamCam.txt", traj_est)

    slamEvalRes = args.dataset_home + "/slamEvalRes.txt"
    file = open(slamEvalRes, "w")
    file.write(str(result))
    file.close()

    print("all finished")
