import torch
import torch.nn.functional as F
import lietorch
import numpy as np
import cv2

from lietorch import SE3
from factor_graph import FactorGraph
from droid_slam.data_readers.mvsnet_dataloader import mvs_loader


class DroidBackend:
    def __init__(self, net, video, mvsnet, args):
        self.video = video
        self.update_op = net.update
        self.mvsnet = mvsnet
        self.args = args

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        print("backend init!")
        
    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value
        if not self.video.stereo and not torch.any(self.video.disps_sens):
             self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t)

        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()

        if self.mvsnet is not None:
            all_poses = SE3(self.video.poses).matrix()
            intrinsics = self.video.intrinsics * 8
            all_intr_matrices = torch.zeros_like(all_poses)
            all_intr_matrices[:, 0, 0], all_intr_matrices[:, 1, 1] = intrinsics[:, 0], intrinsics[:, 1],
            all_intr_matrices[:, :2, 2], all_intr_matrices[:, 2, 2] = intrinsics[:, 2:], 1.0
            for t1 in range(2, t-2):
                ref_id, src_ids = t1, [t1 - 2, t1 - 1, t1 + 1, t1 + 2]
                img_ids = [ref_id] + src_ids
                poses = all_poses[img_ids]
                tstamps = self.video.tstamp[img_ids]
                images, proj_matrices, depth_values = mvs_loader(self.args, tstamps, poses, self.video.disps[ref_id])
                with torch.no_grad():
                    mvs_outputs = self.mvsnet(images, proj_matrices, depth_values.cuda(), temperature=0.01)
                    final_depth = mvs_outputs["refined_depth"]
                    mask = torch.ones_like(final_depth) > 0.0
                    for stage, thresh_conf in zip(["stage1", "stage2", "stage3"], [0.1, 0.2, 0.3]):
                        conf_stage = F.interpolate(mvs_outputs[stage]["photometric_confidence"].unsqueeze(1),
                                                   (mask.size(1), mask.size(2))).squeeze(1)
                        mask = mask & (conf_stage > thresh_conf)
                    final_depth[~mask] = 1e-6
                self.video.disps_up[ref_id] = final_depth.squeeze(0)
                self.video.ref_image[0] = images[0, 0]

                import cv2
                large_depth = 1 / (final_depth + 1e-6)
                print("frame depth is: ", t1, " size: ", large_depth.size()," >>> ", large_depth)
                imgNumpy = large_depth.squeeze(0).cpu().numpy()
                imgNumpy2 = final_depth.squeeze(0).cpu().numpy()
                # cv2.imwrite("/home/nash5/prjs/DROID-SLAM/data/enlarge_e6_" + str(t1) + ".jpg", imgNumpy)
                # cv2.imwrite("/home/nash5/prjs/DROID-SLAM/data/ori_" + str(t1) + ".jpg", imgNumpy2)
                import cv2
                cv2.imwrite("/home/nash5/prjs/DROID-SLAM/data/back_" + str(t1) + ".jpg",
                    (imgNumpy2 * 256.0).astype(np.uint16),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

                self.video.dirty[max(0, ref_id - 3):(ref_id+1)] = True
            
            # import matplotlib.pyplot as plt
            # tmpVisual = 1 / (self.video.disps_up[2] + 1e-6)
            # tmpVisual2 = self.video.disps_up[2]
            # plt.imshow(tmpVisual.squeeze(0).cpu().numpy())
            # plt.colorbar()
            # plt.show()
            # plt.imshow(tmpVisual2.squeeze(0).cpu().numpy())
            # plt.colorbar()
            # plt.show()

