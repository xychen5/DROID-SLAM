import torch
import lietorch
import numpy as np
import torch.nn.functional as F

from lietorch import SE3
from factor_graph import FactorGraph
from cdsmvsnet import CDSMVSNet


class DroidFrontend:
    def __init__(self, net, video, mvsnet, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48)
        self.mvsnet = mvsnet

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # refine depths
        if self.mvsnet is not None:
            ref_id, src_ids = self.t1 - 3, [self.t1-5, self.t1-4, self.t1-2, self.t1-1]
            img_ids = [ref_id] + src_ids
            intrinsics = self.video.intrinsics[img_ids] * 8
            poses = SE3(self.video.poses[img_ids]).matrix()

            intr_matrices = torch.zeros_like(poses)
            intr_matrices[:, 0, 0], intr_matrices[:, 1, 1] = intrinsics[:, 0], intrinsics[:, 1],
            intr_matrices[:, :2, 2], intr_matrices[:, 2, 2] = intrinsics[:, 2:], 1.0
            proj_stage3 = torch.stack((poses, intr_matrices), dim=1)
            proj_stage2 = proj_stage3.clone()
            proj_stage2[:, 1, :2] *= 0.5
            proj_stage1 = proj_stage2.clone()
            proj_stage1[:, 1, :2] *= 0.5
            proj_stage0 = proj_stage1.clone()
            proj_stage0[:, 1, :2] *= 0.5
            proj_matrices = {"stage1": proj_stage0.unsqueeze(0),
                             "stage2": proj_stage1.unsqueeze(0),
                             "stage3": proj_stage2.unsqueeze(0),
                             "stage4": proj_stage3.unsqueeze(0)}

            ref_depth = 1 / self.video.disps[ref_id]
            val_depths = ref_depth[(ref_depth > 0.001) & (ref_depth < 1000)]
            min_d, max_d = val_depths.min(), val_depths.max()
            d_interval = 0.05 #(max_d - min_d) / 256
            depth_values = torch.arange(0, 384, dtype=torch.float32, device=min_d.device).unsqueeze(0) * d_interval + min_d

            images = self.video.images[img_ids].unsqueeze(0) / 255.
            with torch.no_grad():
                final_depth = self.mvsnet(images, proj_matrices, depth_values.cuda(), temperature=0.01)["refined_depth"]
            disp_up = 1 / (final_depth.squeeze(0) + 1e-6)
            self.video.disps_up[ref_id] = disp_up.clamp(min=0.001)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        # self.video.dirty[self.graph.ii.min():self.t1] = True
        self.video.dirty[self.graph.ii.min():(self.t1 - 2)] = True

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """
        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        
