# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
import torch.nn as nn
from mmpose.core.evaluation.top_down_eval import _get_max_preds
from ..builder import LOSSES
from matplotlib import pyplot as plt


def vis_keypoints_car_rainbow(img, kps, kp_thresh=10, alpha=0.7):
    kp_lines = [[0, 2],
                [1, 3],
                [0, 1],
                [2, 3],
                [9, 11],
                [10, 12],
                [9, 10],
                [11, 12],
                [4, 0],
                [4, 9],
                [4, 5],
                [5, 1],
                [5, 10],
                [6, 2],
                [6, 11],
                [7, 3],
                [7, 12],
                [6, 7]
                ]

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = np.int32([kps[0, i1], kps[1, i1]])
        p2 = np.int32([kps[0, i2], kps[1, i2]])
        if p1[0] == -1 and p1[1] == -1 or p2[0] == -1 and p2[1] == -1:
            continue
        # if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
        cv2.line(
            kp_mask, p1, p2,
            color=colors[l], thickness=1, lineType=cv2.LINE_AA)
        # if kps[2, i1] > kp_thresh:
        cv2.circle(
            kp_mask, p1,
            radius=1, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(kp_mask, str(i1),
                    p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
        # if kps[2, i2] > kp_thresh:
        cv2.circle(
            kp_mask, p2,
            radius=1, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(kp_mask, str(i2),
                    p2,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)

    # Blend the keypoints.
    return kp_mask
    # return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


@LOSSES.register_module()
class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        print('target.shape:', target.shape)
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight

@LOSSES.register_module()
class JointsMSELossVis(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.vis_criterion = nn.CrossEntropyLoss(size_average=True)

    """
    !!! This is used for debugging !!!
    """
    def vis_input_output(self, pred, pred_vis, gt_vis, input_imgs):
        print(gt_vis.shape)
        batch_size = pred.size(0)
        num_joints = pred.size(1)

        pred_np = pred.detach().cpu().numpy()
        preds, maxvals = _get_max_preds(pred_np)

        intersection_loss = 0.0

        for bs in range(batch_size):
            print('>>>> bs:',  bs)
            # input image load and un-normalize to visualize
            input_img = input_imgs[bs]
            input_img = input_img.detach().cpu().numpy().transpose((1, 2, 0))
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            for i in range(3):
                input_img[:, :, i] *= std[i]
                input_img[:, :, i] += mean[i]
            input_img = np.ascontiguousarray((input_img * 255.0).astype(np.uint8))
            input_img = cv2.resize(input_img, (72, 96))
            # plt.imshow(input_img)
            # plt.show()

            curr_car_kps = preds[bs]
            curr_car_kps = np.transpose(curr_car_kps, (1, 0))
            # img = np.zeros((pred.shape[2], pred.shape[3], 3), dtype=np.uint8)
            img = vis_keypoints_car_rainbow(input_img, curr_car_kps)
            print(gt_vis[bs])
            plt.imshow(img)
            plt.show()

    def forward(self, output, output_vis, target, target_weight, vis_gts, input_imgs=None):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        if input_imgs is not None:
            self.vis_input_output(target, output_vis, vis_gts, input_imgs)

        vis_preds = output_vis.split(1, 1)
        vis_gts = vis_gts.split(1, 1)

        loss = 0.
        loss_vis = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)

            vis_pred = vis_preds[idx].squeeze(1)
            vis_gt = vis_gts[idx].squeeze(1)

            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

            l_vis = 1.5e-3 * self.vis_criterion(vis_pred, vis_gt)
            loss_vis += l_vis

        # return loss / num_joints * self.loss_weight

        loss /= num_joints
        loss_vis /= num_joints

        return loss, loss_vis


@LOSSES.register_module()
class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.
        CombinedTarget: The combination of classification target
        (response map) and regression target (offset map).
        Paper ref: Huang et al. The Devil is in the Details: Delving into
        Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        loss = 0.
        num_joints = num_channels // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                heatmap_pred = heatmap_pred * target_weight[:, idx]
                heatmap_gt = heatmap_gt * target_weight[:, idx]
            # classification loss
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            # regression loss
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred,
                                         heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred,
                                         heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


@LOSSES.register_module()
class JointsOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        topk (int): Only top k joint losses are kept.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, topk=8, loss_weight=1.):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, loss):
        """Online hard keypoint mining."""
        ohkm_loss = 0.
        N = len(loss)
        for i in range(N):
            sub_loss = loss[i]
            _, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= N
        return ohkm_loss

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)
        if num_joints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not '
                             f'larger than num_joints ({num_joints}).')
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        losses = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                losses.append(
                    self.criterion(heatmap_pred * target_weight[:, idx],
                                   heatmap_gt * target_weight[:, idx]))
            else:
                losses.append(self.criterion(heatmap_pred, heatmap_gt))

        losses = [loss.mean(dim=1).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight