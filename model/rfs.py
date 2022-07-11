# must import open3d before pytorch.
import os
import sys
import trimesh
from util.icp import icp

import numpy as np
import pandas as pd
import pyquaternion
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from model.bsp import CompNet, PolyMesh
from util.bbox import BBoxUtils

BBox = BBoxUtils()

from util.consts import RFS2CAD_arr, CAD_weights


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

class AttributeDict(dict):
    def __init__(self, d=None):
        if d is not None:
            for k, v in d.items():
                self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.LeakyReLU(0.01),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.LeakyReLU(0.01),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.LeakyReLU(0.01),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) 
                  for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.LeakyReLU(0.01),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.LeakyReLU(0.01),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                if i == 0:
                    blocks_tail['block{}'.format(i)] = block(nPlanes[0] * 2, nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
                else:
                    blocks_tail['block{}'.format(i)] = block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))

            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output


class RfSNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        m = cfg.m # 16 or 32
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs
        self.prepare_epochs_2 = cfg.prepare_epochs_2

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        input_c = 0
        if cfg.use_coords:
            input_c += 3
        if cfg.use_rgb:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, 2*m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        
        self.unet = UBlock([2*m, 2*m, 4*m, 4*m, 6*m, 6*m, 8*m], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(2*m),
            nn.LeakyReLU(0.01)
        )

        #### semantic segmentation branch
        self.linear = nn.Linear(2*m, classes) # bias(default): True

        #### offset branch
        self.offset = nn.Sequential(
            nn.Linear(2*m, m, bias=True),
            norm_fn(m),
            nn.LeakyReLU(0.01),
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        ### angle branch
        self.angle = nn.Sequential(
            nn.Linear(2*m, 2*m, bias=True),
            norm_fn(2*m),
            nn.LeakyReLU(0.01),
        )
        self.angle_linear = nn.Linear(2*m, 12*2)

        ### zs + score + bbox branch
        self.z_in = spconv.SparseSequential(
            spconv.SubMConv3d(2*m+3, 4*m, kernel_size=3, padding=1, bias=False, indice_key='z_subm1')
        )

        self.z_net = UBlock([4*m, 4*m, 6*m, 8*m], norm_fn, 2, block, indice_key_id=1)

        self.z_out = spconv.SparseSequential(
            norm_fn(4*m),
            nn.LeakyReLU(0.01),
        )

        self.z_score = nn.Sequential(
            nn.Linear(4*m+6, 1),
        )        

        self.z_linear = nn.Sequential(
            nn.Linear(4*m+6, 8*m),
            nn.LeakyReLU(0.01),
            nn.Linear(8*m, 16*m),
            nn.LeakyReLU(0.01),
            nn.Linear(16*m, 32*m),
            nn.LeakyReLU(0.01),
            nn.Linear(32*m, 256), # zs
        )

        self.z_box = nn.Sequential(
            nn.Linear(4*m+6, 8*m),
            nn.LeakyReLU(0.01),
            nn.Linear(8*m, 16*m),
            nn.LeakyReLU(0.01),
            nn.Linear(16*m, 56),
        )

        #### init

        self.apply(self.set_bn_init)

        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
                      'angle': self.angle, 'angle_linear': self.angle_linear,
                      'z_net': self.z_net, 'z_linear': self.z_linear, 'z_in': self.z_in, 'z_out': self.z_out, 'z_score': self.z_score, 'z_box': self.z_box,
                      }
        
        #### fix weights
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Loaded pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))
        
        ### BSP Net
        bsp_args = AttributeDict({
            'num_classes': 8,
            'mise_resolution_0': 32, # for evaluation
            'mise_upsampling_steps': 0, # for evaluation
            'mise_threshold': 0.5, # bsp assume occ in [0, 1] (not applying sigmoid!)
            'num_planes': 4096,
            'num_convexes': 256,
            'num_feats': 32,
            'mesh_gen': 'bspt', # 'bspt' or 'mcubes'
            'sample': cfg.sample,
        })
        self.comp_net = CompNet(bsp_args)

        ### fix bsp generator
        for param in self.comp_net.parameters():
            param.requires_grad = False
        
        ### load pretrained bsp net
        bsp_pretrain_dict = torch.load(os.path.join('datasets/bsp', 'model.pth'))
        self.comp_net.load_state_dict(bsp_pretrain_dict, strict=False)
        print(f"Loaded pretrained BSP-Net: #params = {sum([p.numel() for p in self.comp_net.parameters()])}")


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, semantics, angles, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda().long()

        clusters_points_feats = feats[c_idxs] # [sumNPoint, C]
        clusters_points_angles_label = angles[c_idxs, :12] # [sumNPoint, 12]
        clusters_points_angles_residual = angles[c_idxs, 12:] # [sumNPoint, 12]
        clusters_points_coords = coords[c_idxs] # [sumNPoint, 3]
        clusters_points_semantics = semantics[c_idxs] # [sumNPoint, 8]

        # get the semantic label of each proposal
        clusters_semantics = pointgroup_ops.sec_mean(clusters_points_semantics, clusters_offset.cuda()) # [nCluster, 8]

        # get mean angle as the bbox angle
        clusters_points_angles_label = torch.softmax(clusters_points_angles_label, dim=1)
        clusters_angles_label_mean = pointgroup_ops.sec_mean(clusters_points_angles_label, clusters_offset.cuda())  # (nCluster, 12), float
        clusters_angles_residual_mean = pointgroup_ops.sec_mean(clusters_points_angles_residual, clusters_offset.cuda())  # (nCluster, 12), float

        # decode angles
        clusters_angles_label_mean = torch.argmax(clusters_angles_label_mean, dim=1) # [nCluster, ] long
        clusters_angles_residual_mean = torch.gather(clusters_angles_residual_mean * np.pi / 12, 1, clusters_angles_label_mean.unsqueeze(1)).squeeze(1)
        # detach !!!
        clusters_angles = BBox.class2angle_cuda(clusters_angles_label_mean, clusters_angles_residual_mean).detach()

        clusters_points_angles = torch.index_select(clusters_angles, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint,), float

        clusters_coords_min_ori = pointgroup_ops.sec_min(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max_ori = pointgroup_ops.sec_max(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        #clusters_coords_mean = pointgroup_ops.sec_mean(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_centroid = (clusters_coords_max_ori + clusters_coords_min_ori) / 2 # (nCluster, 3), float

        clusters_points_centroid = torch.index_select(clusters_centroid, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float

        
        # center coords
        clusters_points_coords -= clusters_points_centroid

        cos_ = torch.cos(-clusters_points_angles)
        sin_ = torch.sin(-clusters_points_angles)
        
        clusters_points_coords[:, 0], clusters_points_coords[:, 1] = clusters_points_coords[:, 0] * cos_ - clusters_points_coords[:, 1] * sin_, clusters_points_coords[:, 0] * sin_ + clusters_points_coords[:, 1] * cos_

        # concat canonical coords
        clusters_points_feats = torch.cat([clusters_points_feats, clusters_points_coords], dim=1)

        clusters_coords_min = pointgroup_ops.sec_min(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_bbox_size = clusters_coords_max - clusters_coords_min

        clusters_scale = 1 / (clusters_bbox_size / fullscale).max(1)[0]  # (nCluster), float
        clusters_scale = clusters_scale.unsqueeze(-1)
        
        min_xyz = clusters_coords_min * clusters_scale  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale
        
        clusters_points_coords = clusters_points_coords * torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        range_xyz = max_xyz - min_xyz
        offset = - min_xyz

        
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_points_coords += offset

        clusters_points_coords = clusters_points_coords.long()
        clusters_points_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_points_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_points_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_points_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map, clusters_angles, clusters_centroid, clusters_bbox_size, clusters_semantics


    def forward(self, data, training_mode='train'):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda

        phase1: semantic segmentation, point-wise offset, point-wise angle.
        phase2: instance clustering, confidence score, bbox params.
        '''
        input = data['input']
        input_map = data['input_map']
        coords = data['coords']
        batch_idxs = data['batch_idxs']
        #batch_offsets = data['batch_offsets']
        epoch = data['epoch']

        ret = {}

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long

        semantic_preds_CAD = torch.from_numpy(RFS2CAD_arr[semantic_preds.cpu()]).long().cuda()
        semantic_preds_CAD[semantic_preds_CAD == -1] = 8
        semantic_scores_CAD = F.one_hot(semantic_preds_CAD, 9).float()

        ret['semantic_scores'] = semantic_scores
        ret['semantic_scores_CAD'] = semantic_scores_CAD

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)   # (N, 3), float32

        ret['pt_offsets'] = pt_offsets

        #### angle
        pt_angles_feats = self.angle(output_feats)
        pt_angles = self.angle_linear(pt_angles_feats) # (N, 24) float32

        ### mask non-CAD instances' pt_angles to 0.
        angle_mask = (semantic_preds_CAD != 8).to(semantic_preds.device).float()
        pt_angles = pt_angles * angle_mask.unsqueeze(-1)

        ret['pt_angles'] = pt_angles

        if epoch > self.prepare_epochs:
            #### get prooposal clusters
            object_idxs = torch.nonzero(semantic_preds_CAD != 8).view(-1) # only get CAD objects, mask out floor and wall and non-CADs.

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = coords[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            semantic_preds_ = semantic_preds_CAD[object_idxs].int().cpu()


            # single-scale proposal gen (pointgroup)
            if self.training:

                ### BFS clustering on shifted coords
                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int() # remap: sumNPoint --> N

                # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: (nProposal + 1), int, start/end index for each proposal, e.g., [0, c1, c1+c2, ..., c1+...+c_nprop = sumNPoint], same information as cluster_id, just in the convinience of cuda operators.

                ### BFS clustering on original coords
                idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
                proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                ### merge two type of clusters
                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
            
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)#.long().cuda()
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]), dim=0)#.cuda()
                # why [1:]: offset is (0, c1, c2), offset_shift is (0, d1, d2) + c2, output is (0, c1, c2, c2+d1, c2+d2)

            # multi-scale proposal gen (naive, maskgroup)
            else:

                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, 0.01, self.cluster_shift_meanActive)
                proposals_idx_shift_0, proposals_offset_shift_0 = pointgroup_ops.bfs_cluster(semantic_preds_, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift_0[:, 1] = object_idxs[proposals_idx_shift_0[:, 1].long()].int()
                
                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, 0.03, self.cluster_shift_meanActive)
                proposals_idx_shift_1, proposals_offset_shift_1 = pointgroup_ops.bfs_cluster(semantic_preds_, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift_1[:, 1] = object_idxs[proposals_idx_shift_1[:, 1].long()].int()
                
                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, 0.05, self.cluster_shift_meanActive)
                proposals_idx_shift_2, proposals_offset_shift_2 = pointgroup_ops.bfs_cluster(semantic_preds_, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift_2[:, 1] = object_idxs[proposals_idx_shift_2[:, 1].long()].int()
                    
                idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, 0.03, self.cluster_meanActive)
                proposals_idx_0, proposals_offset_0 = pointgroup_ops.bfs_cluster(semantic_preds_, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx_0[:, 1] = object_idxs[proposals_idx_0[:, 1].long()].int()
                    
                _offset = proposals_offset_0.size(0) - 1
                proposals_idx_shift_0[:, 0] += _offset
                proposals_offset_shift_0 += proposals_offset_0[-1]
                
                _offset += proposals_offset_shift_0.size(0) - 1
                proposals_idx_shift_1[:, 0] += _offset
                proposals_offset_shift_1 += proposals_offset_shift_0[-1]
                
                _offset += proposals_offset_shift_1.size(0) - 1
                proposals_idx_shift_2[:, 0] += _offset
                proposals_offset_shift_2 += proposals_offset_shift_1[-1]
                
                proposals_idx = torch.cat((proposals_idx_0, proposals_idx_shift_0, proposals_idx_shift_1, proposals_idx_shift_2), dim=0)
                proposals_offset = torch.cat((proposals_offset_0, proposals_offset_shift_0[1:], proposals_offset_shift_1[1:], proposals_offset_shift_2[1:]))

        
            #### proposals voxelization again
            input_feats, inp_map, proposal_angle, point_center, point_scale, proposal_semantics = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, semantic_scores_CAD, pt_angles, coords, self.score_fullscale, self.score_scale, self.mode)

            ### zs
            proposal_out = self.z_in(input_feats)
            proposal_out = self.z_net(proposal_out)
            proposal_out = self.z_out(proposal_out)
            proposal_out = proposal_out.features[inp_map.long()] # (sumNPoint, C)
            proposal_out = pointgroup_ops.roipool(proposal_out, proposals_offset.cuda())  # (nProposal, C) proposal-wise max_pooling

            # concat cell & semantics
            proposal_out = torch.cat([proposal_out, point_center, point_scale], dim=1)

            ### score
            scores = self.z_score(proposal_out)
            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset, proposal_semantics)

            ### CAD bbox
            if epoch > self.prepare_epochs_2:
                proposal_zs = self.z_linear(proposal_out) # (nProposal, 256+...)

                residual_bbox = self.z_box(proposal_out)
                residual_center = residual_bbox[:, 0: 8*3]
                residual_scale = residual_bbox[:, 8*3: 8*6]
                residual_angle = residual_bbox[:, 8*6: 8*7]

                ret['proposal_bbox'] = (proposal_zs, proposal_angle, point_center, point_scale, residual_center, residual_scale, residual_angle)

        return ret

def huber_loss(error, delta=1.0):
    """
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:`
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """

    # handle (N,C) (M,C)
    if len(pc1.shape) == 2: pc1 = pc1.unsqueeze(0)
    if len(pc2.shape) == 2: pc2 = pc2.unsqueeze(0)
    
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
    return dist1, idx1, dist2, idx2    

def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()
    label_criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        ### assert batch_size == 1

        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        rgbs = batch['feats'].cuda()              # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = coords_float

        if cfg.use_rgb:
            feats = torch.cat((rgbs, feats), 1)

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        model_inp = {
            'input': input_,
            'input_map': p2v_map,
            'coords': coords_float,
            'batch_idxs': coords[:, 0].int(),
            'batch_offsets': batch_offsets,
            'epoch': epoch,
        }

        ret = model(model_inp, 'test')

        semantic_scores = ret['semantic_scores_CAD']  # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda

        pt_angles = ret['pt_angles'] # [N, 24]
        pt_angles_label = torch.argmax(F.softmax(pt_angles[:, :12], dim=1), dim=1) # [N,]
        pt_angles_residual = torch.gather(pt_angles[:, 12:] * np.pi / 12, 1, pt_angles_label.unsqueeze(-1)).squeeze(1) # [N, 12] --> [N, 1] --> [N,]
        pt_angles = BBox.class2angle_cuda(pt_angles_label, pt_angles_residual).detach() # [N]

        semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda

        if epoch > cfg.prepare_epochs:
            scores, proposals_idx, proposals_offset, proposals_semantics = ret['proposal_scores']
            
            scores_pred = torch.sigmoid(scores.view(-1))

            N = coords.shape[0]
            proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) # (nProposal, N), int, cuda
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

            cluster_semantic_score = proposals_semantics
            cluster_semantic_id = cluster_semantic_score.max(1)[1]

            ##### score threshold

            score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
            scores_pred = scores_pred[score_mask]
            proposals_pred = proposals_pred[score_mask]
            cluster_semantic_id = cluster_semantic_id[score_mask]
            cluster_semantic_score = cluster_semantic_score[score_mask]
            
            ##### npoint threshold
            proposals_pointnum = proposals_pred.sum(1)
            npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
            scores_pred = scores_pred[npoint_mask]
            proposals_pred = proposals_pred[npoint_mask]
            cluster_semantic_id = cluster_semantic_id[npoint_mask]
            cluster_semantic_score = cluster_semantic_score[npoint_mask]

            ##### nms
            if cluster_semantic_id.shape[0] == 0:
                pick_idxs = np.empty(0)
            else:
                proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), cfg.TEST_NMS_THRESH)  # int, (nCluster, N)

            clusters = proposals_pred[pick_idxs] # (nProp, N)
            cluster_scores = scores_pred[pick_idxs] # (nProp, )
            cluster_semantic_id = cluster_semantic_id[pick_idxs] # (nProp, ) RFS id
            cluster_semantic_score = cluster_semantic_score[pick_idxs] # (nProp, C)

            n_clusters = clusters.shape[0]

            if epoch > cfg.prepare_epochs_2:
                cluster_zs, cluster_angle, cluster_center, cluster_scale, cluster_residual_center, cluster_residual_scale, cluster_residual_angle = ret['proposal_bbox']

                cluster_zs = cluster_zs[score_mask][npoint_mask][pick_idxs]
                cluster_angle = cluster_angle[score_mask][npoint_mask][pick_idxs]
                cluster_center = cluster_center[score_mask][npoint_mask][pick_idxs]
                cluster_scale = cluster_scale[score_mask][npoint_mask][pick_idxs]
                cluster_residual_center = cluster_residual_center[score_mask][npoint_mask][pick_idxs]
                cluster_residual_scale = cluster_residual_scale[score_mask][npoint_mask][pick_idxs]
                cluster_residual_angle = cluster_residual_angle[score_mask][npoint_mask][pick_idxs]

                ### generate meshes

                # restore predicted bbox
                cluster_angle = cluster_angle.detach().cpu().numpy() # [nProp, 3]
                cluster_center = cluster_center.detach().cpu().numpy() # [nProp, 3]
                cluster_scale = cluster_scale.detach().cpu().numpy() # [nProp, 3]
                cluster_residual_center = cluster_residual_center.detach().cpu().numpy().reshape(n_clusters, 8, 3) # [nProp, 8, 3]
                cluster_residual_scale = cluster_residual_scale.detach().cpu().numpy().reshape(n_clusters, 8, 3) # [nProp, 8, 3]
                cluster_residual_angle = cluster_residual_angle.detach().cpu().numpy().reshape(n_clusters, 8) # [nProp, 8]

                cluster_bboxes = np.zeros((n_clusters, 7))

                cluster_semantic_id_CAD = cluster_semantic_id

                for cid in range(n_clusters):

                    # ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']
                    cluster_label = cluster_semantic_id_CAD[cid]

                    ### center
                    cluster_bboxes[cid, 0:3] = cluster_center[cid] + cluster_residual_center[cid, cluster_label]

                    ### scale (soft)
                    residual_scale = np.maximum(cluster_residual_scale[cid, cluster_label], - 0.1 * cluster_scale[cid])
                    cluster_bboxes[cid, 3:6] = cluster_scale[cid] + residual_scale
                    
                    ### rotation (w/o residual)
                    cluster_bboxes[cid, 6] = cluster_angle[cid] #+ cluster_residual_angle[cid, cluster_label]
                
       

                data = {
                    'feats': cluster_zs, # (nProposal, 256)
                    'bboxes': cluster_bboxes, # (nProposal, 7)
                    'labels': cluster_semantic_id, # (nProposal, )
                }

                if cfg.retrieval:
                    bsp_out = model.comp_net(data, phase=1, return_mesh='retrieve') # (nProposal_valid, )
                    cluster_meshes = bsp_out['meshes']
                    # prepare infos for retrieval alignment evaluation.
                    cluster_mesh_ids = bsp_out['mesh_ids'] # ['cat_id/id', ...]
                    cluster_alignment = []
                    for cid in range(clusters.shape[0]):
                        if cluster_mesh_ids[cid] is None:
                            continue
                        else:
                            cat_id, obj_id = cluster_mesh_ids[cid].split('/')
                            tx, ty, tz, sx, sy, sz, rot = cluster_bboxes[cid, :7]
                            qw, qx, qy, qz = pyquaternion.Quaternion(axis=[0, 0, 1], radians=rot).elements
                            alignment = [cat_id, obj_id, tx, ty, tz, qw, qx, qy, qz, sx, sy, sz]
                            cluster_alignment.append(alignment)
                    cluster_alignment = pd.DataFrame(cluster_alignment)
                    
                else:
                    cluster_meshes = model.comp_net(data, phase=1, return_mesh='generate', projection_k=cfg.k_projection)['meshes'] # (nProposal_valid, )
                
                ### mesh post-processings.

                # ICP 

                for cid in range(clusters.shape[0]):
                    mesh = cluster_meshes[cid]

                    if mesh is None: 
                        continue

                    target_pc = coords_float[clusters[cid].byte()].detach().cpu().numpy()
                    tmesh = mesh.to_trimesh() if isinstance(mesh, PolyMesh) else mesh

                    # o3d ICP
                    source_pc, _ = trimesh.sample.sample_surface_even(tmesh, 2048)
                    
                    transformed_vertices, fitness = icp(source_pc, target_pc, mesh.vertices)
                    mesh.vertices = transformed_vertices

                    cluster_meshes[cid] = mesh

        ##### preds
        preds = {}
        preds['semantic_pred'] = semantic_pred
        preds['pt_offsets'] = pt_offsets
        preds['pt_angles'] = pt_angles
        if epoch > cfg.prepare_epochs:
            preds['clusters'] = clusters
            preds['cluster_scores'] = cluster_scores
            preds['cluster_semantic_id'] = cluster_semantic_id
            preds['cluster_semantic_score'] = cluster_semantic_score
            if epoch > cfg.prepare_epochs_2:
                preds['cluster_meshes'] = cluster_meshes
                if cfg.retrieval:
                    preds['cluster_alignment'] = cluster_alignment

        return preds


    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        rgbs = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        instance_zs = batch['instance_zs'].cuda()  # (total_nInst, 256), float, cuda
        instance_zs_valid = batch['instance_zs_valid'].cuda()  # (total_nInst), int, cuda
        
        instance_bbox_size = batch['instance_bbox_size'].cuda()
        instance_bboxes = batch['instance_bboxes'].cuda()

        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = coords_float
            
        if cfg.use_rgb:
            feats = torch.cat((rgbs, feats), 1)

            
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ### model call 
        model_inp = {
            'input': input_,
            'input_map': p2v_map,
            'coords': coords_float,
            'batch_idxs': coords[:, 0].int(),
            'batch_offsets': batch_offsets,
            'epoch': epoch,
        }

        ret = model(model_inp)

        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
        pt_angles = ret['pt_angles']             # (N, 24)

        if epoch > cfg.prepare_epochs:
            scores, proposals_idx, proposals_offset, proposals_semantics = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            if epoch > cfg.prepare_epochs_2:
                pred_zs, pred_angle, pred_center, pred_scale, pred_residual_center, pred_residual_scale, pred_residual_angle = ret['proposal_bbox']

        loss_inp = {}
        loss_inp['semantic_scores'] = (semantic_scores, labels)
        loss_inp['pt_offsets'] = (pt_offsets, pt_angles, coords_float, instance_info, instance_labels)

        if epoch > cfg.prepare_epochs:
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)
            if epoch > cfg.prepare_epochs_2:
                loss_inp['proposal_bbox'] = (pred_zs, instance_zs, instance_zs_valid, \
                                             pred_angle, pred_center, pred_scale, pred_residual_center, pred_residual_scale, pred_residual_angle, instance_bboxes, instance_bbox_size)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if epoch > cfg.prepare_epochs:
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, pt_angles, coords_float, instance_info, instance_labels = loss_inp['pt_offsets']

        gt_offsets = instance_info[:, 0:3] - coords_float   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        #pt_valid = (instance_labels != cfg.ignore_label).float()
        pt_valid = (instance_info[:, 0] != -100).float()
        offset_norm_loss = torch.sum(pt_dist * pt_valid) / (torch.sum(pt_valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * pt_valid) / (torch.sum(pt_valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, pt_valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, pt_valid.sum())

        '''angle loss'''
        pt_valid = (instance_info[:, 9] != -100).float() # only supervise CAD-instances, class balanced.
        gt_angle_label = instance_info[:, 9].long()
        gt_angle_label[gt_angle_label == -100] = 0 # invalid angles, will be masked out later
        gt_angle_residual = instance_info[:, 10]

        angle_label = pt_angles[:, :12]
        angle_residual = pt_angles[:, 12:]

        angle_label_loss = label_criterion(angle_label, gt_angle_label)
        angle_label_loss = (angle_label_loss * pt_valid).sum() / (pt_valid.sum() + 1e-6)

        gt_angle_label_onehot = F.one_hot(gt_angle_label, 12).float()
        angle_residual = (angle_residual * gt_angle_label_onehot).sum(1) # [nProp, 12] --> [nProp, ]
        gt_angle_residual = gt_angle_residual / (np.pi/12) # normalized residual
        angle_residual_loss = huber_loss(angle_residual - gt_angle_residual)
        angle_residual_loss = (angle_residual_loss * pt_valid).sum() / (pt_valid.sum() + 1e-6)

        loss_out['angle_label_loss'] = (angle_label_loss, pt_valid.sum())
        loss_out['angle_residual_loss'] = (angle_residual_loss, pt_valid.sum())

        if epoch > cfg.prepare_epochs:
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int
            # pred_zs: (nProposal, 256)

            ### match GT by max IoU (point cloud)
            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long

            # score loss
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh) # [nProp]

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores).mean()
            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

            if epoch > cfg.prepare_epochs_2:
                
                pred_zs, instance_zs, instance_zs_valid, \
                pred_angle, pred_center, pred_scale, pred_residual_center, pred_residual_scale, pred_residual_angle, instance_bboxes, instance_bbox_size = loss_inp['proposal_bbox']

                # decode center & scale
                gt_bboxes = instance_bboxes[gt_instance_idxs] # [nProp, 7], the original GT bbox.
                gt_bbox_label = instance_bbox_size[gt_instance_idxs] # [nProp, ], the GT bbox label

                pred_center = pred_center + torch.gather(pred_residual_center.view(-1, 8, 3), 1, gt_bbox_label.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3)).squeeze(1) # [nProp, 3]
                pred_scale = pred_scale + torch.gather(pred_residual_scale.view(-1, 8, 3), 1, gt_bbox_label.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3)).squeeze(1) # [nProp, 3]
                class_balance_weights = torch.from_numpy(CAD_weights[gt_bbox_label.detach().cpu().numpy()]).float().to(gt_bbox_label.device)

                # z loss
                gt_zs = instance_zs[gt_instance_idxs]
                gt_zs_valid = instance_zs_valid[gt_instance_idxs]

                z_loss = huber_loss(pred_zs - gt_zs).mean(dim=1) # [nProposal, 264] --> [nProposal]
                z_loss = z_loss * class_balance_weights
                z_loss = (z_loss * gt_scores * gt_zs_valid).sum() / ((gt_scores * gt_zs_valid).sum() + 1e-6) # masked loss, also * gt_scores to make it soft.

                loss_out['z_loss'] = (z_loss, ((gt_scores * gt_zs_valid).sum() + 1e-6))


                # center loss: single directional. Not bi-directional (do not use gt_bboxes, use unmatched bboxes)
                center_loss = huber_loss(pred_center - gt_bboxes[:, 0:3]).mean(1)
                center_loss = center_loss * class_balance_weights
                center_loss = (center_loss * gt_scores * gt_zs_valid).sum() / ((gt_scores * gt_zs_valid).sum() + 1e-6)

                # class-balanced scale loss:
                scale_loss = huber_loss(pred_scale - gt_bboxes[:, 3:6]).mean(1) # [nProp, 3] --> [nProp]
                scale_loss = scale_loss * class_balance_weights
                scale_loss = (scale_loss * gt_scores * gt_zs_valid).sum() / ((gt_scores * gt_zs_valid).sum() + 1e-6)
                
                loss_out['center_loss'] = (center_loss, ((gt_scores * gt_zs_valid).sum() + 1e-6))
                loss_out['scale_loss'] = (scale_loss, ((gt_scores * gt_zs_valid).sum() + 1e-6))


        # total loss
        loss = semantic_loss + offset_norm_loss + offset_dir_loss + angle_label_loss + angle_residual_loss

        if epoch > cfg.prepare_epochs:
            loss += score_loss
            if epoch > cfg.prepare_epochs_2:
                loss += z_loss
                loss += center_loss
                loss += scale_loss

        return loss, loss_out, infos

    ### gt_scores, 
    def get_segmented_scores(scores, fg_thresh=0.75, bg_thresh=0.25):
        '''
        :param scores: (N), float, 0~1, the max IoU !!!
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh # fg == positive, valid match
        bg_mask = scores < bg_thresh # bg == negative, not a valid match
        interval_mask = (fg_mask == 0) & (bg_mask == 0) # between fg & bg, maybe a valid match

        segmented_scores = (fg_mask > 0).float() # hard threshold, score --> 1 for valid match

        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b # soft threshold

        return segmented_scores


    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
