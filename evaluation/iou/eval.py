# offline mesh IoU eval

import open3d as o3d
import numpy as np
import os
import glob
import trimesh

from metrics import *

def extract_label(f):
    clsname = f[:-4].split('_')[3]
    if clsname == 'trash': clsname = 'trash_bin'
    return CAD_labels.index(clsname)


def extract_score(f):
    # scene0652_00_cabinet_0.1308.ply --> 0.1308
    return float(f[:-4].split('_')[-1])


def eval(gt_dir, pred_dir, voxel_size, threshs=[0.25, 0.5]):
    
    log_file = open(f'eval_log_iou.txt', 'a')

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]

    # collect meshes (ply)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.ply')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))

    data = {}

    for f in gt_files:
        scene_name = os.path.basename(f)[:12]
        if scene_name not in data: 
            data[scene_name] = {}
        if 'gt' not in data[scene_name]: 
            data[scene_name]['gt'] = []
        if 'pred' not in data[scene_name]: 
            data[scene_name]['pred'] = []
        data[scene_name]['gt'].append(f)

    for f in pred_files:
        scene_name = os.path.basename(f)[:12]
        data[scene_name]['pred'].append(f)

    scene_names = data.keys()
    #scene_names = ['scene0011_00']

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):

        # gt mesh
        gt_files_scene = data[scene_name]['gt']

        gt_meshes = []
        gt_labels = []
        gt_bboxes = []
        for f in gt_files_scene:
            mesh = trimesh.load(f, process=False)
            
            # mesh expansion !!!!!!
            #mesh.vertices = mesh.vertices + 0.01 * mesh.vertex_normals

            gt_meshes.append(mesh)
            gt_bboxes.append(mesh.bounds.reshape(-1))
            gt_labels.append(extract_label(os.path.basename(f)))
        
        gt_valid_mask = np.ones(len(gt_meshes))[None, :] # [B, M]
        gt_labels = np.array(gt_labels)[None, :] # [B, M], in CAD ids [0, 7]
        gt_meshes = [gt_meshes] # [B, M]
        gt_bboxes = np.array(gt_bboxes)[None, :] # [B, M, 6]

        info_mesh_gts = batched_prepare_gt(gt_valid_mask, gt_labels, gt_bboxes, gt_meshes, voxel_size=voxel_size)


        # pred mesh
        pred_files_scene = data[scene_name]['pred']
        pred_meshes = []
        pred_labels = []
        pred_bboxes = []
        pred_scores = []
        for f in pred_files_scene:
            pred_labels.append(extract_label(os.path.basename(f)))
            pred_scores.append(extract_score(os.path.basename(f)))

            mesh = trimesh.load(f, process=False)

            pred_meshes.append(mesh)
            pred_bboxes.append(mesh.bounds.reshape(-1))
        
        pred_valid_mask = np.ones(len(pred_meshes))[None, :] # [B, M]
        pred_labels = np.array(pred_labels)[None, :] # [B, M], in CAD ids [0, 7]
        pred_meshes = [pred_meshes] # [B, M]
        pred_bboxes = np.array(pred_bboxes)[None, :] # [B, M, 6]
        pred_scores = np.array(pred_scores)[None, :] # [B, M]

        info_mesh_preds = batched_prepare_pred(pred_valid_mask, pred_labels, pred_bboxes, pred_scores, pred_meshes, voxel_size=voxel_size)

        # record
        for calc in ap_calculator_list:
            calc.step(info_mesh_preds, info_mesh_gts)

        print(f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(gt_files_scene)},  #pred = {len(pred_files_scene)}')

    # output
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- IoU thresh = {threshs[i]} -----')
        print(f'----- IoU thresh = {threshs[i]} -----', file=log_file)
        metrics_dict = calc.compute_metrics()
        for k, v in metrics_dict.items():
            if 'Q_mesh' in k: continue
            if 'mesh' not in k: continue
            print(f"{k: <50}: {v}")
            print(f"{k: <50}: {v}", file=log_file)
    
    log_file.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str)
    parser.add_argument('pred_dir', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.047)
    args = parser.parse_args()

    eval(args.gt_dir, args.pred_dir, args.voxel_size)


