# offline mesh IoU eval

import open3d as o3d
import numpy as np
import os
import glob
from torch import gt
import trimesh

from metrics import *

def extract_label(f):
    clsname = f[:-4].split('_')[3]
    if clsname == 'trash': clsname = 'trash_bin'
    return CAD_labels.index(clsname)

def extract_score(f):
    # scene0652_00_cabinet_0.1308.ply --> 0.1308
    return float(f[:-4].split('_')[-1])

def read_txt(file):
    with open(file, 'r') as f:
        output = [x.strip() for x in f.readlines()]
    return output

def eval(pred_dir, threshs=[0.5]):

    log_file = open(f'eval_log_pcr.txt', 'a')

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]

    # collect meshes (ply)
    test_scans = read_txt(os.path.join('datasets/splits/', 'val.txt'))
    
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))

    data = {}

    for scene_name in test_scans:
        data[scene_name] = []
        
    for f in pred_files:
        scene_name = os.path.basename(f)[:12]
        data[scene_name].append(f)

    scene_names = data.keys()
    #scene_names = ['scene0011_00']

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):

        # gt points
        info_mesh_gts = []
        scan_data = np.load(f'datasets/scannet/processed_data/{scene_name}/data.npz')
        xyz = scan_data['mesh_vertices'].astype(np.float32)[:, :3]
        semantic_label = scan_data['semantic_labels'].astype(np.int32)
        instance_label = scan_data['instance_labels'].astype(np.int32) - 1
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_) # returns a one-element tuple, like ([0,1,29,43,...],)
            if len(inst_idx_i[0]) == 0: continue # null instance
            xyz_i = xyz[inst_idx_i] # [Ni, 3]
            label_i = semantic_label[inst_idx_i[0]][0]
            if label_i == 255: continue # ignored class
            label_i = RFS2CAD[label_i]
            if label_i == -1: continue # non-CAD class
            info_mesh_gts.append((label_i, xyz_i))


        # pred mesh
        info_mesh_preds= []
        pred_files_scene = data[scene_name]
        for f in pred_files_scene:
            mesh = trimesh.load(f, process=False)
            info_mesh_preds.append((extract_label(os.path.basename(f)), mesh, extract_score(os.path.basename(f))))
    
        # record
        for calc in ap_calculator_list:
            calc.step([info_mesh_preds], [info_mesh_gts])

        print(f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(info_mesh_gts)},  #pred = {len(info_mesh_preds)}')

    # output
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- thresh = {threshs[i]} -----')
        print(f'----- thresh = {threshs[i]} -----', file=log_file)
        metrics_dict = calc.compute_metrics()
        for k, v in metrics_dict.items():
            print(f"{k: <50}: {v}")
            print(f"{k: <50}: {v}", file=log_file)
    
    log_file.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir', type=str)
    args = parser.parse_args()

    eval(args.pred_dir)


