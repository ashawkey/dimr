'''
Visualization
Written by Li Jiang
'''

import numpy as np
import os, glob, argparse
import torch
from operator import itemgetter
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm

# instance
COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])


SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                        'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 
                        'kitchen_cabinet', 'display', 'trash_bin', 'other_shelf', 'other_table'])


CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177], 
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160],
    'kitchen_cabinet': [0, 60, 88],
    'display': [128, 128, 0],
    'trash_bin': [128, 128, 128],
    'other_shelf': [75, 51, 92],
    'other_table': [128, 76, 76],
}

def read_txt(file):
    with open(file, 'r') as f:
        output = [x.strip() for x in f.readlines()]
    return output


def get_coords_color(opt):

    scan_data = np.load(os.path.join('./datasets/scannet/processed_data', opt.room_name, 'data.npz'))
    #scan_data = np.load(os.path.join('./datasets_old/scannet/processed_data', opt.room_name, 'rfs_data.npz'))
        
    point_cloud = scan_data['mesh_vertices']
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]
    print(xyz.shape)
    
    if opt.room_split != 'test':
        inst_label = scan_data['instance_labels'].astype(np.int32) - 1
        inst_label[inst_label == -1] = -100
        label = scan_data['semantic_labels'].astype(np.int32)
        label[label == 255] = -100

    if (opt.task == 'semantic_gt'):
        assert opt.room_split != 'test'
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'instance_gt'):
        assert opt.room_split != 'test'
        inst_label = inst_label.astype(int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb
    
    elif (opt.task == 'angle_gt'):
        assert opt.room_split != 'test'
        angle_rgb = np.zeros(rgb.shape)
        angle_file = os.path.join('./gt_angles', opt.room_name + '.npy')
        assert os.path.isfile(angle_file), 'No angle result - {}.'.format(angle_file)
        angle = np.load(angle_file) # [N], radian in [-pi, pi]
        angle = (angle + np.pi) / (2 * np.pi) # in [0, 1]
        angle_rgb = (cm.bwr(angle)[:, :3] * 255).astype(int)
        rgb = angle_rgb        

    elif (opt.task == 'angle_pred'):
        assert opt.room_split != 'train'
        angle_file = os.path.join(opt.result_root, opt.room_split, 'angles', opt.room_name + '.npy')
        assert os.path.isfile(angle_file), 'No angle result - {}.'.format(angle_file)
        angle = np.load(angle_file) # [N], radian in [-pi, pi]
        angle = (angle + np.pi) / (2 * np.pi) # in [0, 1]
        angle_rgb = (cm.bwr(angle)[:, :3] * 255).astype(int)
        rgb = angle_rgb        

    elif (opt.task == 'semantic_pred'):
        assert opt.room_split != 'train'
        semantic_file = os.path.join(opt.result_root, opt.room_split, 'semantic', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (opt.task == 'instance_pred'):
        assert opt.room_split != 'train'
        instance_file = os.path.join(opt.result_root, opt.room_split, opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #
        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(opt.result_root, opt.room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(int)
            #print('{} {}: {} pointnum: {}'.format(i, masks[i], int(masks[i][1]), mask.sum()))
            inst_label_pred_rgb[mask == 1] = COLOR20[i % len(COLOR20)]
        rgb = inst_label_pred_rgb

    # if opt.room_split != 'test':
    #     sem_valid = (label != -100)
    #     xyz = xyz[sem_valid]
    #     rgb = rgb[sem_valid]

    return xyz, rgb


def save(opt):
    points, colors = get_coords_color(opt)

    vertices = np.empty(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = points[:, 0].astype('f4')
    vertices['y'] = points[:, 1].astype('f4')
    vertices['z'] = points[:, 2].astype('f4')
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')

    # save as ply
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)

    ply.write(os.path.join(outdir, f"{opt.room_name}_{opt.task}.ply"))
    print(f"[visualize] saved {opt.room_name}_{opt.task}.ply")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', help='path to the predicted results', default='exp/scannetv2/rfs/rfs_phase2_scannet/result/epoch200_nmst0.3_scoret0.05_npointt100/')
    parser.add_argument('--room_split', help='train / val / test', default='val')
    parser.add_argument('--room_name', help='room_name', default='scene0553_01')
    parser.add_argument('--task', help='input / semantic_gt / semantic_pred / instance_gt / instance_pred', default='semantic_gt')
    opt = parser.parse_args()

    outdir = './visualization'
    os.makedirs(outdir, exist_ok=True)

    if opt.room_name == 'all':
        import glob
        
        room_names = read_txt(os.path.join('./datasets/splits/', 'val.txt'))
        for name in room_names:
            opt.room_name = name
            print(f'[INFO] process {name}...')
            save(opt)

    else:
        save(opt)
