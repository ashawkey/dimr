import sys
import os
import re
import csv
import json
import pickle
import quaternion
import numpy as np
import pandas as pd
from glob import glob
from functools import partial
from multiprocessing import Pool
from plyfile import PlyData, PlyElement
from shapely.geometry.polygon import Polygon  


ShapeNetLabels = ['void', # for non-exist classes, e.g., wall and floor
                   'table', 'jar', 'skateboard', 'car', 'bottle',
                   'tower', 'chair', 'bookshelf', 'camera', 'airplane',
                   'laptop', 'basket', 'sofa', 'knife', 'can',
                   'rifle', 'train', 'pillow', 'lamp', 'trash_bin',
                   'mailbox', 'watercraft', 'motorbike', 'dishwasher', 'bench',
                   'pistol', 'rocket', 'loudspeaker', 'file cabinet', 'bag',
                   'cabinet', 'bed', 'birdhouse', 'display', 'piano',
                   'earphone', 'telephone', 'stove', 'microphone', 'bus',
                   'mug', 'remote', 'bathtub', 'bowl', 'keyboard',
                   'guitar', 'washer', 'bicycle', 'faucet', 'printer',
                   'cap', 'clock', 'helmet', 'flowerpot', 'microwaves']

ShapeNetIDMap = {'4379243': 'table', '3593526': 'jar', '4225987': 'skateboard', '2958343': 'car', '2876657': 'bottle', '4460130': 'tower', '3001627': 'chair', '2871439': 'bookshelf', '2942699': 'camera', '2691156': 'airplane', '3642806': 'laptop', '2801938': 'basket', '4256520': 'sofa', '3624134': 'knife', '2946921': 'can', '4090263': 'rifle', '4468005': 'train', '3938244': 'pillow', '3636649': 'lamp', '2747177': 'trash_bin', '3710193': 'mailbox', '4530566': 'watercraft', '3790512': 'motorbike', '3207941': 'dishwasher', '2828884': 'bench', '3948459': 'pistol', '4099429': 'rocket', '3691459': 'loudspeaker', '3337140': 'file cabinet', '2773838': 'bag', '2933112': 'cabinet', '2818832': 'bed', '2843684': 'birdhouse', '3211117': 'display', '3928116': 'piano', '3261776': 'earphone', '4401088': 'telephone', '4330267': 'stove', '3759954': 'microphone', '2924116': 'bus', '3797390': 'mug', '4074963': 'remote', '2808440': 'bathtub', '2880940': 'bowl', '3085013': 'keyboard', '3467517': 'guitar', '4554684': 'washer', '2834778': 'bicycle', '3325088': 'faucet', '4004475': 'printer', '2954340': 'cap', '3046257': 'clock', '3513137': 'helmet', '3991062': 'flowerpot', '3761084': 'microwaves'}

CAD2ShapeNet = np.array([ 1,  7,  8, 13, 20, 31, 34, 43])
ShapeNet2CAD = {class_id: i for i, class_id in enumerate(list(CAD2ShapeNet))}

# label_map: raw --> rfs --> NYU40 / Scan2CAD

def gen_rfs_labelmap(m):
    # m is 'scannetv2-labels.combined.tsv'
    m = m[['id', 'raw_category', 'ShapeNetCore55', 'synsetoffset', 'nyu40class', 'nyu40id']].copy()

    # fix different names
    m.loc[m['ShapeNetCore55'] == 'tv or monitor', 'ShapeNetCore55'] = 'display'
    m.loc[m['ShapeNetCore55'] == 'tub', 'ShapeNetCore55'] = 'bathtub'

    
    
    # only keep relevant ids
    nyu_labels = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    cad_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']

    nyu_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    cad_ids = [4379243, 3001627, 2871439, 4256520, 2747177, 2933112, 3211117, 2808440]

    m.loc[~m['nyu40class'].isin(nyu_labels), 'nyu40class'] = 'void'
    m.loc[~m['ShapeNetCore55'].isin(cad_labels), 'ShapeNetCore55'] = 'void'

    
    m['nyu_ids'] = -1
    m['cad_ids'] = -1
    for i, j in enumerate(nyu_ids):
        m.loc[m['nyu40id'] == j, 'nyu_ids'] = i
    for i, j in enumerate(cad_ids):
        m.loc[m['synsetoffset'] == j, 'cad_ids'] = i


    # fix Scan2CAD special cases...
    m.loc[m['nyu40class'] == 'toilet', 'ShapeNetCore55'] = 'chair'
    m.loc[m['nyu40class'] == 'toilet', 'cad_ids'] = 1
    m.loc[m['nyu40class'] == 'sink', 'ShapeNetCore55'] = 'bathtub'
    m.loc[m['nyu40class'] == 'sink', 'cad_ids'] = 7
    m.loc[m['nyu40class'] == 'counter', 'ShapeNetCore55'] = 'cabinet'
    m.loc[m['nyu40class'] == 'counter', 'cad_ids'] = 5
    m.loc[m['nyu40class'] == 'refridgerator', 'ShapeNetCore55'] = 'cabinet'
    m.loc[m['nyu40class'] == 'refridgerator', 'cad_ids'] = 5
    
    # make rfs label map
    m['rfs_labels'] = 'void'
    m['rfs_ids'] = -1

    for i in nyu_ids:
        m.loc[m['nyu40id'] == i, 'rfs_labels'] = m.loc[m['nyu40id'] == i, 'nyu40class']
    
    # manual fix: 5 new classes to the 20 original classes, to make it compatible with CAD.
    m.loc[m['raw_category'] == 'kitchen cabinet', 'rfs_labels'] = 'kitchen_cabinet'
    m.loc[m['raw_category'] == 'open kitchen cabinet', 'rfs_labels'] = 'kitchen_cabinet'

    m.loc[m['ShapeNetCore55'] == 'display', 'rfs_labels'] = 'display'
    m.loc[m['ShapeNetCore55'] == 'trash_bin', 'rfs_labels'] = 'trash_bin'

    m.loc[((m['ShapeNetCore55'] == 'bookshelf') & (m['nyu40class'] == 'void')), 'rfs_labels'] = 'other_shelf'
    m.loc[((m['ShapeNetCore55'] == 'table') & (m['nyu40class'] == 'otherfurniture')), 'rfs_labels'] = 'other_table'


    # reassign id
    rfs_labels = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'] + ['kitchen_cabinet', 'display', 'trash_bin', 'other_shelf', 'other_table']
    for i, j in enumerate(rfs_labels):
        m.loc[m['rfs_labels'] == j, 'rfs_ids'] = i

    #print(m)
    
    return m




class PathConfig(object):
    def __init__(self):

        self.metadata_root = 'datasets/scannet'
        self.ShapeNetv2_path = 'datasets/ShapeNetCore.v2'
        
        self.ScanNet_OBJ_CLASS_IDS = np.array([ 1,  7,  8, 13, 20, 31, 34, 43])
        
        self.scan2cad_annotation_path = os.path.join(self.metadata_root, 'scan2cad/full_annotations.json')
        self.processed_data_path = os.path.join(self.metadata_root, 'processed_data')

        self.raw_label_map_file = os.path.join(self.metadata_root, 'rfs_label_map.csv')

        # generate scannet2shapenetcore label map and save it
        if not os.path.exists(self.raw_label_map_file):
            LABEL_MAP_FILE = os.path.join(self.metadata_root, 'scannetv2-labels.combined.tsv')
            assert os.path.exists(LABEL_MAP_FILE)

            m = pd.read_csv(LABEL_MAP_FILE, sep='\t')
            m = gen_rfs_labelmap(m)
            m.to_csv(os.path.join(self.metadata_root, 'rfs_label_map.csv'))
            print(f'[INFO] generated RFS label map file at {os.path.join(self.metadata_root, "rfs_label_map.csv")}')

        if not os.path.exists(self.processed_data_path):
            os.mkdir(self.processed_data_path)


def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_json(file):
    '''
    read json file
    :param file: file path.
    :return:
    '''
    with open(file, 'r') as f:
        output = json.load(f)
    return output

def write_json(file, data):
    '''
    read json file
    :param file: file path.
    :param data: dict content
    :return:
    '''
    assert os.path.exists(os.path.dirname(file))

    with open(file, 'w') as f:
        json.dump(data, f)

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = row[label_to]
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M



def normalize(a, axis=-1, order=2):

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1

    if len(a.shape) == 1:
        return a / l2
    else:
        return a / np.expand_dims(l2, axis)


def get_iou_cuboid(cu1, cu2):

    # 2D projection on the horizontal plane (x-y plane)
    polygon2D_1 = Polygon(
        [(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])

    polygon2D_2 = Polygon(
        [(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])

    # 2D intersection area of the two projections.
    intersect_2D = polygon2D_1.intersection(polygon2D_2).area

    # the volume of the intersection part of cu1 and cu2
    inter_vol = intersect_2D * max(0.0, min(cu1[4][2], cu2[4][2]) - max(cu1[0][2], cu2[0][2]))

    # the volume of cu1 and cu2
    vol1 = polygon2D_1.area * (cu1[4][2] - cu1[0][2])
    vol2 = polygon2D_2.area * (cu2[4][2] - cu2[0][2])

    # return 3D IoU
    return inter_vol / (vol1 + vol2 - inter_vol)

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def read_obj(model_path, flags = ('v')):
    fid = open(model_path, 'r', encoding="utf-8")

    data = {}

    for head in flags:
        data[head] = []

    for line in fid:
        line = line.strip()
        if not line:
            continue
        line = re.split('\s+', line)
        if line[0] in flags:
            data[line[0]].append(line[1:])

    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(float)

    return data



def get_box_corners(center, vectors):
    '''
    Convert box center and vectors to the corner-form
    :param center:
    :param vectors:
    :return: corner points related to the box
    '''
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

    corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
    corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    return corner_pnts

def get_points_inside_bbox(xyz, box, expansion=0.05):
    # xyz: [N, 3]
    # box: [7], center, scale, rotz

    # subtract center
    xyz -= box[:3]
    # rotate xyz 
    cos_ = np.cos(-box[6])
    sin_ = np.sin(-box[6])
    xyz[:, 0], xyz[:, 1] = xyz[:, 0] * cos_ - xyz[:, 1] * sin_, xyz[:, 0] * sin_ + xyz[:, 1] * cos_
    # mask by scale
    mask_x = np.abs(xyz[:, 0]) <= box[3] / 2 + expansion
    mask_y = np.abs(xyz[:, 1]) <= box[4] / 2 + expansion
    mask_z = np.abs(xyz[:, 2]) <= box[5] / 2 + expansion
    mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

    return mask

def export(mesh_file, agg_file, seg_file, meta_file, label_map):
    
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint8)  # -1 --> 255 = ignore_label
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
            
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint8)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]

    #instance_bboxes = np.zeros((num_instances, 7))
    instance_bboxes = []
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2,
                         xmax - xmin, ymax - ymin, zmax - zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        #instance_bboxes[obj_id - 1, :] = bbox
        instance_bboxes.append(bbox)

    return mesh_vertices, label_ids, instance_ids, instance_bboxes, object_id_to_label_id

def generate(scan2cad_annotation):
    scene_name = scan2cad_annotation['id_scan']
    print('[INFO] start processing %s' % scene_name)

    output_dir = os.path.join(path_config.processed_data_path, scene_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(output_dir, 'data.npz')
    output_file_bbox = os.path.join(output_dir, 'bbox.pkl')

    if os.path.isfile(output_file):
        print('[INFO] File already exists. skipping.')
        return None

    '''read orientation file'''
    meta_file = os.path.join(path_config.metadata_root, 'scans', scene_name, scene_name + '.txt')  # includes axis
    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    Mscan = make_M_from_tqs(scan2cad_annotation["trs"]["translation"],
                            scan2cad_annotation["trs"]["rotation"],
                            scan2cad_annotation["trs"]["scale"])
    R_transform = np.array(axis_align_matrix).reshape((4, 4)).dot(np.linalg.inv(Mscan))

    ### read scan file
    scene_folder = os.path.join(path_config.metadata_root, 'scans', scene_name)
    
    ### label map
    raw_label_map = pd.read_csv(path_config.raw_label_map_file)
    RAW2RFS = {} # raw --> rfs
    RFS2CAD = {}
    for i in range(len(raw_label_map)):
        row = raw_label_map.iloc[i]
        RAW2RFS[row['raw_category']] = row['rfs_ids']
        RFS2CAD[row['rfs_ids']] = row['cad_ids']

    mesh_file = os.path.join(scene_folder, scene_name + '_vh_clean_2.ply')
    agg_file = os.path.join(scene_folder, scene_name + '.aggregation.json')
    seg_file = os.path.join(scene_folder, scene_name + '_vh_clean_2.0.010000.segs.json')

    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, _ = export(mesh_file, agg_file, seg_file, meta_file, RAW2RFS)

    ### relabel
    instance_relabels = np.zeros_like(instance_labels) # [N,], uint8, 0 means un-annotated
    instance_rebboxes = []
    
    relabel_cnt = 1

    # WARN: scene0217_00: num_instance_scannet = 62, len(instance_bboxes) = 31, seems wrongly annotated data?

    for i in range(len(instance_bboxes)):
        mask = instance_labels == i + 1
        bbox = instance_bboxes[i] # [7]

        if bbox[6] == 255: continue # ignored class

        cad_lbl = RFS2CAD[bbox[6]] # rfs id
        
        if cad_lbl != -1:
            # these three classes will be relabeled based on bbox.
            if cad_lbl not in [2, 5, 7]:
                instance_rebboxes.append(bbox)
                instance_relabels[mask] = relabel_cnt
                relabel_cnt += 1
        else:
            # keep the box for possible inaccurate match
            instance_rebboxes.append(bbox)
            instance_relabels[mask] = relabel_cnt
            relabel_cnt += 1


    #print(f'[relabel] bbox w/o 2&5 {len(instance_bboxes)} --> {len(instance_rebboxes)}')

    ### preprocess boxes
    shapenet_instances = []

    for model in scan2cad_annotation['aligned_models']:
        # read corresponding shapenet scanned points
        catid_cad = model["catid_cad"]
        cls_id = ShapeNetLabels.index(ShapeNetIDMap[catid_cad[1:]])

        if cls_id not in path_config.ScanNet_OBJ_CLASS_IDS:
            continue

        id_cad = model["id_cad"]
        sym = model['sym']

        obj_path = os.path.join(path_config.ShapeNetv2_path, catid_cad, id_cad + '/models/model_normalized.obj')
        assert os.path.exists(obj_path)
        obj_points = read_obj(obj_path)['v']
        '''transform shapenet obj to scannet'''
        t = model["trs"]["translation"]
        q = model["trs"]["rotation"]
        s = model["trs"]["scale"]
        Mcad = make_M_from_tqs(t, q, s)
        transform_shape = R_transform.dot(Mcad)
        '''get transformed axes'''
        center = (obj_points.max(0) + obj_points.min(0)) / 2.
        axis_points = np.array([center,
                                center - np.array([0, 0, 1]),
                                center - np.array([1, 0, 0]),
                                center + np.array([0, 1, 0])])

        axis_points_transformed = np.hstack([axis_points, np.ones((axis_points.shape[0], 1))]).dot(transform_shape.T)[
                                  ..., :3]
        center_transformed = axis_points_transformed[0]
        forward_transformed = axis_points_transformed[1] - axis_points_transformed[0]
        left_transformed = axis_points_transformed[2] - axis_points_transformed[0]
        up_transformed = axis_points_transformed[3] - axis_points_transformed[0]
        forward_transformed = normalize(forward_transformed)
        left_transformed = normalize(left_transformed)
        up_transformed = normalize(up_transformed)
        axis_transformed = np.array([forward_transformed, left_transformed, up_transformed])
        '''get rectified axis'''
        axis_rectified = np.zeros_like(axis_transformed)
        up_rectified_id = np.argmax(axis_transformed[:, 2])
        forward_rectified_id = 0 if up_rectified_id != 0 else (up_rectified_id + 1) % 3
        left_rectified_id = np.setdiff1d([0, 1, 2], [up_rectified_id, forward_rectified_id])[0]
        up_rectified = np.array([0, 0, 1])
        forward_rectified = axis_transformed[forward_rectified_id]
        forward_rectified = np.array([*forward_rectified[:2], 0.])
        forward_rectified = normalize(forward_rectified)
        left_rectified = np.cross(up_rectified, forward_rectified)
        axis_rectified[forward_rectified_id] = forward_rectified
        axis_rectified[left_rectified_id] = left_rectified
        axis_rectified[up_rectified_id] = up_rectified
        if np.linalg.det(axis_rectified) < 0:
            axis_rectified[left_rectified_id] *= -1

        ### deploy points
        obj_points = np.hstack([obj_points, np.ones((obj_points.shape[0], 1))]).dot(transform_shape.T)[..., :3]
        coordinates = (obj_points - center_transformed).dot(axis_transformed.T)
        # obj_points = coordinates.dot(axis_rectified) + center_transformed
        
        ### define bounding boxes
        # [center, edge size, orientation]
        sizes = (coordinates.max(0) - coordinates.min(0))
        box3D = np.hstack([center_transformed, sizes[[forward_rectified_id, left_rectified_id, up_rectified_id]], np.array([np.arctan2(forward_rectified[1], forward_rectified[0])])])

        axis_rectified = np.array([[np.cos(box3D[6]), np.sin(box3D[6]), 0], [-np.sin(box3D[6]), np.cos(box3D[6]), 0], [0, 0, 1]])
        vectors = np.diag(box3D[3:6]/2.).dot(axis_rectified)
        scan2cad_corners = np.array(get_box_corners(box3D[:3], vectors))

        ### relabel
        cad_lbl = ShapeNet2CAD[cls_id]
        if cad_lbl in [2, 5, 7]:
            # get the points within the 7D box3D...
            mask = get_points_inside_bbox(mesh_vertices[:, :3].copy(), box3D)
            # exclude already labeled areas
            mask = np.logical_and(mask, instance_relabels == 0)
            #print(f'[relabel] found {mask.sum()} points for instance {relabel_cnt} of {cad_lbl}')
            instance_relabels[mask] = relabel_cnt
            best_instance_id = relabel_cnt
            relabel_cnt += 1
        else:
            # find corresponding instance id & relabel cabinet
            best_iou_score = 0
            best_instance_id = 0 # means background points
            for inst_id, instance_bbox in enumerate(instance_rebboxes):
                
                center = instance_bbox[:3]
                vectors = np.diag(instance_bbox[3:6]) / 2.
                scannet_corners = np.array(get_box_corners(center, vectors))
                iou_score = get_iou_cuboid(scan2cad_corners, scannet_corners)

                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_instance_id = inst_id + 1

        if best_instance_id == 0:
            print(f'[WARN] {scene_name} unmatched CAD instance of class {cad_lbl}')
        else:
            shapenet_instances.append({
                'box3D': box3D, 
                'cls_id': cls_id, 
                'shapenet_catid': catid_cad, 
                'shapenet_id': id_cad,
                'instance_id': best_instance_id, 
                'box_corners': scan2cad_corners,
                'sym': sym,
            })

    # summarize
    #print(f'[relabel] instance number {len(instance_bboxes)} / {len(scan2cad_annotation["aligned_models"])} --> {relabel_cnt-1}')

    # save files
    with open(output_file_bbox, 'wb') as file:
        pickle.dump(shapenet_instances, file, protocol=pickle.HIGHEST_PROTOCOL)

    np.savez(output_file, mesh_vertices=mesh_vertices, semantic_labels=semantic_labels, instance_labels=instance_relabels)


def batch_export():

    scan2cad_annotations = read_json(path_config.scan2cad_annotation_path)

    # for i, anno in enumerate(scan2cad_annotations):
    #     #generate(anno)
    #     scan_name = anno['id_scan']
    #     if scan_name == 'scene0011_00':
    #         generate(anno)
    
    p = Pool(processes=16)
    p.map(generate, scan2cad_annotations)
    p.close()
    p.join()


if __name__ == '__main__':
    path_config = PathConfig()
    batch_export()

