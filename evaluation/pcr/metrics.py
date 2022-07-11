import os
import numpy as np
import pandas as pd
import torch
import trimesh
from functools import partial
from multiprocessing import Pool
from trimesh.exchange.binvox import voxelize_mesh

RFS_labels = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'kitchen_cabinet', 'display', 'trash_bin', 'other_shelf', 'other_table']
CAD_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']
CAD2ShapeNetID = ['4379243', '3001627', '2871439', '4256520', '2747177', '2933112', '3211117', '2808440']
CAD2ShapeNet = {k: v for k, v in enumerate([1, 7, 8, 13, 20, 31, 34, 43])} # selected 8 categories from SHAPENETCLASSES
ShapeNet2CAD = {v: k for k, v in CAD2ShapeNet.items()}

# assert exist, label map file.
raw_label_map_file = './datasets/scannet/rfs_label_map.csv'
raw_label_map = pd.read_csv(raw_label_map_file)

RFS2CAD = {} # RFS --> cad
for i in range(len(raw_label_map)):
    row = raw_label_map.iloc[i]
    RFS2CAD[int(row['rfs_ids'])] = row['cad_ids']


def voc_ap(rec, prec, use_07_metric):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            print('[ap]', t, p)
            ap = ap + p / 11.
    else:
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        print('[ap] rec: ', mrec[i+1] - mrec[i])
        print('[ap] pre: ', mpre[i+1])
    return ap

###################
# eval function

def eval_det_cls_w_mesh(pred, gt, ovthresh, use_07_metric):

    # per-class! 
    # pred = {scan_id: [[mesh, score], ...]}
    # gt = {scan_id: [points, ...]}

    # construct gt objects
    class_recs = {}

    npos = 0
    for scan_id in gt.keys():
        point_instances = gt[scan_id]
        det = [False] * len(point_instances)
        npos += len(point_instances)
        class_recs[scan_id] = {'points': point_instances, 'det_mesh': det}

    for scan_id in pred.keys():
        if scan_id not in gt:
            class_recs[scan_id] = {'points': np.array([]), 'det_mesh': []}

    # construct dets
    scan_ids = []
    confidence = []
    meshes = []

    for scan_id in pred.keys():
        for mesh, score in pred[scan_id]:
            scan_ids.append(scan_id)
            confidence.append(score)
            meshes.append(mesh)

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    meshes = [meshes[x] for x in sorted_ind]
    scan_ids = [scan_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(scan_ids) # number of detected boxes
    tp_mesh = np.zeros(nd)
    fp_mesh = np.zeros(nd)
    tp_mesh_iou = np.zeros(nd)
    
    # for all detected bboxes
    for d in range(nd):
        #if d % 100 == 0: print(d)
        print('[eval mesh]', d + 1, '/', nd, 'from', scan_ids[d])

        R = class_recs[scan_ids[d]] # the GT scan data dict
        mesh_pred = meshes[d]

        ovmax_mesh = -np.inf

        points_gt = R['points']

        # compare one-by-one, find max-iou match
        if len(points_gt) > 0:
            # compute overlaps
            for j in range(len(points_gt)):

                # this gt is already assigned with a pred as a TP pair
                if R['det_mesh'][j] == 1: continue

                # very slow...
                #print(mesh_pred.vertices.shape, mesh_pred.faces.shape, points_gt[j].shape)
                if (mesh_pred.vertices.shape[0] > 10000):
                    print(f'[WARN] mesh_pred very large {mesh_pred.vertices.shape}, {mesh_pred.faces.shape}')
                #    continue
                
                # it returns absolute distance (no negatives)
                closest, distance, _ = trimesh.proximity.closest_point(mesh_pred, points_gt[j])
                #closest, distance, _ = trimesh.proximity.closest_point_naive(mesh_pred, points_gt[j])

                pcr_mesh = (distance < 0.047).sum() / len(distance)

                if pcr_mesh > ovmax_mesh:
                    ovmax_mesh = pcr_mesh
                    jmax_mesh = j

            print(f'[eval mesh] best mesh PCR {ovmax_mesh}, from {jmax_mesh} in {j}')

        if ovmax_mesh > ovthresh:
            tp_mesh[d] = 1.
            tp_mesh_iou[d] = ovmax_mesh
            R['det_mesh'][jmax_mesh] = 1
            print(f'[eval mesh] mesh TP ++')
        else:
            fp_mesh[d] = 1.

    # for mesh
    fp_mesh = np.cumsum(fp_mesh)
    tp_mesh = np.cumsum(tp_mesh)
    rec_mesh = tp_mesh / np.maximum(npos, np.finfo(np.float64).eps)
    prec_mesh = tp_mesh / np.maximum(tp_mesh + fp_mesh, np.finfo(np.float64).eps)
    ap_mesh = voc_ap(rec_mesh, prec_mesh, use_07_metric)
    
    tp_mesh = tp_mesh[-1]
    fp_mesh = fp_mesh[-1]
    fn_mesh = npos - tp_mesh
    RQ_mesh = tp_mesh / (tp_mesh + fp_mesh/2 + fn_mesh/2)
    SQ_mesh = np.sum(tp_mesh_iou) / np.maximum(tp_mesh, np.finfo(np.float64).eps)
    PQ_mesh = SQ_mesh * RQ_mesh

    print(f'[eval mesh] tp = {tp_mesh}, fp = {fp_mesh}, fn = {fn_mesh}, npos = {npos}')

    return (rec_mesh[-1], prec_mesh[-1], ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)


def eval_det_cls_wrapper_w_mesh(arguments):
    # pred: {scan_id: (bbox, score, mesh)}
    # gt: {scan_id: (bbox, mesh)}
    pred, gt, ovthresh, use_07_metric = arguments
    (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh) = eval_det_cls_w_mesh(pred, gt, ovthresh, use_07_metric)
    return (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)


def eval_det_multiprocessing_w_mesh(pred_all, gt_all, ovthresh, use_07_metric):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {scan_id: [(classname, bbox, score, mesh)]}
            gt_all: map of {scan_id: [(classname, bbox, mesh)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}

    # scan_id == scan_id
    for scan_id in pred_all.keys():
        for classname, mesh, score in pred_all[scan_id]:
            # default dict behaviour
            if classname not in pred: pred[classname] = {}
            if scan_id not in pred[classname]: pred[classname][scan_id] = []
            if classname not in gt: gt[classname] = {}
            if scan_id not in gt[classname]: gt[classname][scan_id] = []
            # record
            pred[classname][scan_id].append((mesh, score))

    for scan_id in gt_all.keys():
        for classname, points in gt_all[scan_id]:
            # default dict behaviour
            if classname not in gt: gt[classname] = {}
            if scan_id not in gt[classname]: gt[classname][scan_id] = []
            # record
            gt[classname][scan_id].append(points)

    rec_mesh = {}
    prec_mesh = {}
    ap_mesh = {}

    PQ_mesh = {}
    SQ_mesh = {}
    RQ_mesh = {}

    # parallel for all classes
    #p = Pool(processes=8)
    #ret_values = p.map(eval_det_cls_wrapper_w_mesh, [(pred[classname], gt[classname], ovthresh, use_07_metric) for classname in gt.keys() if classname in pred])
    #p.close()
    #p.join()

    # fallback to single-thread
    ret_values = []
    for classname in sorted(gt.keys()):
        
        # tmp
        #if CAD_labels[classname] != 'sofa': 
        #    ret_values.append(((0, 0, 0), (0, 0, 0)))
        #    continue

        print(f'[eval mesh] class {CAD_labels[classname]}')
        if classname not in pred:
            continue
        ret_value = eval_det_cls_wrapper_w_mesh((pred[classname], gt[classname], ovthresh, use_07_metric))
        ret_values.append(ret_value)

    for i, classname in enumerate(sorted(gt.keys())):
        if classname in pred:
            (rec_mesh[classname], prec_mesh[classname], ap_mesh[classname]), (PQ_mesh[classname], SQ_mesh[classname], RQ_mesh[classname]) = ret_values[i]
        else:
            rec_mesh[classname] = 0
            prec_mesh[classname] = 0
            ap_mesh[classname] = 0
            PQ_mesh[classname] = 0
            SQ_mesh[classname] = 0
            RQ_mesh[classname] = 0

        #print(classname, 'box', ap[classname])
        #print(classname, 'mesh', ap_mesh[classname])

    return (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)


###################
# mAP calculation

class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
    
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        # pred: [<scan>, ...], <scan> = [(clsname, mesh, score)]
        # gt: [<scan>, ...], <scan> = [(clsname, points)]
        bsize = len(batch_pred_map_cls)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh) = eval_det_multiprocessing_w_mesh(self.pred_map_cls,
                                                                                          self.gt_map_cls,
                                                                                          ovthresh=self.ap_iou_thresh,
                                                                                          use_07_metric=True,
                                                                                          )
        ret_dict = {}

        # for mesh
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['AP_mesh        %s' % (clsname)] = ap_mesh[key]
            ret_dict['Precision_mesh %s' % (clsname)] = prec_mesh[key]
            ret_dict['Recall_mesh    %s' % (clsname)] = rec_mesh[key]
        ret_dict['mAP_mesh'] = np.mean(list(ap_mesh.values()))
        ret_dict['mean Precision_mesh'] = np.mean(list(prec_mesh.values()))
        ret_dict['mean Recall_mesh'] = np.mean(list(rec_mesh.values()))

        # for PQ
        for key in sorted(PQ_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['PQ_mesh %s' % (clsname)] = PQ_mesh[key]
            ret_dict['SQ_mesh %s' % (clsname)] = SQ_mesh[key]
            ret_dict['RQ_mesh %s' % (clsname)] = RQ_mesh[key]

        ret_dict['PQ_mesh'] = np.mean(list(PQ_mesh.values()))
        ret_dict['SQ_mesh'] = np.mean(list(SQ_mesh.values()))
        ret_dict['RQ_mesh'] = np.mean(list(RQ_mesh.values()))

        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0