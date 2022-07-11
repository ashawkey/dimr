import open3d as o3d

import torch
import torch.nn.functional as F
import time
import numpy as np
import random
import os

from util.config import cfg
cfg.task = 'test'

from util.log import logger
import util.utils as utils

from util.consts import *
from model.bsp import PolyMesh


def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, cfg.result_path, 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    
    semantic_label_idx = np.arange(25) # use pico ids

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            from data.scannetv2_inst import Dataset
            dataset = Dataset(test=True)
            dataset.testLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

    dataloader = dataset.test_data_loader


    with torch.no_grad():
        model = model.eval()
        start = time.time()

        for i, batch in enumerate(dataloader):

            # test batch is 1
            N = batch['feats'].shape[0]

            test_scene_name = dataset.test_files[int(batch['id'][0])]

            start1 = time.time()
            with torch.no_grad():
                preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1

            ##### get predictions
            semantic_pred = preds['semantic_pred'] # CAD
            pt_offsets = preds['pt_offsets']
            pt_angles = preds['pt_angles']

            if epoch > cfg.prepare_epochs:
                clusters = preds['clusters']
                cluster_scores = preds['cluster_scores']
                cluster_semantic_id = preds['cluster_semantic_id']
                cluster_semantic_score = preds['cluster_semantic_score']
                cluster_meshes = preds['cluster_meshes']
                if cfg.retrieval:
                    cluster_alignment = preds['cluster_alignment'] # dataframe

                nclusters = clusters.shape[0]

            ##### save files
            start3 = time.time()

            # save semantics.
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)
            
            # save offsets
            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch['locs_float'].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)

            # save angles
            if cfg.save_pt_angles:
                os.makedirs(os.path.join(result_dir, 'angles'), exist_ok=True)
                pt_angles_np = pt_angles.cpu().numpy() # (N, )
                np.save(os.path.join(result_dir, 'angles', test_scene_name + '.npy'), pt_angles_np)

            # save instances.
            if epoch > cfg.prepare_epochs and cfg.save_instance:
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    #semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    semantic_label = cluster_semantic_id[proposal_id].item()
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')
                    np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                f.close()
            
            # save meshes.
            if epoch > cfg.prepare_epochs and cfg.save_mesh:
                os.makedirs(os.path.join(result_dir, 'meshes'), exist_ok=True)
                os.makedirs(os.path.join(result_dir, 'trimeshes'), exist_ok=True)
                for proposal_id in range(nclusters):
                    mesh = cluster_meshes[proposal_id]
                    # not valid CAD label, skip.
                    if mesh is None: 
                        continue
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    
                    semantic_label = cluster_semantic_id[proposal_id].item()
                    #semantic_label_ = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))

                    print(f'[Mesh] save {proposal_id} / {nclusters}, label = {CAD_labels[semantic_label]}')

                    score = cluster_scores[proposal_id]
                    
                    mesh.export(os.path.join(result_dir, "meshes", f"{test_scene_name}_{proposal_id}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))
                    if isinstance(mesh, PolyMesh):
                        tmesh = mesh.to_trimesh()
                        tmesh.export(os.path.join(result_dir, "trimeshes", f"{test_scene_name}_{proposal_id}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))

            # save alignments
            if epoch > cfg.prepare_epochs and cfg.retrieval:
                os.makedirs(os.path.join(result_dir, 'alignment'), exist_ok=True)

            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()

            ##### print
            if epoch > cfg.prepare_epochs:
                logger.info("instance iter: {}/{} point_num: {} ncluster: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(batch['id'][0] + 1, len(dataset.test_files), N, nclusters, end, end1, end3))
            else:
               logger.info("instance iter: {}/{} point_num: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(batch['id'][0] + 1, len(dataset.test_files), N, end, end1, end3))


if __name__ == '__main__':
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    from model.rfs import RfSNet as Network
    from model.rfs import model_fn_decorator
    
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    ##### load model
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, data_name, cfg.test_epoch)
