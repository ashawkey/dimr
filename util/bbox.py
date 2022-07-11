
import numpy as np
import torch

SHAPENETCLASSES = ['void',
                   'table', 'jar', 'skateboard', 'car', 'bottle',
                   'tower', 
                   'chair', 
                   'bookshelf',
                    'camera', 'airplane',
                   'laptop', 'basket', 'sofa', 'knife', 'can',
                   'rifle', 'train', 'pillow', 'lamp', 'trash_bin',
                   'mailbox', 'watercraft', 'motorbike', 'dishwasher', 'bench',
                   'pistol', 'rocket', 'loudspeaker', 'file cabinet', 'bag',
                   'cabinet', 'bed', 'birdhouse', 'display', 'piano',
                   'earphone', 'telephone', 'stove', 'microphone', 'bus',
                   'mug', 'remote', 'bathtub', 'bowl', 'keyboard',
                   'guitar', 'washer', 'bicycle', 'faucet', 'printer',
                   'cap', 'clock', 'helmet', 'flowerpot', 'microwaves']

OBJ_CLASS_IDS = np.array([ 1,  7,  8, 13, 20, 31, 34, 43])                   

class BBoxUtils(object):
    def __init__(self):
        self.num_class = len(OBJ_CLASS_IDS)
        self.num_heading_bin = 12
        self.num_size_cluster = len(OBJ_CLASS_IDS)

        self.type2class = {SHAPENETCLASSES[cls]:index for index, cls in enumerate(OBJ_CLASS_IDS)}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.class_ids = OBJ_CLASS_IDS
        self.shapenetid2class = {class_id: i for i, class_id in enumerate(list(self.class_ids))}
        self.mean_size_arr = np.array([[0.72613623, 1.24456995, 0.66353637],
                                       [0.57895266, 0.55146825, 0.84949912],
                                       [0.33791219, 1.06731947, 1.33759765],
                                       [0.89405706, 1.69241158, 0.76549946],
                                       [0.27877716, 0.36634103, 0.45592777],
                                       [0.56651502, 0.96013238, 1.00018008],
                                       [0.16438198, 0.6067032 , 0.47594247],
                                       [0.51612009, 0.85305383, 0.43925024]]
                                       )
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert (angle >= 0) and (angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = np.int16(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def class2angle_cuda(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.'''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls.float() * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            angle = angle - 2*np.pi*(angle>np.pi).float()
        return angle

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def class2size_cuda(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size_arr = torch.from_numpy(self.mean_size_arr).to(residual.device).float()
        return mean_size_arr[pred_cls.view(-1), :].view(*pred_cls.size(), 3) + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle
        return obb