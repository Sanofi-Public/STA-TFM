""" Conversions """
import numpy as np

def coco_to_canonical(coco_kpts):
    """Convert COCO 17 keypoints to canonical 17 format with root, neck, torso."""
    kpts = np.zeros((17, 3), dtype=np.float32)

    lhip, rhip = coco_kpts[11], coco_kpts[12]
    root = (lhip + rhip) / 2.0
    kpts[0] = root               # root
    kpts[1] = rhip               # RHip
    kpts[2] = coco_kpts[14]      # RKnee
    kpts[3] = coco_kpts[16]      # RAnkle
    kpts[4] = lhip               # LHip
    kpts[5] = coco_kpts[13]      # LKnee
    kpts[6] = coco_kpts[15]      # LAnkle
    lsho, rsho = coco_kpts[5], coco_kpts[6]
    neck = (lsho + rsho)/2.0
    kpts[8] = neck               # neck
    kpts[7] = (root + neck)/2.0  # torso
    leye, reye = coco_kpts[1], coco_kpts[2]
    kpts[9]  = coco_kpts[0]      # nose
    kpts[10] = (leye + reye)/2.0 # head 
    kpts[11] = lsho
    kpts[12] = coco_kpts[7]      # LElbow
    kpts[13] = coco_kpts[9]      # LWrist
    kpts[14] = rsho
    kpts[15] = coco_kpts[8]      # RElbow
    kpts[16] = coco_kpts[10]     # RWrist

    return kpts

def mediapipe_to_canonical(mp_kpts):
    """Convert one frame of Mediapipe 33 keypoints → canonical 17 joints."""
    kpts = np.zeros((17, 3), dtype=np.float32)

    lhip, rhip = mp_kpts[23], mp_kpts[24]
    root = (lhip + rhip) / 2.0
    kpts[0] = root
    kpts[1] = rhip
    kpts[2] = mp_kpts[26]        # RKnee
    kpts[3] = mp_kpts[28]        # RAnkle
    kpts[4] = lhip
    kpts[5] = mp_kpts[25]        # LKnee
    kpts[6] = mp_kpts[27]        # LAnkle

    lsho, rsho = mp_kpts[11], mp_kpts[12]
    neck = (lsho + rsho) / 2.0
    kpts[8] = neck
    kpts[7] = (root + neck) / 2.0

    kpts[9]  = mp_kpts[0]        # nose
    kpts[10] = (mp_kpts[2] + mp_kpts[5]) / 2.0  # head (avg eyes)

    kpts[11] = lsho
    kpts[12] = mp_kpts[13]       # LElbow
    kpts[13] = mp_kpts[15]       # LWrist
    kpts[14] = rsho
    kpts[15] = mp_kpts[14]       # RElbow
    kpts[16] = mp_kpts[16]       # RWrist

    return kpts