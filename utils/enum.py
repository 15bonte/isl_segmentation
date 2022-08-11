from enum import Enum


class LossType(Enum):
    PCC = 1
    IoU = 2
    mAP = 3
    ap_50 = 4
    none = -1


class NormalizeMethods(Enum):
    ZeroAndOne = 1
    Standardize = 2
    none = -1


class ProjectMethods(Enum):
    Maximum = 1
    Mean = 2
    Focus = 3
    none = -1
