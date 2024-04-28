from os import name
from pathlib import Path
from typing import Optional, overload
import cv2
import imageio
import numpy as np
import ants
import pickle
from loguru import logger
from utils import to_ants, to_numpy
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

from dataclasses import dataclass


ANTS_METRIC_TYPES = '''MeanSquares
Correlation
ANTSNeighborhoodCorrelation
MattesMutualInformation
JointHistogramMutualInformation
Demons'''.splitlines()

@dataclass
class MetricResult:
    metric_name: str
    wm: float
    '''warped with moving'''
    wf: float
    '''warped with fixed'''

    def from_images(self, warped: ants.ANTsImage, moving: ants.ANTsImage, fixed: ants.ANTsImage):
        try:
            self.wm = ants.image_similarity(warped, moving, metric_type=self.metric_name)
            self.wf = ants.image_similarity(warped, fixed, metric_type=self.metric_name)
        except RuntimeError as e:
            logger.warning(f'ants image_similarity failed: {e}, {self.metric_name=}')
            self.wm = 0
            self.wf = 0

        return self

@dataclass
class MetricResults:
    MeanSquares                    : MetricResult
    Correlation                    : MetricResult
    ANTSNeighborhoodCorrelation    : MetricResult
    MattesMutualInformation        : MetricResult
    JointHistogramMutualInformation: MetricResult
    Demons                         : MetricResult
    IOU                            : MetricResult

    @staticmethod
    def zero():
        return MetricResults(
            MeanSquares                     = MetricResult('MeanSquares', 0, 0),
            Correlation                     = MetricResult('Correlation', 0, 0),
            ANTSNeighborhoodCorrelation     = MetricResult('ANTSNeighborhoodCorrelation', 0, 0),
            MattesMutualInformation         = MetricResult('MattesMutualInformation', 0, 0),
            JointHistogramMutualInformation = MetricResult('JointHistogramMutualInformation', 0, 0),
            Demons                          = MetricResult('Demons', 0, 0),
            IOU                             = MetricResult('IOU', 0, 0),
        )

@dataclass
class MatchResult:
    idx_f: int
    idx_m: int
    
    metric_results: MetricResults
    type_of_transform: str

    warped: Optional[np.ndarray]


    # def to_file(self, base_path: Path):
    #     p = base_path / f'{self.idx_f:03d}_{self.idx_m:03d}_match_result.pkl'
    #     with open(p, 'wb') as f:
    #         pickle.dump(self, f)
    #     return p
    
    # @staticmethod
    # def from_file(base_path: Path, idx_f: int, idx_m: int):
    #     with open(base_path / f'{idx_f:03d}_{idx_m:03d}_match_result.pkl', 'rb') as f:
    #         return pickle.load(f)

    def store_warped(self, base_path: Path):
        if self.warped is not None:
            p = base_path / f'{self.idx_f:03d}_{self.idx_m:03d}_warped.tif'
            imageio.v3.imwrite(p, self.warped)
            self.warped = None
        return self

    def load_warpd(self, base_path: Path):
        p = base_path / f'{self.idx_f:03d}_{self.idx_m:03d}_warped.tif'
        self.warped = imageio.v3.imread(p)
        return self



def to_mask_path(base_path: Path, idx_f: Optional[int]=None, idx_m: Optional[int]=None, ext='.tif'):
    assert idx_f is not None or idx_m is not None
    assert idx_f is None or idx_m is None
    if idx_f is not None:
        return base_path / f'f{idx_f:03d}{ext}'
    else:
        return base_path / f'm{idx_m:03d}{ext}'

def calc_iou(a: np.ndarray, b: np.ndarray):
    min_hw = np.min([a.shape, b.shape], axis=0)
    a = a[:min_hw[0], :min_hw[1]]
    b = b[:min_hw[0], :min_hw[1]]

    并 = (a + b) > 0
    交 = (a * b) > 0
    return float(np.sum(交) / np.sum(并))



def match_region(
    fimg_np: np.ndarray, mimg_np: np.ndarray, 
    idx_f: int, idx_m: int, 
    type_of_transform='SyN', outprefix='/tmp/ants/res_{idx_f:03d}_{idx_m:03d}'
):
    fimg = to_ants(fimg_np)
    mimg = to_ants(mimg_np)    

    metrics = MetricResults.zero()

    try:
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

        outs = ants.registration(
            fimg, mimg, type_of_transform=type_of_transform, 
            aff_metric='GC',
            verbose=False,
            outprefix=outprefix.format(idx_f=idx_f, idx_m=idx_m),
        )

    except RuntimeError as e:
        logger.warning(f'ants registration failed: {e}, {idx_f=:03d} {idx_m=:03d}')
        warpedmovout = np.zeros_like(fimg_np)


    else:
        for metric_name in ANTS_METRIC_TYPES:
            metrics.__dict__[metric_name].from_images(
                warped = outs['warpedmovout'],
                moving = mimg,
                fixed = fimg
            )

        warpedmovout = to_numpy(outs['warpedmovout'], equalize_hist=False)
        metrics.IOU = MetricResult(
            metric_name = 'IOU',
            wm = calc_iou(warpedmovout, fimg_np),
            wf = calc_iou(warpedmovout, mimg_np),
        )

    return MatchResult(
        idx_f = idx_f,
        idx_m = idx_m,
        warped = warpedmovout,

        metric_results = metrics,

        type_of_transform = type_of_transform
    )

def match_region_from_id(idx_f: int, idx_m: int, base_path: Path, type_of_transform='SyN') -> MatchResult: 
    fimg_path = to_mask_path(base_path, idx_f=idx_f)
    mimg_path = to_mask_path(base_path, idx_m=idx_m)
    assert fimg_path.exists(), f'{fimg_path} not exists'
    assert mimg_path.exists(), f'{mimg_path} not exists'

    fimg_np = cv2.imread(str(fimg_path), -1)
    mimg_np = cv2.imread(str(mimg_path), -1)
    # print(fimg_np.shape, fimg_np.dtype, fimg_path, mimg_path)
    res = match_region(fimg_np, mimg_np, idx_f, idx_m, type_of_transform=type_of_transform)

    return res.store_warped(base_path)
