import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from functools import cache, lru_cache
from pathlib import Path
from turtle import left
from typing import Callable, Literal, Optional, Sequence, overload

import ants
import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import orjson
import pandas as pd
import polars as pl
import torch
import yaml
from bidict import bidict
from dipy.align.imwarp import DiffeomorphicMap
from joblib import Parallel, delayed
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nhp_utils.image_correction import CorrectionPara
from ntp_manager import NTPRegion, SliceMeta, from_dict, parcellate
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.ops import linemerge, polygonize_full, unary_union
from tqdm import tqdm

dye_to_color = bidict({
    'FB': 'blue',
    'CTB555': 'yellow',
    'CTB647': 'red',
    'CTB488': 'green',
})

def to_numpy(x: ants.ANTsImage | np.ndarray | torch.Tensor, to_u8=True, equalize_hist=True):
    if isinstance(x, ants.ANTsImage):
        x = x.numpy()
    elif isinstance(x, torch.Tensor):
        x_: np.ndarray = x.detach().cpu().numpy()
        if len(x_.shape) == 4:
            x_ = x_[0]
        if x_.shape[0] == 3:
            x_ = x_.transpose(1, 2, 0)

        x = x_

    if to_u8 and x.dtype != 'uint8':
        max_ = x.max()
        if max_ <= 1:
            x = (x * 255).astype('uint8')
        elif max_ <= 255:
            x = x.astype('uint8')
        elif max_ <= 65535:
            x = (x / 255).astype('uint8')

    if equalize_hist and x.dtype == 'uint8' and len(x.shape) == 2:
        x = cv2.equalizeHist(x)

    return x
def to_ants(x: ants.ANTsImage | np.ndarray):
    if isinstance(x, ants.ANTsImage):
        return x
    max_ = x.max()
    if max_ <= 1:
        pass
    elif max_ <= 255:
        x = x / 255
    elif max_ <= 65535:
        x = x / 65535

    return ants.from_numpy(x)

def scale_image(s: float, image: np.ndarray):
    return cv2.resize(image, None, fx=s, fy=s)

# def pad_or_crop(
#     image: np.ndarray, w: int, h: int, 
#     pad_method: Literal['center', 'top']='center'
# ):
#     w = int(w)
#     h = int(h)

#     src_h, src_w = image.shape[:2]
#     dim = len(image.shape) # 2 or 3

#     for drct in ('h', 'w'):
#         if locals()[f'src_{drct}'] > locals()[f'{drct}']:
#             if pad_method == 'center' or drct != 'h':
#                 d = locals()[f'src_{drct}'] - locals()[f'{drct}']
#                 dd2 = d // 2
#             elif pad_method == 'top':
#                 d = locals()[f'src_{drct}']
#                 dd2 = 0

#             slice_ = (
#                 slice(dd2, locals()[f'src_{drct}']-dd2),
#                 slice(None), 
#             )
#             if drct == 'w': slice_ = tuple(reversed(slice_))
#             image = image[slice_]

#     for drct in ('h', 'w'):
#         if locals()[f'src_{drct}'] < locals()[f'{drct}']:
#             if pad_method == 'center' or drct != 'h':
#                 d = locals()[f'{drct}'] - locals()[f'src_{drct}']
#                 dd2 = d // 2
#                 pad_width = ((dd2, d-dd2), (0, 0))
#             elif pad_method == 'top':
#                 d = locals()[f'{drct}'] - locals()[f'src_{drct}']
#                 pad_width = ((0, d), (0, 0))

#             if drct == 'w': pad_width = tuple(reversed(pad_width))
#             if dim == 3:
#                 pad_width = (*pad_width, (0, 0))

#             image = np.pad(image, pad_width, 'constant')

#     return image[:h, :w]


@overload
def pad_or_crop(
    image: np.ndarray, dst_w: int, dst_h: int, 
    pad_method: Literal['center', 'top']='center',
    non_center_pad_value: int=0,
) -> np.ndarray: ...

@overload
def pad_or_crop(
    image: pl.DataFrame, dst_w: int, dst_h: int, 
    pad_method: Literal['center', 'top']='center',
    non_center_pad_value: int=0, src_w: int=0, src_h: int=0,
) -> pl.DataFrame: ...

def pad_or_crop(
    image: np.ndarray | pl.DataFrame, dst_w: int, dst_h: int, 
    pad_method: Literal['center', 'top']='center',
    non_center_pad_value: int=0, src_w: int=0, src_h: int=0,
):
    dst_w = int(dst_w)
    dst_h = int(dst_h)

    if isinstance(image, np.ndarray):
        src_h, src_w = image.shape[:2]
    elif isinstance(image, pl.DataFrame):
        if pad_method == 'center':
            assert src_w != 0 and src_h != 0
        elif src_w == 0 or src_h == 0:
            r = image.max().to_dicts()[0]
            src_h, src_w = r['y'], r['x']
    else:
        raise TypeError(f'`image` must be `np.ndarray` or `pl.DataFrame`, but got `{type(image)}`')

    src = {
        'w': src_w,
        'h': src_h,
    }
    dst = {
        'w': dst_w,
        'h': dst_h,
    }


    dim = len(image.shape) # 2 or 3

    for drct, v_name in zip(('h', 'w'), ('y', 'x')):
        if src[drct] > dst[drct]:
            if pad_method == 'center' or drct != 'h':
                d = src[drct] - dst[drct]
                dd2 = d // 2
            elif pad_method == 'top':
                # d = src[drct] - non_center_pad_value
                dd2 = non_center_pad_value

            slice_ = (
                slice(dd2, src[drct]-dd2),
                slice(None), 
            )
            if drct == 'w': slice_ = tuple(reversed(slice_))
            if isinstance(image, np.ndarray):
                image = image[slice_]
            elif isinstance(image, pl.DataFrame):
                image = image.filter(
                    (pl.col(v_name) >= dd2) & (pl.col(v_name) < src[drct]-dd2)
                )

    for drct, v_name in zip(('h', 'w'), ('y', 'x')):
        if src[drct] < dst[drct]:
            if pad_method == 'center' or drct != 'h':
                d = dst[drct] - src[drct]
                dd2 = d // 2
                raw_pad_width = ((dd2, d-dd2), (0, 0))
            elif pad_method == 'top':
                d = dst[drct] - src[drct]
                pad_v = non_center_pad_value
                if d - pad_v < 0:
                    pad_v = d

                raw_pad_width = ((pad_v, d-pad_v), (0, 0))

            pad_width = raw_pad_width
            if drct == 'w': pad_width = tuple(reversed(raw_pad_width))
            if dim == 3: pad_width = (*raw_pad_width, (0, 0))

            if isinstance(image, np.ndarray):
                image = np.pad(image, pad_width, 'constant')
            elif isinstance(image, pl.DataFrame):
                image = image.with_columns([
                    pl.col(v_name) + raw_pad_width[0][0],
                ])
    if isinstance(image, np.ndarray):
        return image[:dst_h, :dst_w]
    elif isinstance(image, pl.DataFrame):
        return image.filter(
            (pl.col('y') < dst_h) & (pl.col('x') < dst_w)
        )
    else:
        raise TypeError(f'`image` must be `np.ndarray` or `pl.DataFrame`, but got `{type(image)}`')

def pad_df_and_mask(
    item: dict | np.ndarray, cell_df: pl.DataFrame, 
    target_shape: tuple[int, int]=(256, 256),
    pad_method: Literal['center', 'top']='center',
):
    res: dict[Literal['left', 'right'], tuple[np.ndarray, pl.DataFrame]] = {}

    for side in ('left', 'right'):
        if isinstance(item, dict):
            img = item['splited'][side].image
        else:
            img = item
        src_h, src_w = img.shape[:2]

        new_mask = pad_or_crop(img, *target_shape, pad_method=pad_method)
        new_df = pad_or_crop(cell_df.filter(side), *target_shape, pad_method=pad_method, src_h=src_h, src_w=src_w)

        res[side] = (new_mask, new_df)
    return res


def draw_images(*images: np.ndarray, shape: tuple[int, int] | None = None, figsize=(5, 5), titles: list[str] | None = None, sharexy=False):
    if shape is None:
        nrow = int(np.ceil(len(images) ** 0.5))
        ncol = int(np.ceil(len(images) / nrow))
        shape = (nrow, ncol)
    if titles is None:
        titles = list(map(str, range(len(images))))

    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=sharexy, sharey=sharexy)
    axes = axes.flatten()
    for ax, image, title in zip(axes, images, titles):
        ax.axis('off')
        ax.imshow(image)
        ax.set_title(title)

    fig.tight_layout()
    return fig, axes

def draw_image_with_spacing(
    images: list[np.ndarray] | np.ndarray, size: int, 
    shape: tuple[int, int] | None = None, 
    figsize=(5, 5), axis=0, sharexy=False,
    titles: list[str] | None=None,
):
    if isinstance(images, np.ndarray):
        assert len(images.shape) == 3

    to_show = []
    if isinstance(images, np.ndarray):
        idxs = split_into_n_parts_slice(images.shape[axis], n=size)

        for i in idxs:
            to_show.append(images.take(i, axis=axis))
    else:
        idxs = split_into_n_parts_slice(len(images), n=size)

        for i in idxs:
            to_show.append(images[i])

    if titles is None:
        titles = [str(i) for i in idxs]
    else:
        titles = [titles[i] for i in idxs]
    return draw_images(
        *to_show, shape=shape, figsize=figsize, 
        titles=titles, sharexy=sharexy
    )


def ndarray_info(x: np.ndarray):
    return {
        'shape': x.shape,
        'dtype': x.dtype,
        # 'max'  : x.max(),
        # 'min'  : x.min(),
        # 'mean' : x.mean(),
    }


@dataclass
class ImageInfo:
    animal_id: str
    slice_id: int
    image: np.ndarray
    mask: np.ndarray
    cells: dict[Literal['red', 'green', 'blue', 'yellow'], np.ndarray] | None

    def __post_init__(self):
        if (ishape := self.image.shape) != (mshape := self.mask.shape):
            logger.warning(f'animal_id={self.animal_id}, slice_id={self.slice_id}, image.shape={ishape} != mask.shape={mshape}')
        if self.cells is not None:
            for color in self.cells:
                img = self.cells[color]
                if img is None: 
                    continue

                if (ishape := self.image.shape) != (cshape := img.shape):
                    logger.warning(f'animal_id={self.animal_id}, slice_id={self.slice_id}, image.shape={ishape} != cells[{color}].shape={cshape}')


    @staticmethod
    def from_df(df: pl.DataFrame | dict, i: int=0):
        if isinstance(df, pl.DataFrame):
            current_meta = df[i].to_dicts()[0]
        elif not isinstance(df, dict):
            raise TypeError('need dict or df')
        else:
            current_meta = df

        return ImageInfo(
            animal_id=current_meta['animal_id'],
            slice_id=current_meta['slice_id'],
            image=current_meta.get('image', current_meta.get('mask')),
            mask=current_meta['mask'],
            cells=current_meta.get('cells'),
        )


    def __repr__(self) -> str:
        return f'ImageInfo(animal_id={self.animal_id}, slice_id={self.slice_id}, image={ndarray_info(self.image)}, mask={ndarray_info(self.mask)}, cells={list(self.cells.keys()) if self.cells else None})'

@dataclass
class RegistInfo(ImageInfo):
    warped_image: np.ndarray
    regist_outs : dict
    fixed_info  : Optional['RegistInfo']

    method: str = ''
    @staticmethod
    def from_df(df: pl.DataFrame | dict, i: int=0):
        if isinstance(df, pl.DataFrame):
            current_meta = df[i].to_dicts()[0]
        elif not isinstance(df, dict):
            raise TypeError('need dict or df')
        else:
            current_meta = df

        res = RegistInfo(
            animal_id=current_meta['animal_id'],
            slice_id=current_meta['slice_id'],
            image=current_meta['image'],
            mask=current_meta['mask'],

            regist_outs  = {},
            warped_image = current_meta['image'],
            fixed_info   = None, 

            cells=current_meta['cells'],
        )
        res.fixed_info = res
        return res

    def __repr_with__(self, parent=False) -> str:
        if self.fixed_info is self:
            fixed_info = 'self'
        elif parent and self.fixed_info is not None:
            fixed_info = self.fixed_info.__repr_with__(parent=False)
        else:
            fixed_info = ''

        return f'RegistInfo(animal_id={self.animal_id}, slice_id={self.slice_id}, image={ndarray_info(self.image)}, mask={ndarray_info(self.mask)}, warped_image={ndarray_info(self.warped_image)}, regist_outs={self.regist_outs}, fixed_info={fixed_info})'

    def __repr__(self) -> str:
        return self.__repr_with__(parent=True)

def coloring(img: np.ndarray, c=(1, 1, 1)):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * np.array(c)

def draw_fix_and_mov(
    fixed_meta: RegistInfo | np.ndarray | ants.ANTsImage, 
    moving_meta: RegistInfo | np.ndarray | ants.ANTsImage,
    warped_mov_image: np.ndarray | ants.ANTsImage | None = None,
    fixed_slice_id: int | None = None,
    moving_slice_id: int | None = None,

    fcoloe: tuple[float, float, float] = (1, 0, 0),
    mcoloe: tuple[float, float, float] = (0, 1, 0),
):
    has_warped_mov_image = warped_mov_image is not None

    fig, axes = plt.subplots(2, 2, figsize=(5, 5))

    fixed_image      = fixed_meta.image if isinstance(fixed_meta, RegistInfo) else fixed_meta
    moving_image     = moving_meta.image if isinstance(moving_meta, RegistInfo) else moving_meta

    if not has_warped_mov_image:
        warped_mov_image = moving_meta.warped_image if isinstance(moving_meta, RegistInfo) else None
        has_warped_mov_image = warped_mov_image is not None

    fixed_image  = to_numpy(fixed_image)
    fixed_image  = coloring(fixed_image, fcoloe)
    moving_image = to_numpy(moving_image)
    moving_image = coloring(moving_image, mcoloe)

    if has_warped_mov_image and warped_mov_image is not None:
        warped_mov_image = to_numpy(warped_mov_image)
        warped_mov_image = coloring(warped_mov_image, mcoloe)

    fixed_slice_id   = fixed_slice_id or (fixed_meta.slice_id if isinstance(fixed_meta, RegistInfo) else None)
    moving_slice_id  = moving_slice_id or (moving_meta.slice_id if isinstance(moving_meta, RegistInfo) else None)

    for ax in axes.flatten():
        ax.axis('off')

    axes[0, 0].imshow(fixed_image)
    axes[0, 0].set_title(f'{fixed_slice_id} fixed')

    axes[0, 1].imshow(moving_image)
    axes[0, 1].set_title(f'{moving_slice_id} moving')

    if has_warped_mov_image:
        axes[1, 0].imshow(warped_mov_image)
        axes[1, 0].set_title(f'{moving_slice_id} warped')
    else:
        axes[1, 0].set_title(f'no warped')


    axes[1, 1].imshow(fixed_image, alpha=0.5)
    if has_warped_mov_image:
        axes[1, 1].imshow(warped_mov_image, alpha=0.5)
        axes[1, 1].set_title(f'{fixed_slice_id} + {moving_slice_id}')
    else:
        axes[1, 1].set_title(f'{fixed_slice_id} + no warped')

    fig.tight_layout()
    return fig, axes


@dataclass(frozen=True, eq=True)
class AnimalSection:
    animal_id: str
    slice_id: str

    @property
    def slice_id_int(self):
        return int(self.slice_id)
    
    @property
    def slice_id_str(self):
        return self.slice_id.zfill(3)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return f'{self.animal_id}-{self.slice_id}' == __value.replace('_', '-')
        if isinstance(__value, tuple) and len(__value) == 2:
            return (
                (self.animal_id == __value[0] and self.slice_id_str == str(__value[1])) or 
                (self.animal_id == __value[0] and self.slice_id_int == __value[1])
            )
        if isinstance(__value, AnimalSection):
            return self.animal_id == __value.animal_id and self.slice_id == __value.slice_id
        return super().__eq__(__value)

    def __repr__(self) -> str:
        return f'AniSec(animal_id={self.animal_id}, slice_id={self.slice_id})'

@dataclass(frozen=True, eq=True)
class AnimalSectionSide(AnimalSection):
    side: Literal['left', 'right', '']

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return f'{self.animal_id}-{self.slice_id}-{self.side}' == __value.replace('_', '-')
        if isinstance(__value, tuple) and len(__value) == 3:
            return (
                (self.animal_id == __value[0] and self.slice_id_str == str(__value[1]) and self.side == __value[2]) or 
                (self.animal_id == __value[0] and self.slice_id_int == __value[1] and self.side == __value[2])
            )
        if isinstance(__value, AnimalSectionSide):
            return self.animal_id == __value.animal_id and self.slice_id == __value.slice_id and self.side == __value.side
        return False

@dataclass(frozen=True, eq=True)
class AnimalSectionWithDistance(AnimalSection):
    side: Literal['left', 'right', '']
    image_path: str
    need_flip: bool
    distance: float = 0

    @property
    def image(self):
        img = cv2.imread(self.image_path, -1)
        if self.need_flip:
            img = cv2.flip(img, 1)
        return img

    def to_sec(self, with_side=True):
        animal_id = self.animal_id
        side = ''
        if self.animal_id[-1].lower() in 'lr':
            animal_id = self.animal_id[:-2]
            side = self.animal_id[-2:]

        if with_side:
            animal_id = f'{animal_id}{side}'

        return AnimalSection(
            animal_id=animal_id,
            slice_id=self.slice_id,
        )

    def to_sec_side(self):
        return AnimalSectionSide(
            animal_id=self.animal_id,
            slice_id=self.slice_id,
            side=self.side
        )

def read_exclude_sections(project_name: str) -> list[AnimalSection]:
    '''example: ['C052-017', 'C057-235']'''

    p = '/mnt/97-macaque/projects/segment/ignore-sections.xlsx'
    df = pl.read_excel(p)
    if project_name not in df:
        raise ValueError(f'project_name=`{project_name}` not in `{p}`, all projects: {df.columns}')

    res: list[AnimalSection] = []
    for i in df.select(project_name).drop_nulls()[project_name].to_list():
        animal_id, slice_id = i.replace('_', '-').split('-')
        res.append(AnimalSection(animal_id, slice_id))
    return res


def read_labeled_sections(project_name: str) -> list[AnimalSection]:
    p = '/mnt/97-macaque/projects/segment/labeled-sections.xlsx'
    df = pl.read_excel(p)
    if project_name not in df:
        raise ValueError(f'project_name=`{project_name}` not in `{p}`, all projects: {df.columns}')

    res: list[AnimalSection] = []
    for i in df.select(project_name).drop_nulls()[project_name].to_list():
        animal_id, slice_id = i.replace('_', '-').split('-')
        res.append(AnimalSection(animal_id, slice_id))
    return res


def read_region_ids() -> dict[str, int]:
    p = '/mnt/97-macaque/projects/segment/region-ids.xlsx'
    df = pl.read_excel(p)
    return {i['region_name']: i['region_id'] for i in df.drop_nulls().to_dicts()}


def merge_draw(
    image, mask, 
    fig: Figure | None = None,
    axes: list[Axes] | None = None, 
    title: str=''
) -> tuple[Figure, list[Axes]]:
    if axes is None or fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        if axes is None: raise RuntimeError
    assert len(axes) == 3

    try:
        import torch
    except ImportError:
        pass
    else:
        if isinstance(image, torch.Tensor):
            image = image.data.cpu().numpy().transpose(1, 2, 0)
        if isinstance(mask, torch.Tensor):
            mask = mask.data.cpu().numpy().squeeze(0)

    for i, cur in enumerate([[image], [mask], [image, mask]]):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])
        for im, alpha in zip(cur, [1, 0.5]):
            ax.imshow(im, alpha=alpha)
    
    fig.suptitle(title)
    return fig, axes

def crop_black(img: np.ndarray):
    nonzeros = np.nonzero(img)
    if len(nonzeros[0]) == 0:
        return img[0:0, 0:0]

    ss = []
    for i in nonzeros:
        ss.append(slice(i.min(), i.max()))

    return img[*ss]

def split_into_n_parts_slice(seq_or_seq_count: Sequence | int, n: int) -> list[int]:
    if isinstance(seq_or_seq_count, int):
        seq = range(seq_or_seq_count)
    else:
        seq = seq_or_seq_count

    part_size = len(seq) / n

    indices = []
    i       = 0
    while i < len(seq) and len(indices) < n:
        indices.append(int(i))
        i += part_size
        
    if len(indices) > 1:
        indices[-1] = len(seq) - 1
    return indices

def zcrop_center(rimg: np.ndarray | list[np.ndarray], target_shape: tuple[int, ...] | None=None, scale: float=1, flip: int|None=None) -> np.ndarray:
    """_summary_

    Args:
        img (np.ndarray | list[np.ndarray]): (x, y, z)  
        target_shape (tuple[int, ...] | None, optional): (w, h). Defaults to None.  
        scale (int, optional): _description_. Defaults to 1.  

    Returns:
        np.ndarray: _description_
    """

    is_list_images = isinstance(rimg, list)
    if target_shape is None:
        if is_list_images:
            shapes = [(*crop_black(i).shape[:2], len(rimg)) for i in rimg]
        else:
            shapes = [crop_black(rimg).shape[:2]]
        target_shape = tuple(np.array(shapes).max(axis=0))
        print(f'target_shape: {target_shape}')



    w, h = target_shape[:2]
    res_arr = np.zeros(target_shape, dtype=rimg[0].dtype)

    for z in range(target_shape[2]):
        img = rimg[:, :, z].copy() if not is_list_images else rimg[z].copy()
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filled_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
        cv2.drawContours(img, filled_cnts, -1, (0, 0, 0), thickness=cv2.FILLED)

        if scale != 1:
            img = scale_image(scale, img)
        if flip is not None:
            img = cv2.flip(img, flip)

        img = crop_black(img)
        img = pad_or_crop(img, h, w)

        res_arr[:, :, z] = img

    return res_arr

@cache
def _read_connectome_masks_from_pickle(p: Path) -> list[dict]:
    with open(p, 'rb') as f:
        return pickle.load(f)

def read_connectome_masks(
    connectome_mask_base_path: Path, animal_id: str, 
    scale: float, exclude_slices: list[AnimalSection] = [],
    force: bool = False, min_sum=0
):
    def read_mask(p: Path) -> dict | None:
        slice_id = int(p.stem.split('_')[-1])
        if (animal_id, slice_id) in exclude_slices: return
        mask = cv2.imread(p.as_posix(), -1) # 这里的原始数据就是 bin20 的
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = scale_image(scale, mask)
        if (s := (mask != 0).sum()) == 0: return

        return {
            'animal_id': animal_id,
            'slice_id' : slice_id,
            'mask'     : mask,
            'sum'      : s
        }

    pkl_path = connectome_mask_base_path / f'{animal_id}_{scale}_raw_cntm_masks.pkl'

    if not force and pkl_path.exists():
        print('read from pkl')
        raw_cntm_masks: list[dict] = _read_connectome_masks_from_pickle(pkl_path)
    else:
        print('read from files')
        cntm_mask_ps = list(connectome_mask_base_path.glob(f'{animal_id}_*.tif'))

        raw_cntm_masks: list[dict] = [i for i in tqdm(Parallel(
            n_jobs=16, return_as='generator_unordered', backend='threading')(
            delayed(read_mask)(p) for p in cntm_mask_ps
        ), total=len(cntm_mask_ps)) if i is not None]
        print('write to pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(raw_cntm_masks, f)
    raw_cntm_masks = [
        m for m in raw_cntm_masks 
        if m['sum'] > min_sum and (m['animal_id'], m['slice_id']) not in exclude_slices
    ]
    raw_cntm_masks.sort(key=lambda x: x['slice_id'])
    return raw_cntm_masks

@cache
def poly_to_bitmap(
    poly: Polygon, w=0, h=0, buffer_x=50, buffer_y=50, 
    minus_minimum=True, scale=1.0
):
    minx, miny, maxx, maxy = poly.bounds
    if minx < 0 or miny < 0:
        print('Polygon has negative coordinates')

    coords = np.array(poly.exterior.coords, dtype=np.int32)
    if minus_minimum:
        coords[:, 0] -= int(minx) - buffer_x
        coords[:, 1] -= int(miny) - buffer_y
        true_w = int(maxx - minx) + buffer_x * 2
        true_h = int(maxy - miny) + buffer_y * 2
    else:
        true_w = int(maxx) + buffer_x
        true_h = int(maxy) + buffer_y

    if w == 0: w = true_w
    if h == 0: h = true_h

    bitmap = np.zeros((int(h * scale), int(w * scale)), dtype=np.uint8)

    cv2.fillPoly(bitmap, [(coords * scale).astype(int)], color=(255, ))
    return bitmap

def read_stereo_masks(
    chips: Sequence[str], ntp_version: str, scale: float, 
    min_sum=0, target_shape: tuple[int, int]=(128, 128),
):
    ps = (Path('/data/sde/ntp/macaque') / ntp_version / 'region-mask').glob('*.json')
    ps_with_chip = {}
    for p in ps:
        match = re.search(r'T\d+', p.stem)
        if match:
            ps_with_chip[match.group()] = p

    res = []

    for chip in chips:
        if chip not in ps_with_chip: continue
        slice_meta_p = ps_with_chip[chip]

        slice_meta = orjson.loads(slice_meta_p.read_bytes())
        regions: list[NTPRegion] = [
            from_dict(NTPRegion, r) for r in slice_meta['regions']
        ]
        corr = CorrectionPara.select(chip)[0]

        all_cla_list = []
        all_cla_list_without_warped = []
        raw_points = []
        for i, r in enumerate(regions):
            if 'cla' not in r.label.name.lower(): continue
            print(chip, r.label.name)
            # if 'lim' in r.label.name: continue

            points = np.array(r.polygon.exterior.xy).T
            raw_points.append(np.hstack([
                points, np.repeat(i, len(points)).reshape(-1, 1)
            ]))

            all_cla_list_without_warped.append(Polygon(points * scale))
            points = corr.wrap_point(points + corr.offset) * scale
            all_cla_list.append(Polygon(points))

        all_cla: Polygon | MultiPolygon = unary_union(all_cla_list)
        if isinstance(all_cla, MultiPolygon):
            print(f'{chip} has MultiPolygon, use the largest one')
            for gi, geom in enumerate(all_cla.geoms):
                print(f"\t[{gi}]: {geom.area}")
            all_cla = max(all_cla.geoms, key=lambda x: x.area)

        all_cla_without_warped: Polygon | MultiPolygon = unary_union(all_cla_list_without_warped)
        if isinstance(all_cla_without_warped, MultiPolygon):
            all_cla_without_warped = max(all_cla_without_warped.geoms, key=lambda x: x.area)
        
        raw_points = np.vstack(raw_points) + np.array([*corr.offset, 0])

        raw_df = pl.DataFrame(raw_points, schema=['x', 'y', 'region_index']).with_columns(
            pl.lit(True).alias('left'), pl.lit(False).alias('right'),
            pl.int_range(0, raw_points.shape[0]).alias('index')
        ) # 与转录组测序结果一致
        new_df = pl.DataFrame(corr.wrap_point(raw_points[:, :2]) * scale, schema=['x', 'y']).with_columns(
            pl.lit(True).alias('left'), pl.lit(False).alias('right'),
            pl.int_range(0, raw_points.shape[0]).alias('index'),
            raw_df['region_index']
        ) # 在处理中，与图像上的坐标一致，与缩放有关

        img = poly_to_bitmap(all_cla, buffer_x=0, buffer_y=0)
        if (img != 0).sum() < min_sum:
            continue
        bounds = all_cla.bounds

        new_df = new_df.with_columns(pl.col('x') - bounds[0], pl.col('y') - bounds[1])
        # img = pad_or_crop(img, *target_shape)
        pad_res = pad_df_and_mask(img, new_df, target_shape=target_shape)
        img, new_df = pad_res['left']
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # print(raw_df)
        raw_df = raw_df.filter(pl.col('index').is_in(new_df['index'])) # 去除掉被删去的点

        transformation_matrix, _ = cv2.findHomography(
            np.array(raw_df[['x', 'y']]), 
            np.array(new_df[['x', 'y']]), 
            cv2.RANSAC, 5.0
        )
        res.append({
            'chip': chip,
            'chip_num': int(chip[1:]),
            'polygon': all_cla,
            'polygon_without_warped': all_cla_without_warped,
            'image': img,
            'raw_df': raw_df,
            'new_df': new_df,
            'transformation_matrix': transformation_matrix # 从生数据到转后数据的变换矩阵
        })
    return res

@dataclass
class SplitedResult:
    image: np.ndarray
    cnt  : tuple[int, int, int, int]
    '''(x, y, w, h)'''

@dataclass
class SplitedResults:
    left : SplitedResult
    right: SplitedResult

    def __getitem__(self, key: Literal['left', 'right']) -> SplitedResult:
        return getattr(self, key)

def split_left_right(
    image: np.ndarray
) -> SplitedResults | None:
    m = image

    cnts, hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    if len(cnts) < 2:
        logger.warning(f'len(cnts) < 2')
        return None
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0]) # 用外接矩形的x排序

    res = {}

    for side, cnt in zip(('left', 'right'), cnts):
        new_side_mask = np.zeros_like(m)

        cv2.drawContours(new_side_mask, [cnt], -1, (255, ), thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(cnt)
        x = max(0, x-1)
        y = max(0, y-1)
        w = min(w+2, m.shape[1]-x)
        h = min(h+2, m.shape[0]-y)

        new_side_mask = new_side_mask[y:y+h, x:x+w] / 255

        res[side] = (SplitedResult(new_side_mask, (x, y, w, h)))

    return SplitedResults(**res)


@overload
def split_left_right_masks(
    images: list[np.ndarray] | list[dict], zcenter_args=None
) -> tuple[list[np.ndarray] | list[dict], dict[Literal['left', 'right'], np.ndarray]]: ...

@overload
def split_left_right_masks(
    images: list[np.ndarray] | list[dict], zcenter_args={}
) -> tuple[list[np.ndarray] | list[dict], dict[Literal['left', 'right'], np.ndarray]]: ...


def split_left_right_masks(images: list[np.ndarray] | list[dict], zcenter_args: dict|None=None):
    res = {}
    to_pop_idxs = []
    
    for i, image in enumerate(images):
        if isinstance(image, dict):
            image = image['mask']
            assert isinstance(image, np.ndarray)

        res_ = split_left_right(image)
        if res_ is None:
            to_pop_idxs.append(i)
            continue

        for side in ('left', 'right'):
            res.setdefault(side, [])
            res[side].append(res_[side].image)
    
    for i in to_pop_idxs[::-1]:
        images.pop(i)
    if zcenter_args is not None:
        for side in res:
            res_len = len(res[side])
            zcenter_args['target_shape'] = (*zcenter_args['target_shape'][:2], res_len)
            res[side] = zcrop_center(res[side], **zcenter_args)

    return images, res



@nb.njit
def tri_argmin(i, j, k):
    if i < j:
        if i < k:
            return 0
        else:
            return 2
    else:
        if j < k:
            return 1
        else:
            return 2

@nb.njit
def dtw(distance_matrix: np.ndarray):
    '''
    _, path = dtw(confusion_matrix)

    plt.imshow(((confusion_matrix)), cmap='Blues')
    plt.plot(path[:, 1], path[:, 0], color='orange', linewidth=2)
    '''
    m, n = distance_matrix.shape
    dtw_matrix = np.zeros((m+1, n+1))
    
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 1:] = np.inf

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 1 - distance_matrix[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],      # 向上
                                          dtw_matrix[i, j-1],      # 向左
                                          dtw_matrix[i-1, j-1])    # 斜向上左

    i, j = m, n
    path = [(i, j)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            # next_step = np.argmin([
            #     dtw_matrix[i-1, j], 
            #     dtw_matrix[i, j-1], 
            #     dtw_matrix[i-1, j-1]
            # ])
            next_step = tri_argmin(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1],
            )

            if next_step == 0:
                i -= 1
            elif next_step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))

    return np.array(path[::-1]) - 1


@lru_cache(maxsize=10000)
def load_pickled_data(p: Path | str):
    with open(p, 'rb') as f:
        return pickle.load(f)

@dataclass
class PariRegistInfo:
    fix_sec: AnimalSectionWithDistance
    mov_sec: AnimalSectionWithDistance

    mapping_path: str

    warped_img_path: str
    warped_grid_path: str = ''
    iou: float = -1

    @property
    def warped_image(self):
        return cv2.imread(self.warped_img_path, -1)

    @property
    def mapping(self) -> DiffeomorphicMap:
        return load_pickled_data(self.mapping_path)

    def transform_points(self, points: np.ndarray | pl.DataFrame) -> np.ndarray:
        '''points: {x, y}'''
        if isinstance(points, pl.DataFrame):
            points = points.to_numpy()


        dst_pts = self.mapping.transform_points_inverse(points[:, [1, 0]]) # 交换为 y x 顺序以便转换
        return dst_pts[:, [1, 0]] # 交换回 x y 顺序 # type: ignore


@cache
def get_corr(chip: str, source=Literal['db'] | str, corr_scale=100.0, bin_size=1.0):
    if source == 'db':
        return CorrectionPara.select(chip)[0]
    sp = Path(source) # type: ignore
    if not sp.exists(): raise FileNotFoundError(sp)

    config = read_yaml(sp)
    sec_para = get_sec_para_from_db(chip)

    for k, v in config.items():
        if chip in k:
            item = v
            break
    else:
        raise ValueError(f'Unknown chip: {chip}')
    
    corr_para = CorrectionPara(
        chip, '1', datetime.now(),
        item['flip_x'], item['flip_y'],
        item['degree'], item['scale'],
        item['top'], item['left'],
        item['w'], item['h'],
        sec_para["offset_x"], sec_para["offset_y"]
    ).with_scale(corr_scale).with_scale(1 / bin_size)
    return corr_para

dye_to_color = {
    'FB': 'blue',

    'CTB555': 'yellow',
    'CtbY': 'yellow',
    'Ctb_Y': 'yellow',

    'CTB647': 'red',
    'Ctb_R': 'red',

    'CTB488': 'green',
    'Ctb_G': 'green',
}
color_to_dye = {
    'blue': 'FB',
    'yellow': 'CTB555',
    'red': 'CTB647',
    'green': 'CTB488',
}


@cache
def read_corr_configs(path: str):
    try:
        return yaml.safe_load(open(path))
    except FileNotFoundError:
        return {}
def in_polys_par(p: Polygon, points: np.ndarray):
    tasks = [delayed(p.contains)(Point(*i)) for i in points]
    return np.array(Parallel(1, backend='threading')(tasks))

@cache
def get_connectome_corr_para_from_file(
    corr_path: str, animal_id: str, slice_id: str | int, bin_size: int,
    /, file_format: str = '{animal_id}-{slice_id}.png',
    corr_scale=100,
):
    name = file_format.format(animal_id=animal_id, slice_id=slice_id)
    item = read_corr_configs(corr_path).get(name)
    if item is None: return None
    
    corr_para = CorrectionPara(
        f'{animal_id}-{slice_id}', '1', datetime.now(),
        item['flip_x'], item['flip_y'],
        item['degree'], item['scale'],
        item['top'], item['left'],
        item['w'], item['h'],
        0, 0,
    ).with_scale(corr_scale).with_scale(1 / bin_size)
    return corr_para
