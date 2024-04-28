from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import cv2
import duckdb
import numpy as np
import pandas as pd
import polars as pl
import stackprinter
import torch
from connectome_utils import find_connectome_ntp, get_czi_size, read_ntp
from joblib import Memory
from ntp_manager import SliceMeta, parcellate
from scipy.interpolate import splev, splprep
from shapely.geometry import Polygon
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from utils import (
    AnimalSection,
    pad_df_and_mask,
    read_connectome_masks,
    split_left_right,
    split_left_right_masks,
    zcrop_center,
    to_numpy,
)

stackprinter.set_excepthook(style='darkbg2')

memory = Memory('/home/myuan/projects/cla/pwvy/cache', verbose=0)

pc = pl.col

animal_default_paths = {
    line.split(' ')[0]: line.split(' ')[1]
    for line in '''
C006 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset004_CLA_6.8.57/imagesInfer20231031/
C007 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset004_CLA_6.8.57/imagesInfer20231031/
C008 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset004_CLA_6.8.57/imagesInfer20231031/
C011 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset005_CLA_11-20/imagesInfer20231031/
C012 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset005_CLA_11-20/imagesInfer20231031/
C013 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset005_CLA_11-20/imagesInfer20231031/
C015 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset005_CLA_11-20/imagesInfer20231031/
C018 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset005_CLA_11-20/imagesInfer20231031/
C057 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Claustrum_infer/imagesInferResult20231020/
C042 /mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Claustrum_infer/imagesInferC042Result/
'''.strip().splitlines()
}


def get_base_path(animal: str):
    default_path = '/mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Claustrum_infer/imagesInfer20231117Result/'
    # return Path(animal_default_paths.get(animal, default_path))
    return Path(default_path)

@dataclass
class ClaDataset:
    animal_id    : str
    raw_images   : list[np.ndarray]
    raw_slice_ids: list[int]
    dataset      : Dataset
    dataloader   : DataLoader

    side: Literal['left', 'right', 'left-right', ''] = ''
    animal_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.animal_ids:
            self.animal_ids = [self.animal_id] * len(self.raw_images)

    def add(self, *others: 'ClaDataset', batch_size=64, shuffle=True):
        images     = []
        slice_ids  = []
        datasets   = []
        animal_ids = []

        for other in (self, *others):
            images.extend(other.raw_images)
            slice_ids.extend(other.raw_slice_ids)
            datasets.append(other.dataset)
            animal_ids.extend(other.animal_ids)

        dataset    = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        return ClaDataset(
            animal_id     = "|".join(sorted(set(animal_ids))),
            raw_images    = images,
            raw_slice_ids = slice_ids,
            dataset       = dataset,
            dataloader    = dataloader,
            animal_ids    = animal_ids,
        )

    def __repr__(self) -> str:
        side_str = f", side={self.side}" if self.side else ''

        return f"ClaDataset(animal_id={self.animal_id}, len={len(self.raw_images)}{side_str})"


def stereo_masks_to_cla_dataset(animal_id: str, res: list[dict], device = 'cuda', shuffle=False, batch_size=128, min_sum=0, side: Literal['left', 'right']='left'):
    res = [i for i in res if (i['image'][..., 0] != 0).sum() > min_sum]
    raw_images = [i['image'] / 255 for i in res]
    raw_slice_ids = [i['chip_num'] for i in res]
    

    dataset = TensorDataset(
        (torch.tensor(np.array(raw_images)).float().permute(0, 3, 1, 2) / 1).to(device),
        (torch.tensor(raw_slice_ids).long()).to(device),
    )
    dataloaders = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )

    return ClaDataset(
        animal_id     = animal_id,
        raw_images    = raw_images,
        raw_slice_ids = raw_slice_ids,
        dataset       = dataset,
        dataloader    = dataloaders,
        side          = side
    )



def get_dataset(
    side: Literal['left', 'right', 'left-right'], 
    raw_cntm_masks: list[dict], batch_size: int=64, shuffle=True, 
    animal_id='', target_shape: tuple[int, int]=(128, 128), device='cuda'
):
    raw_images = []
    raw_slice_ids = []

    new_raw_cntm_masks, side_splited = split_left_right_masks(raw_cntm_masks, zcenter_args=dict(
        target_shape=(*target_shape, len(raw_cntm_masks))
    ))
    assert isinstance(new_raw_cntm_masks, list) and isinstance(new_raw_cntm_masks[0], dict)
    raw_cntm_masks = cast(list[dict], new_raw_cntm_masks)

    for s in side.split('-'): # type: ignore
        s: Literal['left', 'right']

        curr_raw_images = np.repeat(
            zcrop_center(
                side_splited[s], (*target_shape, len(raw_cntm_masks))
            )[:, :, np.newaxis, ...], 3, axis=2)
        raw_images.extend([
            curr_raw_images[:, :, :, i] for i in range(curr_raw_images.shape[-1])
        ])
        raw_slice_ids.extend([i['slice_id'] for i in raw_cntm_masks])

    dataset = TensorDataset(
        (torch.tensor(np.array(raw_images)).float().permute(0, 3, 1, 2) / 1).to(device),
        (torch.tensor(raw_slice_ids).long()).to(device),
    )
    dataloaders = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    # return raw_images, raw_slice_ids, dataset, dataloaders
    return ClaDataset(
        animal_id     = animal_id,
        raw_images    = raw_images,
        raw_slice_ids = raw_slice_ids,
        dataset       = dataset,
        dataloader    = dataloaders,

        side = side,
    )

@lru_cache(maxsize=1024)
def extract_connectome_slice_meta(
    sec: AnimalSection, /, *, 
    bin_size: int=20, # 用于分割的底图就是 bin20 的，因此为与之对齐，此处亦 bin20
    ntp_base_path='/mnt/90-connectome/finalNTP-layer4-parcellation91-106/'
):
    ps = find_connectome_ntp(
        sec.animal_id, 
        base_path=ntp_base_path, 
        slice_id=sec.slice_id,
    )
    sm = read_ntp(ps, sec.animal_id, sec.slice_id, bin_size=bin_size)
    return sm


def extract_cell_df(
    item: dict, scale: float, with_all_cells=False, 
):
    '''
    Parameters
    ----------
    item : dict
        {
            'animal_id': str,
            'slice_id': int,
            'mask': np.ndarray,
            'splited': dict,
        }
        从 read_connectome_masks 返回的结果
    scale : float
        缩放比例
    '''
    animal_id = item['animal_id']
    slice_id = item['slice_id']

    sm = extract_connectome_slice_meta(
        AnimalSection(animal_id, slice_id), 
        ntp_base_path='/mnt/90-connectome/finalNTP/',
    )
    if sm is None:
        return 
    assert isinstance(sm, SliceMeta)
    assert sm.cells is not None

    cell_dfs = []

    for c in sm.cells.colors:
        cells = getattr(sm.cells, c) * scale
        cell_df = pl.DataFrame(cells, schema=['raw_x', 'raw_y']).with_columns(
            pl.lit(c).alias('color'), 
        )

        # if len(cells):
        #     print(c, cells.shape, sep='\t')

        side_dfs = []
        for side, side_other in zip(('left', 'right'), reversed(('left', 'right'))):
            cnt = item['splited'][side].cnt # x, y, w, h
            side_df = cell_df.with_columns(
                ((pc('raw_x') >= cnt[0]) & (pc('raw_x') <= cnt[0] + cnt[2]) &
                (pc('raw_y') >= cnt[1]) & (pc('raw_y') <= cnt[1] + cnt[3])).alias(side),

                (pc('raw_x') - cnt[0]).alias('x'),
                (pc('raw_y') - cnt[1]).alias('y'),
                pl.lit(cnt[0]).alias('cnt_x'),
                pl.lit(cnt[1]).alias('cnt_y'),
                pl.lit(cnt[2]).alias('cnt_w'),
                pl.lit(cnt[3]).alias('cnt_h'),
                pl.lit(False).alias(side_other),
            ).filter(side)
            side_dfs.append(side_df)
            # print(cnt[0], side_df)

        cell_df_filter = pl.concat(side_dfs, how='diagonal')
        if not with_all_cells:
            cell_df_filter = cell_df_filter.filter(pc('left') | pc('right'))
        cell_dfs.append(cell_df_filter)

    return pl.concat(cell_dfs)



@memory.cache(ignore=['tqdm'])
def read_regist_datasets(
    animal_id: str, scale: float, exclude_slices: list[AnimalSection]=[], 
    force=False, tqdm=lambda x: x, target_shape: tuple[int, int] = (256, 256),
    with_all_cells=False, min_sum=400, 
):
    regist_datasets: dict[AnimalSection, dict] = {}
    raw_cntm_masks = read_connectome_masks(
        get_base_path(animal_id), animal_id, scale,
        exclude_slices=exclude_slices, force=force, 
        min_sum=min_sum,
    )

    for info in tqdm(raw_cntm_masks):
        image = info['mask']
        assert isinstance(image, np.ndarray)

        res_ = split_left_right(image)
        if res_ is None:
            continue

        info['splited'] = res_
        df = extract_cell_df(info, scale, with_all_cells=with_all_cells)
        if df is None:
            print(f'{info["animal_id"]}-{info["slice_id"]} has no cells')
            continue
    
        pad_res = pad_df_and_mask(info, df, target_shape=target_shape)
        info['pad_res'] = pad_res

        sec = AnimalSection(
            animal_id=info['animal_id'],
            slice_id=f"{info['slice_id']:03d}",
        )

        regist_datasets[sec] = info
    return regist_datasets

def calc_mask_contours(mask: np.ndarray | Polygon, points_num=100) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mask, np.ndarray):
        mask = to_numpy(mask)
        cnts, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
        cnt = cnts[0].reshape(-1, 2)
    elif isinstance(mask, Polygon):
        cnt = np.array(mask.exterior.coords)
    else:
        raise TypeError(f'Unknown type: {type(mask)}')

    tck, u = splprep(cnt.T, per=1)
    u_new = np.linspace(u.min(), u.max(), points_num)
    x_new, y_new = splev(u_new, tck, der=0)
    smooth_cnt = np.stack([x_new, y_new], axis=1).astype(int)

    return cnt, smooth_cnt

@memory.cache(ignore=['datasets'])
def get_mask_cnts_and_raw(
    animal_id: str, slice_id: str, side: str, 
    datasets: dict[AnimalSection, dict] = {},
    exclude_slices: list[AnimalSection] = [],
    p配准时缩放 = 0.1,
    v展示时缩放 = 0.5,
):
    j计算缩放 = int(v展示时缩放 / p配准时缩放)

    if not datasets:
        datasets = read_regist_datasets(
            animal_id, v展示时缩放, exclude_slices=exclude_slices, 
            target_shape=(128*j计算缩放, 128*j计算缩放)
        )
    mask, _ = datasets[AnimalSection(
        animal_id=animal_id,
        slice_id=slice_id,
    )]['pad_res'][side]

    raw_cnt, smooth_cnt = calc_mask_contours(mask)
    return mask, raw_cnt, smooth_cnt



@dataclass
class SimilarityMatrix:
    fix_animal_id: str
    src_animal_id: str
    side: Literal['left', 'right']

    save_root: Path = Path('./output/similarity_matrixs')
    @property
    def p(self):
        return self.save_root / f'{self.fix_animal_id}-{self.src_animal_id}-{self.side}.csv'
    
    @property
    def exists(self):
        return self.p.exists()
    
    def save(self, similarity_matrix: np.ndarray, row_labels: list, col_labels: list):
        df = pd.DataFrame(
            similarity_matrix, 
            index=row_labels, columns=col_labels
        )
        df.to_csv(self.p)

    def load(self):
        return pd.read_csv(self.p, index_col=0)
    

# 读入转录组细胞
def read_cell_type_result(
    chip: str, 
    cell_type_root = Path('/data/sdf/to_zhang/snRNA/spatialID-20240115/'),
    cell_type_version = 'macaque1_CLA_res1.2_49c_csv',
    total_gene_2d_version = 'macaque-20240125-Mq179-cla'
) -> pl.DataFrame:
    cell_type_p = next((cell_type_root / cell_type_version).glob(f'*{chip}*.csv'))
    res_p = cell_type_p.with_name(f'{chip}_with_total_gene2d.parquet')
    if res_p.exists():
        res = pl.read_parquet(res_p)
    else:
        res = duckdb.sql(f'''
            select * from "/data/sde/total_gene_2D/{total_gene_2d_version}/total_gene_{chip}_macaque_f001_2D_{total_gene_2d_version}.parquet" as t1 
            join "{cell_type_p}" as t2
            on t1.cell_label = t2.cell
        ''').pl()
        res.write_parquet(res_p)
    return res
