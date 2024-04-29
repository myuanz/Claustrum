# %%
import itertools
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from copy import copy
from datetime import datetime
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Literal, TypedDict, cast

if os.path.exists('pwvy'):
    os.chdir((Path('.') / 'pwvy').absolute())

import cairo
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shutil
import pyinstrument
import yaml
from bidict import bidict
from bokeh.io import output_file, save
from bokeh.plotting import figure, output_notebook, show
from connectome_utils import find_connectome_ntp, get_czi_size
from dataset_utils import get_base_path, read_regist_datasets
from joblib import Memory, Parallel, delayed
from loguru import logger
from nhp_utils.image_correction import CorrectionPara
from ntp_manager import SliceMeta, parcellate
from range_compression import RangeCompressedMask, mask_encode
from scipy.interpolate import splev, splprep
from shapely import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from utils import (
    AnimalSection,
    AnimalSectionSide,
    AnimalSectionWithDistance,
    PariRegistInfo,
    poly_to_bitmap,
    read_connectome_masks,
    read_exclude_sections,
    split_into_n_parts_slice,
    to_numpy,
)
import regist_utils
import importlib
importlib.reload(regist_utils)


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

def draw_cnt(ctx: cairo.Context, cnt: np.ndarray, fill=False):
    ctx.move_to(*cnt[0])
    for i in cnt[1:]:
        ctx.line_to(*i)
    ctx.move_to(*cnt[0])
    ctx.close_path()
    if fill:
        ctx.fill()
    else:
        ctx.stroke()

T_SIDE = Literal['left', 'right', '']

pc = pl.col

PROJECT_NAME = 'CLA'
assets_path = Path('/mnt/97-macaque/projects/cla/injections-cells/assets/raw_image/')
exclude_slices = read_exclude_sections(PROJECT_NAME)
memory = Memory('./cache', verbose=0)
regist_results_raw = pickle.load(open(assets_path.parent / 'regist_results_20240129_connectome_to_C042.pkl', 'rb'))
regist_results_raw = [i for i in regist_results_raw if isinstance(i, PariRegistInfo)]

regist_to_mq179_results = pickle.load(open(assets_path.parent / 'regist_results_20240118_C042_to_Mq179.pkl', 'rb'))
regist_to_mq179_results = [i for i in regist_to_mq179_results if isinstance(i, PariRegistInfo)]

# <为单注射位点计算分布特用>
# target_animal_id = 'C074'
# regist_results = []

# for i in regist_results_raw:
#     if i.mov_sec.animal_id == target_animal_id:
#         i = copy(i)
#         i.fix_sec = i.mov_sec
#         regist_results.append(i)

# </为单注射位点计算分布特用>
regist_results = regist_results_raw

mov_animal_to_regist_results: defaultdict[tuple[str, T_SIDE], list[PariRegistInfo]] = defaultdict(list)
for i in regist_results:
    mov_animal_to_regist_results[(i.mov_sec.animal_id, i.mov_sec.side)].append(i)
for k in mov_animal_to_regist_results:
    mov_animal_to_regist_results[k].sort(key=lambda x: x.mov_sec.slice_id_int)

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

target_zone_id_df = pl.read_excel('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/merged-clustermap-reordered_ind-0419.xlsx').filter(pc('no_repeat_58_inds').is_not_null())
target_zone_ids = target_zone_id_df['inverted_clutered_Names'].to_list()


cluster_df_p = './cluster_df.csv'
cluster_df_p = './cluster_df_to_draw.csv'
cluster_df_p = '/mnt/97-macaque/projects/cla/injections-cells/injection-zone/cluster_dfd_0424.xlsx'

cluster_df = pl.read_excel(cluster_df_p).select(
    pc('zone_id').str.replace("'", '').alias('zone_id'),
    'cluster'
).with_columns(
    pc('zone_id').str.split('-').list.get(0).alias('animal_id'),
    pc('zone_id').str.split('-').list.get(1).alias('dye'),
    pc('cluster').str.split('-').list.get(0).alias('big-cluster')
).with_columns(
    pc('dye').replace(dye_to_color, default=None).alias('color')
)
# .filter(
#     pc('zone_id').is_in(target_zone_ids)
# )

# cluster_df = target_zone_id_df.select(
#     pc('inverted_clutered_Names').alias('zone_id'),
#     pc('eight cluster').alias('cluster'),
#     pc('eight cluster').alias('big-cluster'),
# ).with_columns(
#     pc('zone_id').str.split('-').list.get(0).alias('animal_id'),
#     pc('zone_id').str.split('-').list.get(1).alias('dye'),
# ).with_columns(
#     pc('dye').replace(dye_to_color, default=None).alias('color')
# )


assert len(cluster_df.filter(pc('color').is_null())) == 0
cluster_df
# %%
p配准时缩放 = 0.1
v展示时缩放 = 0.5
j计算缩放 = int(v展示时缩放 / p配准时缩放)
animal_id = 'C042'
side = 'left'

fix_datasets = read_regist_datasets(
    'C042', v展示时缩放, exclude_slices=exclude_slices,
    target_shape=(128*j计算缩放, 128*j计算缩放),
    min_sum=100,
)
mov_datasets = read_regist_datasets(
    animal_id, v展示时缩放, exclude_slices=exclude_slices, 
    target_shape=(128*j计算缩放, 128*j计算缩放)
)

def calc_mask_contours(mask: np.ndarray, points_num=100) -> tuple[np.ndarray, np.ndarray]:
    mask = to_numpy(mask)
    cnts, _ = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
    cnt = cnts[0].reshape(-1, 2)

    tck, u = splprep(cnt.T, per=1)
    u_new = np.linspace(u.min(), u.max(), points_num)
    x_new, y_new = splev(u_new, tck, der=0)
    smooth_cnt = np.stack([x_new, y_new], axis=1).astype(int)

    return cnt, smooth_cnt

@memory.cache(ignore=['datasets'])
def get_mask_cnts_and_cell(
    animal_id: str, slice_id: str, side: str, 
    datasets: dict[AnimalSection, dict] = {}
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    if not datasets:
        datasets = read_regist_datasets(
            animal_id, v展示时缩放, exclude_slices=exclude_slices, 
            target_shape=(128*j计算缩放, 128*j计算缩放), 
        )
    mask, cells = datasets[AnimalSection(
        animal_id=animal_id,
        slice_id=slice_id,
    )]['pad_res'][side]

    raw_cnt, smooth_cnt = calc_mask_contours(mask)
    return mask, raw_cnt.astype(np.float32), smooth_cnt.astype(np.float32), cells


mask, raw_cnt, smooth_cnt, cells = get_mask_cnts_and_cell(
    animal_id, '181', side, mov_datasets
)
cells
# %%

C042_corr_conf_p = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240308-C042Template/myuan-precision3650tower-config.yaml')
# C042_corr_conf_p = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240318-optimized/C077-R/DESKTOP-QH967BF-config.yaml')


# C042_corr_conf_p = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240308-C042Template/DESKTOP-NLB3J50-config.yaml')


# import yaml

# config = yaml.safe_load((C042_corr_conf_p.with_suffix('.yaml.bck')).read_text())
# new_config = {}
# for key in config:
#     if key.startswith('C042'):
#         item = config[key]
#         item['left'] -= 500 
#         # item['left'] = item['left'] - 500
#         new_config[key] = item

# yaml.safe_dump(new_config, C042_corr_conf_p.open('w'))
# %%

# %%

enable_regist = True

def export_connectome_cells(animal_id: str, export_root: Path):
    @cache
    def read_corr_configs(path: str):
        try:
            return yaml.safe_load(open(path))
        except FileNotFoundError:
            return {}

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

    try:
        mov_datasets = read_regist_datasets(
            animal_id, v展示时缩放, exclude_slices=exclude_slices, 
            target_shape=(128*j计算缩放, 128*j计算缩放)
        )
    except Exception as e:
        print(e)
        return animal_id


    for side in ('left', 'right'):
        print(animal_id, side)
        target_p = export_root / f'{animal_id}-{side}-cells.parquet'
        # if target_p.exists(): continue

        res = []
        plt.figure(figsize=(20, 3))


        regist_pairs = mov_animal_to_regist_results[(animal_id, side)]

        for ani_sec, item in mov_datasets.items():
            if ani_sec.slice_id_int < 100: continue
            if ani_sec.slice_id_int > 250: continue
            # if ani_sec.slice_id_int > 136: break
            regist_matchs = [
                i for i in regist_pairs if i.mov_sec.slice_id_int == ani_sec.slice_id_int
            ]
            if not regist_matchs: 
                print(f'No match for {ani_sec}')
                continue
            regist_best_match = max(regist_matchs, key=lambda x: x.iou)

            corr = get_connectome_corr_para_from_file(
                C042_corr_conf_p, 
                regist_best_match.fix_sec.animal_id, 
                regist_best_match.fix_sec.slice_id, 
                int(20 / p配准时缩放 / j计算缩放), 
                corr_scale=200
            )
            if corr is None: 
                print(f'No corr for {ani_sec}')
                continue


            slice_id = ani_sec.slice_id
            mov_mask, mov_raw_cnt, mov_smooth_cnt, mov_cell_df = get_mask_cnts_and_cell(
                animal_id, slice_id, side, mov_datasets
            )

            # print(regist_best_match.mov_sec)
            if regist_best_match.mov_sec.need_flip:
                mov_mask = cv2.flip(mov_mask, 1)
                w = mov_mask.shape[1]
                mov_raw_cnt[:, 0] = w - mov_raw_cnt[:, 0]
                mov_smooth_cnt[:, 0] = w - mov_smooth_cnt[:, 0]

                # before_max_x = mov_cell_df['x'].max()
                mov_cell_df = mov_cell_df.with_columns(
                    (w - pc('x')).alias('x')
                )
                after_max_x = mov_cell_df['x'].max()
                # print('flip', w, before_max_x, after_max_x)
            fix_mask, fix_raw_cnt, fix_smooth_cnt, _ = get_mask_cnts_and_cell(
                regist_best_match.fix_sec.animal_id, 
                regist_best_match.fix_sec.slice_id, 
                regist_best_match.fix_sec.side, 
                datasets=fix_datasets
            )

            mov_cells = mov_cell_df[['x', 'y']].to_numpy()
            if len(mov_cells) > 2: 
                mov_cells = regist_best_match.transform_points(mov_cells / j计算缩放) * j计算缩放
                # plt.scatter(*mov_cells.T * j计算缩放)
                mov_cells_in_target = fix_mask[mov_cells[:, 1].astype(int), mov_cells[:, 0].astype(int)] != 0
                # if mov_cells_in_target.any():
                #     print(f'In target {ani_sec}')
                # mov_cells *= j计算缩放
                new_cells_xy = corr.wrap_point(mov_cells)
                new_mov_cell_df = mov_cell_df.with_columns(
                    pc('x').alias('raw_x'),
                    pc('y').alias('raw_y'),
                    pl.Series(new_cells_xy[:, 0]).alias('x'),
                    pl.Series(new_cells_xy[:, 1]).alias('y'),
                    pl.Series(mov_cells_in_target).alias('in_target'),
                )
            else:
                new_mov_cell_df = mov_cell_df.with_columns(
                    pc('x').alias('raw_x'),
                    pc('y').alias('raw_y'),
                    pl.lit(False).alias('in_target')
                )
            new_mov_cell_df = new_mov_cell_df.with_columns(
                # mov animal, mov slice, mov side
                # fix animal, fix slice, fix side
                pl.lit(animal_id).alias('mov_animal_id'),
                pl.lit(slice_id).alias('mov_slice_id'),
                pl.lit(side).alias('mov_side'),
                pl.lit(regist_best_match.fix_sec.animal_id).alias('fix_animal_id'),
                pl.lit(regist_best_match.fix_sec.slice_id).alias('fix_slice_id'),
                pl.lit(regist_best_match.fix_sec.side).alias('fix_side'),
                pc('color').replace({
                    'blue': 'FB',
                    'yellow': 'CTB555',
                    'red': 'CTB647',
                    'green': 'CTB488',
                }, default=None).alias('dye')
            )

            new_cnt = regist_best_match.transform_points(mov_raw_cnt / j计算缩放) * j计算缩放

            new_cnt = corr.wrap_point(new_cnt)

            p = Polygon(new_cnt)
            text_p = p.representative_point()
            plt.plot(*new_cnt.T)
            # plt.text(*[*np.array(text_p.xy).reshape(-1)], regist_best_match.fix_sec.slice_id)
            if len(new_mov_cell_df) > 2:
                for (in_target, ), g in new_mov_cell_df.group_by(['in_target']):
                    # if not in_target: continue
                    plt.scatter(g['x'], g['y'], s=1, c='r' if in_target else 'b', alpha=0.5 if in_target else 0.1)

            # print(ani_sec)

            res.append({
                'animal_id': animal_id,
                'slice_id': slice_id,
                'fix_sec': regist_best_match.fix_sec,
                'mov_sec': regist_best_match.mov_sec,
                'corr': corr,
                'cell_df': new_mov_cell_df,
            })

            # break
        plt.axis('equal')
        plt.savefig(export_root / f'{animal_id}-{side}-cells.svg')
        plt.close()
        if not res: 
            print(f'No res for {animal_id}-{side}')
            continue
        total_cells: pl.DataFrame = pl.concat([i['cell_df'] for i in res])
        total_cells.write_parquet(target_p, compression='snappy')
    return animal_id

animals = [f'C{i:03d}' for i in range(6, 100)]
# animals = ['C042']

export_root = Path('/data/sdf/to_wangML/injection-zone/connectome-cells-20240321-wml/')
export_root.mkdir(exist_ok=True, parents=True)


# tasks = [delayed(export_connectome_cells)(i, export_root) for i in animals]
# for i in (pbar := tqdm(Parallel(
#     48, return_as='generator_unordered'
# )(tasks), total=len(tasks))):
#     pbar.set_postfix_str(i)

# %%
# 下面是按分组合并绘制数据
# %%
export_root = Path('/data/sdf/to_wangML/injection-zone/connectome-cells-20240321-wml/')

parquets = list(export_root.glob('*.parquet'))

all_dfs = []
for i in parquets:
    df = pl.read_parquet(i).select(pl.all().exclude('literal'))
    all_dfs.append(df)
all_df: pl.DataFrame = pl.concat(all_dfs).with_row_index()
all_df
# %%
fix_slice_ids = sorted(set(
    i.fix_sec.slice_id_int for i in regist_results
    if i.fix_sec.slice_id_int <= 265
))
# fix_slice_ids = [i.slice_id_int for i in fix_datasets.keys() if i.slice_id_int < 215]

fix_slice_ids_to_index = bidict({i: idx for idx, i in enumerate(fix_slice_ids)})

fix_animal_id = 'C042'
# fix_animal_id = 'C074'

added_w = 256 # 每张片子之间多加这么宽


plt.figure(figsize=(10, 1))

fix_cnts = []
for i, slice_id in enumerate(sorted(fix_slice_ids)):
    corr = get_connectome_corr_para_from_file(
        C042_corr_conf_p, 
        fix_animal_id, 
        f'{slice_id:03d}', 
        int(20 / p配准时缩放 / j计算缩放), 
        corr_scale=200
    )
    # <为单注射位点计算分布特用>
    # corr = get_connectome_corr_para_from_file(
    #     C042_corr_conf_p, 
    #     fix_animal_id, 
    #     f'{slice_id:03d}', 
    #     int(80 / p配准时缩放 / j计算缩放), 
    #     corr_scale=100
    # )
    # </为单注射位点计算分布特用>

    if corr is None: continue
    assert corr is not None, f'{fix_animal_id}-{slice_id}'

    fix_mask, fix_raw_cnt, fix_smooth_cnt, _ = get_mask_cnts_and_cell(
        fix_animal_id, 
        f'{slice_id:03d}', 
        'left', 
        datasets=fix_datasets,
    )
    new_cnt = corr.wrap_point(fix_raw_cnt)
    new_cnt[:, 0] += i*added_w

    fix_cnts.append(new_cnt)
    plt.plot(*new_cnt.T)
plt.axis('equal')
plt.gca().invert_yaxis()

pdf_w = max(fix_cnts, key=lambda x: x[:, 0].max())[:, 0].max() + 200
pdf_h = max(fix_cnts, key=lambda x: x[:, 1].max())[:, 1].max() + 200

pdf_w, pdf_h
# %%

def calc_kde(X: np.ndarray, min_x: float, max_x: float, points=1000, bw=75):
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X.reshape(-1, 1))

    x = np.linspace(min_x, max_x, points)
    k_values = np.exp(kde.score_samples(x.reshape(-1, 1)))
    k_values = k_values / k_values.max()
    return x, k_values


def draw_kde(
    ctx: cairo.Context, x: np.ndarray, y: np.ndarray, 
    kde_w: float, pdf_w: float, pdf_h: float, 
    bw_x=30, bw_y=10
):
    x_distributed, x_k_values = calc_kde(x, 0, pdf_w-kde_w, bw=bw_x)
    y_distributed, y_k_values = calc_kde(y, 0, pdf_h-kde_w, bw=bw_y)

    x_k_values = x_k_values * kde_w/3 + pdf_h-kde_w
    y_distributed, y_k_values = y_k_values * kde_w/3 + pdf_w-kde_w, y_distributed

    ctx.set_source_rgba(0, 0, 0, 0.5)
    ctx.set_line_width(5)

    ctx.line_to(0, pdf_h-kde_w)
    ctx.line_to(x_distributed[0], x_k_values[0])
    for x, y in zip(x_distributed, x_k_values):
        ctx.line_to(x, y)
    ctx.line_to(0, pdf_h-kde_w)
    ctx.fill()

    ctx.line_to(pdf_w-kde_w, 0)
    ctx.line_to(y_distributed[0], y_k_values[0])
    for x, y in zip(y_distributed, y_k_values):
        ctx.line_to(x, y)
    ctx.line_to(pdf_w-kde_w, 0)

    ctx.fill()
# %%
def draw_cluster_pdf(
    group: pl.DataFrame, group_name: str, 
    export_root: Path, pdf_w: float, pdf_h: float,
    show_outer=False
):
    with cairo.PDFSurface(str(export_root / f'cluster-{group_name}.pdf'), pdf_w, pdf_h) as surface:
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

        ctx.set_source_rgb(0, 0, 0)
        for fix_cnt in fix_cnts:
            draw_cnt(ctx, fix_cnt)

        target_animals = []

        for animal_id, color in group.select('animal_id', 'color').iter_rows():

            target_animals.append({
                'animal_id': animal_id,
                'color': color,
                'side': 'left' if animal_id.endswith('L') else 'right'
            })
        
        target_animals_df = pl.DataFrame(target_animals)
        # target_cells_df = all_df.join(target_animals_df, left_on=['mov_animal_id', 'mov_side', 'color'], right_on=['animal_id', 'side', 'color']).filter('in_target')
        # print(k, target_cells_df[['x', 'y']].shape)
        # ctx.set_source_rgba(0, 0, 0, 1)

        # for x, y in target_cells_df[['x', 'y']].iter_rows():
        #     ctx.arc(x, y, 3, 0, 2 * np.pi)
        #     ctx.fill()

        target_cells_df = all_df.join(target_animals_df, left_on=['mov_animal_id', 'mov_side', 'color'], right_on=['animal_id', 'side', 'color'])
        for (in_target, ), g in target_cells_df.group_by(['in_target']):
            print(group_name, in_target, g.shape)

            for x, y, fix_slice_id in g[['x', 'y', 'fix_slice_id']].iter_rows():
                if in_target:
                    ctx.set_source_rgba(0, 0, 0, 0.4)
                else:
                    if not show_outer:
                        continue
                    ctx.set_source_rgba((int(fix_slice_id) - 100) / 167, 0, 1, 1)

                x += fix_slice_ids_to_index[int(fix_slice_id)] * added_w
                ctx.arc(x, y, 3, 0, 2 * np.pi)
                ctx.fill()

        in_target_cells = target_cells_df.filter('in_target')
        if len(in_target_cells):
            draw_kde(
                ctx, in_target_cells['x'].to_numpy() + fix_slice_ids_to_index[int(fix_slice_id)] * added_w, 
                in_target_cells['y'].to_numpy(), 
                200, pdf_w, pdf_h, bw_x=128
            )
        # ctx.set_source_rgba(0, 0, 0, 1 / len(group)*3)

    
export_root = Path('/data/sdf/to_wangML/injection-zone/connectome-cells-20240424/')
export_root.mkdir(exist_ok=True, parents=True)

# for (k, ), g in cluster_df.group_by(['cluster']):
#     draw_cluster_pdf(g, k, export_root, pdf_w, pdf_h)
#     draw_cluster_pdf(g, f'outer-{k}', export_root, pdf_w, pdf_h, show_outer=True)
#     # break

# for (k, ), g in cluster_df.group_by(['big-cluster'], maintain_order=True):
#     draw_cluster_pdf(g, k, export_root, pdf_w, pdf_h)
#     draw_cluster_pdf(g, f'outer-{k}', export_root, pdf_w, pdf_h, show_outer=True)
#     break

# %%
# from sklearn.cluster._hdbscan.hdbscan import (
#     HDBSCAN,  # from sklearn.cluster import HDBSCAN
# )
# from concave_hull import concave_hull
# from matplotlib.axes import Axes
# import seaborn as sns

# for (k, ), g in cluster_df.group_by(['big-cluster'], maintain_order=True):
#     print(k, g.shape)
#     target_animals = []

#     for animal_id, color in g.select('animal_id', 'color').iter_rows():

#         target_animals.append({
#             'animal_id': animal_id,
#             'color': color,
#             'side': 'left' if animal_id.endswith('L') else 'right'
#         })
    
#     target_animals_df = pl.DataFrame(target_animals)
#     target_cells_df = all_df.join(target_animals_df, left_on=['mov_animal_id', 'mov_side', 'color'], right_on=['animal_id', 'side', 'color']).filter('in_target')


#     fig, ax = plt.subplots(figsize=(20, 3))
#     ax.invert_yaxis()

#     for (_, fix_slice_id), slice_g in tqdm(target_cells_df.group_by(
#         ['fix_animal_id', 'fix_slice_id'], maintain_order=True
#     )):
#         # if fix_slice_id > '150': continue

#         # print(fix_slice_id, slice_g.shape)
#         cell_points = slice_g[['x', 'y']].to_pandas()

#         ax.scatter(*cell_points.to_numpy().T + np.array([0, 800]).reshape(2, -1), s=1, alpha=0.1, c='black')
#         # ax.scatter(*cell_points.to_numpy().T, s=1, alpha=0.1, c='black')
#         i = fix_slice_ids_to_index[int(fix_slice_id)]
#         fix_cnt = fix_cnts[i] - np.array([i*added_w, 0])
#         ax.plot(*fix_cnt.T, c='black')
#         ax.plot(*fix_cnt.T + np.array([0, 800]).reshape(2, -1), c='black')

#         sns.kdeplot(
#             data=cell_points, x='x', y='y', fill=True, 
#             thresh=0.5, levels=30, ax=ax, color='green'
#         )


#     plt.savefig(export_root / f'cluster-{k}-kde.pdf')
#     plt.close()
# %%
import cairo
from scipy.interpolate import UnivariateSpline


def points_kde(
    all_points: np.ndarray, 
    selected_index: np.ndarray,
    mesh_size_scale=1, bandwidth=0.025, 
    zz_factor=lambda x: x, atol=0.5
):
    points = all_points[selected_index]

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    ps = (points - min_xy) / (max_xy - min_xy)
    mesh_w, mesh_h = ((max_xy - min_xy) * mesh_size_scale).astype(int)
    
    try:
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, atol=atol).fit(ps)
        xx, yy = np.meshgrid(np.linspace(0, 1, mesh_w),
                            np.linspace(0, 1, mesh_h))
        zz = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except ValueError:
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, atol=atol)
        zz = np.zeros((mesh_h, mesh_w))
    
    ps = (all_points - min_xy) / (max_xy - min_xy)
    scores = np.exp(kde.score_samples(ps))
    zz = zz_factor(zz)
    return scores, kde, zz, min_xy, max_xy

@memory.cache
def cached_points_kde(    
    all_points: np.ndarray, 
    selected_index: np.ndarray,
    mesh_size_scale=1, bandwidth=0.025
):
    return points_kde(all_points, selected_index, mesh_size_scale, bandwidth)

# %%
@cache
def get_cluster_animal_and_channels(cluster_name: str):
    return cluster_df.filter(pc('big-cluster') == cluster_name).select(
        'animal_id', 'color'
    ).with_columns(
        pl.when(pc('animal_id').str.ends_with('L')).then(
            pl.lit('left')
        ).otherwise(pl.lit('right')).alias('side')
    )

@cache
def get_cluster_cells(cluster_name: str):
    target_animals_df = get_cluster_animal_and_channels(cluster_name).with_columns(
        pc('animal_id').str.slice(0, 4)
    )
    target_cells_df = all_df.join(
        target_animals_df, 
        left_on=['mov_animal_id', 'mov_side', 'color'], 
        right_on=['animal_id', 'side', 'color']
    ).filter('in_target').sort('fix_slice_id')
    return target_cells_df

# @memory.cache()
def calc_all_mean_dists(cluster_df: pl.DataFrame):
    all_dists: dict[tuple[str, str], np.ndarray] = {} # {(cluster_name, fix_slice_id): avg_dists}

    for (k, ), g in cluster_df.group_by(['big-cluster'], maintain_order=True):
        k = cast(str, k)
        # if k != 'P4':
        #     continue

        for (fix_slice_id, ), df in get_cluster_cells(k).sort(
            'fix_slice_id'
        ).group_by(['fix_slice_id'], maintain_order=True):
            fix_slice_id = cast(str, fix_slice_id)

            points = df[['x', 'y']].to_numpy()
            dist_mtx = pairwise_distances(points, n_jobs=-1)
            dists = np.triu(dist_mtx, k=1).flatten()
            dists = dists[dists > 0]


            mtx = dist_mtx.copy()
            np.fill_diagonal(mtx, np.nan)
            if np.isnan(mtx).all():
                continue
            avg_dists = np.nanmean(mtx, axis=1)
            all_dists[(k, fix_slice_id)] = avg_dists
    return all_dists
# %%
@dataclass
class KDEResultItem:
    scores: np.ndarray
    kde: KernelDensity
    zz: np.ndarray
    min_xy: np.ndarray
    max_xy: np.ndarray
    fix_slice_id: str
    cell_count: int

@dataclass
class KDEResults:
    scores: list[np.ndarray]
    kde: list[KernelDensity]
    zzs: list[np.ndarray]
    min_xys: list[np.ndarray]
    max_xys: list[np.ndarray]
    fix_slice_ids: list[str]
    cell_counts: list[int]
    selected_cells: list[np.ndarray]
    '''使用了哪些细胞参与kde计算，是 index 数组'''

    @staticmethod
    def new() -> 'KDEResults':
        return KDEResults([], [], [], [], [], [], [], [])

    def append(
        self, 
        scores: np.ndarray,
        kde: KernelDensity,
        zz: np.ndarray, min_xy: np.ndarray, max_xy: np.ndarray, 
        fix_slice_id: str, cell_count: int, selected_cells: np.ndarray
    ):
        self.scores.append(scores)
        self.kde.append(kde)
        self.zzs.append(zz)
        self.min_xys.append(min_xy)
        self.max_xys.append(max_xy)
        self.fix_slice_ids.append(fix_slice_id)
        self.cell_counts.append(cell_count)
        self.selected_cells.append(selected_cells)

    def max_xy(self):
        return np.max(self.max_xys, axis=0)

    def min_xy(self):
        return np.min(self.min_xys, axis=0)

    def sort(self):
        idxs = np.argsort(self.fix_slice_ids)
        self.zzs = [self.zzs[i] for i in idxs]
        self.min_xys = [self.min_xys[i] for i in idxs]
        self.max_xys = [self.max_xys[i] for i in idxs]
        self.fix_slice_ids = [self.fix_slice_ids[i] for i in idxs]
        self.cell_counts = [self.cell_counts[i] for i in idxs]
        self.selected_cells = [self.selected_cells[i] for i in idxs]
        self.scores = [self.scores[i] for i in idxs]
        self.kde = [self.kde[i] for i in idxs]
        return self

    def __len__(self):
        return len(self.zzs)

    def __getitem__(self, idx):
        return KDEResultItem(
            self.scores[idx],
            self.kde[idx],
            self.zzs[idx], self.min_xys[idx], self.max_xys[idx], 
            self.fix_slice_ids[idx], self.cell_counts[idx]
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        if len(self):
            return f'KDEResults({len(self)}, fix:{self.fix_slice_ids[0]}-{self.fix_slice_ids[-1]})'
        else:
            return f'KDEResults(0)'

# %%
@memory.cache()
def calc_cell_kde_by_slice(
    cluster_name: str, bandwidth=0.05, pv=95, 
):
    target_cells_df = get_cluster_cells(cluster_name)
    mean_dists = calc_all_mean_dists(cluster_df)
    zz_res = KDEResults.new()

    def f(mean_dists, fix_slice_id: str, slice_g):
        curr_avg_dists = mean_dists.get((cluster_name, fix_slice_id))
        if curr_avg_dists is None: return

        p = np.percentile(curr_avg_dists, pv)
        less_p_idx = curr_avg_dists < p

        cell_points = slice_g[['x', 'y']].sort('x', 'y').to_numpy()
        less_p_cell_points = cell_points[less_p_idx]
        target_cell_index = slice_g['index'].to_numpy()[less_p_idx]
        if len(less_p_cell_points) < 3: return
        # print(less_p_cell_points.shape, fix_slice_id)
        scores, kde, zz, min_xy, max_xy = points_kde(
            cell_points, less_p_idx, bandwidth=bandwidth,
        )
        return scores, kde, zz, min_xy, max_xy, fix_slice_id, less_p_cell_points.shape[0], target_cell_index

    tasks = [
        delayed(f)(mean_dists, k[1], slice_g)
        for k, slice_g in target_cells_df.group_by(
            ['fix_animal_id', 'fix_slice_id'], maintain_order=True
        )
    ]
    for res in tqdm(Parallel(48, return_as='generator_unordered')(tasks), total=len(tasks)):
        if res is None: continue
        zz_res.append(*res)
    
    return zz_res
# kde_res = calc_cell_kde_by_slice(1)
kde_res = calc_cell_kde_by_slice(
    'P1',
    bandwidth=0.09,
    pv=99,
).sort()

kde_res
# %%
n = len(kde_res)
plt.figure(figsize=(n, 2))

for i in range(0, n):
    plt.subplot(1, n, i+1)
    # plt.imshow(((kde_res.zzs[i] > 0.01) * 255).astype(np.uint8))
    plt.imshow(kde_res.zzs[i])

    plt.axis('off')

    # plt.subplot(2, n, (i)*(n + 1) + 1)
    # plt.imshow(kde_res.zzs[i] > 0)

    plt.axis('off')
    # plt.gca().invert_yaxis()
plt.tight_layout()
# %%
landmark_base_path = Path('/mnt/97-macaque/projects/cla/stereo-cla-highlight-20240327-landmarker/')
landmark_csv_pths = sorted((landmark_base_path / f'{fix_animal_id}-L').glob('*.csv'))
landmark_csv_pths
landmark_warps: dict[int, regist_utils.ItkTransformWarp] = {}

for i in landmark_csv_pths:
    s = int(i.stem.split('-')[1])
    landmark_warps[s] = regist_utils.ItkTransformWarp.from_big_warp_df(i)

landmark_warps
# %%
export_root = Path('/data/sdf/to_wangML/injection-zone/20240426-kdezone-test-alpha/')
export_root.mkdir(exist_ok=True, parents=True)

def exectract_polygons(p):
    corrds = []
    if isinstance(p, Polygon):
        corrds.append(np.array(p.exterior.coords))
    elif isinstance(p, MultiPolygon):
        for pp in p.geoms:
            corrds.append(np.array(pp.exterior.coords))
    elif isinstance(p, GeometryCollection):
        for pp in p.geoms:
            if isinstance(pp, Polygon):
                corrds.append(np.array(pp.exterior.coords))
            elif isinstance(pp, MultiPolygon):
                for ppp in pp.geoms:
                    corrds.append(np.array(ppp.exterior.coords))
    return corrds


@dataclass
class DrawKDEArgs:
    zz_cnt_res: list[np.ndarray]
    inner_cell_n: int
    '''在 kde 分区的数量'''
    total_cell_n: int
    '''所有细胞数量'''

    pv: int
    '''首步去噪比值'''
    selected_cell_n: int
    '''按照pv选出来的细胞数量'''

    titles: list[str] = field(default_factory=list)
    cell_alphas: list[float] = field(default_factory=list)

def xy_to_index_wrapper(rcm: RangeCompressedMask, binary_search=False):
    def f(xy):
        x = (xy.struct.field('x').to_numpy())
        y = (xy.struct.field('y').to_numpy())

        out = rcm.find_index(x, y, binary_search=binary_search)
        return pl.Series(out)
    return f


def draw_cnt(ctx: cairo.Context, cnt: np.ndarray, fill=False):
    ctx.move_to(*cnt[0])
    for i in cnt[1:]:
        ctx.line_to(*i)
    ctx.move_to(*cnt[0])
    if fill:
        ctx.fill()
    # else:
    #     # 
    ctx.close_path()
    ctx.stroke()
    
def draw_kde_cnts(
    output_p: Path, 
    fix_cnts: list[np.ndarray], 
    zz_cnt_res: dict[float, DrawKDEArgs],
):
    max_xy = np.max([i.max(axis=0) for i in fix_cnts], axis=0)
    min_xy = np.min([i.min(axis=0) for i in fix_cnts], axis=0)
    single_w, single_h = (max_xy - min_xy).astype(int)

    margin_h = 100
    w = single_w + margin_h
    h = single_h + margin_h

    h *= len(zz_cnt_res) # 总共有 len(fix_cnts) 个平铺结果，每个平铺结果之间有 margin_h 的间隔
    
    with cairo.PDFSurface(str(output_p), w, h) as surface:
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(5)
        for i, (thr, draw_ked_args) in enumerate(zz_cnt_res.items()):
            base_x = 0
            base_y = i * single_h + i * margin_h
            d = np.array([base_x, base_y])

            ctx.set_source_rgba(0, 1, 0, 0.9)            
            ctx.set_font_size(12)

            for i, (cnt, title) in enumerate(zip(draw_ked_args.zz_cnt_res, draw_ked_args.titles)):
                draw_cnt(ctx, cnt + d - min_xy)
            
                title_xy = cnt[0] + d - min_xy
                ctx.move_to(*title_xy)
                ctx.show_text(title)

            ctx.set_source_rgba(0, 1, 0, 0.9)
            for cnt in draw_ked_args.zz_cnt_res:
                draw_cnt(ctx, cnt + d - min_xy, fill=True)

            ctx.set_source_rgba(0, 0, 0, 0.5)
            for cnt in fix_cnts:
                draw_cnt(ctx, cnt + d - min_xy)

            ctx.set_font_size(64)
            ctx.move_to(*d + np.array([10, 300]))
            ctx.show_text(f'阈2={thr:.3f}, {draw_ked_args.inner_cell_n}/{draw_ked_args.total_cell_n} ~= {draw_ked_args.inner_cell_n/draw_ked_args.total_cell_n:.3f}')
            ctx.move_to(*d + np.array([10, 400]))
            ctx.show_text(f'阈1={draw_ked_args.pv} {draw_ked_args.inner_cell_n}/{draw_ked_args.selected_cell_n} ~= {draw_ked_args.inner_cell_n/draw_ked_args.selected_cell_n:.3f}')

def draw_cells_with_kde(
    output_p: Path,
    fix_cnts: list[np.ndarray],
    zz_cnt_res: dict[float, DrawKDEArgs],
    cell_df: pl.DataFrame,
    cell_r = 1,
):
    max_xy = np.max([i.max(axis=0) for i in fix_cnts], axis=0)
    min_xy = np.min([i.min(axis=0) for i in fix_cnts], axis=0)
    single_w, single_h = (max_xy - min_xy).astype(int)

    margin_h = 100
    w = single_w + margin_h
    h = single_h + margin_h

    h *= len(zz_cnt_res) # 总共有 len(fix_cnts) 个平铺结果，每个平铺结果之间有 margin_h 的间隔
    
    with cairo.PDFSurface(str(output_p), w, h) as surface:
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(5)
        cell_points = cell_df[['x', 'y']].to_numpy()

        for i, (thr, draw_ked_args) in enumerate(zz_cnt_res.items()):
            base_x = 0
            base_y = i * single_h + i * margin_h
            d = np.array([base_x, base_y])



            ctx.set_source_rgba(0, 1, 0, 0.9)            
            ctx.set_font_size(12)

            for j, (cnt, title) in enumerate(zip(draw_ked_args.zz_cnt_res, draw_ked_args.titles)):
                draw_cnt(ctx, cnt + d - min_xy)
            
                title_xy = cnt[0] + d - min_xy
                ctx.move_to(*title_xy)
                ctx.show_text(title)

            # ctx.set_source_rgba(0, 1, 0, 0.9)
            # for cnt in draw_ked_args.zz_cnt_res:
            #     draw_cnt(ctx, cnt + d - min_xy, fill=True)

            ctx.set_source_rgba(0, 0, 0, 0.5)
            for cnt in fix_cnts:
                draw_cnt(ctx, cnt + d - min_xy)

            slice_kde = np.concatenate(draw_ked_args.cell_alphas)
            print(slice_kde.shape, slice_kde.shape)
            print(slice_kde, '\n', draw_ked_args.cell_alphas)
            for s, (x, y) in zip(slice_kde, cell_points):
                ctx.set_source_rgba(1, 0, 0, s)
                ctx.arc(
                    x + d[0] - min_xy[0], y + d[1] - min_xy[1], 
                    cell_r, 
                    0, 2 * np.pi
                )
                ctx.fill()

pv = 99
bandwidth = 0.09

def trans(warper: regist_utils.ItkTransformWarp, points: np.ndarray, fix_cnt_i: int):
    return points
    points = points / j计算缩放
    points = warper.transform_points_to_mov(points)
    points = points * 2

    points[:, [0, 1]] = points[:, [1, 0]]
    min_xy = fix_cnts[fix_cnt_i].min(axis=0)
    points = points - min_xy

    points[:, 0] += fix_cnt_i*added_w/3
    return points

for (k, ), g in cluster_df.group_by(['big-cluster'], maintain_order=True):
    k = cast(str, k)
    if k not in 'P1':
        continue
    target_cells_df = get_cluster_cells(k).with_columns(
        (pc('x') // 2 * 2).cast(pl.Int64).alias('x_trunc'),
        (pc('y') // 2 * 2).cast(pl.Int64).alias('y_trunc'),
    ).unique(['x_trunc', 'y_trunc'])

    target_cells_df = get_cluster_cells(k)
    fix_to_mov_slice_id_dict = dict(zip(*target_cells_df[['fix_slice_id', 'mov_slice_id']].unique(['mov_slice_id', 'fix_slice_id']).to_numpy().T))

    # thr = {
    #     'P1': 0.55,
    #     'P2': 0.25,
    #     'P3': 0.25,
    #     'P4': 0.4,
    #     'P74FB': 0.25,
    #     'P77R': 0.25,
    # }[k]

    print(k, g.shape)
    zzs_data = calc_cell_kde_by_slice(k, pv=pv, bandwidth=bandwidth).sort()
    all_score_max = np.max([np.max(i.scores) for i in zzs_data])
    all_score_min = np.min([np.min(i.scores) for i in zzs_data])

    zzs_data.scores = [(i.scores - all_score_min) / (all_score_max - all_score_min) for i in zzs_data]

    selected_cell_ids = np.concatenate(zzs_data.selected_cells)
    zz_cell_count_mov_mean = np.convolve(zzs_data.cell_counts, np.ones(3) / 3, mode='valid')
    zz_cell_count_mov_mean = np.pad(zz_cell_count_mov_mean, (1, 1), 'edge')
    zz_cell_count_mov_mean = np.maximum(zz_cell_count_mov_mean, zzs_data.cell_counts)
    # zz_mov_mean = zzs_sum

    total_max_xy = np.round(zzs_data.max_xy()).astype(int)
    total_min_xy = np.round(zzs_data.min_xy()).astype(int)

    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(zzs_data.cell_counts, label='raw')
    # ax.plot(np.array(zz_cell_count_mov_mean), label='mov mean')

    zz_cnt_res: dict[float, DrawKDEArgs] = {} # {thr: [zz_cnts]}
    final_fix_cnts = []

    for thr in (np.linspace(0, 0.5, 1)):
        zz_cnt_res[thr] = DrawKDEArgs([], 0, 0, pv, 0, [], [])

        for i in range(len(zzs_data.zzs)):
            fix_slice_id = zzs_data.fix_slice_ids[i]
            # print(fix_slice_id)

            # <转换到转录组>
            # 由于不是所有landmark都画了
            # closest_landmark = min(landmark_warps.keys(), key=lambda x:abs(x-int(fix_slice_id)))

            # landmark_warp = landmark_warps[closest_landmark]
            # print('keep', fix_slice_id)
            # </转换到转录组>
            # mov_slice_id = fix_to_mov_slice_id_dict.get(fix_slice_id)
            # if mov_slice_id not in '115, 123, 131, 139, 147, 155, 163, 171,179, 187, 195, 203, 211, 219, 227, 235, 243, 251, 259, 267': continue

            scores = zzs_data.scores[i]
            # scores = (
            #     (scores - scores.min()) / (scores.max() - scores.min())
            # )
            zz_cnt_res[thr].cell_alphas.append(scores)

            fix_cnt_i = fix_slice_ids_to_index[int(fix_slice_id)]
            raw_fix_cnt = (fix_cnts[fix_cnt_i] - np.array([fix_cnt_i*added_w, 0]))
            fix_cnt = raw_fix_cnt.copy()
            # <转换到转录组>

            # fix_cnt = trans(landmark_warp, fix_cnt, fix_cnt_i)
            # fix_cnt_min_xy = fix_cnt.min(axis=0)
            # fix_cnt -= fix_cnt_min_xy
            # # plt.plot(*fix_cnt.T, c='r')

            # csv_p = [i.stem for i in landmark_csv_pths if f'{closest_landmark:03d}' in i.name]

            # </转换到转录组>
            zz_cnt_res[thr].titles.append(f'{fix_cnt_i}')

            if len(final_fix_cnts) <= len(fix_cnts):
                final_fix_cnts.append(fix_cnt)
            raw_fix_cnt_p = Polygon(raw_fix_cnt)
            fix_cnt_p = Polygon(fix_cnt)

            raw_zz = zzs_data.zzs[i]
            zz = raw_zz * zz_cell_count_mov_mean[i] / raw_fix_cnt_p.area
            min_xy = zzs_data.min_xys[i]

            zz_cnts, _ = cv2.findContours(
                (zz > thr).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            for zz_cnt in zz_cnts:
                if len(zz_cnt) < 3: 
                    # print(f'[{thr:.3f}] 1 continue')
                    continue
                p = Polygon(zz_cnt.reshape(-1, 2) + min_xy)
                try:
                    p = raw_fix_cnt_p.intersection(p)
                except Exception as e:
                    # print(f'[{thr:.3f}] {e} continue')
                    continue

                # if p.area < 300: continue
                # print(p.area)
                corrds = exectract_polygons(p)
                for _zz_cnt in corrds:
                    if len(_zz_cnt) < 3: 
                        # print(f'[{thr:.3f}] 2 continue')
                        continue

                    # <转换到转录组>
                    # _zz_cnt = trans(landmark_warp, _zz_cnt, fix_cnt_i) - fix_cnt_min_xy

                    # _zz_cnt += total_min_xy
                    # plt.plot(*_zz_cnt.T)
                    # print(_zz_cnt)
                    # </转换到转录组>

                    zz_cnt_res[thr].zz_cnt_res.append(_zz_cnt)
            

        region_mask = np.zeros((
            int(target_cells_df['y'].max() - target_cells_df['y'].min()) + 2, 
            int(target_cells_df['x'].max() - target_cells_df['x'].min()) + 2,
        ), dtype=np.uint8)
        for corrd in zz_cnt_res[thr].zz_cnt_res:
            corrd = corrd.astype(int) - total_min_xy
            if corrd.max(axis=0)[0] > region_mask.shape[1] or corrd.max(axis=0)[1] > region_mask.shape[0]:
                print(f'Out of range {corrd.max(axis=0)} {region_mask.shape}')
                continue

            if corrd.min(axis=0)[0] < 0 or corrd.min(axis=0)[1] < 0:
                print(f'Out of range {corrd.min(axis=0)} {region_mask.shape}')
                continue

            cv2.fillPoly(region_mask, [corrd], (255, ))

        
        X = (target_cells_df['x'].to_numpy() - total_min_xy[0]).astype(int)
        Y = (target_cells_df['y'].to_numpy() - total_min_xy[1]).astype(int)
        
        in_polys = region_mask[Y, X] != 0

        zz_cnt_res[thr].inner_cell_n = sum(in_polys)
        zz_cnt_res[thr].total_cell_n = len(target_cells_df)
        zz_cnt_res[thr].selected_cell_n = len(selected_cell_ids)
        print(thr, len(zz_cnt_res[thr].zz_cnt_res))

    kde_pdf_p = export_root / f'{k}-{pv}-all-thr-bw{bandwidth:.4f}2.pdf'
    shutil.rmtree(kde_pdf_p, ignore_errors=True)
    # draw_kde_cnts(
    #     kde_pdf_p, 
    #     final_fix_cnts, 
    #     zz_cnt_res
    # )


    for cell_r in range(1, 3):
        kde_pdf_p = export_root / f'with-cells-{k}-{pv}-all-thr-bw{bandwidth:.4f}-cellr{cell_r}-global.pdf'
        shutil.rmtree(kde_pdf_p, ignore_errors=True)
        draw_cells_with_kde(
            kde_pdf_p, 
            final_fix_cnts, 
            zz_cnt_res,
            cell_df=target_cells_df,
            cell_r=cell_r,
        )

    # inner_cell_ns = [n.inner_cell_n for thr, n in zz_cnt_res.items()]
    # total_cell_ns = [n.total_cell_n for thr, n in zz_cnt_res.items()]
    # # Create a new figure and axis
    # fig, ax1 = plt.subplots()
    # ax1.set_title(k)

    # ax1.plot(inner_cell_ns)
    # ax1.plot(total_cell_ns)
    # ax1.set_ylim(0, max(total_cell_ns) * 1.1)

    # xticks = [f'{thr:.2f}' for thr in zz_cnt_res.keys()]
    # ax1.set_xticks(range(len(xticks)), xticks, rotation=45)
    # # Plot the ratio of inner_cell_ns to total_cell_ns
    # ratio = [i / j for i, j in zip(inner_cell_ns, total_cell_ns)]
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.plot(ratio)
    # ax2.set_ylim(0, 1 * 1.1)

    # # Label the axes
    # ax1.set_xlabel('thr')
    # ax1.set_ylabel('Cell Counts', color='black')
    # ax2.set_ylabel('Ratio')
    # plt.savefig(export_root / f'{k}-cell-counts-bw{bandwidth:.4f}.png')
# %%
points = target_cells_df.filter(
    pl.lit('115, 123, 131, 139, 147, 155, 163, 171,179, 187, 195, 203, 211, 219, 227, 235, 243, 251, 259, 267').str.contains(pc('mov_slice_id'))
)[['x', 'y']].to_numpy()
points = target_cells_df[['x', 'y']].to_numpy()

plt.figure(figsize=(10, 3))
plt.scatter(*points.T, s=1, )
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()

# %%
all_dists = calc_all_mean_dists(cluster_df)

        # pvs = list(range(99, 80, -2))
        # fig, axes = plt.subplots(1, len(pvs), figsize=(10, 3))
        # for pv, ax in zip(pvs, axes):
        #     p = np.percentile(avg_dists, pv)

        #     less_p_idx = avg_dists < p

        #     ax.scatter(*points[less_p_idx].T, s=1)
        #     ax.scatter(*points[~less_p_idx].T, s=1, c='red')

        #     ax.set_aspect('equal', adjustable='box')
        #     ax.invert_yaxis()
        #     ax.set_title(f'{pv=}')
        #     ax.axis('off')

        # plt.savefig(export_root / f'{k}-{fix_slice_id}-cell-dist.png')
        # plt.close()

# %%
pv = 90

p = np.percentile(all_mean_dists, pv)
plt.title(f'{p=}')
plt.hist(all_mean_dists, bins=200)
plt.axvline(p, color='red')
rather_close = all_mean_dists < p
print(f'rather_close: {np.sum(rather_close)}')
# %%


mtx = dist_mtx.copy()
np.fill_diagonal(mtx, np.nan)
avg_dists = np.nanmean(mtx, axis=1)
less_p_idx = avg_dists < p

curr_cells = target_cells_df[['x', 'y']].to_numpy()
plt.scatter(*curr_cells[less_p_idx].T, s=1)
plt.scatter(*curr_cells[~less_p_idx].T, s=1, c='red')
# Flip y-axis and maintain aspect ratio
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()


# %%
# 下面是读取动物各自的平铺结果 然后分细胞通道绘制在一起的

ntp_base_path = Path('/mnt/90-connectome/finalNTP/')
ntp_base_path = Path('/mnt/90-connectome/finalNTP-layer4-parcellation91-106/')

base_config_path = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240318-optimized/')
base_config_path = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240419')
base_config_path = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240425')


# base_config_path = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240407')

bin_size = 100

all_animals = [i.name for i in base_config_path.iterdir() if i.is_dir()]
all_animals
# %%
# %%

class AnimalPDFExportor:
    def __init__(self, animal: str):
        '''
        animal: str, like 'C074-R'
        '''
        self.animal = animal
        self.animal_id, self.side = animal.split('-')


    def find_config_p_and_secs(self, config_root: Path):
        selected_p = list((config_root / self.animal / 'Selected').glob('*-config.yaml'))

        if selected_p:
            self.arrange_config_p = selected_p[0]
        else:
            self.arrange_config_p = list((config_root / self.animal).glob('*-config.yaml'))[0]    

        png_ps = list(self.arrange_config_p.parent.glob('*.png'))
        self.secs = [
            AnimalSection(self.animal_id, i.stem.split('-')[1]) 
            for i in png_ps
        ]
        self.corr_config = read_corr_configs(self.arrange_config_p)

        return self.arrange_config_p, self.secs

    @staticmethod
    def _find_single_cnts_and_cells(
        corr_config: dict, 
        sec: AnimalSection, side: Literal['R', 'L'], 
        ntp_base_path: Path,
        bin_size: int=100, 
    ):

        @cache
        def get_connectome_corr_para_from_file(
            animal_id: str, slice_id: str | int, bin_size: int,
            /, file_format: str = '{animal_id}-{slice_id}.png',
            corr_scale=100,
        ):
            name = file_format.format(animal_id=animal_id, slice_id=slice_id)
            item = corr_config.get(name)
            if item is None: return None
            
            corr_para = CorrectionPara(
                f'{animal_id}-{slice_id}', '1', datetime.now(),
                item['flip_x'], item['flip_y'],
                item['degree'], item['scale'],
                item['top'], item['left'],
                item['w'], item['h'],
                0, 0,
            )
            # print(f"{corr_scale} ?= {bin_size} {corr_scale == bin_size}")
            if corr_scale != bin_size:
                corr_para = corr_para.with_scale(corr_scale).with_scale(1 / bin_size)
            return corr_para

        print(sec.animal_id, sec.slice_id)
        try:
            ntp = find_connectome_ntp(
                sec.animal_id, base_path=str(ntp_base_path), slice_id=sec.slice_id, 
                skip_pre_pro=True
            )[0]
        except IndexError:
            print(f'{sec} ntp 未找到，从 {ntp_base_path}')
            raise

        w, h = get_czi_size(sec.animal_id, sec.slice_id)
        sm = parcellate(ntp, um_per_pixel=0.65, w=w//bin_size, h=h//bin_size, bin_size=bin_size, return_bytes=False, export_position_policy='none')

        assert isinstance(sm, SliceMeta)

        corr = get_connectome_corr_para_from_file(sec.animal_id, sec.slice_id, bin_size)
        try:
            assert corr is not None, f'No corr for {sec.animal_id}-{sec.slice_id} bin{bin_size}'
        except AssertionError as ex:
            print(f'{sec.animal_id}-{sec.slice_id} bin{bin_size} corr not found')
            return [], {}

        curr_cells = sm.cells
        assert curr_cells is not None

        cnts = []
        cells = {
            'red': np.zeros((0, 3)),
            'green': np.zeros((0, 3)),
            'blue': np.zeros((0, 3)),
            'yellow': np.zeros((0, 3)),
        }

        for r in sm.regions:
            if f'{side}-Cla' not in r.label.name: continue
            print(r.label.name)
            cnt = np.array(r.polygon.exterior.coords)
            poly = Polygon(cnt)

            for color in curr_cells.colors:
                if not curr_cells[color].any(): continue

                in_p = in_polys_par(poly, curr_cells[color][:, :2])
                curr_color_cells = np.hstack((
                    corr.wrap_point(curr_cells[color]), 
                    in_p.reshape(-1, 1)
                ))

                cells[color] = np.vstack((cells[color], curr_color_cells))

            cnt_corr = corr.wrap_point(cnt)
            cnts.append(cnt_corr)
        return cnts, cells
    

    def find_cnts_and_cells(
        self, ntp_base_path: Path, bin_size: int=100
    ):
        cnts: list[np.ndarray] = []
        cells: dict[str, np.ndarray] = {
            'red': np.zeros((0, 3)),
            'green': np.zeros((0, 3)),
            'blue': np.zeros((0, 3)),
            'yellow': np.zeros((0, 3)),
        }

        tasks = [
            delayed(self._find_single_cnts_and_cells)(self.corr_config, sec, self.side, ntp_base_path, bin_size)
            for sec in self.secs
        ]
        for cnts_, cells_ in tqdm(Parallel(
            1, return_as='generator_unordered', 
            backend='threading',
        )(tasks), total=len(tasks)):
            if not cnts_ or not cells_: continue
            cnts.extend(cnts_)
            for color in cells:
                cells[color] = np.vstack((cells[color], cells_[color]))

        self.cnts  = cnts
        self.cells = cells
        return cnts, cells


    def draw_pdf(self, /, *, export_root: Path|None=None, kde_w: int=200):
        if export_root is None:
            export_root = self.arrange_config_p.parent.parent
        export_root = export_root / 'pdfs'
        export_root.mkdir(exist_ok=True, parents=True)

        min_x = min(self.cnts, key=lambda x: x[:, 0].min())[:, 0].min()
        min_y = min(self.cnts, key=lambda x: x[:, 1].min())[:, 1].min()
        max_x = max(self.cnts, key=lambda x: x[:, 0].max())[:, 0].max()
        max_y = max(self.cnts, key=lambda x: x[:, 1].max())[:, 1].max()

        cnts_new = [
            (i - [min_x, min_y])
            for i in self.cnts
        ]


        pdf_w = max_x - min_x
        pdf_h = max_y - min_y

        
        pdf_w += kde_w
        pdf_h += kde_w


        for color in self.cells:
            output_svg_p = export_root / f'{color}-{animal}-cells-.pdf'

            with cairo.PDFSurface(str(output_svg_p), pdf_w, pdf_h) as surface:
                ctx = cairo.Context(surface)
                ctx.set_source_rgb(1, 1, 1)
                ctx.paint()

                ctx.set_source_rgb(0, 0, 0)
                ctx.set_line_width(1)
                for fix_cnt in cnts_new:
                    draw_cnt(ctx, fix_cnt)
                ctx.set_source_rgba(0, 0.4, 0, 0.5)

                curr_cells = self.cells[color] - np.array((min_x, min_y, 0))
                curr_cells = curr_cells[curr_cells[:, -1] > 0]
                if len(curr_cells) < 2:
                    continue

                for x, y, in_p in curr_cells:
                    ctx.arc(x, y, 1, 0, 2 * np.pi)
                    ctx.fill()


                draw_kde(ctx, curr_cells[:, 0], curr_cells[:, 1], kde_w, pdf_w, pdf_h)


for animal_p in (list(base_config_path.glob('C*'))):
    animal = animal_p.name
    # animal = 'C075-L'
    # animal = 'C063-R'
    # animal = 'C056-L'
    # if animal[:4] not in r'C098R\C054R\C032R\C063R\C059L\C094R\C075L\C060R': continue
    if animal[:4] not in 'C077': continue
    print(animal, animal_p)
    bin_size = 100
    try:
        e = AnimalPDFExportor(animal)
        print(e.find_config_p_and_secs(base_config_path))
        try:
            cnts, _ = e.find_cnts_and_cells(ntp_base_path, bin_size)
        except Exception:
            e.find_cnts_and_cells(Path('/mnt/90-connectome/finalNTP/'), bin_size)
        e.draw_pdf(export_root=base_config_path)
    except IndexError:
        pass
    except Exception as ex:
        print(animal_p, ex)
        import traceback
        traceback.print_exception(ex)
        continue


# %%
# 这一部分是读取老版svg然后提取点画出来的
import xml.etree.ElementTree as ET

from svgpathtools import parse_path


@cache
def get_polygon(tag: ET.Element, curve_n=500):
    attrs = tag.attrib
    if 'd' in attrs:
        p = parse_path(attrs['d'])

        lines = []
        for i in range(curve_n+1):
            po = p.point(i / curve_n)
            lines.append((po.real, po.imag))
        lines.append(lines[0])
        lines = np.array(lines)
    elif 'points' in attrs:
        points = attrs['points']
        lines = np.fromiter(map(float, points.split(' ')), dtype=float).reshape(-1, 2)
    else:
        raise ValueError(f'unknown tag: {tag}')

    return Polygon(lines)


import_svg_p = '/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/four-channel.svg'
svg = ET.parse(import_svg_p)

base_svg_pths = svg.findall('.//{http://www.w3.org/2000/svg}polygon')
svg_points = svg.findall('.//{http://www.w3.org/2000/svg}path')
# %%
from collections import Counter

classes = [i.attrib['class'] for i in base_svg_pths]
Counter(classes)
# %%
from itertools import groupby

base_pths = [get_polygon(i) for i in base_svg_pths]
points = {
    k: [get_polygon(i) for i in g]
    for k, g in groupby(svg_points, key=lambda x: x.attrib['class'])
}
len(base_pths), [len(i) for i in points.values()]
# %%
points = []

for p in svg_points:
    x, y = get_polygon(p).centroid.xy
    points.append({
        'class': p.attrib['class'],
        'x': x[0],
        'y': y[0],
    })
# %%
point_df = pd.DataFrame(points)
point_df
# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

joint = sns.jointplot(
    point_df,
    x='x',
    y='y',

    hue='class',
    kind='scatter', 
    # marginal_kws={
    #     'kde': True,
    # },
    height=10,
)
joint.ax_joint.invert_yaxis()
# joint.ax_joint.set_aspect('equal', adjustable='box')
# save fig
plt.savefig(Path(import_svg_p).parent.parent / 'jointplot.pdf')