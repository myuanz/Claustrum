'''
之前在 pwvy/20240308-create-region-svg.py 画了很多配准后的 KDE，但是有些问题：

1. 往下一步转录组配准很麻烦
2. 架构无法轻易修改为全切片共同配准
3. 那个文件代码太多了

这个文件主要负责: 

读取原始细胞、在全切片范围计算KDE和边界、绘制
'''
# %%
import itertools
import os
import pickle
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Literal, TypedDict, cast

if os.path.exists('pwvy'):
    os.chdir((Path('.') / 'pwvy').absolute())

import importlib
import shutil

import cairo
import cairo_utils
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyinstrument
import regist_utils
import yaml
from bidict import bidict
from bokeh.io import output_file, save
from bokeh.plotting import figure, output_notebook, show
from connectome_utils import find_connectome_ntp, get_czi_size
from dataset_utils import get_base_path, read_regist_datasets
from joblib import Memory, Parallel, delayed
from loguru import logger
from nhp_utils.image_correction import CorrectionPara
from ntp_manager import NTPRegion, SliceMeta, parcellate
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
    color_to_dye,
    dye_to_color,
    get_connectome_corr_para_from_file,
    in_polys_par,
    poly_to_bitmap,
    read_connectome_masks,
    read_exclude_sections,
    split_into_n_parts_slice,
    to_numpy,
)

importlib.reload(regist_utils)
importlib.reload(cairo_utils)

draw_cnt = cairo_utils.draw_cnt
memory = Memory('./cache', verbose=0)

pc = pl.col
PROJECT_NAME = 'CLA'
exclude_slices = read_exclude_sections(PROJECT_NAME)

raw_scale = 1/50
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

# %%

mov_animal_id = 'C063'
mov_animal_corr_config_p = '/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240318-optimized/C063-R/DESKTOP-GQEVEF3-config.yaml'
target_side = 'R'
target_color = 'red' # CTB647

mov_animal_id = 'C077'
mov_animal_corr_config_p = '/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240318-optimized/C077-R/DESKTOP-QH967BF-config.yaml'
target_side = 'R'
target_color = 'blue' # FB

corr_config = yaml.load(open(mov_animal_corr_config_p), Loader=yaml.FullLoader)
mov_secs: list[AnimalSection] = []
for i in corr_config.keys():
    animal, slice_id = i.replace('.png', '').split('-')
    if animal != mov_animal_id:
        continue
    mov_secs.append(AnimalSection(animal, slice_id))
mov_secs
# %%
cells = []

regions: defaultdict[AnimalSection, list[NTPRegion]] = defaultdict(list)
rregions: defaultdict[AnimalSection, list[NTPRegion]] = defaultdict(list)

for sec in tqdm(mov_secs):
    ntp_p = find_connectome_ntp(sec.animal_id, slice_id=sec.slice_id, base_path='/mnt/90-connectome/finalNTP-layer4-parcellation91-106')[0]
    w, h = get_czi_size(sec.animal_id, sec.slice_id)
    corr = get_connectome_corr_para_from_file(
        mov_animal_corr_config_p, 
        sec.animal_id, 
        sec.slice_id, 
        1, 
        corr_scale=100,
    )
    assert corr is not None

    sm = parcellate(ntp_p, um_per_pixel=0.65, w=w, h=h, export_position_policy='none')
    for r in sm.regions:
        regions[sec].append(r)
        rregion = Polygon(corr.wrap_point(
            np.array(r.polygon.exterior.coords) + corr.offset
        ) * raw_scale)
        rregions[sec].append(NTPRegion(label=r.label, polygon=rregion))

    for color in sm.cells.colors:
        cell_points: np.ndarray = getattr(sm.cells, color)
        corred_points = corr.wrap_point(cell_points)

        for i, (cell, corr_cell) in enumerate(zip(cell_points, corred_points)):
            cells.append([
                sec.animal_id, 
                sec.slice_id_int, 
                color, 
                *cell * raw_scale,
                *corr_cell * raw_scale,
            ])
# %%
cell_df = pl.DataFrame(
    cells, schema=('animal_id', 'slice_id', 'color', 'x', 'y', 'rx', 'ry')
).cast({'slice_id': pl.UInt32}).with_row_index()
cell_df

# %%
all_in_polys = []
all_target_regions: dict[AnimalSection, Polygon] = {} # This is the param of regions

for (animal, slice_id), g in cell_df.group_by(['animal_id', 'slice_id'], maintain_order=True): # type: ignore
    sec = AnimalSection(animal, f'{slice_id:03d}') # type: ignore
    g = g.filter(pc('color') == target_color)
    cell_points = g[['rx', 'ry']].to_numpy()

    curr_sec_regions = [r for r in rregions[sec] if r.label.name == f'{target_side}-Cla-s']
    curr_sec_region = unary_union([r.polygon for r in curr_sec_regions])

    in_polys = in_polys_par(curr_sec_region, cell_points)
    print(sec, sum(in_polys), len(in_polys), [r for r in rregions[sec] if 'Cla' in r.label.name])
    all_target_regions[sec] = curr_sec_region

    for cell, rcell, in_poly in zip(cell_points, g[['rx', 'ry']].to_numpy(), in_polys):
        all_in_polys.append([
            animal, 
            slice_id, 
            *cell, 
            *rcell, 
            in_poly,
        ])

all_cell_df = pl.DataFrame(
    all_in_polys, schema=('animal_id', 'slice_id', 'x', 'y', 'rx', 'ry', 'in_poly')
).cast({'slice_id': pl.UInt32, 'in_poly': pl.Boolean}).with_row_index() # This is all param of cells
# %%
all_cell_df.filter(pc('in_poly'))
# %%
for (slice_id, ), g in all_cell_df.group_by(['slice_id'], maintain_order=True): # type: ignore
    points = g.filter(pc('in_poly'))[['rx', 'ry']].to_numpy()
    plt.scatter(*points.T, s=1)
for sec, p in all_target_regions.items():
    for cnt in exectract_polygons(p):
        plt.plot(*cnt.T, color='black', linewidth=0.1)

plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()

# %%
def points_kde(
    points: np.ndarray, 
    cnt_min_xy: np.ndarray, cnt_max_xy: np.ndarray,
    mesh_size_scale=1, bandwidth=0.025, 
    zz_factor=lambda x: x, atol=0.5,
):
    # min_xy = points.min(axis=0)
    # max_xy = points.max(axis=0)
    ps = (points - cnt_min_xy) / (cnt_max_xy - cnt_min_xy)
    mesh_w, mesh_h = ((cnt_max_xy - cnt_min_xy) * mesh_size_scale).astype(int)

    try:
        kde = KernelDensity(
            kernel='gaussian', bandwidth=bandwidth, atol=atol,
            leaf_size=40
        ).fit(ps)
        xx, yy = np.meshgrid(np.linspace(0, 1, mesh_w),
                             np.linspace(0, 1, mesh_h))
        zz = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        # zz = (kde.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

    except ValueError:
        zz = np.zeros((mesh_h, mesh_w))
    zz = zz_factor(zz)
    return zz, cnt_min_xy, cnt_max_xy

@memory.cache()
def cached_points_kde(points: np.ndarray, 
    cnt_min_xy: np.ndarray, cnt_max_xy: np.ndarray,
    mesh_size_scale=1, bandwidth=0.025
):
    return points_kde(points, cnt_min_xy, cnt_max_xy, mesh_size_scale, bandwidth)


plt.figure(figsize=(20, 5))

thr = 1.2
bw = 0.05
for (slice_id, ), g in all_cell_df.group_by( # type: ignore
    ['slice_id'], maintain_order=True
):
    # if slice_id != 171: continue
    g = g.filter('in_poly')

    r = (
        g.with_columns(
            (pc('rx') // 2 * 2).cast(pl.Int64).alias('rx_trunc'),
            (pc('ry') // 2 * 2).cast(pl.Int64).alias('ry_trunc'),
        ).unique(['rx_trunc', 'ry_trunc'])
    )
    
    points = r[['rx', 'ry']].to_numpy()



    if len(points) < 2: continue
    poly = all_target_regions[AnimalSection(animal_id=mov_animal_id, slice_id=f'{slice_id:03d}')]
    cnt_min_xy = np.array(poly.bounds[:2])
    cnt_max_xy = np.array(poly.bounds[2:])

    zz, min_xy, max_xy = points_kde(
        points, cnt_min_xy, cnt_max_xy,
        bandwidth=bw, atol=0.5
    )
    print(slice_id, zz.shape, zz.max(), len(points), cnt_min_xy, cnt_max_xy)
    print(slice_id, g.shape, r.shape)
    mask = zz > thr
    cnts, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )

    # plt.imshow(zz > thr, extent=(
    #     min_xy[0], max_xy[0], min_xy[1], max_xy[1]
    # ), origin='lower', cmap='viridis', alpha=0.5)

    plt.imshow(zz, extent=(
        min_xy[0], max_xy[0], min_xy[1], max_xy[1]
    ), origin='lower', cmap='viridis', alpha=1)

    for cnt in cnts:
        if cv2.contourArea(cnt) < 10:
            continue
        cnt = cnt.squeeze() + min_xy
        plt.fill(cnt[:, 0], cnt[:, 1], color='lightblue', alpha=0.1)

    # plt.scatter(*points.T, s=2, alpha=0.5, c='black')
plt.title(f'{bw=}')
plt.colorbar()
for sec, p in all_target_regions.items():
    final_fix_cnts = exectract_polygons(p)
    for cnt in final_fix_cnts:
        plt.plot(*cnt.T, color='black', linewidth=1, alpha=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
# %%
import seaborn as sns

plt.figure(figsize=(20, 5))
sns.kdeplot(
    data=all_cell_df.filter(pc('in_poly')).to_pandas(), x='rx', y='ry', 
    levels=5, thresh=0.019,
    # gridsize=50,
    # bw_method=0.1,
    alpha=0.5,
    fill=True,
)
for (slice_id, ), g in all_cell_df.group_by(['slice_id'], maintain_order=True): # type: ignore
    points = g.filter(pc('in_poly'))[['rx', 'ry']].to_numpy()
    plt.scatter(*points.T, s=1, alpha=0.5)
for sec, p in all_target_regions.items():
    for cnt in final_fix_cnts:
        plt.plot(*cnt.T, color='black', linewidth=0.1, alpha=1)

plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
# %%
# #####################################################################################
# #####################################################################################
# #####################################################################################
# #####################################################################################
# #####################################################################################
# 下面主要来自从 pwvy/20240308-create-region-svg.py 复制


# @memory.cache()
def calc_all_mean_dists(target_cells_df: pl.DataFrame):
    all_dists: dict[AnimalSection, np.ndarray] = {}
    print(target_cells_df)
    for (animal_id, slice_id), df in target_cells_df.sort( # type: ignore
        'slice_id'
    ).group_by(['animal_id', 'slice_id'], maintain_order=True):
        slice_id = cast(str, slice_id)

        points = df[['x', 'y']].to_numpy()
        dist_mtx = pairwise_distances(points, n_jobs=-1)
        dists = np.triu(dist_mtx, k=1).flatten()
        dists = dists[dists > 0]


        mtx = dist_mtx.copy()
        np.fill_diagonal(mtx, np.nan)
        if np.isnan(mtx).all():
            continue
        avg_dists = np.nanmean(mtx, axis=1)
        sec = AnimalSection(animal_id=animal_id, slice_id=f'{slice_id:03d}') # type: ignore
        all_dists[sec] = avg_dists
    return all_dists
# %%
@dataclass
class KDEResultItem:
    zz: np.ndarray
    min_xy: np.ndarray
    max_xy: np.ndarray
    fix_sec: AnimalSection
    cell_count: int

@dataclass
class KDEResults:
    zzs: list[np.ndarray]
    min_xys: list[np.ndarray]
    max_xys: list[np.ndarray]
    fix_secs: list[AnimalSection]
    cell_counts: list[int]
    selected_cells: list[np.ndarray]
    '''使用了哪些细胞参与kde计算，是 index 数组'''

    @staticmethod
    def new() -> 'KDEResults':
        return KDEResults([], [], [], [], [], [])

    def append(
        self, 
        zz: np.ndarray, min_xy: np.ndarray, max_xy: np.ndarray, 
        fix_sec: AnimalSection, cell_count: int, selected_cells: np.ndarray
    ):
        self.zzs.append(zz)
        self.min_xys.append(min_xy)
        self.max_xys.append(max_xy)
        self.fix_secs.append(fix_sec)
        self.cell_counts.append(cell_count)
        self.selected_cells.append(selected_cells)

    def max_xy(self):
        return np.max(self.max_xys, axis=0)

    def min_xy(self):
        return np.min(self.min_xys, axis=0)

    def sort(self):
        idxs = np.argsort([i.slice_id_int for i in self.fix_secs])
        self.zzs = [self.zzs[i] for i in idxs]
        self.min_xys = [self.min_xys[i] for i in idxs]
        self.max_xys = [self.max_xys[i] for i in idxs]
        self.fix_secs = [self.fix_secs[i] for i in idxs]
        self.cell_counts = [self.cell_counts[i] for i in idxs]
        self.selected_cells = [self.selected_cells[i] for i in idxs]
        return self

    def __len__(self):
        return len(self.zzs)

    def __getitem__(self, idx):
        return KDEResultItem(
            self.zzs[idx], self.min_xys[idx], self.max_xys[idx], 
            self.fix_secs[idx], self.cell_counts[idx]
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f'KDEResults({len(self)}, fix:{self.fix_secs[0]}-{self.fix_secs[-1]})'
# %%
@memory.cache()
def calc_cell_kde_by_slice(
    target_cells_df: pl.DataFrame, bandwidth=0.05, pv=95, 
):

    mean_dists = calc_all_mean_dists(target_cells_df)
    zz_res = KDEResults.new()

    def f(mean_dists, sec: AnimalSection, slice_g):
        curr_avg_dists = mean_dists.get(sec)
        if curr_avg_dists is None: return

        p = np.percentile(curr_avg_dists, pv)
        less_p_idx = curr_avg_dists < p

        cell_points = slice_g[['x', 'y']].sort('x', 'y').to_numpy()
        less_p_cell_points = cell_points[less_p_idx]
        target_cell_index = slice_g['index'].to_numpy()[less_p_idx]
        if len(less_p_cell_points) < 3: return
        # print(sec, less_p_cell_points.shape, less_p_cell_points)

        poly = all_target_regions[sec]
        cnt_min_xy = np.array(poly.bounds[:2])
        cnt_max_xy = np.array(poly.bounds[2:])

        zz, min_xy, max_xy = points_kde(
            less_p_cell_points, cnt_min_xy, cnt_max_xy,
            bandwidth=bandwidth,
        )
        return zz, min_xy, max_xy, sec, less_p_cell_points.shape[0], target_cell_index

    tasks = [
        delayed(f)(
            mean_dists, 
            AnimalSection(k[0], f'{k[1]:03d}'), # type: ignore
            slice_g
        )
        for k, slice_g in target_cells_df.group_by(
            ['animal_id', 'slice_id'], maintain_order=True
        )
    ]
    for res in tqdm(Parallel(24, return_as='generator_unordered')(tasks), total=len(tasks)):
        if res is None: continue
        zz_res.append(*res)
    
    return zz_res

kde_res = calc_cell_kde_by_slice(
    all_cell_df,
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
export_root = Path('/data/sdf/to_wangML/injection-zone/20240426-C077-kdezone/')
export_root.mkdir(exist_ok=True, parents=True)



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


pv = 99

bandwidth = 0.05



target_cells_df = all_cell_df.filter('in_poly')
target_cells_df = (
    target_cells_df.with_columns(
        (pc('rx') // 2 * 2).cast(pl.Int64).alias('rx_trunc'),
        (pc('ry') // 2 * 2).cast(pl.Int64).alias('ry_trunc'),
    ).unique(['rx_trunc', 'ry_trunc'])
)
zzs_data = calc_cell_kde_by_slice(target_cells_df, pv=pv, bandwidth=bandwidth).sort()

selected_cell_ids = np.concatenate(zzs_data.selected_cells)
zz_cell_count_mov_mean = np.convolve(zzs_data.cell_counts, np.ones(5) / 5, mode='valid')
zz_cell_count_mov_mean = np.pad(zz_cell_count_mov_mean, (2, 2), 'edge')
zz_cell_count_mov_mean = np.maximum(zz_cell_count_mov_mean, zzs_data.cell_counts)
zz_cell_count_mov_mean = zzs_data.cell_counts


total_max_xy = np.round(zzs_data.max_xy()).astype(int)
total_min_xy = np.round(zzs_data.min_xy()).astype(int)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(zzs_data.cell_counts, label='raw')
ax.plot(np.array(zz_cell_count_mov_mean), label='mov mean')

zz_cnt_res: dict[float, DrawKDEArgs] = {} # {thr: [zz_cnts]}

for thr in (np.linspace(-0.01, 0.4, 21)):
    # thr = np.exp(-thr)
    zz_cnt_res[thr] = DrawKDEArgs([], 0, 0, pv, 0, [])

    for i in range(len(zzs_data.zzs)):
        sec = zzs_data.fix_secs[i]
        zz_cnt_res[thr].titles.append(f'{i}')
        fix_cnt_p = all_target_regions[sec]

        raw_zz = zzs_data.zzs[i]
        zz = raw_zz / (fix_cnt_p.area / zz_cell_count_mov_mean[i])**0.5
        min_xy = zzs_data.min_xys[i]

        zz_cnts, _ = cv2.findContours(
            (zz > thr).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        for zz_cnt in zz_cnts:
            if len(zz_cnt) < 3: 
                print(f'[{thr:.3f}] 1 continue')
                continue
            p = Polygon(zz_cnt.reshape(-1, 2) + min_xy)
            try:
                p = fix_cnt_p.intersection(p)
            except Exception as e:
                print(f'[{thr:.3f}] {e} continue')
                continue

            corrds = exectract_polygons(p)
            for _zz_cnt in corrds:
                if len(_zz_cnt) < 3: 
                    print(f'[{thr:.3f}] 2 continue')
                    continue


                zz_cnt_res[thr].zz_cnt_res.append(_zz_cnt)
        

    region_mask = np.zeros((
        int(target_cells_df['y'].max() - target_cells_df['y'].min()) + 2, # type: ignore
        int(target_cells_df['x'].max() - target_cells_df['x'].min()) + 2, # type: ignore
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
    
    # in_polys = region_mask[Y, X] != 0

    # zz_cnt_res[thr].inner_cell_n = sum(in_polys)
    # zz_cnt_res[thr].total_cell_n = len(target_cells_df.filter('in_poly'))
    # zz_cnt_res[thr].selected_cell_n = len(selected_cell_ids)
    zz_cnt_res[thr].inner_cell_n = 1
    zz_cnt_res[thr].total_cell_n = 1
    zz_cnt_res[thr].selected_cell_n = 1
    print(thr, len(zz_cnt_res[thr].zz_cnt_res))

kde_pdf_p = export_root / f'{mov_animal_id}-{pv}-all-thr-bw{bandwidth:.4f}2.pdf'
shutil.rmtree(kde_pdf_p, ignore_errors=True)

final_fix_cnts = [
    [
        p
        for p in exectract_polygons(all_target_regions[i])
    ]
    for i in zzs_data.fix_secs
]
final_fix_cnts = sum(final_fix_cnts, [])
draw_kde_cnts(
    kde_pdf_p, 
    final_fix_cnts, 
    zz_cnt_res
)

# %%

plt.hist(np.concatenate([i.reshape(-1) for i in zzs_data.zzs], 0))