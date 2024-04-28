# %%
import itertools
import pickle
from collections import defaultdict
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Literal
import yaml
from datetime import datetime
from shapely.affinity import translate
from scipy.optimize import minimize_scalar
from dataclasses import dataclass

import cairo
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from bidict import bidict
from dataset_utils import get_base_path, read_regist_datasets
from joblib import Memory, Parallel, delayed
from loguru import logger
from scipy.interpolate import splev, splprep
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from utils import (
    AnimalSection,
    AnimalSectionWithDistance,
    PariRegistInfo,
    read_connectome_masks,
    read_exclude_sections,
    split_into_n_parts_slice,
    to_numpy,
    read_stereo_masks,
    pad_or_crop
)
from connectome_utils import find_connectome_ntp, ntp_p_to_slice_id, find_stereo_ntp
import matplotlib.colors as colors
from nhp_utils import CorrectionPara, NHPPathHelper
import duckdb
from joblib import Parallel, delayed
import orjson

from shapely import Polygon
pc = pl.col

def points_kde(
    points: np.ndarray, image_size: int, mesh_size: int=0, bandwidth=0.025, 
    zz_factor=lambda x: x, atol=0.5
) -> np.ndarray:
    ps = points / image_size # normalize to [0, 1]
    if mesh_size == 0: mesh_size = image_size // 4

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, atol=atol).fit(ps)
    xx, yy = np.meshgrid(np.linspace(0, 1, mesh_size),
                        np.linspace(0, 1, mesh_size))
    zz = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zz = zz_factor(zz)
    zz = cv2.resize(zz, (image_size, image_size))

    return zz
def mask_image_by_contours(image: np.ndarray, contours: list[np.ndarray], v=1.0) -> np.ndarray:
    mask = np.zeros_like(image, dtype='uint16')
    mask = cv2.fillPoly(
        mask, [i.astype(int) for i in contours], 
        (v, )
    )
    ret = (image * mask)
    return ret

def mask_points_by_contours(points: np.ndarray, *contours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(len(points), dtype=bool)
    for cnt in contours:
        for i, point in enumerate(points):
            mask[i] = cv2.pointPolygonTest(cnt, point, False) >= 0
    return points[mask], mask

def draw_contours(
    ctx: cairo.Context, cnt: np.ndarray, 
    dx: float=0.0, dy: float=0.0,
    clip=True, stroke=True,
):
    def d():
        ctx.move_to(cnt[0][0] - dx, cnt[0][1] - dy)
        for x, y in cnt[1:]:
            ctx.line_to(x - dx, y - dy)
    if clip:
        ctx.new_path()
        d()
        ctx.clip()
    if stroke:
        ctx.new_path()
        d()
        ctx.stroke()

def image_lut(
    base_image: np.ndarray, 
    local_color_map: np.ndarray, 
    background_color: tuple[float,float,float,float]=(.0,.0,.0,.0),
):
    # local_color_map[0] = background_color

    img_r = cv2.LUT(base_image, local_color_map[:, 0])
    img_g = cv2.LUT(base_image, local_color_map[:, 1])
    img_b = cv2.LUT(base_image, local_color_map[:, 2])
    img_a = cv2.LUT(base_image, local_color_map[:, 3])
    base_image = cv2.merge([img_b, img_g, img_r, img_a]) # BGRA
    return base_image

PROJECT_NAME = 'CLA'
assets_path = Path('/mnt/97-macaque/projects/cla/injections-cells/assets/raw_image/')
exclude_slices = read_exclude_sections(PROJECT_NAME)
memory = Memory('./cache', verbose=0)

# %%
regist_results = pickle.load(open(assets_path.parent / 'regist_results.pkl', 'rb'))
# regist_results = pickle.load(open(assets_path.parent / 'regist_results_Mq179.pkl', 'rb'))
regist_results = pickle.load(open(assets_path.parent / 'regist_results_20231219.pkl', 'rb'))

regist_results = [i for i in regist_results if isinstance(i, PariRegistInfo)]
all_mov_animals = sorted(set([i.mov_sec.animal_id for i in regist_results]))
all_mov_animals
# %%
dye_to_color = bidict({
    'FB': 'blue',
    'CTB555': 'yellow',
    'CTB647': 'red',
    'CTB488': 'green',
})
def find_cell(regist_datasets: dict[AnimalSection, dict], sec: AnimalSectionWithDistance) -> pl.DataFrame:
    side = sec.side

    return regist_datasets[AnimalSection(
        animal_id=sec.animal_id,
        slice_id=sec.slice_id,
    )]['pad_res'][side][1]

# %%

combine_file_p = '/mnt/97-macaque/projects/cla/injections-cells/Combine-20231220-mergecla.xlsx'
combine_file_p = '/mnt/97-macaque/projects/cla/injections-cells/Combine-20240116.xlsx'

input_df = pd.read_excel(combine_file_p).rename(columns={
    'Combine3': 'combine',
    'Combine_area': 'combine_area', 
    'Animal': 'animal_id',
    'injectionSites': 'region',
    'hemisphere': 'side',
    'Dye': 'tracer',
}).sort_values(['combine', 'animal_id', 'region'], ignore_index=True)


input_df['color'] = input_df['tracer'].map(dye_to_color)
input_df['side'] = input_df['side'].map({'L': 'left', 'R': 'right', 'left': 'left', 'right': 'right'})
input_df['animal_id_side'] = input_df['animal_id'] + '-' + input_df['side']
# input_df['combine'] = 'all-combine'
# target_info
print('groups', list(i[0] for i in input_df.groupby('combine')))
input_df

# %%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%
# %%
# 绘制连接组的 Begin

@memory.cache(ignore=['datasets'])
def get_mask_cnts(
    animal_id: str, slice_id: str, side: str, 
    datasets: dict[AnimalSection, dict] = {}
):
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
    return raw_cnt, smooth_cnt

# def calc_mask_contours(mask: np.ndarray, points_num=100) -> tuple[np.ndarray, np.ndarray]:
#     mask = to_numpy(mask)
#     cnts, _ = cv2.findContours(
#         mask, 
#         cv2.RETR_EXTERNAL, 
#         cv2.CHAIN_APPROX_SIMPLE
#     )
#     cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
#     cnt = cnts[0].reshape(-1, 2)

#     tck, u = splprep(cnt.T, per=1)
#     u_new = np.linspace(u.min(), u.max(), points_num)
#     x_new, y_new = splev(u_new, tck, der=0)
#     smooth_cnt = np.stack([x_new, y_new], axis=1).astype(int)

#     return cnt, smooth_cnt
per_image = 8
delta_image_distance = 0


p配准时缩放 = 0.1
v展示时缩放 = 0.5
j计算缩放 = int(v展示时缩放 / p配准时缩放)
per_image = 1
delta_image_distance = 110 * j计算缩放
image_size = 128 * j计算缩放
# %%

dfs = []

for p in tqdm(assets_path.parent.parent.glob('saved-*.pkl')):
    total_records = pickle.load(open(p, 'rb'))
    dxs = total_records['dxs']
    records = total_records['record_cells']
    df = pl.DataFrame({
        **records,
        **dxs,
    })
    dfs.append(df)
    break
df: pl.DataFrame = pl.concat(dfs)
# %%
df.columns
# %%
animal_id = 'C006'

target_rows = df.filter(
    (pc('animal_id') == animal_id)
)
rows = target_rows.to_dicts()
target_rows.select(['animal_id', 'slice_id', 'side'])
# %%

fix_datasets = read_regist_datasets(
    'C042', v展示时缩放, exclude_slices=exclude_slices,
    target_shape=(128*j计算缩放, 128*j计算缩放)
)
# %%
# 导出所有连接组配准细胞
import orjson

regist_results = pickle.load(open(assets_path.parent / 'regist_results.pkl', 'rb'))
regist_to_mq179_results = pickle.load(open(assets_path.parent / 'regist_results_20240109_C042_to_Mq179.pkl', 'rb'))

regist_results = [i for i in regist_results if isinstance(i, PariRegistInfo)]
regist_to_mq179_results = [i for i in regist_to_mq179_results if isinstance(i, PariRegistInfo)]

export_base = Path('/mnt/97-macaque/projects/cla/injections-cells/20240116-data-share/')
export_base.mkdir(exist_ok=True, parents=True)

fix_cnts = {}

for mov_animal_id, g in itertools.groupby(regist_results, lambda x: x.mov_sec.animal_id):
    print(mov_animal_id)
    mov_regist_datasets = read_regist_datasets(
        mov_animal_id, v展示时缩放, exclude_slices=exclude_slices, 
        target_shape=(128*j计算缩放, 128*j计算缩放)
    )
    curr_animal_cells = []

    for pair in g:
        to_mq179_pair = [
            i for i in regist_to_mq179_results
            if (
                i.mov_sec.animal_id == pair.fix_sec.animal_id
                and i.mov_sec.slice_id == pair.fix_sec.slice_id
                and i.mov_sec.side == pair.fix_sec.side
            )
        ]
        if not to_mq179_pair:
            print(f'missing: {pair.mov_sec}')
            continue

        best_to_mq179_pair = max(to_mq179_pair, key=lambda x: x.iou)

        mov_side = pair.mov_sec.side

        mov_mask, mov_cells_df = mov_regist_datasets[AnimalSection(
            animal_id = pair.mov_sec.animal_id,
            slice_id  = pair.mov_sec.slice_id,
        )]['pad_res'][mov_side]
        raw_cnt, smooth_cnt = get_mask_cnts(
            pair.fix_sec.animal_id, 
            pair.fix_sec.slice_id, 
            pair.fix_sec.side,
            datasets=fix_datasets
        )
        fix_cnts[f'{pair.fix_sec.animal_id}-{pair.fix_sec.slice_id}-{pair.fix_sec.side}'] = ({
            'animal_id': pair.fix_sec.animal_id,
            'slice_id': pair.fix_sec.slice_id,
            'side': pair.fix_sec.side,
            'raw_cnt': raw_cnt.tolist(),
            'smooth_cnt': smooth_cnt.tolist(),
        })
        fix_cnts[f'{best_to_mq179_pair.fix_sec.animal_id}-{best_to_mq179_pair.fix_sec.slice_id}-{best_to_mq179_pair.fix_sec.side}'] = ({
            'animal_id': best_to_mq179_pair.fix_sec.animal_id,
            'slice_id': best_to_mq179_pair.fix_sec.slice_id,
            'side': best_to_mq179_pair.fix_sec.side,
            'raw_cnt': raw_cnt.tolist(),
            'smooth_cnt': smooth_cnt.tolist(),
        })


        mov_cells = mov_cells_df.select('x', 'y').to_numpy()
        if pair.mov_sec.need_flip:
            mov_cells[:, 0] = image_size - mov_cells[:, 0]
        mov_cells = pair.transform_points(mov_cells / j计算缩放) * j计算缩放
        mov_cells_to_mq179 = best_to_mq179_pair.transform_points(mov_cells / j计算缩放) * j计算缩放

        # mov_cells_filtered, filtered_mask = mask_points_by_contours(mov_cells, raw_cnt)
        curr_animal_cells.append({
            'meta': pair,
            'meta_to_mq179': best_to_mq179_pair,
            'mov_side': mov_side, 
            'mov_cells': np.ascontiguousarray(mov_cells),
            'cell_color': mov_cells_df['color'].to_list(),
            # 'in_contour': filtered_mask,
            'mov_cells_to_mq179': np.ascontiguousarray(mov_cells_to_mq179),
        })

    with open(export_base / f'{mov_animal_id}.json', 'wb') as fp:
        fp.write(orjson.dumps(
            curr_animal_cells, 
            option=orjson.OPT_INDENT_2|orjson.OPT_SERIALIZE_NUMPY
        ))
# %%
regist_results = pickle.load(open(assets_path.parent / 'regist_results_Mq179.pkl', 'rb'))

# %%
fix_cnts = dict(sorted(fix_cnts.items(), key=lambda x: x[0]))

(export_base / 'fix_cnts.json').write_bytes(orjson.dumps(
    fix_cnts, 
    option=orjson.OPT_INDENT_2|orjson.OPT_SERIALIZE_NUMPY
))

# %%
stereo_scale = 0.015
stereo_image_size = 640
ntp_version = 'Mq179-CLA-20230505'

masks = read_stereo_masks(
    [chip], ntp_version, scale=stereo_scale, 
    target_shape=(stereo_image_size, stereo_image_size),
    buffer_x=0, buffer_y=0
)

# %%
item = curr_animal_cells[53]
mov_cells = item['mov_cells']
mov_cells_to_mq179 = item['mov_cells_to_mq179']

print(mov_cells.shape)
pair = item['meta']

raw_cnt, smooth_cnt = get_mask_cnts(
    pair.fix_sec.animal_id, 
    pair.fix_sec.slice_id, 
    pair.fix_sec.side,
    datasets=fix_datasets
)

plt.plot(*raw_cnt.T)
plt.plot(*smooth_cnt.T)
plt.scatter(*mov_cells.T, s=1, c='red')
plt.scatter(*mov_cells_to_mq179.T, s=1, c='blue')


# plt.scatter(*mov_cells_filtered.T, s=1, c='blue')

# %%

original_cmap = mpl.colormaps['RdYlBu']
colors_array = original_cmap(np.linspace(0, 1, original_cmap.N))[::-1, ...]

colors_array[0] = [0, 0, 0, 0]
new_cmap = colors.ListedColormap(colors_array)

combine_to_draws: dict[str, defaultdict[str, list]] = {}

groupby_col = 'Combine3'

for combine_name, meta_group in input_df.groupby(groupby_col):
    if not isinstance(combine_name, str): continue
    # if not combine_name.startswith('CLA'): continue
    if 'Claustrum' not in combine_name: continue
    print(f'{combine_name=}')

    target_info = meta_group.to_dict(orient='records')

    current_combine_result = defaultdict(list)

    for combine_item in target_info:
        print(combine_item)
        mov_animal_id   = combine_item['animal_id']
        mov_side        = combine_item['side']
        mov_regist_datasets = read_regist_datasets(
            mov_animal_id, v展示时缩放, exclude_slices=exclude_slices, 
            target_shape=(128*j计算缩放, 128*j计算缩放)
        )
        pair_regist_infos = [
            i for i in regist_results
            if (
                i.mov_sec.animal_id == mov_animal_id
                and i.mov_sec.side == mov_side
            )
        ]
        for pair in tqdm(pair_regist_infos):
            # print(pair)
            mov_mask, mov_cells_df = mov_regist_datasets[AnimalSection(
                animal_id = pair.mov_sec.animal_id,
                slice_id  = pair.mov_sec.slice_id,
            )]['pad_res'][mov_side]
            mov_cells_df = mov_cells_df.filter(
                pc('color') == combine_item['color']
            )
            if not len(mov_cells_df): continue

            raw_cnt, smooth_cnt = get_mask_cnts(
                pair.fix_sec.animal_id, 
                pair.fix_sec.slice_id, 
                pair.fix_sec.side,
                datasets=fix_datasets
            )

            mov_cells = mov_cells_df.select('x', 'y').to_numpy()
            if pair.mov_sec.need_flip:
                mov_cells[:, 0] = image_size - mov_cells[:, 0]
            mov_cells = pair.transform_points(mov_cells / j计算缩放) * j计算缩放
            mov_cells_filtered, _ = mask_points_by_contours(mov_cells, smooth_cnt)
            # kde_img   = points_kde(mov_cells_filtered, image_size, mesh_size=image_size // 4, bandwidth=0.03)
            # kde_img   = mask_image_by_contours(kde_img, [smooth_cnt], v=1.0)

            current_combine_result[pair.fix_sec.slice_id].append({
                'smooth_cnt'        : smooth_cnt,
                'raw_cnt'           : raw_cnt,
                'mov_cells'         : mov_cells,
                'mov_cells_filtered': mov_cells_filtered,
                'sec'               : pair.mov_sec,
                'fix_slice_id'      : pair.fix_sec.slice_id,
            })
            # plt.imshow(kde_img, cmap=new_cmap)
            # plt.plot(*smooth_cnt.T, linewidth=1, c='black')
            # plt.colorbar()
            # plt.scatter(*mov_cells.T, s=1, c='red')
            # break
        # break
    # break
    combine_to_draws[combine_name] = current_combine_result
    # break
# %%
combine_to_draws_with_kde = {}
# 合并配准到一张目标的 cells 并计算 mean_cell_number和max_cells
output_root = Path('/mnt/97-macaque/projects/cla/injections-cells/20231220-kde-cla-merge/')
output_root.mkdir(exist_ok=True, parents=True)

filtered_cells = []


for current_combine in combine_to_draws:
    to_draws = combine_to_draws[current_combine]
    merged_to_draws = []

    max_cells = 0

    for fix_slice_id, mov_items in tqdm(
        sorted(to_draws.items(), key=lambda x: int(x[0])), 
        desc=current_combine, total=len(to_draws)
    ):
        first_mov_item = mov_items[0]
        mov_cells = np.vstack([i['mov_cells'] for i in mov_items])

        all_mov_cells_filtered = [i['mov_cells_filtered'] for i in mov_items]
        for mov_item, cells in zip(mov_items, all_mov_cells_filtered):
            # filtered_cells[current_combine][
            #     (mov_item['sec'].animal_id, mov_item['sec'].side, fix_slice_id)
            # ].append(len(cells))
            filtered_cells.append({
                'combine': current_combine,
                'animal_id': mov_item['sec'].animal_id,
                'animal_id_side': f'{mov_item["sec"].animal_id}-{mov_item["sec"].side}',
                'side': mov_item['sec'].side,
                'fix_slice_id': fix_slice_id,
                'celln': len(cells),
            })

        mov_cells_filtered = np.vstack(all_mov_cells_filtered)
        mean_cell_number = np.median([len(i) for i in all_mov_cells_filtered])

        max_cells = max(max_cells, mean_cell_number)

        merged = {
            'smooth_cnt'        : first_mov_item['smooth_cnt'],
            'row_cnt'           : first_mov_item['smooth_cnt'],
            'mov_cells'         : mov_cells,
            'mov_cells_filtered': mov_cells_filtered,
            'fix_slice_id'      : fix_slice_id,
            'mean_cell_number'  : mean_cell_number
        }

        print(fix_slice_id, len(mov_items), merged['mean_cell_number'])
        merged_to_draws.append(merged)
    for i in merged_to_draws:
        i['max_cells'] = max_cells

    combine_to_draws_with_kde[current_combine] = merged_to_draws
filtered_cells = pl.DataFrame(filtered_cells)
filtered_cells
# for current_combine in filtered_cells:
#     for (animal_id, side, fix_slice_id), cells in filtered_cells[current_combine].items():
#         plt.plot(cells)
# %%
for current_combine, combine_group in filtered_cells.groupby('combine'):
    plt.figure(figsize=(10, 5))
    plt.title(str(current_combine))
    for animal_id_side, celln_group in combine_group.groupby('animal_id_side'):
        celln_group = celln_group.sort('fix_slice_id')
        if animal_id_side != 'C080-right': continue
        plt.plot(
            list(map(int, celln_group['fix_slice_id'])), 
            celln_group['celln'], 
            label=animal_id_side
        )
    
    agged = combine_group.group_by('fix_slice_id').agg(
        pc('celln').mean()
    ).sort('fix_slice_id')
    plt.plot(
        list(map(int, agged['fix_slice_id'])),
        agged['celln'],
        label='mean'
    )

    plt.legend()

# %%

# %%
# 计算 KDE

for current_combine, merged_to_draws in combine_to_draws_with_kde.items():
    # if current_combine != 'posterior': continue

    for merged in tqdm(merged_to_draws, desc=current_combine):
        mov_cells_filtered = merged['mov_cells_filtered']
        if len(mov_cells_filtered) > 1:
            # factor = np.sqrt(merged['mean_cell_number'] / merged['max_cells'])
            merged['kde_img'] = points_kde(
                merged['mov_cells_filtered'], image_size, 
                mesh_size=image_size // 4, bandwidth=0.03,
                # zz_factor=lambda x: x * factor
            )
            # print(factor, merged['kde_img'].max())

        else:
            merged['kde_img'] = np.zeros((image_size, image_size))
# %%


original_cmap = mpl.colormaps['RdYlBu']
# original_cmap = mpl.colormaps['magma']

colors_array = original_cmap(np.linspace(0, 1, original_cmap.N))[::-1, ...]
colors_array = (colors_array * 255).astype('uint8')

for current_combine in combine_to_draws_with_kde:
    # if current_combine != 'posterior': continue

    pading_px = 10
    total_image_w = 0
    total_image_h = 0
    all_to_draws = []

    merged_to_draws = combine_to_draws_with_kde[current_combine]
    merged_to_draws.sort(key=lambda x: x['fix_slice_id'])
    # for to_draw_index in split_into_n_parts_slice(merged_to_draws, 42):
    for to_draw_index in range(len(merged_to_draws)):
        d = merged_to_draws[to_draw_index]

        x, y, cnt_w, cnt_h = cv2.boundingRect(d['smooth_cnt'].astype(int))
        d['cnt_box'] = (x, y, cnt_w, cnt_h)
        d['current_dx'] = total_image_w
        d['current_dy'] = pading_px

        total_image_w += cnt_w + pading_px
        total_image_h = max(total_image_h, cnt_h)
        all_to_draws.append(d)
    total_image_h += pading_px * 2
    print(f'{total_image_w=}, {total_image_h=}')



    with cairo.PDFSurface((output_root / f'{current_combine}.pdf').as_posix(), total_image_w, total_image_h) as surface:
    # with cairo.ImageSurface(cairo.FORMAT_ARGB32, total_image_w, total_image_h) as surface:
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1, 1, 1)
        ctx.rectangle(0, 0, total_image_w, total_image_h)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(0.1)
        ctx.translate(pading_px, pading_px)

        factors = np.array([
            np.sqrt(i['mean_cell_number'] / i['max_cells'])
            for i in all_to_draws
        ])
        # print(factors.shape)
        smooth_n = 2
        # factors = np.convolve(factors, np.ones(smooth_n) / smooth_n, mode='same')
        print(factors.shape)

        for to_draw, factor in zip(all_to_draws, factors):
            x, y, cnt_w, cnt_h = to_draw['cnt_box']
            dx, dy = to_draw['current_dx'], to_draw['current_dy']
            # print(dx, dy, x, y, cnt_w, cnt_h, sep='\t')
            ctx.set_source_rgb(0, 0, 0)
            draw_contours(ctx, to_draw['smooth_cnt'], x, y)
            # ctx.clip()

            kde_img = to_draw['kde_img']
            # kde_img = ((kde_img / kde_img.max()) * 200).astype('uint8')
            # factor = np.sqrt(to_draw['mean_cell_number'] / to_draw['max_cells'])

            kde_img = ((kde_img / kde_img.max()) * factor * 255).astype('uint8')

            kde_img = image_lut(kde_img, colors_array,)
            cv2.imwrite(f'/data/sdf/kde_img_{to_draw["fix_slice_id"]}.png', kde_img)

            # kde_img = kde_img[..., [2, 1, 0, 3]]
            kde_img = np.ascontiguousarray(kde_img)
            # print(kde_img.dtype, kde_img.max())

            img_surface = cairo.ImageSurface.create_for_data(
                (kde_img).data, cairo.FORMAT_ARGB32, kde_img.shape[1], kde_img.shape[0]
            )
            ctx.set_source_surface(img_surface, -x, -y)
            ctx.paint()

            # ctx.set_source_rgba(1, 0, 0, 0.5)
            # for cx, cy in to_draw['mov_cells']:
            #     ctx.arc(cx - (x), cy - (y), 1, 0, 2 * np.pi)
            #     ctx.fill()

            ctx.reset_clip()


            ctx.translate(cnt_w, 0)
        surface.write_to_png((output_root / f'{current_combine}.png').as_posix())
        # img = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape((total_image_h, total_image_w, 4)).copy()[..., [2, 1, 0, 3]]

    img = cv2.imread((output_root / f'{current_combine}.png').as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30, 10))
    plt.imshow(img)
# %%
# 绘制连接组 End

# %%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntp_version = 'Mq179-CLA-20230505'
ntp_base_path = Path('/data/sde/ntp/macaque')
chips_dict = {
    'Mq179': 'T105 T103 T101 T99 T97 T95 T93 T91 T89 T87 T85 T83 T81 T79 T77 T75 T73 T71 T67 T65 T63 T61 T59 T57 T55 T53 T51 T49 T47 T45 T43 T41 T39 T37 T33 T31 T29 T27 T25 T26 T28 T30 T32 T34'.split(),
    'Mq179-half': 'T101 T105 T25 T28 T29 T32 T33 T39 T43 T47 T49 T55 T59 T63 T67 T73 T77 T81 T85 T89 T93 T97 '.split(),
}




# 读入转录组细胞
def read_cell_type_result(
    chip: str, 
    cell_type_root = Path('/mnt/97-macaque/projects/cell-type-results/SpatialID_20231225_55c_mq1&mq4/'),
    cell_type_version = 'macaque1_CLA_55c_csv'                   
) -> pl.DataFrame:
    cell_type_p = next((cell_type_root / cell_type_version).glob(f'*{chip}*'))
    res_p = cell_type_p.with_name(f'{chip}_with_total_gene2d.parquet')
    if res_p.exists():
        res = pl.read_parquet(res_p)
    else:
        res = duckdb.sql(f'''
            select * from "/data/sde/total_gene_2D/macaque-20231204-all/total_gene_{chip}_macaque_f001_2D_macaque-20231204-all.parquet" as t1 
            join "{cell_type_p}" as t2
            on t1.cell_label = t2.cell
        ''').pl()
        res.write_parquet(res_p)
    return res
# %%

@cache
def get_sec_para_from_db(chip: str):
    return NHPPathHelper.get_sec_para_from_db(None, chip) # type: ignore

@cache
def read_yaml(p: str | Path):
    return yaml.load(open(p, 'r'), Loader=yaml.FullLoader)



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
# %%
chip = 'T67'
cell_type_df = read_cell_type_result(chip)
cell_type_df
# %%
stereo_scale = 0.015
stereo_image_size = 640

masks = read_stereo_masks(
    [chip], ntp_version, scale=stereo_scale, 
    target_shape=(stereo_image_size, stereo_image_size),
    buffer_x=0, buffer_y=0
)
mask = masks[0]
polygon: Polygon = mask['polygon']
minx, miny, maxx, maxy = polygon.bounds

plt.imshow(mask['image'])
# %%
corr = get_corr(chip, source='db')
draw_df = cell_type_df.sample(5000)
points = corr.wrap_point(draw_df[['x', 'y']].to_numpy()) * stereo_scale
points -= np.array([minx, miny])

res = pad_or_crop(
    pl.DataFrame(points, schema=['x', 'y']), 
    dst_w=stereo_image_size, dst_h=stereo_image_size,
    src_w=polygon.bounds[2] - polygon.bounds[0], src_h=polygon.bounds[3] - polygon.bounds[1],
).to_numpy()

plt.imshow(mask['image'])


plt.scatter(*res.T, s=1)
# %%
# 导出所有 Mq179 的配准细胞
p配准时缩放 = 1/315
v展示时缩放 = 5/315
j计算缩放 = int(v展示时缩放 / p配准时缩放)


export_base = Path('/mnt/97-macaque/projects/cla/injections-cells/20231227-data-share/')
export_base.mkdir(exist_ok=True, parents=True)

regist_results = pickle.load(open(assets_path.parent / 'regist_results_20231228_Mq179.pkl', 'rb'))
regist_results = [i for i in regist_results if isinstance(i, PariRegistInfo)]
regist_results
# %%
for mov_animal_id, g in itertools.groupby(regist_results, lambda x: x.mov_sec.animal_id):
    print(mov_animal_id)

    stereo_masks = read_stereo_masks(
        chips_dict[mov_animal_id], ntp_version='Mq179-CLA-20230505', 
        scale=v展示时缩放, min_sum=140
    )
    mov_regist_datasets = {
        AnimalSection(
            animal_id=mov_animal_id, 
            slice_id=item['chip'], 
        ): item for item in stereo_masks
    }

    curr_animal_cells = []

    for pair in tqdm(g):
        mov_side = pair.mov_sec.side
        mask_item = mov_regist_datasets.get(AnimalSection(
            animal_id = pair.mov_sec.animal_id,
            slice_id  = f'T{pair.mov_sec.slice_id}',
        ))
        if mask_item is None: 
            print(f'missing: {pair.mov_sec}')
            continue

        polygon: Polygon = mask_item['polygon']
        pminx, pminy, pmaxx, pmaxy = polygon.bounds
        chip = mask_item['chip']
        corr = get_corr(chip, source='db')

        raw_cnt, smooth_cnt = get_mask_cnts(
            pair.fix_sec.animal_id, 
            pair.fix_sec.slice_id, 
            pair.fix_sec.side,
            datasets=fix_datasets
        )
        
        cell_type_df = read_cell_type_result(chip)

        mov_cells = cell_type_df.select('x', 'y').to_numpy()
        mov_cells = corr.wrap_point(mov_cells) * v展示时缩放
        mov_cells -= np.array([pminx, pminy])
        print(mov_cells.shape)
        cell_type_df = cell_type_df.with_columns(
            pl.lit(mov_cells[:, 0]).alias('x'),
            pl.lit(mov_cells[:, 1]).alias('y'),
        )

        cell_type_df = pad_or_crop(
            cell_type_df, 
            dst_w=stereo_image_size, dst_h=stereo_image_size,
            src_w=pmaxx - pminx, src_h=pmaxy - pminy,
        )
        mov_cells = cell_type_df.select('x', 'y').to_numpy()
        
        mov_cells = pair.transform_points(mov_cells / j计算缩放) * j计算缩放
        mov_cells_filtered, filtered_mask = mask_points_by_contours(mov_cells, raw_cnt)
        print(mov_cells.shape)

        curr_animal_cells.append({
            'meta': pair,
            'mov_side': mov_side, 
            'mov_cells': np.ascontiguousarray(mov_cells),
            'cell_color': cell_type_df['celltype_pred'].to_list(),
            'in_contour': filtered_mask,
        })

    with open(export_base / f'{mov_animal_id}.json', 'wb') as fp:
        fp.write(orjson.dumps(
            curr_animal_cells, 
            option=orjson.OPT_INDENT_2|orjson.OPT_SERIALIZE_NUMPY
        ))


# %%
# for chip in chips_dict['Mq179']:
#     res = read_cell_type_result(chip)
# %%
# 读入转录组 mask
stereo_scale = 0.015
stereo_image_size = 640

stereo_masks = read_stereo_masks(chips_dict['Mq179'], ntp_version, scale=stereo_scale, target_shape=(stereo_image_size, stereo_image_size))
stereo_masks_dic = {
    i['chip']: i for i in stereo_masks
}

# %%

# 读取分区、细胞，生成核密度估计任务

curr_cell_types = []

for chip in chips_dict['Mq179']:
    # if chip not in 'T67T65T63T61T69T71T33T44T57': continue

    print(chip)
    sec_para = get_sec_para_from_db(chip)
    polygon: Polygon = stereo_masks_dic[chip]['polygon_without_warped'] # 这里已经是缩放过的

    cell_type_df = read_cell_type_result(chip)

    # 多边形先乘到目标大小, 然后上offset以变换
    corr = get_corr(chip, source='db')
    # corr = get_corr(chip, source='/data/sdf/corr/macaque1-cla-flatten.yaml')

    polygon_xy = np.array(polygon.exterior.coords) / stereo_scale + np.array([sec_para['offset_x'], sec_para['offset_y']])

    polygon_xy = corr.wrap_point(polygon_xy) * stereo_scale # 转换后再回到缩放过的状态
    cells = corr.wrap_point(cell_type_df[['x', 'y']].to_numpy()) * stereo_scale # 转换后再回到缩放过的状态

    polygon = Polygon(polygon_xy)
    minx, miny, maxx, maxy = polygon.bounds
    pol_w, pol_h = maxx - minx, maxy - miny

    cell_type_df = cell_type_df.with_columns(
        pl.lit(cells[:, 0] - minx).alias('x'),
        pl.lit(cells[:, 1] - miny).alias('y'),
    )
    cell_type_df = pad_or_crop(
        cell_type_df, stereo_image_size, stereo_image_size, 
        src_w=pol_w, src_h=pol_h
    )

    polygon_xy = pl.DataFrame({
        'x': np.array(polygon.exterior.xy[0]) - minx,
        'y': np.array(polygon.exterior.xy[1]) - miny,
    })
    polygon_xy = pad_or_crop(
        polygon_xy, stereo_image_size, stereo_image_size, 
        src_w=pol_w, src_h=pol_h
    )

    for curr_cell_type, curr_cell_type_g in tqdm(cell_type_df.group_by('celltype_pred'), desc=f'{chip} 分组'):
        # if curr_cell_type != 'Excit_Neuron_5': continue # 跳过 cell type
        cells = curr_cell_type_g[['x', 'y']].to_numpy()
        
        curr_cell_types.append((curr_cell_type, cells, polygon_xy, chip))
# %%
        
# 根据参数生成任务，计算核密度估计

tasks = []

for curr_cell_type, cells, polygon_xy, chip in curr_cell_types:
    tasks.append(delayed(points_kde)(
        cells, stereo_image_size, 
        mesh_size=stereo_image_size // 4, 
        bandwidth=0.03, # 带宽
        atol=0.5,
    ))

to_draws = defaultdict(list)

for (curr_cell_type, cells, polygon_xy, chip), kde_img in tqdm(zip(
    curr_cell_types, 
    Parallel(n_jobs=32, verbose=0, return_as='generator')(tasks)
), desc=f'计算 kde', total=len(curr_cell_types)):
    to_draws[curr_cell_type].append({
        'raw_cnt': polygon_xy.to_numpy(),
        'mov_cells': cells,
        'mov_cells_filtered': cells, 
        'kde_img': kde_img,
        'fix_slice_id': chips_dict['Mq179'].index(chip),
        'chip': chip, 
        'mean_cell_number': len(cells),
    })
# %%
# if input("要覆盖吗？").lower() == 'y':
#     with open('/mnt/97-macaque/projects/cla/injections-cells/stereo-data2.pkl', 'wb') as f:
#         pickle.dump(to_draws, f)
# %%
def cost_function(shift, poly_a, poly_b):
    shifted_poly_b = translate(poly_b, xoff=shift)
    a = poly_a.intersection(shifted_poly_b).area
    res = a * shift + shift + 1 / (1 if a == 0 else a)

    return res

def calc_min_noninter_shift(poly_a: Polygon, poly_b: Polygon):
    result = minimize_scalar(cost_function, args=(poly_a, poly_b), bounds=(0, 1000))
    return result


@dataclass
class PolygonBound:
    p: Polygon

    def __post_init__(self):
        self.minx, self.miny, self.maxx, self.maxy = self.p.bounds
        self.w, self.h = self.maxx - self.minx, self.maxy - self.miny

    @staticmethod
    def from_np(p: np.ndarray):
        return PolygonBound(Polygon(p))

    @property
    def cnt_box(self):
        return self.minx, self.miny, self.w, self.h

# 绘制出来
original_cmap = mpl.colormaps['RdYlBu'] # colormap 

colors_array = original_cmap(np.linspace(0, 1, original_cmap.N))[::-1, ...]
colors_array = (colors_array * 255).astype('uint8')
output_root = Path('/mnt/97-macaque/projects/cla/injections-cells/20231225-kde-celltype-bd0.3-all/')
output_root.mkdir(exist_ok=True, parents=True)

target_chips = 'T105 T103 T101 T99 T97 T95 T93 T91 T89 T87 T85 T83 T81 T79 T77 T75 T73 T71 T67 T65 T63 T61 T59 T57 T55 T53 T51 T49 T47 T45 T43 T41 T39 T37 T33 T31 T29 T27 T25 T26 T28 T30 T32 T34'.split()

for curr_cell_type in tqdm(to_draws, desc=f'cell type: '):
    pading_px = 40
    total_image_w = 0
    total_image_h = 0

    to_draws[curr_cell_type].sort(key=lambda x: x['fix_slice_id'])
    curr_type_to_draws = [i for i in to_draws[curr_cell_type] if i['chip'] in target_chips]
    all_to_draws = []


    for d_index in range(len(curr_type_to_draws) - 1):
        d1 = curr_type_to_draws[d_index].copy()
        d2 = curr_type_to_draws[d_index + 1].copy()

        # polygon = Polygon(d['raw_cnt'])
        # minx, miny, maxx, maxy = polygon.bounds
        # # x, y, cnt_w, cnt_h = cv2.boundingRect(d['smooth_cnt'].astype(int))
        # x, y = minx, miny
        # cnt_w, cnt_h = maxx - minx, maxy - miny
        cnt1 = PolygonBound.from_np(d1['raw_cnt'])
        cnt2 = PolygonBound.from_np(d2['raw_cnt'])


        d1['cnt_box']    = cnt1.cnt_box
        d1['current_dx'] = total_image_w
        d1['current_dy'] = cnt1.h / 2

        total_image_w += calc_min_noninter_shift(cnt1.p, cnt2.p).x * 1
        d2['cnt_box'] = cnt2.cnt_box
        d2['current_dx'] = total_image_w
        d2['current_dy'] = cnt2.h / 2

        if not all_to_draws:
            all_to_draws.append(d1)
    
        total_image_h = max(total_image_h, cnt1.h, cnt2.h)
        all_to_draws.append(d2)
    total_image_w += stereo_image_size
    total_image_h += pading_px * 4

    # min_dy = min([i['current_dy'] for i in all_to_draws])
    # min_dx = min([i['current_dx'] for i in all_to_draws])
    # for i in all_to_draws:
    #     i['current_dy'] -= min_dy
    #     i['current_dx'] -= min_dx
    
    # total_image_h += pading_px * 2 - min_dy/2

    with cairo.PDFSurface((output_root / f'{curr_cell_type}.pdf').as_posix(), total_image_w, total_image_h) as surface:
    # with cairo.ImageSurface(cairo.FORMAT_ARGB32, total_image_w, total_image_h) as surface:
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle(0, 0, total_image_w, total_image_h)
        ctx.fill()
        ctx.set_line_width(0.1)
        ctx.translate(pading_px, pading_px)

        cell_ns = np.array([i['mean_cell_number'] for i in all_to_draws])
        factors = np.sqrt(cell_ns / max(cell_ns))
        # factors = (cell_ns / max(cell_ns))
        # factors = np.ones_like(factors)

        # smooth_n = 3
        # factors = np.convolve(factors, np.ones(smooth_n) / smooth_n, mode='same')


        for to_draw, factor in zip(all_to_draws, factors):
            x, y, cnt_w, cnt_h = to_draw['cnt_box']
            dx, dy = to_draw['current_dx'], to_draw['current_dy']
            ctx.identity_matrix()
            ctx.translate(dx, 0)

            # print(dx, dy, x, y, cnt_w, cnt_h, sep='\t')
            ctx.set_source_rgb(1, 1, 1)
            ctx.show_text(f'{to_draw["chip"]}')

            ctx.set_source_rgb(1, 0, 0)
            draw_contours(ctx, to_draw['raw_cnt'], 0, 0, clip=True)

            kde_img = to_draw['kde_img']
            kde_img = ((kde_img / kde_img.max()) * factor * 255).astype('uint8')

            kde_img = image_lut(kde_img, colors_array,)
            # cv2.imwrite(f'/data/sdf/kde_img_{to_draw["fix_slice_id"]}_{to_draw["chip"]}.png', kde_img)

            kde_img = np.ascontiguousarray(kde_img)
            img_surface = cairo.ImageSurface.create_for_data(
                (kde_img).data, cairo.FORMAT_ARGB32, kde_img.shape[1], kde_img.shape[0]
            )
            ctx.set_source_surface(img_surface, 0, 0)
            ctx.paint()


            # ctx.set_source_rgba(1, 0, 0, 0.5)
            # for cx, cy in to_draw['mov_cells'][:1000]:
            #     ctx.arc(cx , cy , 1, 0, 2 * np.pi)
            #     ctx.fill()

            ctx.reset_clip()

        surface.write_to_png((output_root / f'{curr_cell_type}.png').as_posix())
    # break
# %%
to_draw = [i for i in all_to_draws if i['chip'] == 'T75'][0]
cells = to_draw['mov_cells']

# plt.imshow(to_draw['kde_img'] * 10)
kde_img = points_kde(
    cells, stereo_image_size, 
    mesh_size=stereo_image_size // 4, 
    bandwidth=0.03
)
kde_img = ((kde_img / kde_img.max()) * 255).astype('uint8')

kde_img_color = image_lut(kde_img, colors_array,)
# %%
plt.imshow(kde_img_color)
plt.scatter(*to_draw['mov_cells'].T, s=0.1, alpha=0.01)
# %%
p1 = Polygon(to_draws[curr_cell_type][2]['raw_cnt'])
p2 = Polygon(to_draws[curr_cell_type][3]['raw_cnt'])
plt.plot(*p1.exterior.xy)
plt.plot(*p2.exterior.xy)
# %%
res = calc_min_noninter_shift(p1, p2)
p2_mov = translate(p2, xoff=res.x)
plt.plot(*p1.exterior.xy)
plt.plot(*p2_mov.exterior.xy)