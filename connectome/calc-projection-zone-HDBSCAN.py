# %%
import os
import pickle
import re
import traceback
from dataclasses import dataclass
from functools import cache
from itertools import groupby
from pathlib import Path
from typing import Literal, cast
import random

if os.path.exists('pwvy'):
    os.chdir((Path('.') / 'pwvy').absolute())

import cairo
import cv2
import dataset_utils
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import utils
from concave_hull import concave_hull
from dataset_utils import (
    get_base_path,
    read_regist_datasets,
)
from matplotlib.axes import Axes
from ntp_manager import SliceMeta
from range_compression import mask_encode
from shapely import Polygon
from shapely.geometry import Polygon
from sklearn.cluster._hdbscan.hdbscan import (
    HDBSCAN,  # from sklearn.cluster import HDBSCAN
)
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from utils import (
    AnimalSection,
    PariRegistInfo,
    pad_or_crop,
    read_connectome_masks,
    read_exclude_sections,
    split_into_n_parts_slice,
)

color_to_tracer = {
    'blue'  : 'FB',
    'yellow': 'CTB555',
    'red'   : 'CTB647',
    'green' : 'CTB488',
}

pc = pl.col


# %%
animal_id = 'C074'
# animal_id = 'C042'

scale = 0.1
base_shape = (128, 128)
PROJECT_NAME = 'CLA'
SRC_IMG_BIN_SIZE = 20

exclude_slices = read_exclude_sections(PROJECT_NAME)

raw_cntm_masks = read_connectome_masks(
    get_base_path(animal_id), animal_id, scale, 
    exclude_slices=exclude_slices, force=False, 
    min_sum=400
)
len(raw_cntm_masks)
fix_regist_datasets = read_regist_datasets(
    animal_id, scale, exclude_slices=exclude_slices, force=False, 
    tqdm=tqdm, target_shape=base_shape
)
# %%
# stereo_masks = {i['chip']: i for i in read_stereo_masks('T105 T103 T101 T99 T97 T95 T93 T91 T89 T87 T85 T83 T81 T79 T77 T75 T73 T71 T67 T65 T63 T61 T59 T57 T55 T53 T51 T49 T47 T45 T43 T41 T39 T37 T33 T31 T29 T27 T25 T26 T28 T30 T32 T34'.split(),ntp_version='Mq179-CLA-20230505', scale=1/315, min_sum=140)}

# %%
# %%



def read_regist_results():
    assets_path = Path('/mnt/97-macaque/projects/cla/injections-cells/assets/raw_image/')

    connectome_regist_results = pickle.load(open(assets_path.parent / 'regist_results_20240129_connectome_to_C042.pkl', 'rb'))
    connectome_to_mq179_results = pickle.load(open(assets_path.parent / 'regist_results_20240118_C042_to_Mq179.pkl', 'rb'))

    connectome_regist_results = [i for i in connectome_regist_results if isinstance(i, PariRegistInfo)]
    connectome_to_mq179_results = [i for i in connectome_to_mq179_results if isinstance(i, PariRegistInfo)]
    return connectome_regist_results, connectome_to_mq179_results

def move_to_stereo(
    points: np.ndarray, /, *, 
    mov_animal: str, mov_slice_id: int, mov_side: Literal['left', 'right'],
):
    mov_configs = []
    mov_points = points.copy()
    connectome_regist_results, connectome_to_mq179_results = read_regist_results()

    if mov_animal != 'C042':
        to_conn_mov_configs = [
            i for i in connectome_regist_results 
            if i.mov_sec.animal_id == mov_animal and i.mov_sec.slice_id == f'{mov_slice_id:03d}' and i.mov_sec.side==mov_side
        ]
        best_match = max(to_conn_mov_configs, key=lambda x: x.iou)
        mov_configs.append(best_match)
        mov_slice_id = best_match.fix_sec.slice_id_int

        if best_match.mov_sec.need_flip:
            mov_image = best_match.mov_sec.image
            mov_points[:, 0] = mov_image.shape[1] - mov_points[:, 0]

        mov_points = best_match.transform_points(mov_points)


    to_stereo_mov_configs = [
        i for i in connectome_to_mq179_results 
        if i.mov_sec.animal_id == 'C042' and i.mov_sec.slice_id == f'{mov_slice_id:03d}' and i.mov_sec.side==mov_side
    ]

    best_match = max(to_stereo_mov_configs, key=lambda x: x.iou)
    mov_configs.append(best_match)
    # if best_match.mov_sec.need_flip:
    #     mov_image = best_match.mov_sec.image
    #     mov_points[:, 0] = mov_image.shape[1] - mov_points[:, 0]
    mov_points = best_match.transform_points(mov_points)
    return mov_points, mov_configs


move_to_stereo(np.random.random((10, 2)) * 100, mov_animal='C040', mov_slice_id=167, mov_side='right')
# %%
# %%pyinstrument
@dataclass
class LRData:
    left: Polygon
    right: Polygon

    def __iter__(self):
        return zip(['left', 'right'], [self.left, self.right])

def line_iou(points: np.ndarray, img: np.ndarray):
    img = img.astype(np.uint8)
    points_to_img = np.zeros_like(img, dtype=img.dtype)

    cv2.fillPoly(points_to_img, [points.astype(int)], color=(1, ))
    return np.sum(points_to_img & img) / np.sum(points_to_img | img)

@cache
def match_regex(s: str, regex: str):
    return re.search(regex, s, re.IGNORECASE) is not None


def extract_target_regions_and_split_left_right(sm: SliceMeta, taget_regions_regex: str='cla'):
    target_regions: list[Polygon] = []
    for r in sm.regions:
        if not match_regex(r.label.name, taget_regions_regex):
            continue
        target_regions.append(r.polygon)
    if len(target_regions) != 2:
        print(f"target_regions: {len(target_regions)}!=2")
        return None
    
    target_regions = sorted(target_regions, key=lambda x: x.area, reverse=True)[:2]
    target_regions = sorted(target_regions, key=lambda x: x.bounds[0])
    return LRData(target_regions[0], target_regions[1])

@cache
def pad_polygon_to_exterior_lines(polygon: Polygon, base_shape: tuple[int, int]):
    bounds = polygon.bounds
    ps = np.array(polygon.exterior.xy).T - np.array([bounds[0], bounds[1]])-1
    ext_df = pl.DataFrame(ps, schema=['x', 'y']).with_row_index()
    ext_df = pad_or_crop(
        ext_df, *base_shape, 
        src_w=int(bounds[2]-bounds[0]), src_h=int(bounds[3]-bounds[1])
    )
    return ext_df


def check_polygon_contains_points(polygon: Polygon):
    bounds = polygon.bounds
    w, h = int(bounds[2]-bounds[0]), int(bounds[3]-bounds[1])
    least_w = 3000

    scale = 1
    if w < least_w or h < least_w:
        scale = int(least_w / max(w, h))
        w, h = int(w*scale), int(h*scale)
    
    mask = utils.poly_to_bitmap(polygon, *base_shape, buffer_x=0, buffer_y=0, scale=scale, minus_minimum=False)
    # plt.imshow(mask)
    # plt.show()
    rcm = mask_encode(mask)
    
    # print(rcm, scale)
    def f(xy):
        xs = (cast(pl.Series, xy.struct.field('x')) * scale).cast(pl.Int32)
        ys = (cast(pl.Series, xy.struct.field('y')) * scale).cast(pl.Int32)

        x = xs.to_numpy()
        y = ys.to_numpy()
        # print(x, y)
        rcm.encodings = np.ascontiguousarray(rcm.encodings[:, :3])
        out = rcm.find_index(x, y)
        return pl.Series(out)

    return pl.struct('x', 'y').map_batches(f).eq(0).not_().alias('in_poly')

# force_slice_id = 184
# # force_slice_id = 178
# force_slice_id = 191

# force_slice_id = None


class Clusterer:
    def __init__(
        self, points: np.ndarray | pl.DataFrame, /, *,  
        min_cluster_size: int=10, min_samples: int=10,
        cluster_selection_epsilon: int=10,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        if isinstance(points, pl.DataFrame):
            points = points[['x', 'y']].to_numpy()
        self.points = points
        self.hdb = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            alpha=0.4

        )
        
    def run(self):
        self.hdb.fit(self.points)
        return self

    @property
    def labels(self):
        return self.hdb.labels_

    @property
    def df_with_label(self):
        df = pl.DataFrame([self.points[:, 0], self.points[:, 1], self.hdb.labels_], schema=['x', 'y', 'label'])
        return df

    @property
    def df_without_noise(self):
        return self.df_with_label.filter(pl.col('label') >= 0)

    # @cache
    def __iter__(self):
        for gs, g in self.df_with_label.group_by(['label']):
            gs = cast(tuple[int], gs)
            yield gs[0], g

    # @cache
    def concave_hull(self, label: int, *, concavity: float=2.0, length_threshold: float=6, buffer:float=1):
        points = self.df_with_label.filter(pl.col('label') == label)[['x', 'y']].to_numpy()
        new_points = concave_hull(points, concavity=concavity, length_threshold=length_threshold)
        pol = Polygon(new_points).buffer(buffer)
        return pol

    # @cache
    def concave_hulls(self, *, concavity: float=2.0, length_threshold: float=6, buffer:float=1):
        for label, g in self:
            if label < 0: continue
            points = g[['x', 'y']].to_numpy()
            new_points = concave_hull(points, concavity=concavity, length_threshold=length_threshold)
            pol = Polygon(new_points).buffer(buffer)
            yield label, pol

    def plot(
        self, ax: Axes|None=None, 
        show_noise: bool=True, noise_color: str='gray', s=3,
        show_concave_hull: bool=False, length_threshold: float=6, 
        mask: np.ndarray|None=None, ext_lines: pl.DataFrame|None=None, side: str=''
    ):
        if ax is None:
            ax = plt.gca()

        if mask is not None:
            ax.imshow(mask)
        if ext_lines is not None:
            ax.plot(ext_lines['x'], ext_lines['y'], label=side)

        for label, g in self:
            # print('1', label)
            if label < 0:
                if show_noise:
                    ax.scatter(g['x'], g['y'], s=5, c=noise_color, marker='x')
            else:
                ax.scatter(g['x'], g['y'], s=s)

        if show_concave_hull:
            for label, pol in self.concave_hulls(length_threshold=length_threshold):
                # print('2', label)

                ax.plot(*pol.exterior.xy)
        return ax


@cache
def read_ntp_with_cache(ps: Path | tuple[Path], animal: str, slice_id: str, bin_size: int):
    psarg = []
    if isinstance(ps, tuple):
        psarg = list(ps)
    else:
        psarg = [ps]

    return dataset_utils.read_ntp(psarg, animal, slice_id, bin_size=bin_size)


@dataclass
class ZoneCellsDs: 
    animal_id: str
    slice_id : str
    side     : str
    ch       : str
    mask     : np.ndarray
    cell_df  : pl.DataFrame
    clusterer: Clusterer
    poly     : Polygon
    hulls    : list[Polygon]


class CalcInjectionZone:
    def __init__(self, animal_id: str, scale: float, base_shape: tuple[int, int]) -> None:
        self.animal_id = animal_id
        self.scale = scale
        self.base_shape = base_shape

        self.raw_cntm_masks = read_connectome_masks(
            get_base_path(self.animal_id), self.animal_id, self.scale, 
            exclude_slices=exclude_slices, force=False, 
            min_sum=400
        )

        self.fix_regist_datasets = read_regist_datasets(
            self.animal_id, self.scale, exclude_slices=exclude_slices, force=False, 
            tqdm=tqdm, target_shape=self.base_shape
        )

    def extract_slice_zone(self, slice_id: str):
        ansec = AnimalSection(self.animal_id, str(slice_id))
        if (item := self.fix_regist_datasets.get(ansec)) is None:
            return
        ntp_ps = dataset_utils.find_connectome_ntp(
            self.animal_id, slice_id=slice_id, 
            # base_path='/mnt/90-connectome/finalNTP-transnew/ntp-2024-02-23-v0.2/'
            # base_path='/mnt/90-connectome/finalNTP-layer4-parcellation91-106/'
            # base_path='/mnt/90-connectome/finalNTP-transnew/ntp-2024-02-28/'
            base_path='/mnt/90-connectome/finalNTP-with-cla/'
        )

        sm = dataset_utils.read_ntp(ntp_ps, self.animal_id, slice_id, bin_size=int(SRC_IMG_BIN_SIZE/self.scale))
        assert sm is not None

        lr_poly = extract_target_regions_and_split_left_right(sm)
        if lr_poly is None:
            print(f"slice_id: {slice_id} not found")
            return


        for i, (side, poly) in enumerate(lr_poly):
            assert side in ('left', 'right')
            mask, cell_df = item['pad_res'][side]
            cell_df = cast(pl.DataFrame, cell_df)

            ext_lines = pad_polygon_to_exterior_lines(poly, self.base_shape)
            paded_poly = Polygon(ext_lines[['x', 'y']].to_numpy())

            cells_in_poly = cell_df.with_columns(
                check_polygon_contains_points(paded_poly)
            ).filter(pl.col('in_poly'))
            for key, g in cells_in_poly.group_by(['color']):
                ch = cast(tuple[str], key)[0]
                if len(g) < 7:
                    continue

                cell_points = g[['x', 'y']].to_numpy()
                cluster = Clusterer(cell_points, min_cluster_size=4, min_samples=20, cluster_selection_epsilon=7)
                
                if len(cell_points) >= cluster.min_cluster_size and len(cell_points) >= cluster.min_samples:
                    cluster = cluster.run()
                else:
                    continue

                yield ZoneCellsDs(
                    animal_id=self.animal_id,
                    slice_id=slice_id,
                    side=side,
                    ch=ch, 
                    mask=mask,
                    cell_df=g.with_columns(pl.Series(cluster.labels).alias('label')),
                    clusterer=cluster,
                    poly=paded_poly,
                    hulls=[poly for label, poly in cluster.concave_hulls()],
                )


    def extract_slice_zones(self, n_parts: int=0, tqdm=lambda x, *args, **kargs: x):
        if n_parts == 0:
            n_parts = len(self.raw_cntm_masks)

        res: list[ZoneCellsDs] = []

        for i in tqdm(
            split_into_n_parts_slice(self.raw_cntm_masks, n_parts),
            desc=f"extract_slice_zones: {self.animal_id}"
        ):
            slice_id = self.raw_cntm_masks[i]['slice_id']
            for zone_cells_ds in self.extract_slice_zone(slice_id):
                res.append(zone_cells_ds)
        res.sort(key=lambda x: int(x.slice_id))
        return res

# %%
animal_id = 'C077'

ciz = CalcInjectionZone(animal_id, scale=scale, base_shape=base_shape)
zones = ciz.extract_slice_zones(tqdm=tqdm)
len(zones)
# %%
def color_name_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    return {
        'red'   : (0, 0, 255, alpha),
        'green' : (0, 255, 255, alpha),
        'blue'  : (255, 0, 0, alpha),
        'yellow': (0, 255, 0, alpha),
    }[color]

def draw_zones(
    animal_id: str, side: Literal['left', 'right'], ch: str, 
    zones: list[ZoneCellsDs], *, 
    margin_x: int, margin_y: int, 
    w: int, h: int,
    output_base: Path
):

    start_xs = [0]
    for z in zones:
        start_xs.append(start_xs[-1] + z.poly.bounds[2] - z.poly.bounds[0] + margin_x)

    pdf_w = start_xs[-1] - margin_x*16 + 500
    pdf_h = h + margin_y*2
    dye = utils.dye_to_color.inv.get(ch, ch)
    output_p = output_base / f'{dye}-{side}-{animal_id}.svg'

    with cairo.SVGSurface(output_p.as_posix(), pdf_w, pdf_h) as surface:
        ctx = cairo.Context(surface)

        for z, start_x in zip(zones, start_xs):
            z: ZoneCellsDs
            if z.side == side:
                continue

            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(1)

            coords = list(z.poly.exterior.coords)

            ctx.move_to(coords[0][0]+start_x, margin_y)
            ctx.set_font_size(6)
            ctx.show_text(f"{z.slice_id}")

            ctx.move_to(coords[0][0]+start_x, coords[0][1]+margin_y)

            for x, y in coords[1:]:
                ctx.line_to(x+start_x, y+margin_y)
            ctx.close_path()
            ctx.stroke()



            color = color_name_to_rgba(ch, alpha=200) # type: ignore
            ctx.set_source_rgba(*color)

            for x, y in z.cell_df.select('x', 'y').iter_rows():
                ctx.move_to(x+start_x, y+margin_y)
                ctx.arc(x+start_x, y+margin_y, 0.5, 0, 2*np.pi)
            ctx.fill()

            ctx.set_source_rgba(0, 0, 0, 0.5)
            for hull in z.hulls:
                coords = list(hull.exterior.coords)
                ctx.move_to(coords[0][0]+start_x, coords[0][1]+margin_y)

                for x, y in coords[1:]:
                    ctx.line_to(x+start_x, y+margin_y)
                ctx.close_path()
                ctx.stroke()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ###############################################################################
# ###############################################################################
# ###############################################################################
# ###############################################################################
# %%
auto_zone_base_path = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/auto-zone-20240322/')
auto_zone_base_path.mkdir(exist_ok=True, parents=True)

def write_log(text: str):
    (auto_zone_base_path / 'log.log').open('a+').write(text)

# for animal_id in (pbar := tqdm([
#     f'C{i:03d}' for i in range(74, 75)
# ], desc='animal_id: ')):
#     # animal_id = 'C006'

#     pbar.set_postfix_str(animal_id)
#     try:
#         ciz = CalcInjectionZone(animal_id, scale=scale, base_shape=base_shape)
#         zones = ciz.extract_slice_zones(tqdm=tqdm, n_parts=40)
#     except Exception as e:
#         stack_str = traceback.format_stack()

#         write_log(f"计算 zone 时异常：{animal_id} {e}\n{stack_str}\n\n")
#         continue

#     for ch, g_zones in groupby(
#         sorted(zones, key=lambda x: x.ch), 
#         key=lambda x: x.ch
#     ):
#         g_zones = sorted(g_zones, key=lambda x: int(x.slice_id))
#         print(ch, len(g_zones))
#         for side in ('left', 'right'):
#             try:
#                 draw_zones(
#                     animal_id, side, ch, g_zones, 
#                     margin_x=10, margin_y=10, w=128, h=128, 
#                     output_base=auto_zone_base_path
#                 )
#             except Exception as e:
#                 stack_str = traceback.format_stack()
#                 write_log(f'绘制 zone 时异常：{animal_id} {side} {e}\n{stack_str}\n\n')

#                 output_p = Path(f'/mnt/97-macaque/projects/cla/injections-cells/injection-zone/auto-zone/{animal_id}-{side}.svg')


# %%
force_slice_id = '209'
force_slice_id = None


def read_animal_cells(animal_id: str, tqdm=lambda x: x):
    all_cell_dfs = []

    raw_cntm_masks = read_connectome_masks(
        get_base_path(animal_id), animal_id, scale, 
        exclude_slices=exclude_slices, force=False, 
        min_sum=400
    )
    fix_regist_datasets = read_regist_datasets(
        animal_id, scale, exclude_slices=exclude_slices, force=False, 
        tqdm=tqdm, target_shape=base_shape
    )

    for i in tqdm(range(len(raw_cntm_masks))):
        slice_id = raw_cntm_masks[i]['slice_id']

        ansec = AnimalSection(animal_id, str(slice_id))
        if (item := fix_regist_datasets.get(ansec)) is None:
            continue

        ntp_ps = dataset_utils.find_connectome_ntp(
            animal_id, slice_id=slice_id, 
            # base_path='/mnt/90-connectome/finalNTP-transnew/ntp-2024-02-23-v0.2/'
            # base_path='/mnt/90-connectome/finalNTP-layer4-parcellation91-106/'
            # base_path='/mnt/90-connectome/finalNTP-transnew/ntp-2024-02-28/'
            base_path='/mnt/90-connectome/finalNTP-with-cla/'
        )
        sm = dataset_utils.read_ntp(ntp_ps, animal_id, slice_id, bin_size=int(SRC_IMG_BIN_SIZE/scale))
        assert sm is not None
        # if sm is None:
        #     print(f"slice_id: {slice_id} not found")
        #     continue


        lr_poly = extract_target_regions_and_split_left_right(sm)
        if lr_poly is None:
            print(f"slice_id: {slice_id} not found")
            continue


        for i, (side, poly) in enumerate(lr_poly):
            assert side in ('left', 'right')
            mask, cell_df = item['pad_res'][side]
            cell_df = cast(pl.DataFrame, cell_df)

            ext_lines = pad_polygon_to_exterior_lines(poly, base_shape)
            paded_poly = Polygon(ext_lines[['x', 'y']].to_numpy())

            cell_df = cell_df.with_columns(
                check_polygon_contains_points(paded_poly)
            ).with_columns(
                pc('color').replace(color_to_tracer, default=None).alias('tracer'),
                pl.lit(animal_id).alias('animal_id'),
                pl.lit(slice_id).alias('slice_id'),
                pl.lit(side).alias('side'),
            )

            all_cell_dfs.append(cell_df)
    return pl.concat(all_cell_dfs)

all_cell_dfs = []
for animal_id in tqdm([
    'C025',
    'C081',
    'C042',
    'C074',
]):
    all_cell_dfs.append(read_animal_cells(animal_id, tqdm=tqdm))
# %%
all_cell_df: pl.DataFrame = pl.concat(all_cell_dfs)
# %%

@dataclass
class SliceCells:
    animal_id: str
    slice_id: str
    side: str
    tracer: str
    cell_df: pl.DataFrame
    mtx: np.ndarray

target_animal = 'C025'
target_side = 'right'


raw_cell_df: pl.DataFrame = all_cell_df.filter(
    pc('side') == target_side,
    pc('animal_id') == target_animal,
    'in_poly'
)

all_dist_mtx: dict[
    tuple[str, str, str, str], SliceCells
] = {}

for (tracer, ), g in raw_cell_df.group_by(['tracer']):
    for (slice_id, ), g2 in g.group_by(['slice_id']):
        print(tracer, slice_id, len(g2))
        points = g2[['x', 'y']].to_numpy()
        mtx = dist_matrix = pairwise_distances(points)
        sc = SliceCells(
            animal_id=target_animal,
            slice_id=slice_id,
            side=target_side,
            tracer=tracer,
            cell_df=g2,
            mtx=mtx
        )
        all_dist_mtx[(target_animal, target_side, tracer, slice_id)] = sc

# %%
all_dists = []
all_mean_dists = []

for (animal_id, side, tracer, slice_id), sc in all_dist_mtx.items():
    dists = np.triu(sc.mtx, k=1).flatten()
    dists = dists[dists > 0]
    all_dists.append(dists)


    mtx = sc.mtx.copy()
    np.fill_diagonal(mtx, np.nan)
    if np.isnan(mtx).all():
        continue
    avg_dists = np.nanmean(mtx, axis=1)
    all_mean_dists.append(avg_dists)

all_dists = np.concatenate(all_dists)
all_mean_dists = np.concatenate(all_mean_dists)
# %%
p = np.percentile(all_dists, 99)
plt.title(f'{p=}')
plt.hist(all_dists, bins=1000)
plt.axvline(p, color='red')
# %%
p = np.percentile(all_mean_dists, 99)
plt.title(f'{p=}')
plt.hist(all_mean_dists, bins=200)
plt.axvline(p, color='red')
rather_close = all_mean_dists < p
print(f'rather_close: {np.sum(rather_close)}')
#%%
sc = random.choice(list(all_dist_mtx.values()))
for sc in tqdm(all_dist_mtx.values()):
    plt.figure(figsize=(10, 5))
    plt.title(f'{sc.tracer} {sc.animal_id}-{sc.slice_id}-{sc.side} n={len(sc.cell_df)}')
    plt.axis('off')

    plt.subplot(131)
    plt.imshow(sc.mtx)
    plt.colorbar()

    plt.subplot(132)
    plt.hist(sc.mtx.flatten())
    plt.axvline(p, color='red', label=f'p99={p}')
    plt.xlim(0, 60)

    plt.subplot(133)

    mtx = sc.mtx.copy()
    np.fill_diagonal(mtx, np.nan)
    avg_dists = np.nanmean(mtx, axis=1)
    less_p_idx = avg_dists < p

    all_cells = sc.cell_df[['x', 'y']].to_numpy()
    plt.scatter(*all_cells[less_p_idx].T, s=1)
    plt.scatter(*all_cells[~less_p_idx].T, s=1, c='red')

    ntp_ps = dataset_utils.find_connectome_ntp(
        sc.animal_id, slice_id=sc.slice_id, 
        # base_path='/mnt/90-connectome/finalNTP-transnew/ntp-2024-02-23-v0.2/'
        # base_path='/mnt/90-connectome/finalNTP-layer4-parcellation91-106/'
        # base_path='/mnt/90-connectome/finalNTP-transnew/ntp-2024-02-28/'
        base_path='/mnt/90-connectome/finalNTP-with-cla/'
    )
    sm = dataset_utils.read_ntp(ntp_ps, sc.animal_id, sc.slice_id, bin_size=int(SRC_IMG_BIN_SIZE/scale))
    lr_poly = extract_target_regions_and_split_left_right(sm)
    ext_lines = pad_polygon_to_exterior_lines(lr_poly.__dict__[sc.side], base_shape)[['x', 'y']].to_numpy()

    plt.plot(*ext_lines.T, color='blue')
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.savefig(
        '/mnt/97-macaque/projects/cla/injections-cells/injection-zone/mean-dist/'
        f'{sc.tracer}-{sc.animal_id}-{sc.slice_id}-{sc.side}.png'
    )
    plt.close()
