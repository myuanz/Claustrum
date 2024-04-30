# %%
from dataclasses import dataclass
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from shapely import Polygon, Point
import numpy as np
import cairo
from functools import cache
import re
import xml.etree.ElementTree as ET
import shutil
from functools import cache
import os
from collections import defaultdict
from functools import reduce
from itertools import groupby
import seaborn as sns
import pandas as pd
from shapely.errors import GEOSException
import json

if os.path.exists('pwvy'):
    os.chdir((Path('.') / 'pwvy').absolute())

import cv2
import orjson
from ntp_manager import SliceMeta, from_dict
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm
from utils import read_stereo_masks
from range_compression import RangeCompressedMask, mask_encode
import xarray as xr

import utils
import duckdb

@dataclass
class MovAnimalMeta:
    # {
    #         'chip': chip,
    #         'min_x': min_x,
    #         'min_y': min_y,
    #         'max_x': max_x,
    #         'max_y': max_y,
    #         'offset_x': i * (w + margin_x),
    #         'i': i
    #     }
    chip: str
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    offset_x: int
    i: int
    region_index: int = -1

@dataclass
class PolygonWithText:
    polygon: Polygon
    text: str

    def __repr__(self) -> str:
        return f'Polygon(text={self.text}, area={self.polygon.area})'

    @property
    def poly_xy(self):
        return self.polygon.exterior.xy

    @property
    def meta(self):
        return MovAnimalMeta(**json.loads(self.text))


pc = pl.col

def flatten_polygon(p: MultiPolygon | Polygon) -> list[Polygon]:
    if isinstance(p, Polygon): return [p]
    res = []
    for i in p.geoms:
        res.extend(flatten_polygon(i))
    return res

# _chips_str = '''T105 T103 T101 T99 T97 T95 T93 T91 T89 T87 T85 T83 T81 T79 T77 T75 T73 T71 T67 T65 T63 T61 T59 T57 T55 T53 T51 T49 T47 T45 T43 T41 T39 T37 T33 T31 T29 T27 T25 T26 T28 T30 T32 T34'''
# ALL_CHIPS = _chips_str.split(' ')


# assets_path = Path('/mnt/97-macaque/projects/cla/injections-cells/assets/raw_image/')

# ntp_version = 'Mq-CLA-20240115'
# ntp_version = 'Mq179-CLA-sh-20240204'

# v展示时缩放 = 1/315
# stereo_masks = read_stereo_masks(
#     ALL_CHIPS, 
#     # ntp_version='Mq179-CLA-20230505', 
#     ntp_version=ntp_version,
#     # ntp_version='Mq-All-20230718',
#     scale=v展示时缩放, min_sum=0
# )

# -----

_chips_str = '''T1031 T1032 T1030 T1029 T1027 T1021 T1020 T1018 T1016 T1014 T1012 T1011 T1013 T1015 T1017 T1019 T1022 T1023 T1024 T1025 T1026 T1028 T1033'''
ALL_CHIPS = _chips_str.split(' ')

ntp_version = 'Mq277-CLA-sh-20240204'

v展示时缩放 = 1/315
stereo_masks = read_stereo_masks(
    ALL_CHIPS, 
    # ntp_version='Mq179-CLA-20230505', 
    ntp_version=ntp_version,
    # ntp_version='Mq-All-20230718',
    scale=v展示时缩放, min_sum=0
)


# stereo_masks = pickle.load(open(assets_path.parent / 'stereo_masks_20240118.pkl', 'rb'))
stereo_masks = {v['chip']: v for v in stereo_masks}
# stereo_masks
use_mq277 = 'T33' not in ALL_CHIPS

# %%
chip = 'T73'
chip = 'T1033'

stereo_image_size = 128


mask_item = stereo_masks[chip]

mask_polygon: Polygon = mask_item['polygon']
pminx, pminy, pmaxx, pmaxy = mask_polygon.bounds

def read_cell_type_result(
    chip: str, 
    cell_type_root = Path('/data/sdf/to_zhang/snRNA/spatialID-20240115/'),
    cell_type_version = 'macaque1_CLA_res1.2_49c_csv',
    total_gene_2d_version = 'macaque-20240125-Mq179-cla',
    cla_region_ids = (347, 1355) # cla shCla
) -> pl.DataFrame:
    cell_type_p = next((cell_type_root / cell_type_version).glob(f'*{chip}*.csv'))
    res_p = cell_type_p.with_name(f'{chip}_with_total_gene2d.parquet')
    print(res_p, res_p.exists())
    if res_p.exists():
        res = pl.read_parquet(res_p)
    else:
        res = duckdb.sql(f'''
            select * from "/data/sde/total_gene_2D/{total_gene_2d_version}/total_gene_{chip}_macaque_f001_2D_{total_gene_2d_version}.parquet" as t1 
            join "{cell_type_p}" as t2
            on t1.cell_label = t2.cell
            where t1.gene_area in {cla_region_ids}
        ''').pl()
        res.write_parquet(res_p)
    return res


# total_gene_2d_version = 'macaque-20240125-Mq179-cla'
total_gene_2d_version = 'macaque-20240205-Mq179-277-CLA-sh'

# cell_type_root = Path('/data/sdf/to_zhang/snRNA/spatialID-20240115/')
cell_type_root = Path('/mnt/97-macaque/projects/cell-type-results/SpatialID_boundary_138c_mq1&mq4-20220202/')
cell_type_root = Path(r'/data/sdf/cell-type-results/SpatialID_0218_mq1&mq4')
# cell_type_root = Path(r'/data/sdf/cell-type-results/SpatialID_20240314_merge_74c_mq1&mq4')

cell_type_version = 'macaque1_CLA_49c_csv'
cell_type_version = 'macaque4_CLA_49c_csv'

# cell_type_version = 'macaque1_CLA_merge_74c_csv'


# for chip in tqdm(ALL_CHIPS):
#     cell_type_df = read_cell_type_result(
#         chip, 
#         cell_type_root=cell_type_root,
#         cell_type_version=cell_type_version,
#         total_gene_2d_version=total_gene_2d_version
#     )
#     # break
cell_type_df = read_cell_type_result(
    chip, 
    cell_type_root=cell_type_root,
    cell_type_version=cell_type_version,
    total_gene_2d_version=total_gene_2d_version
)
# %%
# cell_type_df = pl.scan_parquet(f"/data/sde/total_gene_2D/{total_gene_2d_version}/total_gene_{chip}_macaque_f001_2D_macaque-20240125.parquet").limit(1000000).collect().filter(pl.col('gene_area') == 347)

mov_cells = cell_type_df[['x', 'y']].to_numpy()
corr = utils.get_corr(chip, source='db')

mov_cells = corr.wrap_point(mov_cells) * v展示时缩放
mov_cells -= np.array([pminx, pminy])
print(mov_cells.shape)
cell_type_df_corr = cell_type_df.with_columns(
    pl.lit(mov_cells[:, 0]).alias('x'),
    pl.lit(mov_cells[:, 1]).alias('y'),
)
max_xy = cell_type_df_corr[['x', 'y']].max().to_numpy()[0]
min_xy = cell_type_df_corr[['x', 'y']].min().to_numpy()[0]

cell_type_df_corr = utils.pad_or_crop(
    cell_type_df_corr, 
    dst_w=stereo_image_size, dst_h=stereo_image_size,
    # src_w=max_xy[0] - min_xy[0]+1, src_h=max_xy[1] - min_xy[1]+1,
    src_w=int(pmaxx - pminx), src_h=int(pmaxy - pminy),
)

to_draw = cell_type_df_corr.sample(10000)[['x', 'y']].to_numpy()
plt.figure(figsize=(10, 10))
plt.title(chip)
plt.gca().invert_yaxis()
plt.axis('equal')

plt.scatter(*to_draw.T, s=1, alpha=0.1)

new_df = mask_item['new_df']
to_draw = new_df[['x', 'y']].to_numpy()
# plt.plot(*to_draw.T, color='g', label='new_df')

to_draw = np.array(mask_polygon.exterior.coords) - np.array([pminx, pminy])
to_draw = pl.DataFrame({'x': to_draw[:, 0], 'y': to_draw[:, 1]})
to_draw = utils.pad_or_crop(
    to_draw, 
    dst_w=stereo_image_size, dst_h=stereo_image_size,
    src_w=int(pmaxx - pminx), src_h=int(pmaxy - pminy),
).to_numpy()


plt.plot(*to_draw.T, color='r', label='polygon', alpha=0.5)

cell_type_df = read_cell_type_result(
    chip, 
    cell_type_root=cell_type_root,
    cell_type_version=cell_type_version,
    total_gene_2d_version=total_gene_2d_version
).limit(100000)

to_draw = cell_type_df.sample(10000)[['rx', 'ry']].to_numpy() * v展示时缩放 - np.array([pminx, pminy])
to_draw = utils.pad_or_crop(
    pl.DataFrame({'x': to_draw[:, 0], 'y': to_draw[:, 1]}),
    dst_w=stereo_image_size, dst_h=stereo_image_size,
    src_w=int(pmaxx - pminx), src_h=int(pmaxy - pminy),
).to_numpy()

plt.scatter(*to_draw.T, s=1, alpha=0.1)

to_draw = mask_item['image']
# plt.imshow(to_draw, alpha=0.1)

plt.legend()
# 1: totalgene2d rxry、cell type xy手动处理、cell type rxry、读取函数的多边形
# 2: 多边形生成的图像、多边形生成的轮廓df
# %%

total_gene_2d_df = pl.scan_parquet(f"/data/sde/total_gene_2D/{total_gene_2d_version}/total_gene_{chip}_macaque_f001_2D_{total_gene_2d_version}.parquet").limit(1000000).collect().filter(pl.col('gene_area').is_in((347, 1355)))
# %%
corr = utils.get_corr(chip, source='db')

# smj = from_dict(SliceMeta, orjson.loads(Path(f'/data/sde/ntp/macaque/{ntp_version}/region-mask/Mq179-{chip}-P0.json').read_bytes()))
smj = from_dict(SliceMeta, orjson.loads(Path(f'/data/sde/ntp/macaque/{ntp_version}/region-mask/Mq277L-{chip}-P0.json').read_bytes()))

# smj = parcellate(
#     f'/data/sde/ntp/macaque/{ntp_version}/Mq179-{chip}-P0.ntp',
#     um_per_pixel=0.5,
#     export_position_policy='none',
#     w=corr.h, h=corr.w
# )

for r in smj.regions:
    p = r.polygon
    x, y = p.exterior.xy
    plt.plot(y, x, linewidth=1, c='red')


plt.scatter(
    total_gene_2d_df['y'] - corr.offset_y + 0, 
    total_gene_2d_df['x'] - corr.offset_x + 0, 

    s=1, alpha=0.1
)


# %%

region_df: pl.DataFrame = stereo_masks[chip]['new_df']
group_by_ = 'region_index'
region_df.select()
def merge_regions(region_df: pl.DataFrame, group_by_: str='region_index'):
    ps = []
    for i, g in region_df.group_by([group_by_]):
        p = Polygon(g.select('x', 'y').to_numpy())
        ps.append(p)

    all_regions: Polygon | MultiPolygon = unary_union(ps)
    if isinstance(all_regions, MultiPolygon):
        all_regions = max(all_regions.geoms, key=lambda x: x.area)

    return all_regions

merge_regions(region_df)
# %%
margin_x = -100
margin_y = 5

h = w = 128

pdf_h = 128 + margin_y*2
pdf_w = len(ALL_CHIPS) * (w + margin_x) - margin_x

# output_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/stereo_masks_20240206.pdf')
# with cairo.PDFSurface(output_p.as_posix(), pdf_w, pdf_h) as surface:

output_p = Path(f'/mnt/97-macaque/projects/cla/injections-cells/injection-zone/stereo_masks_20240423.from-python-{ntp_version}-{cell_type_version}.pdf')
if output_p.exists():
    output_p = Path('/dev/null')

raw_poly_texts: list[PolygonWithText] = []

with cairo.PDFSurface(output_p.as_posix(), pdf_w, pdf_h) as surface:
    ctx = cairo.Context(surface)

    for i, chip in enumerate(ALL_CHIPS):
        if (item := stereo_masks.get(chip)) is None:
            continue
        # print(chip)
        coords_df: pl.DataFrame = item['new_df']

        min_x, min_y = coords_df[['x', 'y']].min().to_numpy()[0]
        max_x, max_y = coords_df[['x', 'y']].max().to_numpy()[0]
        meta = {
            'chip': chip,
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'offset_x': i * (w + margin_x),
            'i': i
        }

        coords_df = coords_df.with_columns(pl.col('x') + i * (w + margin_x))
        for _, g in coords_df.group_by(['region_index']):
            coords = g[['x', 'y']].to_numpy()
            p = Polygon(coords)
            rp = p.representative_point()

            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(0.25)

            ctx.move_to(rp.x, rp.y)
            ctx.set_font_size(1)
            meta['region_index'] = g['region_index'][0]
            ctx.show_text(json.dumps(meta))

            ctx.move_to(*coords[0])
            for x, y in coords[1:]:
                ctx.line_to(x, y)
            ctx.close_path()
            ctx.stroke()

            plt.plot(*coords.T, gid=chip)
            plt.text(rp.x, rp.y, chip)
            raw_poly_texts.append(PolygonWithText(
                polygon=p,
                text=json.dumps(meta)
            ))

plt.xlim(0, 1200)

if not use_mq277:
    with open(output_p.with_suffix('.json'), 'wb') as f:
        f.write(orjson.dumps({
            'margin_x': margin_x,
            'margin_y': margin_y,
            'h': h,
            'w': w,
            'pdf_h': pdf_h,
            'pdf_w': pdf_w,
            'output_p': str(output_p),
            'ntp_version': ntp_version,
            'cell_type_version': cell_type_version,
            'cell_type_root': str(cell_type_root),
        }))
# %%
# 读入 svg, 确定 chip text 对应的多边形, 以及手绘的投射区域线条
    
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/画板 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_20240221test/SVG/画板 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-20240227/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-20240228/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-20240301/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2-20240305/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-INS/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-0311/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-20240313/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/20240314-fix/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-demo/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-20240313-demo/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all-20240319/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX-20240329/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_20240404/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_20240406-withoutDEN/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_20240406-withoutDEN-DZ_sub/SVG/Artboard 1.svg')

import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_20240404-cla-2/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_withoutDEN_20240408/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_withoutDEN_20240409/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_withoutDEN_20240415/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/stereo_masks_Version2_all_ZoneX_withoutDEN_20240416.svg')

import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_withoutDEN-0418modified/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_withoutDEN-0418modified/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_withoutDEN_153Areas_0423/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_withoutDEN_148Areas_0423-2222/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Mq277/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Mq277_20240425_fix/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/stereo_masks_Version2_all_ZoneX_withoutDEN_153Areas_0426-temp2/SVG/Artboard 1.svg')
import_svg_p = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/SVG/case2-0424-2/SVG/Artboard 1.svg')




svg = ET.parse(import_svg_p)
s_namespaces = {'svg': 'http://www.w3.org/2000/svg'}

base_pths = svg.findall('.//{http://www.w3.org/2000/svg}g[@id="_图层_1"]/{http://www.w3.org/2000/svg}polygon') + svg.findall('.//{http://www.w3.org/2000/svg}g[@id="_图层_1"]/{http://www.w3.org/2000/svg}path')

text_tags = svg.findall('.//{http://www.w3.org/2000/svg}g[@id="_图层_1"]/{http://www.w3.org/2000/svg}text')
zone_pths = svg.findall('.//{http://www.w3.org/2000/svg}g[@id="_图层_2"]/{http://www.w3.org/2000/svg}path')

mq277_elements = svg.findall('.//{http://www.w3.org/2000/svg}g[@id="_图层_4"]/{http://www.w3.org/2000/svg}path')

from svgpathtools import parse_path

def get_pos(tag: ET.Element):
    transform = tag.attrib['transform']
    translate = re.search(r'translate\((.+?)\)', transform).group(1)
    x, y = map(float, translate.split(' '))
    return x, y

def get_text(tag: ET.Element):
    children = tag.findall('*')
    return "".join([c.text or '' for c in children])


@cache
def get_polygon(tag: ET.Element, curve_n=500) -> Polygon:
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


def parse_group_of_textpolygon(tag: ET.Element) -> PolygonWithText:
    text_tag = tag.find('./svg:text', namespaces=s_namespaces)
    assert text_tag is not None
    text = get_text(text_tag)

    poly = tag.find('./svg:polygon', namespaces=s_namespaces)
    return PolygonWithText(get_polygon(poly), text)

mov_animal_groups = svg.findall('.//svg:g[@id="_图层_4"]//svg:g[svg:text]', namespaces=s_namespaces)
mov_animal_polygons = [parse_group_of_textpolygon(i) for i in mov_animal_groups]

mov_animal_polygons

# %%
@dataclass
class MovAnimalPair:
    src: PolygonWithText
    tgt: PolygonWithText

    @property
    def src_points(self):
        return np.array(self.src.poly_xy).T

    @property
    def tgt_points(self):
        return np.array(self.tgt.poly_xy).T

    def __post_init__(self):
        src = self.src_points
        dst = self.tgt_points
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        self.M = M

    def __repr__(self) -> str:
        return f'MovAnimalPair({self.src.meta.chip})'

    def mov_to_tgt(self, src: np.ndarray):
        assert src.shape[1] == 2, f'{src.shape} should be (n, 2)'
        return cv2.transform(src.reshape(1, -1, 2), self.M)


mov_paris: defaultdict[str, list[MovAnimalPair]] = defaultdict(list)

for i in range(len(mov_animal_polygons)):
    mov_poly = mov_animal_polygons[i]
    
    for j in range(len(raw_poly_texts)):
        raw_poly = raw_poly_texts[j]
        if mov_poly.text == raw_poly.text:
            print(f'{i}<->{j}')
            mov_paris[mov_poly.meta.chip].append(MovAnimalPair(src=mov_poly, tgt=raw_poly))
# %%
i = 'T1032'

m = mov_paris[i][0].mov_to_tgt(mov_paris[i][0].src_points)
plt.plot(*mov_paris[i][0].tgt_points.T, label='target points')
plt.scatter(*m.T, label='moved points')
plt.legend()


# %%
all_base_polygons: list[Polygon] = []
chip_to_polygon_list: defaultdict[str, list[Polygon]] = defaultdict(list)

# zone_pth = unary_union([get_polygon(p) for p in zone_pths])

plt.figure(figsize=(10, 2))
for p in base_pths:
    poly = get_polygon(p)

    plt.plot(*poly.exterior.xy)
    all_base_polygons.append(poly)

plt.plot(*get_polygon(zone_pths[len(zone_pths) // 2]).exterior.xy)

for tag in text_tags:
    x, y = get_pos(tag)
    text = get_text(tag)
    # if 'foot' in text:
    #     continue
    # if not text.endswith('-2'):
    #     continue
    # print(text)

    text = re.search(r'(T\d+)', text).group(1)
    print(text, x, y)

    plt.text(x, y, text, fontsize=2)

    for p in all_base_polygons:
        assert p.is_valid
        if p.contains(Point(x, y)):
            chip_to_polygon_list[text].append(p)
            break
    else:
        print(f'{text} not found')
        continue
    all_base_polygons.remove(chip_to_polygon_list[text][-1])


plt.scatter(coords_df['x'], coords_df['y'], s=0.5)

plt.xlim(0, 1500)
# assert all_base_polygons == []
for p in all_base_polygons:
    plt.plot(*p.exterior.xy, color='b')
assert len(all_base_polygons) == 0, len(all_base_polygons)
inz = set()
len(zone_pths)
# %%
if use_mq277:
    chip_to_polygon_list.clear()
    plt.plot(*get_polygon(zone_pths[len(zone_pths) // 2]).exterior.xy)
    for g in mov_animal_polygons:
        plt.plot(*g.poly_xy)
        chip_to_polygon_list[g.meta.chip].append(g.polygon)
    plt.xlim(0, 1500)

# %%
target_zone_id_df = pl.read_excel('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/Selected59Areas.xlsx')
target_zone_ids = target_zone_id_df['CaseName'].to_list()

# if enable, set 58 类，
# inz = set()
# for zone_pth in zone_pths:
#     zone_id = zone_pth.attrib['id']
#     if zone_id not in target_zone_ids:
#         pass
#     else:
#         inz.add(zone_id)
# assert len(inz) == len(target_zone_ids)
# %%
inz = set()
# %%
%load_ext pyinstrument

# %%
%%pyinstrument

save_fig = True
save_fig = False

export_root = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/intermediate-results-20240227-wml/')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240307-wml/')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240318-new-cells/')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240326-zone-x/')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240328-zone-x/')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240329-zone-x/')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240330-zone-x-sub-dzn/')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240404')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240407')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240407-withoutDEN-DZ_sub')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240407-checkvm')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240408-stereo_masks_Version2_all_ZoneX_withoutDEN')

export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240415-stereo_masks_Version2_all_ZoneX_withoutDEN')
export_root = Path('/data/sdf/to_wangML/injection-zone/intermediate-results-20240416-stereo_masks_Version2_all_ZoneX_withoutDEN')
# export_root = Path(f'/data/sdf/to_wangML/injection-zone/intermediate-results-20240418-{len(target_zone_id_df)}-dedup-2')
export_root = Path(f'/data/sdf/to_wangML/injection-zone/intermediate-results-20240422-153-2')
export_root = Path(f'/data/sdf/to_wangML/injection-zone/intermediate-results-20240423-153-total')
export_root = Path(f'/data/sdf/to_wangML/injection-zone/intermediate-results-20240423-2222-148-total')
export_root = Path(f'/data/sdf/to_wangML/injection-zone/intermediate-results-20240424-mq277-test')
export_root = Path(f'/data/sdf/to_wangML/injection-zone/intermediate-results-20240426-mq277')
export_root = Path(f'/data/sdf/to_wangML/injection-zone/intermediate-results-case2-version0424-2-0429')


# export_root = Path('/mnt/97-macaque/projects/cla/injections-cells/injection-zone/intermediate-results-20240221-wml-test')
export_root.mkdir(exist_ok=True, parents=True)
shutil.copyfile(import_svg_p, export_root / 'import.svg')

j检查时分区扩张系数 = 10 # 


def xy_to_index_wrapper(rcm: RangeCompressedMask, binary_search=False):
    def f(xy):
        x = (xy.struct.field('x').to_numpy())
        y = (xy.struct.field('y').to_numpy())

        out = rcm.find_index(x, y, binary_search=binary_search)
        return pl.Series(out)
    return f

polygons = []

for i, chip in enumerate(ALL_CHIPS):
    # if chip != 'T1032': continue

    if (mask_item := stereo_masks.get(chip)) is None:
        continue
    if not use_mq277:
        if (ps := chip_to_polygon_list.get(chip)) is None:
            continue
        p: MultiPolygon | Polygon = unary_union(ps)

    else:
        if (ps := mov_paris.get(chip)) is None:
            continue
        p: MultiPolygon | Polygon = unary_union([i.src.polygon for i in ps])

    print(chip)

    mask_polygon: Polygon = mask_item['polygon']
    pminx, pminy, pmaxx, pmaxy = mask_polygon.bounds

    cell_type_df = read_cell_type_result(
        chip, 
        cell_type_root=cell_type_root,
        cell_type_version=cell_type_version,
        total_gene_2d_version=total_gene_2d_version
    ).with_row_index()
    corr = utils.get_corr(chip, source='db')

    mov_cells = cell_type_df[['x', 'y']].to_numpy()
    mov_cells = corr.wrap_point(mov_cells) * v展示时缩放
    mov_cells -= np.array([pminx, pminy])
    cell_type_df_corr = cell_type_df.with_columns(
        pl.lit(mov_cells[:, 0]).alias('x'),
        pl.lit(mov_cells[:, 1]).alias('y'),
    )
    cell_type_df_corr = utils.pad_or_crop(
        cell_type_df_corr, 
        dst_w=stereo_image_size, dst_h=stereo_image_size,
        src_w=int(pmaxx - pminx), src_h=int(pmaxy - pminy),
    )
    cell_type_df = cell_type_df.join(cell_type_df_corr, on='index', how='inner').select(
        'index', 'gene', 'x', 'y', 'umi_count', 'rx', 'ry', 
        'gene_area', 'cell_label', 'celltype_pred'
    ) # crop 这一步有可能删除一些数据，现在用 join 把原始数据里对应的行也删除掉
    assert len(cell_type_df_corr) == len(cell_type_df)

    polygons.append({
        'chip': chip,
        'zone_id': 'total',
        'polygon': mask_polygon
    })
    if use_mq277:
        pair = max(mov_paris[chip], key=lambda x: x.tgt.polygon.area)
        '''
        cell type df里的xy是跟pair.tgt在同一坐标系下，pair.tgt来自于stereo_masks，
        在前面的代码里叫做raw_poly_texts。
        '''

    zone_dfs = []
    for zone_pth in tqdm(zone_pths):
        zone_id = zone_pth.attrib['id']
        if len(inz) and zone_id not in inz:
            continue

        zone_id = f'projection_zone_{zone_id}'

        zone_pth = get_polygon(zone_pth)
        # print(chip, zone_id)
        # 计算交集
        try:
            p_intersection = p.intersection(zone_pth)
        except GEOSException as e:
            print(chip, zone_id, e)

        polygons.append({
            'chip': chip,
            'zone_id': zone_id,
            'polygon': p_intersection
        })

        if p_intersection.area == 0:
            continue

        mask_lines = []
        if isinstance(p_intersection, MultiPolygon):
            mask_lines = [p.exterior.coords for p in p_intersection.geoms]
        elif isinstance(p_intersection, Polygon):
            mask_lines = [p_intersection.exterior.coords]
        else:
            raise ValueError(f'unknown type: {p_intersection}')
        # p_np = np.array(p.exterior.coords) - np.array([i * (w + margin_x), 0])

        region_mask = np.zeros((stereo_image_size * j检查时分区扩张系数, stereo_image_size * j检查时分区扩张系数), dtype=np.uint8)
        
        mask_lines_np = []

        for mask_line in mask_lines:
            if use_mq277:
                mask_line = (
                    pair.mov_to_tgt(np.array(mask_line)) - np.array([i * (w + margin_x), 0])
                ) * j检查时分区扩张系数
            else:
                mask_line = (np.array(mask_line) - np.array([i * (w + margin_x), 0])) * j检查时分区扩张系数

            mask_lines_np.append(mask_line)

            region_mask = cv2.fillPoly(region_mask, [mask_line.astype(np.int32)], (1, ))
        region_rcm = mask_encode(region_mask)
        
        zone_df = cell_type_df_corr.select(
            pl.struct([
                (pc('x') * j检查时分区扩张系数).cast(pl.Int32), 
                (pc('y') * j检查时分区扩张系数).cast(pl.Int32),
            ]).map_batches(xy_to_index_wrapper(region_rcm)).alias(zone_id)
        )

        # zone_df.write_parquet(
        #     export_root / f'{chip}_cell_type_df_corr_with_{zone_id}.parquet'
        # )
        zone_dfs.append(zone_df)
        if save_fig:
            to_draw_df = pl.concat([cell_type_df_corr, zone_df], how='horizontal').sample(10000)

            plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.resize(region_mask, (stereo_image_size, stereo_image_size), interpolation=cv2.INTER_NEAREST), alpha=0.5)
            for k, g in to_draw_df.group_by([zone_id]):
                plt.scatter(g['x'], g['y'], s=1, alpha=0.1, label=k[0])

            for single_p in ps:
                if isinstance(single_p, Polygon):
                    p_np = np.array(single_p.exterior.coords) - np.array([i * (w + margin_x), 0])
                else:
                    p_np = pair.tgt_points - np.array([i * (w + margin_x), 0])

                plt.plot(*p_np.T, color='r', linewidth=0.5)

            for mask_line_np in mask_lines_np:
                # mask_line_np = pair.mov_to_tgt(mask_line_np)
                plt.plot(*mask_line_np.T / j检查时分区扩张系数, color='g')
            plt.axis('equal')
            plt.axis('off')
            plt.gca().invert_yaxis()
            plt.title(chip)
            plt.legend()

            plt.savefig(export_root / f'{chip}_{zone_id}_绘制检查.svg')
            plt.close()
        # break
    pl.concat([cell_type_df, *zone_dfs], how='horizontal').write_parquet(export_root / f'{chip}_cell_type_df_with_zones.parquet', compression='snappy')



for i in range(len(polygons)):
    item = polygons[i]
    polygons[i]['area'] = item['polygon'].area / v展示时缩放

polygons_df = pl.DataFrame(polygons).select('chip', 'zone_id', 'area').sort('area')
polygons_df.write_excel(export_root / 'polygons.xlsx')
# %%
polygons_dict_list = defaultdict(lambda: defaultdict(list))
polygons_dict_list = []


for p in tqdm(polygons):
    raw_zid = p['zone_id'].replace('projection_zone_', '')
    if raw_zid == 'total': continue

    animal_id, ch, *zids = raw_zid.split('-')

    zid = "-".join(zids)

    pol = p['polygon']
    if pol.is_empty: continue

    # polygons_dict_list[zid][p['chip']].append(pol)
    polygons_dict_list.append({
        **p,
        'zone_id': zid,
        'raw_zone_id': raw_zid, 
        'animal': animal_id,
        'ch': ch
    })
pol_df = pl.DataFrame(polygons_dict_list)
len(pol_df)
# %%
# pol_df_p = pol_df.select(pl.all().exclude('polygon'))
# areas = pol_df_p.group_by('zone_id').agg(
#     pl.sum('area')
# ).sort('area', descending=True).with_columns(
#     pl.col('area').log(2).alias('log_area')
# )
# plt.hist(areas['area'].log(2), bins=50)
# plt.title('log2(area) histogram')

# areas.write_excel(export_root / 'sub-areas.xlsx')
# plt.savefig(export_root / 'sub-areas.png')

# %%
# %%
for p in polygons:
    if p['zone_id'] == 'total': 
        continue

    polys = flatten_polygon(p['polygon'])
    for poly in polys:
        plt.plot(*poly.exterior.xy)
# %%
# for chip_to_polygon in chip_to_polygon_list:
#     for p in chip_to_polygon.values():
#         polys = flatten_polygon(unary_union(p))
#         for poly in polys:
#             plt.plot(*poly.exterior.xy)
# %%

# pol_df_ged = []

# for k, g in pol_df.group_by(['animal', 'ch', 'zone_id']):
#     curr_zone = unary_union(g.select('polygon').to_numpy())
#     pol_df_ged.append({
#         'zone_id': k[2],
#         'polygon': curr_zone
#     })

# polygons_dict = {}

# for k, g in groupby(sorted(pol_df_ged, key=lambda x: x['zone_id']), lambda x: x['zone_id']):
#     ps = [i['polygon'] for i in g]
#     # print(k, len(ps))

#     polygons_dict[k] = reduce(
#         lambda x, y: x.intersection(y), 
#         ps
#     )



# # %%

# @cache
# def get_zone_by_id(zid: str):
#     return unary_union([
#         i['polygon'] for i in 
#         polygons_dict[zid].values()
#     ])
# # %%

# zone_ids = sorted(polygons_dict.keys())
# len(zone_ids)
# # %%
# sim_mtx = np.zeros((len(zone_ids), len(zone_ids)))

# #%%
# for i, zone_id1 in enumerate(zone_ids):
#     for j, zone_id2 in enumerate(tqdm(zone_ids[i:])):
#         j += i
#         zones1 = polygons_dict[zone_id1]
#         zones2 = polygons_dict[zone_id2]
#         # if zones1.is_empty or zones2.is_empty:
#         #     iou = 0
#         # else:
#         #     iou = zones1.intersection(zones2).area / zones1.union(zones2).area

#         iou = zones1.intersection(zones2).area / zones1.union(zones2).area
#         sim_mtx[i, j] = sim_mtx[j, i] = iou
# # %%
# sim_df = pd.DataFrame(sim_mtx, columns=zone_ids, index=zone_ids)

# g = sns.clustermap(
#     sim_df,
#     xticklabels='auto',
#     figsize=(20, 20),
#     # z_score=1,
#     # standard_scale=1, 

#     # metric="correlation",
#     method="weighted",
# )

# # hightlight_idx = zone_ids.index('Cla')
# # new_idx = g.dendrogram_col.reordered_ind.index(hightlight_idx)

# for tick_labels in (
#     g.ax_heatmap.get_xticklabels(), 
#     g.ax_heatmap.get_yticklabels(),
# ):
#     for tick_label in tick_labels:
#         t = tick_label.get_text() 
#         if t.startswith('F'):
#             tick_label.set_backgroundcolor("yellow")
#         elif t.startswith('V'):
#             tick_label.set_backgroundcolor("lightgreen")
#         elif re.match(r'^\d$', t):
#             tick_label.set_backgroundcolor("lightblue")
#         elif t.lower().startswith('cla'):
#             tick_label.set_backgroundcolor("lightcoral")
# # %%
# g.savefig(export_root / f'clustermap-intersection{"-sub" if enable_subcotex else ""}.pdf')

# %%

pol_df_ged = {}

for k, g in pol_df.group_by(['raw_zone_id']):
    curr_zone = unary_union(g.select('polygon').to_numpy())
    pol_df_ged[f'{k[0]}'] = curr_zone



# %%
zone_ids = sorted(pol_df_ged.keys())
zone_ids, len(zone_ids)

# %%
sim_mtx = np.zeros((len(zone_ids), len(zone_ids)))

intersection_mtx = np.zeros((len(zone_ids), len(zone_ids)))
union_mtx = np.zeros((len(zone_ids), len(zone_ids)))

#%%
from joblib import Parallel, delayed

def area_iu(pol_df_ged, zone_id1: str, zone_id2: str, i: int, j: int):
    zones1 = pol_df_ged[zone_id1]
    zones2 = pol_df_ged[zone_id2]

    inter_area = zones1.intersection(zones2).area
    union_area = zones1.union(zones2).area

    return inter_area, union_area, i, j


tasks = []
for i, zone_id1 in enumerate(zone_ids):
    for j, zone_id2 in enumerate(zone_ids[i:]):
        j += i
        tasks.append(delayed(area_iu)(pol_df_ged, zone_id1, zone_id2, i, j))

results = Parallel(n_jobs=64, backend='threading', return_as='generator')(tasks)

for inter_area, union_area, i, j in tqdm(results, total=len(tasks)):
    sim_mtx[i, j] = sim_mtx[j, i] = inter_area / union_area
    intersection_mtx[i, j] = intersection_mtx[j, i] = inter_area
    union_mtx[i, j] = union_mtx[j, i] = union_area
# %%
area_df = pd.DataFrame({
    k: v.area for k, v in pol_df_ged.items()
}, index=['area'])
area_df.to_excel(export_root / 'area_df.xlsx')
area_df

# %%
intersection_df = pd.DataFrame(intersection_mtx, columns=zone_ids, index=zone_ids)
union_df = pd.DataFrame(union_mtx, columns=zone_ids, index=zone_ids)
intersection_df.to_excel(export_root / 'intersection_mtx.xlsx')
union_df.to_excel(export_root / 'union_mtx.xlsx')
# %%
enable_subcotex = False
for enable_subcotex in (False, ):
    if not enable_subcotex:
        keep_indx = [i for i, zid in enumerate(zone_ids) if zid.split('-')[-1].lower() not in 'hippo hippo_sub sub_hippo cla thalamus amygdala putamen caudate dz1 dz2 dz3 dz4'.split(' ')]
    else:
        keep_indx = []

    # shuffler = np.random.permutation(len(sim_mtx))
    
    sim_df = pd.DataFrame(sim_mtx, columns=zone_ids, index=zone_ids)
    sim_df.to_excel(export_root / f'sim_mtx.xlsx')
    if keep_indx:
        # keep_indx.remove(zone_ids.index('C068-Ctb_R-F2_F7'))
        sim_df = sim_df.iloc[keep_indx, keep_indx]

    

    g = sns.clustermap(
        sim_df,
        xticklabels='auto',
        figsize=(50, 50),
        # row_cluster=False,
        metric="correlation", 
        # method="ward",
        # z_score=0,
        # standard_scale=0, 

        # metric="correlation",
        # method="centroid",
        # method="weighted",
    )

    # hightlight_idx = zone_ids.index('Cla')
    # new_idx = g.dendrogram_col.reordered_ind.index(hightlight_idx)

    for tick_labels in (
        g.ax_heatmap.get_xticklabels(), 
        g.ax_heatmap.get_yticklabels(),
    ):
        for tick_label in tick_labels:
            t = tick_label.get_text().split('-')[-1]
            if t.startswith('F'):
                tick_label.set_backgroundcolor("yellow")
            elif t.startswith('V'):
                tick_label.set_backgroundcolor("lightgreen")
            elif re.match(r'^\d$', t):
                tick_label.set_backgroundcolor("lightblue")
            elif t.lower().startswith('cla'):
                tick_label.set_backgroundcolor("lightcoral")
            elif t.lower().startswith('10'):
                tick_label.set_backgroundcolor('lightgrey')

    g.savefig(export_root / f'merged-clustermap{"-sub" if enable_subcotex else ""}.pdf')

    pl.DataFrame(sim_df).write_excel(
        export_root / f'merged-clustermap{"-sub" if enable_subcotex else ""}.xlsx'
    )
    pl.DataFrame({
        'recordered_ind': g.dendrogram_col.reordered_ind
    }).write_excel(
        export_root / f'merged-clustermap-reordered_ind{"-sub" if enable_subcotex else ""}.xlsx'
    )

# %%
cluster_df = pl.read_csv('./cluster_df.csv', new_columns=['zone_id', 'cluster'], has_header=False, separator='\t').select(
    pc('zone_id').str.replace("'", '').alias('zone_id'),
    'cluster'
).with_columns(
    pc('cluster').str.split('-').list.first().alias('cluster1'),
    pc('cluster').str.split('-').list.last().alias('cluster2'),
    pc('zone_id').is_in(zone_ids).alias('in_zone_ids'),
)
# assert len(cluster_df.filter(pc('in_zone_ids').not_())) == 0, cluster_df.filter(pc('in_zone_ids').not_())
# %%
pol_df_ged = {}

for k, g in pol_df.group_by(['raw_zone_id']):
    curr_zone = unary_union(g.select('polygon').to_numpy())
    pol_df_ged[f'{k[0]}'] = curr_zone


# %%

clusters = cluster_df['cluster'].unique().to_list()

fig, axes = plt.subplots(
    len(clusters), 1, figsize=(20, 20), sharex=True, 
)

for ax, ((cluster0, ), g) in zip(axes, cluster_df.group_by(['cluster'])):
    g = g['zone_id'].to_list()
    polys = [v for (k, v) in pol_df_ged.items() if k in g]
    polys = unary_union(polys)

    ax.axis('equal')
    ax.axis('off')
    ax.set_title(cluster0)
    for poly in flatten_polygon(polys):
        ax.plot(*poly.exterior.xy)
    print(cluster0, g)
    
    for polys in chip_to_polygon_list.values():
        for poly in polys:
            ax.plot(*poly.exterior.xy, color='gray', alpha=0.2)
fig.tight_layout()
# save figure
fig.savefig(export_root / f'clustered-zones-union{"-sub" if enable_subcotex else ""}.pdf')
# %%

clusters = cluster_df['cluster'].unique().to_list()

fig, axes = plt.subplots(
    len(clusters), 1, figsize=(20, 20), sharex=True, 
)

for ax, ((cluster0, ), g) in zip(axes, cluster_df.group_by(['cluster'])):
    g = g['zone_id'].to_list()
    polys = [v for (k, v) in pol_df_ged.items() if k in g]
    polys = reduce(
        lambda x, y: x.intersection(y), 
        polys
    )

    ax.axis('equal')
    ax.axis('off')
    ax.set_title(cluster0)
    for poly in flatten_polygon(polys):
        ax.plot(*poly.exterior.xy)
    print(cluster0, g)
    
    for polys in chip_to_polygon_list.values():
        for poly in polys:
            ax.plot(*poly.exterior.xy, color='gray', alpha=0.2)
fig.tight_layout()
# save figure
fig.savefig(export_root / f'clustered-zones-intersection{"-sub" if enable_subcotex else ""}.pdf')
# %%

clusters = cluster_df['cluster'].unique().to_list()

fig, axes = plt.subplots(
    len(clusters), 1, figsize=(20, 20), sharex=True, 
)


for ax, ((cluster0, ), g) in zip(axes, cluster_df.sort('cluster').group_by(['cluster'], maintain_order=True)):
    g = g['zone_id'].to_list()
    polys = [v for (k, v) in pol_df_ged.items() if k in g]

    ax.axis('equal')
    ax.axis('off')
    ax.set_title(f'{cluster0} :{len(polys)}')
    ax.invert_yaxis()
    # for poly in flatten_polygon(unary_union(polys)):
    #     print('1')
    #     ax.fill(*poly.exterior.xy, alpha=0.5, color='lightblue')
    for mpoly in polys:
        for poly in flatten_polygon(mpoly):
            ax.fill(*poly.exterior.xy, alpha=1/len(polys), color='blue')
    print(cluster0, g)
    
    for polys in chip_to_polygon_list.values():
        for poly in polys:
            ax.plot(*poly.exterior.xy, color='gray', alpha=0.2)
fig.tight_layout()
# save figure
fig.savefig(export_root / f'clustered-zones-alpha{"-sub" if enable_subcotex else ""}.pdf')


# %%
