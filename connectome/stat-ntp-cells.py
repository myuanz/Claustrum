# %%
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from ntp_manager import SliceMeta, parcellate, NTPRegion, NTPLabel
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm
from itertools import groupby

import connectome_utils as cutils
output_p = Path('/mnt/97-macaque/projects/cla/cell-stat/20240412-ynn')
output_p = Path('/mnt/90-connectome/finalNTP-layer4-parcellation91-106/落单细胞检查/cla20240415/')
output_p = Path('/mnt/90-connectome/finalNTP-layer4-parcellation91-106/落单细胞检查/cla20240425/')
ref_version_ps = [
    Path('/mnt/90-connectome/finalNTP-layer4-parcellation91-106/落单细胞检查/cla20240415/'),
    Path('/mnt/90-connectome/finalNTP-layer4-parcellation91-106/落单细胞检查/cla20240417-1/'),
]

# output_p = Path('/mnt/97-macaque/projects/cla/cell-stat/20240411-wml/')


output_p.mkdir(exist_ok=True, parents=True)


def in_polys_par(p: Polygon, points: np.ndarray):
    tasks = [delayed(p.contains)(Point(*i)) for i in points]
    return np.array(Parallel(64, backend='threading')(tasks))

color_to_tracer = {
    'blue': 'FB',
    'yellow': 'CTB555',
    'red': 'CTB647',
    'green': 'CTB488',
}

pc = pl.col
# %%
base_ntp_p = Path('/mnt/90-connectome/finalNTP-withcla-20240416/')
base_ntp_p = Path('/mnt/90-connectome/finalNTP-withcla-20240424/')

# base_ntp_p = Path('/mnt/90-connectome/finalNTP/')

animals = [i.name for i in base_ntp_p.glob('C*') if 'Pre' not in i.name][1:]
# 

# animals = ['C077']
# animals = ['C080', 'C075']
# animals = ['C119']
# animals = ['C075', 'C077', 'C080', 'C119', 'C096']
# animals = ['C079']

# base_ntp_p = Path('/mnt/90-connectome/C080-cell Number/')
# animals = [i.name for i in base_ntp_p.glob('C*')]

animals
# %%

def extrace_cells(ntp_p: Path, animal: str):
    slice_id = cutils.ntp_p_to_slice_id(ntp_p)
    w, h = cutils.get_czi_size(animal, slice_id)
    sm = parcellate(ntp_p, um_per_pixel=0.65, w=w, h=h, export_position_policy='none')
    assert isinstance(sm, SliceMeta)
    assert sm.cells is not None

    all_cells: pl.DataFrame = pl.concat([
        pl.DataFrame(sm.cells[c], schema=('x', 'y')).with_columns(
            pl.lit(c).alias('color'),
            pl.lit(slice_id).alias('slice_id'),
            pl.lit(animal).alias('animal')
        )
        for c in ('green', 'red', 'blue', 'yellow')
    ]).with_row_index()

    all_cells_regions: list[dict | None] = [None] * len(all_cells)

    region_groups = groupby(sm.regions, lambda x: x.label.name)
    polys = {
        k: unary_union([i.polygon for i in g])
        for k, g in region_groups
    }
    # print(polys.keys())

    target_regions = []
    cell_points = all_cells[['x', 'y']].to_numpy()
    for name, poly in polys.items():
        if poly.area > 2**26:
            continue
        target_regions.append(NTPRegion(NTPLabel(name, (0, 0)), poly))

        in_polys = in_polys_par(poly, cell_points)

        side = name[0].upper()
        if side not in ('R', 'L'):
            side = 'unknown'
        if side != 'unknown':
            strip_name = "-".join(name.split('-')[1:])
        else:
            strip_name = name

        for i, in_poly in enumerate(in_polys):
            if not in_poly: continue
            if all_cells_regions[i] is not None:
                print(f'[{ntp_p}] {i}, old: {all_cells_regions[i]} new: {name}, {all_cells[i, :]}')
                continue

            all_cells_regions[i] = {
                'index': i,
                'side': side,
                'region': strip_name,
                'full_region_name': name,
            }

    region_df = pl.DataFrame([i for i in all_cells_regions if i is not None])
    if len(region_df):
        all_cells = all_cells.join(region_df.cast({
            'index': pl.UInt32,
        }), on='index', how='left')
    else:
        all_cells = all_cells.with_columns(
            pl.lit(None).alias('side'),
            pl.lit(None).alias('region'),
            pl.lit(None).alias('full_region_name'),
        )

    all_cells = all_cells.cast({
        'x': pl.Float32,
        'y': pl.Float32,
        'slice_id': pl.Int32,
        'animal': pl.String,
        'color': pl.String,
        'side': pl.String,
        'region': pl.String,
        'full_region_name': pl.String,
    })
    return all_cells, target_regions


def find_slices_to_flip(ref_version_ps: list[Path], file_name: str):
    res: set[int] = set()

    for p in ref_version_ps:
        ref_p = p / file_name
        if not ref_p.exists():
            continue

        ref_df = pl.read_excel(ref_p)
        print(ref_df)
        if '列1' in ref_df.columns or 'flipLR' in ref_df.columns:
            ref_df = ref_df.drop_nulls()
            if '列1' in ref_df.columns:
                ref_df = ref_df.rename({'列1': 'flipLR'})
            ref_df = ref_df.cast(
                {'slice_id': pl.Int32, 'flipLR': pl.String}
            ).filter(pc('flipLR').str.len_chars() > 0)
    
            for i in ref_df['slice_id'].to_list():
                res.add(i)
    return res


def swap_lr(dic: dict[tuple[int, str], int], flip_slices: set[int]):
    res = dic.copy()
    for slice_id in flip_slices:
        for col_slice, col_name in dic.keys():
            if col_slice != slice_id:
                continue
            if col_name.startswith('R'):
                new_col = 'L' + col_name[1:]
            elif col_name.startswith('L'):
                new_col = 'R' + col_name[1:]
            else:
                raise ValueError(f'Invalid column name: {col_name}')

            if (slice_id, new_col) in dic:
                res[(slice_id, col_name)], res[(slice_id, new_col)] = dic[(slice_id, new_col)], dic[(slice_id, col_name)]
    return res

for animal in animals:
    target_file_names = {
        True: f"{animal}-%s-distinguish-rl.xlsx",
        False: f"{animal}-%s.xlsx",
    }
    if all((output_p / i).exists() for i in target_file_names.values()):
        continue

    res_dfs = []
    res_regions = []


    ntp_ps = list((base_ntp_p / animal).glob('*.ntp'))
    print(animal, len(ntp_ps))

    tasks = [delayed(extrace_cells)(ntp_p, animal) for ntp_p in ntp_ps]
    for i in tqdm(
        Parallel(64, return_as='generator_unordered')(tasks), 
        total=len(ntp_ps)
    ):
        res_dfs.append(i[0])
        res_regions.append(i[1])
    if res_dfs == []:
        continue

    res_df: pl.DataFrame = pl.concat(res_dfs).filter(pc('side').is_not_null())

    for distinguish_lr in (False, True):
        if distinguish_lr:
            region_col = 'full_region_name'
        else:
            region_col = 'region'
        for color in ('green', 'red', 'blue', 'yellow'):
            df = res_df.filter(pc('color') == color).group_by('animal', 'slice_id', region_col).agg(
                pc('index').count().alias('count')
            )
            total_df = df.group_by('animal', region_col).agg(
                pc('count').sum()
            ).with_columns(
                pl.lit(0).alias('slice_id'),
            ).select('animal', 'slice_id', region_col, 'count')
            if len(total_df) == 0:
                continue

            df = pl.concat([df, total_df]).pivot(
                index='slice_id', columns=region_col, values='count'
            ).sort('slice_id').fill_null(0)
            sorted_cols = sorted(df.columns[1:])
            df = df.select(['slice_id'] + sorted_cols)

            slice_cell_counts: dict[tuple[int, str], int] = {}
            for col in sorted_cols:
                for slice_id, count in df[['slice_id', col]].iter_rows():
                    slice_cell_counts[(slice_id, col)] = count

            file_name = target_file_names[distinguish_lr] % color
            target_p = output_p / file_name
            
            to_flip_slices = find_slices_to_flip(ref_version_ps, file_name)
            print(f'{to_flip_slices=}')
            new_slice_cell_counts = swap_lr(slice_cell_counts, to_flip_slices)

            cols_dict: defaultdict[str, list[int]] = defaultdict(list)
            for (slice_id, col), count in sorted(new_slice_cell_counts.items(), key=lambda x: x[0][0]):
                cols_dict[col].append(count)

            df = pl.DataFrame(cols_dict).with_columns(
                df['slice_id']
            ).select(['slice_id'] + sorted_cols)

            df.write_excel(output_p / file_name)



# %%

def find_slices_to_flip(ref_version_ps: list[Path], file_name: str):
    res: set[int] = set()

    for p in ref_version_ps:
        ref_p = p / file_name
        if not ref_p.exists():
            continue

        ref_df = pl.read_excel(ref_p)
        print(ref_df)
        if '列1' in ref_df.columns or 'flipLR' in ref_df.columns:
            ref_df = ref_df.drop_nulls()
            if '列1' in ref_df.columns:
                red_df = ref_df.rename({'列1': 'flipLR'})
            red_df = red_df.cast(
                {'slice_id': pl.Int32, 'flipLR': pl.String}
            ).filter(pc('flipLR').str.len_chars() > 0)
    
            for i in ref_df['slice_id'].to_list():
                res.add(i)
    return res

find_slices_to_flip(ref_version_ps, file_name)