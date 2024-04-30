# %%
import importlib
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import cast

import cairo
import connectome_utils as cutils
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory, Parallel, delayed
from matplotlib.axes import Axes
from ntp_manager import NTPLabel, NTPRegion, SliceMeta, parcellate
from scipy.interpolate import splev, splprep
from shapely import MultiPoint, Point, Polygon, geometry
from shapely.geometry.polygon import orient
from tqdm import tqdm
import polars as pl
from utils import color_to_dye, dye_to_color
import orjson

pc = pl.col


def draw_cnt(ctx: cairo.Context, cnt: np.ndarray, fill=False):
    ctx.move_to(*cnt[0])
    for i in cnt[1:]:
        ctx.line_to(*i)
    ctx.move_to(*cnt[0])
    if fill:
        ctx.fill()
    ctx.close_path()
    ctx.stroke()


memory = Memory(location='./cache', verbose=0)
# %%
ntp_root = Path('/mnt/90-connectome/finalNTP/')

output_p = Path('/mnt/97-macaque/projects/cla/foot-regist/')
output_p.mkdir(exist_ok=True, parents=True)
# %%
fix_animal = 'C042'
fix_side = 'L'

def read_ntp(animal_id: str, ntp_p: Path|str):
    slice_id = cutils.ntp_p_to_slice_id(ntp_p)
    w, h = cutils.get_czi_size(animal_id, slice_id)
    sm = parcellate(ntp_p, um_per_pixel=0.65, w=w, h=h, export_position_policy='none')
    assert isinstance(sm, SliceMeta)
    assert sm.cells is not None
    return ntp_p, sm
# %%

def find_foot_ntp(animal: str, ntp_root: Path) -> dict[str, list[str]]:
    foot_slices_p = ntp_root / animal / 'foot_slices.json'
    if foot_slices_p.exists():
        return orjson.loads(foot_slices_p.read_text())
    
    ntp_ps = cutils.find_connectome_ntp(animal, base_path=str(ntp_root))
    tasks = [delayed(read_ntp)(animal, p) for p in ntp_ps]

    foot_ntp_ps = []
    for res in Parallel(128, backend='threading', return_as='generator_unordered')(tasks):
        if res is None: continue
        ntp_p, sm = res
        sm = cast(SliceMeta, sm)
        ntp_p = cast(Path, ntp_p)

        if not sm.raw_labels: continue
        for name, p in sm.raw_labels:
            if 'foot' in name:
                foot_ntp_ps.append((name, ntp_p))
    res_dict: defaultdict[str, list[str]] = defaultdict(list)
    for name, p in foot_ntp_ps:
        res_dict[name].append(str(p))
    foot_slices_p.write_bytes(orjson.dumps(res_dict, option=orjson.OPT_INDENT_2))
    return res_dict

find_foot_ntp('C042', ntp_root)
# %%
mov_animal = 'C054'
foot_ntp_dict = find_foot_ntp(mov_animal, ntp_root)
# foot_ntp_dict = find_foot_ntp('C001')

foot_ntp_dict
# %%
fix_foot_ntp_dict = find_foot_ntp(fix_animal, ntp_root)
fix_foot_ntp_dict
# %%
def extract_foot_points(sm: SliceMeta, foot_name: str):
    for r in sm.regions:
        if foot_name == r.label.name:
            # orient_sign = 1.0 if foot_name.startswith('R') else -1.0
            # 右脑顺时针, 左脑逆时针
            # 不要用分区名称里的RL, 而是判断在中轴哪侧
            x, y = r.polygon.representative_point().coords[0]

            orient_sign = -1.0 if x > sm.w / 2 else 1.0
            points = np.array(orient(r.polygon, orient_sign).exterior.coords)
            return orient_sign, points

def find_corneal_points(points: np.ndarray, simplify: int=100):
    angles: list[tuple[int, np.ndarray, float]] = []
    p = Polygon(points)
    if simplify > 0:
        p = p.simplify(simplify, preserve_topology=False)

    sim_points = np.array(p.exterior.coords)
    sim_points = np.array([sim_points[-2], *sim_points])
    # print(f'{sim_points=}')
    for i in range(1, len(sim_points)-1):
        p1 = sim_points[i - 1]
        p2 = sim_points[i]
        p3 = sim_points[i + 1]

        v1 = p1 - p2
        v2 = p3 - p2

        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(angle)
        if np.isnan(angle): continue
        angles.append((i, p2, angle))
        # print(f'[{i}]', p2, angle, p1, p2, p3, v1, v2)

    # plt.plot(*sim_points.T, c='b', linestyle='--')
    # plt.plot(*points.T, c='g', alpha=0.4)
    # for _, p, angle in angles:
    #     print(p, angle)
    #     plt.scatter(*p, c='r', s=1000 / (abs(angle)+1))
    # plt.show()
    return angles

def extract_min_angle_points(points: np.ndarray, region_name: str, orient_sign: float):
    if len(points) < 2:
        raise ValueError("Not enough points to calculate angles.")
    
    # Check if region_name starts with 'R' and flip if necessary
    if orient_sign == -1:
        points[:, 0] *= -1

    for simp in range(0, 1000, 50):
        # print(simp)
        angles = find_corneal_points(points, simplify=simp)
        if len(angles) < 2:
            continue
        # print(angles)
        # plt.scatter(*points.T, c='g', alpha=0.4)
        # for _, p, a in angles:
        #     print(p, a)
        #     plt.scatter(*p, c='r', s=1000 / (abs(a)+1))

        sorted_angles = sorted(angles, key=lambda x: x[2])
        min_angle_points = sorted_angles[:2].copy()
        # print(len(sorted_angles), min_angle_points)
        for a in min_angle_points:
            if a[2] > 90 + simp / 10:
                break
        else:
            break
    else:
        raise ValueError(f"Angle({a[2]:.2f}) is too large, probably not a foot.")

    # print(min_angle_points)
    right_top = max(min_angle_points, key=lambda x: x[1][0] + x[1][1])
    left_bottom = min(min_angle_points, key=lambda x: x[1][0] + x[1][1])

    # Flip back if necessary
    if orient_sign == -1:
        right_top[1][0] *= -1
        left_bottom[1][0] *= -1
        points[:, 0] *= -1

    return right_top[1], left_bottom[1]

def split_line(points: np.ndarray, rt: np.ndarray, lb: np.ndarray):
    # print(points, rt, lb)
    equal_to_lb_index = np.where((points == lb).all(1))[0][0]
    equal_to_rt_index = np.where((points == rt).all(1))[0][0]
    equal_indexes = sorted((equal_to_rt_index, equal_to_lb_index))

    # print(f'{equal_indexes=}')
    line1 = points[equal_indexes[0]:equal_indexes[1] + 1]
    line2 = np.concatenate([points[equal_indexes[1]:], points[1:equal_indexes[0] + 1]]) # 这里第二部分线从1开始, 是因为points是回环的, 首位相同, 因此去除一个
    # plt.plot(*line1.T, c='b')
    # plt.plot(*line2.T, c='r')
    # plt.scatter(*rt, c='r')
    # plt.scatter(*lb, c='b')
    # plt.show()

    line1_spl_y = spl_line(line1, point_n=10)[:, 1]
    line2_spl_y = spl_line(line2, point_n=10)[:, 1]
    line1_spl_y.sort(); line2_spl_y.sort();
    
    if (line1_spl_y - line2_spl_y).mean() < 0:
        return line1, line2
    else:
        return line2, line1

def spl_line(points: np.ndarray, point_n: int=100):
    tck, u = splprep(points.T, s=0)
    x_new, y_new = splev(np.linspace(0, 1, point_n), tck)
    return np.array([x_new, y_new]).T

class FootSplitter:
    def __init__(self, sm: SliceMeta | None, foot_name: str, spl_n: int=20):
        assert sm is not None
        self.sm = sm
        res = extract_foot_points(self.sm, foot_name)
        assert res is not None
        self.orient_sign, self.points = res

        self.rt, self.lb = extract_min_angle_points(self.points, foot_name, self.orient_sign)
        self.top_line, self.bottom_line = split_line(self.points, self.rt, self.lb)
        
        self.top_line_spl = spl_line(self.top_line, spl_n)
        self.bottom_line_spl = spl_line(self.bottom_line, spl_n)

    def plot(self, ax: Axes | None = None, show_legend: bool=True):
        if ax is None:
            ax = plt.figure().add_subplot(111)
        ax.scatter(*self.rt, color='red', label='Right Top')
        ax.scatter(*self.lb, color='blue', label='Left Bottom')
        ax.plot(*self.top_line.T, c='b', label='Top Line')
        ax.plot(*self.bottom_line.T, c='r', label='Bottom Line')
        ax.plot(*self.top_line_spl.T, c='b', linestyle='--', label='Top SplLine')
        ax.plot(*self.bottom_line_spl.T, c='r', linestyle='--', label='Bottom SplLine')
        if show_legend:
            ax.legend()
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()




fig, axes = plt.subplots(1, 2, figsize=(8, 4))

mov_region_name = 'R-foot1-Cla'
mov_ntp_p = foot_ntp_dict[mov_region_name][0]
_, mov_sm = read_ntp(mov_animal, mov_ntp_p)
mov_fs = FootSplitter(mov_sm, mov_region_name)
mov_fs.plot(axes[0])

fix_region_name = 'L-foot1-Cla'
fix_ntp_p = fix_foot_ntp_dict[fix_region_name][0]
_, fix_sm = read_ntp(fix_animal, fix_ntp_p)
fix_fs = FootSplitter(fix_sm, fix_region_name)
fix_fs.plot(axes[1])
# %%
import regist_utils

importlib.reload(regist_utils)


class FootRegist:
    def __init__(self, mov_fs: FootSplitter, fix_fs: FootSplitter):
        self.mov_fs = mov_fs
        self.fix_fs = fix_fs

        self.mov_line = np.vstack([self.mov_fs.top_line_spl, self.mov_fs.bottom_line_spl])
        self.fix_line = np.vstack([self.fix_fs.top_line_spl, self.fix_fs.bottom_line_spl])

        self.itw = regist_utils.ItkTransformWarp(
            self.mov_line, self.fix_line
        )

    def plot(self, mesh_n=50):
        x = np.linspace(self.mov_line[:, 0].min() * 0.99, self.mov_line[:, 0].max() * 1.02, mesh_n)
        y = np.linspace(self.mov_line[:, 1].min() * 0.99, self.mov_line[:, 1].max() * 1.02, mesh_n)
        mesh_x, mesh_y = np.meshgrid(x, y)
        mesh_xy = np.stack([mesh_x.ravel(), mesh_y.ravel()]).T
        # mesh_xy = mesh_xy[mesh_xy[:,0].argsort()]

        moved_mesh_line = self.itw.transform_points_to_fix(mesh_xy)
        moved_mesh_line_colors = mpl.cm.jet(
            np.arange(len(moved_mesh_line)) % (mesh_n + 1) / (mesh_n + 1)
        )

        moved_line = self.itw.transform_points_to_fix(self.mov_line)

        plt.figure(figsize=(16, 4))
        plt.subplot(131)

        mov_fs.plot(plt.gca())
        plt.scatter(*mesh_xy.T, alpha=0.5, s=2, label='Moved Mesh Line', c=moved_mesh_line_colors)
        plt.title(f'Moving Line {Path(self.mov_fs.sm.ntp_path).stem}')
        plt.legend(bbox_to_anchor=(-0.5, 1), loc='upper left', borderaxespad=0.1)

        plt.subplot(132)

        fix_fs.plot(plt.gca())
        plt.title(f'Fixed Line {Path(self.fix_fs.sm.ntp_path).stem}')


        plt.subplot(133)
        plt.scatter(*moved_mesh_line.T, alpha=0.5, s=2, label='Moved Mesh Line', c=moved_mesh_line_colors)
        plt.plot(*moved_line.T, c='b', label='Moved Line', alpha=0.5)
        plt.plot(*self.fix_line.T, c='r', label='Fixed Line', alpha=0.5, linestyle='--')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        plt.title('Moved Line')

fr = FootRegist(mov_fs, fix_fs)
fr.plot()

# %%
output_root = Path('/mnt/97-macaque/projects/cla/foot-regist/')
output_root.mkdir(exist_ok=True, parents=True)

def find_target_region(sm: SliceMeta, name: str):
    for r in sm.regions:
        if r.label.name == name:
            return r
    return None

def color_name_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    return {
        'red'   : (0, 0, 255, alpha),
        'green' : (0, 255, 255, alpha),
        'blue'  : (255, 0, 0, alpha),
        'yellow': (0, 255, 0, alpha),
    }[color]

def draw_foot_regist(ctx: cairo.Context, fr: FootRegist, draw_colors: tuple[str, ...]=('red', 'green', 'blue', 'yellow')):
    # --- mov
    ctx.set_source_rgb(1, 0, 0)

    for region in fr.mov_fs.sm.regions:
        if 'cla' in region.label.name.lower():
            ctx.set_line_width(5)
            ctx.set_source_rgb(1, 0, 1)
        else:
            ctx.set_line_width(1)
            ctx.set_source_rgb(1, 0, 0)
        coords = np.array(region.polygon.exterior.coords)
        coords = fr.itw.transform_points_to_fix(coords)
        # draw_cnt(ctx, coords, fill='foot' in region.label.name)
        draw_cnt(ctx, coords, fill=False)
    
    ctx.set_source_rgb(1, 0, 0)
    ctx.set_line_width(1)

    # for line in fr.mov_fs.sm.raw_lines:
    #     coords = np.array(line.line.coords)
    #     coords = fr.itw.transform_points_to_fix(coords)

    #     draw_cnt(ctx, coords, fill=False)
    
    assert fr.mov_fs.sm.cells is not None
    for cell_color in draw_colors:
        for cell in fr.mov_fs.sm.cells[cell_color]:
            # print(cell, cell.reshape(-1, 2))
            ctx.set_source_rgba(*color_name_to_rgba(cell_color, alpha=200))
            cell = cell.reshape(-1, 2)
            cell = fr.itw.transform_points_to_fix(cell).reshape(-1)
            # print('trs cell', cell)
            ctx.arc(cell[0], cell[1], 50, 0, 2 * np.pi)
            ctx.fill()


animals = [i.name for i in ntp_root.glob('C*') if 'Pre' not in i.name and 'C030' not in i.name]
# %%
# 按照 四个 foot 拆分的
for fix_foot_name in fix_foot_ntp_dict.keys():
    if not fix_foot_name.startswith('L'): continue

    for fix_foot_name_index in range(len(fix_foot_ntp_dict[fix_foot_name])):
        fix_ntp_p = fix_foot_ntp_dict[fix_foot_name][fix_foot_name_index]
        _, fix_sm = read_ntp(fix_animal, fix_ntp_p)
        fix_fs = FootSplitter(fix_sm, fix_foot_name)
        output_path = output_root / f'{fix_foot_name}{fix_foot_name_index}-{Path(fix_sm.ntp_path).stem}.pdf'

        with cairo.PDFSurface(str(output_path), 5000, 5000) as surface:
            ctx = cairo.Context(surface)
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(5)
            ctx.scale(0.05, 0.05)
            for region in fix_fs.sm.regions:
                if 'cla' in region.label.name.lower():
                    ctx.set_line_width(1)
                    ctx.set_source_rgba(0, 0, 0.5, 0.5)
                else:
                    ctx.set_line_width(5)
                    ctx.set_source_rgba(0, 0, 0, 0.5)
                draw_cnt(ctx, np.array(region.polygon.exterior.coords), fill='foot' in region.label.name)

            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(0.1)

            for line in fr.fix_fs.sm.raw_lines:
                draw_cnt(ctx, np.array(line.line.coords), fill=False)

            for mov_animal in tqdm(animals):
                mov_foot_ntp_dict = find_foot_ntp(mov_animal, ntp_root)
                for side in ['R', 'L']:
                    mov_foot_name = fix_foot_name.replace('L', side)
                    ntp_ps = mov_foot_ntp_dict.get(mov_foot_name, [])
                    if not ntp_ps:
                        print('跳过', mov_animal, mov_foot_name, fix_foot_name)
                        continue
                    print(mov_animal, mov_foot_name)

                    _, sm = read_ntp(mov_animal, Path(ntp_ps[0]))
                    mov_fs = FootSplitter(sm, mov_foot_name)
                    fr = FootRegist(mov_fs, fix_fs)
                    draw_foot_regist(ctx, fr)
    #     break
    # break
# %%
# 每个单张导出的
output_path = output_root / f'{fix_animal}-foot-regist.pdf'
with cairo.PDFSurface(str(output_path), 10000, 10000) as surface:
    ctx = cairo.Context(surface)
    for mov_animal in tqdm(animals):
        try:

            mov_foot_sms: dict[str, SliceMeta] = {}

            mov_foot_ntp_dict = find_foot_ntp(mov_animal)

            for foot_name, sms in find_foot_ntp(mov_animal).items():
                # 对于 mov 的保留面积最大的那个 sm
                def _sort_key(x: SliceMeta):
                    target_region = find_target_region(x, foot_name)
                    if target_region is None:
                        return 0
                    return target_region.polygon.area
                sms.sort(key=_sort_key)
                _, sms[-1] = read_ntp(mov_animal, Path(sms[-1].ntp_path))
                mov_foot_sms[foot_name] = sms[-1]
            
            for mov_foot_name, mov_foot_sm in mov_foot_sms.items():
                fix_foot_name = mov_foot_name.replace('R', 'L')
                mov_fs = FootSplitter(mov_foot_sm, mov_foot_name)
                
                for fix_sm in fix_foot_ntp_dict[fix_foot_name]:
                    fix_fs = FootSplitter(fix_sm, fix_foot_name)
                    fr = FootRegist(mov_fs, fix_fs)
                    fr.plot()
                    plt.savefig(output_root / f'preview-{mov_animal}-{mov_foot_name}-{Path(fix_sm.ntp_path).stem}.png')
                    plt.close()
                    export_path = output_root / f'{mov_animal}-{mov_foot_name}-{Path(fix_sm.ntp_path).stem}.pdf'
                    draw_foot_regist(ctx, fr)
        except Exception as e:
            print(f'[{mov_animal}] {e}')
            continue
# %%
# 按照 四个 zone + 四个 foot 拆分的
cluster_df = pl.read_csv('./cluster_df.csv', new_columns=['zone_id', 'cluster'], has_header=False, separator='\t').select(
    pc('zone_id').str.replace("'", '').alias('zone_id'),
    'cluster'
).with_columns(
    pc('cluster').str.split('-').list.first().alias('cluster1'),
    pc('cluster').str.split('-').list.last().alias('cluster2'),
    pc('zone_id').str.split('-').list.get(1).replace(dye_to_color).alias('ch'),
    pc('zone_id').str.split('-').list.get(0).alias('animal_id'),
)
cluster_df
# %%

for fix_foot_name in fix_foot_ntp_dict.keys():
    if not fix_foot_name.startswith('L'): continue

    for fix_foot_name_index in range(len(fix_foot_ntp_dict[fix_foot_name])):
        fix_ntp_p = fix_foot_ntp_dict[fix_foot_name][fix_foot_name_index]
        _, fix_sm = read_ntp(fix_animal, fix_ntp_p)
        fix_fs = FootSplitter(fix_sm, fix_foot_name)

        for (cluster_key, ), df in cluster_df.group_by(['cluster1']):
            print(cluster_key)

            output_path = output_root / f'{cluster_key}-{fix_foot_name}{fix_foot_name_index}-{Path(fix_sm.ntp_path).stem}.pdf'

            with cairo.PDFSurface(str(output_path), 5000, 5000) as surface:
                ctx = cairo.Context(surface)
                ctx.set_source_rgb(0, 0, 0)
                ctx.set_line_width(5)
                ctx.scale(0.05, 0.05)
                for region in fix_fs.sm.regions:
                    if 'cla' in region.label.name.lower():
                        ctx.set_line_width(1)
                        ctx.set_source_rgba(0, 0, 0.5, 0.5)
                    else:
                        ctx.set_line_width(5)
                        ctx.set_source_rgba(0, 0, 0, 0.5)
                    draw_cnt(ctx, np.array(region.polygon.exterior.coords), fill='foot' in region.label.name)

                ctx.set_source_rgb(0, 0, 0)
                ctx.set_line_width(0.1)

                for line in fr.fix_fs.sm.raw_lines:
                    draw_cnt(ctx, np.array(line.line.coords), fill=False)

                for mov_animal in tqdm(animals):
                    if mov_animal not in df['animal_id']: continue

                    mov_foot_ntp_dict = find_foot_ntp(mov_animal, ntp_root)
                    for side in ['R', 'L']:
                        mov_foot_name = fix_foot_name.replace('L', side)
                        ntp_ps = mov_foot_ntp_dict.get(mov_foot_name, [])
                        if not ntp_ps:
                            # print('跳过', mov_animal, mov_foot_name, fix_foot_name)
                            continue
                        draw_colors = tuple(cluster_df.filter((pc('animal_id') == 'C054'))['ch'].to_list())
                        print(mov_animal, mov_foot_name, draw_colors)

                        _, sm = read_ntp(mov_animal, Path(ntp_ps[0]))
                        mov_fs = FootSplitter(sm, mov_foot_name)
                        fr = FootRegist(mov_fs, fix_fs)
                        draw_foot_regist(ctx, fr, draw_colors=draw_colors)
