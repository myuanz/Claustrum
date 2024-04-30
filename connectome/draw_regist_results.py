# %%
import itertools
import pickle
from collections import defaultdict
from io import StringIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
import polars as pl
from bidict import bidict
from bokeh.io import output_file, save
from bokeh.plotting import figure, output_notebook, show
from dataset_utils import get_base_path, read_regist_datasets
from joblib import Memory, Parallel, delayed
from loguru import logger
from tqdm import tqdm
from utils import AnimalSection, AnimalSectionWithDistance, PariRegistInfo, read_connectome_masks, read_exclude_sections, to_numpy, split_into_n_parts_slice, AnimalSectionSide
from scipy.interpolate import splev, splprep
from sklearn.neighbors import KernelDensity
from functools import cache
from nhp_utils.image_correction import CorrectionPara
from cla_gui.synthesizer import get_connectome_corr_para_from_file

pc = pl.col

PROJECT_NAME = 'CLA'
assets_path = Path('/mnt/97-macaque/projects/cla/injections-cells/assets/raw_image/')
exclude_slices = read_exclude_sections(PROJECT_NAME)
memory = Memory('./cache', verbose=0)

# %%
fix_regist_datasets = read_regist_datasets(
    'C042', 0.1, exclude_slices=exclude_slices, force=False, 
    tqdm=tqdm, target_shape=(128, 128)
)
fix_regist_datasets = sorted(fix_regist_datasets.items(), key=lambda x: x[0].slice_id_int)

# %%
# regist_results = pickle.load(open(assets_path.parent / 'regist_results.pkl', 'rb'))
# regist_results = pickle.load(open(assets_path.parent / 'regist_results_Mq179.pkl', 'rb'))
regist_results = pickle.load(open(assets_path.parent / 'regist_results_20240129_connectome_to_C042.pkl', 'rb'))



regist_results = [i for i in regist_results if isinstance(i, PariRegistInfo)]
# %%
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

p = '/mnt/97-macaque/projects/cla/injections-cells/Combine-20240202.xlsx'
p = '/mnt/97-macaque/projects/cla/injections-cells/Combine-20240306.xlsx'

input_df = pd.read_excel(p).rename(columns={
    'Combine3': 'combine',
    'Combine_area': 'combine_area', 
    'Animal': 'animal_id',
    'injectionSites': 'region',
    'hemisphere': 'side',
    'Dye': 'tracer',
    'draw_color': 'draw_color', 
    'Draw_Seq': 'draw_seq',
}).sort_values(['combine', 'animal_id', 'region'])


input_df['color'] = input_df['tracer'].map(dye_to_color)
input_df['side'] = input_df['side'].map({'L': 'left', 'R': 'right', 'left': 'left', 'right': 'right'})
input_df['animal_id_side'] = input_df['animal_id'] + '-' + input_df['side']
input_df['fetch_id'] = input_df['animal_id_side'] + '-' + input_df['color']

# CLA 每一个单独视为一组
# cla_rows = input_df['combine_area'].str.contains('CLA').fillna(False)
# input_df.loc[cla_rows, 'combine_area'] += '_' + input_df[cla_rows]['animal_id'] + '_' + input_df['side']

# input_df['combine'] = 'all-combine'
# target_info
input_df[input_df['combine'].notna()]
# input_df[cla_rows]
# %%

%load_ext pyinstrument
# %%pyinstrument
# 现在基本不再导出网页
# 导出 bokeh 网页形式结果的
# per_image = 8
# delta_image_distance = 0

# per_image = 1
# delta_image_distance = 150


# for combine_name, meta_group in input_df.groupby('combine'):
#     if not isinstance(combine_name, str) and not isinstance(combine_name, float): continue
#     # if combine_name != '8l': continue
#     print(f'{combine_name=}')
#     image_size = 128
#     export_version = f'20240306-groupby-inj-no-warp-per{per_image}-d{delta_image_distance}-{combine_name}-raw-merge-42'
#     export_path = assets_path.parent.parent / f'{export_version}.html'
#     # export_path = Path('./output/') / f'{export_version}.html'

#     export_path.parent.mkdir(exist_ok=True, parents=True)

#     target_info = meta_group.to_dict(orient='list')

#     all_animal_ids = [
#         i[0] for i in itertools.groupby(regist_results, lambda x: x.mov_sec.animal_id)
#         if i[0] in target_info['animal_id']
#     ]
#     hsv_colors = dict(zip(
#         all_animal_ids, 
#         [tuple(i[:3]) for i in (plt.cm.hsv(np.linspace(0, 1, len(all_animal_ids))) * 255).astype(int).tolist()]
#     ))


#     p = figure(width=2400, height=1000)

#     imgs = defaultdict(list)
#     labels = defaultdict(list)
#     dxs = defaultdict(list)
#     record_cells = defaultdict(list)

#     cells = defaultdict(list)

#     # for sec, item in fix_regist_datasets:
#     #     if sec.slice_id_int % per_image != 0: continue

#     #     dx = sec.slice_id_int - fix_regist_datasets[0][0].slice_id_int

#     #     imgs['url'].append(
#     #         (f'assets/raw_image/{sec.animal_id}/{sec.animal_id}-left-{sec.slice_id}.png')
#     #     )
#     #     imgs['x'].append((dx * (image_size - delta_image_distance)) / per_image)
#     #     imgs['y'].append(0)
#     #     imgs['w'].append(image_size)
#     #     imgs['h'].append(image_size)
#     #     imgs['alpha'].append(0.5)


#     #     labels['x'].append(
#     #         (dx * (image_size - delta_image_distance)) / per_image + (image_size - delta_image_distance) * 0.8
#     #     )
#     #     labels['y'].append(-10)
#     #     labels['text'].append(f'{sec.animal_id}-{sec.slice_id}')
#     #     labels['text_color'].append('black')
#     #     labels['text_font_size'].append('10pt')


#     dy_count = 1
#     for mov_animal_index, (mov_animal_id, group) in enumerate(itertools.groupby(regist_results, lambda x: x.mov_sec.animal_id)):
#         if mov_animal_id not in target_info['animal_id']: continue

#         legal_df = meta_group.animal_id == mov_animal_id
#         legal_color = set(meta_group[legal_df].color)
#         legal_side = set(meta_group[legal_df].side)
#         legal_region = set(meta_group[legal_df].region)

#         group = sorted(group, key=lambda x: x.mov_sec.slice_id)

#         print(mov_animal_id, len(group))

#         regist_datasets = read_regist_datasets(mov_animal_id, 0.1, exclude_slices=[], force=False, tqdm=tqdm, target_shape=(128, 128))
#         curr_cells = {
#             'left': [],
#             'right': [],
#         }
#         # break
#         for i, pair_regist_info in enumerate(group):
#             # side = 'right' if pair_regist_info.mov_sec.need_flip else 'left'
#             side = pair_regist_info.mov_sec.side
#             if pair_regist_info.fix_sec.slice_id_int % per_image != 0: continue
#             if f'{mov_animal_id}-{pair_regist_info.mov_sec.side}' not in target_info['animal_id_side']: continue

#             mov_image    = pair_regist_info.mov_sec.image

#             mov_cells_df = find_cell(regist_datasets, pair_regist_info.mov_sec).filter(
#                 pc('color').is_in(legal_color)
#             )
#             # print(f'{mov_animal_id=} {len(mov_cells_df)=} {side=}')


#             mov_cells = mov_cells_df.select('x', 'y').to_numpy().copy()
#             # mov_cells = mov_cells[
#             #     mov_image[mov_cells[:, 1], mov_cells[:, 0]] != 0
#             # ]

#             if len(mov_cells) < 2: continue
#             # print(mov_cells.shape)
#             dx = (pair_regist_info.fix_sec.slice_id_int - group[0].fix_sec.slice_id_int)
#             if pair_regist_info.mov_sec.need_flip:
#                 mov_cells[:, 0] = mov_image.shape[1] - mov_cells[:, 0]

#             mov_cells = pair_regist_info.transform_points(mov_cells)

#             mov_cells_int = mov_cells.astype(int)
#             # mov_cells = mov_cells[
#             #     (mov_cells_int[:, 0] >= 0) & (mov_cells_int[:, 0] < image_size) &
#             #     (mov_cells_int[:, 1] >= 0) & (mov_cells_int[:, 1] < image_size)
#             # ]

#             mov_cells = mov_cells[
#                 pair_regist_info.warped_image[mov_cells_int[:, 1], mov_cells_int[:, 0]] != 0
#             ]
#             # mov_cells = mov_cells[
#             #     pair_regist_info.mov_sec.image[mov_cells_int[:, 1], mov_cells_int[:, 0]] != 0
#             # ]

#             mov_cells[:, 0] += ((dx * (image_size - delta_image_distance)) / per_image)
#             # mov_cells[:, 1] *= -1
#             curr_cells[side].append(mov_cells)
#             record_cells['animal_id'].append(mov_animal_id)
#             record_cells['slice_id'].append(pair_regist_info.mov_sec.slice_id)
#             record_cells['side'].append(side)
#             record_cells['cells'].append(mov_cells)


#             imgs['url'].append(f'./assets/raw_image/{pair_regist_info.fix_sec.animal_id}/warped/{mov_animal_id}/{pair_regist_info.fix_sec.animal_id}-left-{pair_regist_info.fix_sec.slice_id}-warped-{mov_animal_id}-{side}-{pair_regist_info.mov_sec.slice_id}.png')

#             # imgs['url'].append(pair_regist_info.mov_sec.image_path.replace('/home/myuan/projects/cla/pwvy/output/', ''))

#             # imgs['image'].append(pair_regist_info.mov_sec.image)
#             imgs['x'].append((dx * (image_size - delta_image_distance)) / per_image)
#             imgs['y'].append((dy_count) * image_size)
#             imgs['w'].append(image_size)
#             imgs['h'].append(image_size)
#             imgs['alpha'].append(0.1)

#             labels['x'].append(
#                 (dx * (image_size - delta_image_distance)) / per_image + (image_size - delta_image_distance) * 0.8
#             )
#             labels['y'].append(-10 * (mov_animal_index + 2))
#             labels['text'].append(f'{mov_animal_id}-{pair_regist_info.mov_sec.slice_id}-{side}')
#             labels['text_color'].append(hsv_colors[mov_animal_id])
#             labels['text_font_size'].append('10pt')

#             dxs['final'].append((dx * (image_size - delta_image_distance)) / per_image)
#             dxs['raw'].append(dx)

#         for side in curr_cells:
#             if not curr_cells[side]: continue
#             ps = np.vstack(curr_cells[side])
#             # p.circle(x=ps[:, 0], y=ps[:, 1], color=hsv_colors[mov_animal_id], legend_label=f'{mov_animal_id}-{side}', alpha=0.5)
#             cells['x'].append(ps[:, 0])
#             cells['y'].append(ps[:, 1] + (dy_count) * image_size)
#             cells['color'].append(hsv_colors[mov_animal_id])
#             cells['legend_label'].append(f'{mov_animal_id}-{side}-{"|".join(legal_region)}')
#             cells['alpha'].append(0.5)
#         # break
#         # dy_count += 1

#     p.image_url(
#         url=imgs['url'],
#         x=imgs['x'],
#         y=imgs['y'],
#         w=imgs['w'],
#         h=imgs['h'],
#         alpha=imgs['alpha'],
#     )
#     # p.image(
#     #     image=imgs['image'],
#     #     x=imgs['x'],
#     #     y=imgs['y'],
#     #     dw=imgs['w'],
#     #     dh=imgs['h'],
#     #     alpha=imgs['alpha'],
#     # )

#     p.text(
#         x=labels['x'],
#         y=labels['y'],
#         text=labels['text'],
#         text_color=labels['text_color'],
#         text_font_size=labels['text_font_size'],
#     )
#     for i in range(len(cells['x'])):
#         p.circle(
#             x=cells['x'][i],
#             y=cells['y'][i],
#             color=cells['color'][i],
#             legend_label=cells['legend_label'][i],
#             alpha=cells['alpha'][i],
#             # set size 
#             radius=0.3,
#         )


#     p.legend.location = "top_left"
#     p.legend.click_policy = "hide"
#     p.match_aspect = True
#     p.y_range.flipped = True


#     export_path.unlink(missing_ok=True)
#     output_file(export_path)
#     save(p, export_path)

#     to_save = {
#         # 'cells': cells,
#         'record_cells': record_cells,
#         'imgs': imgs,
#         'labels': labels,
#         'dxs': dxs
#     }
#     pickle.dump(to_save, open(export_path.parent / f'saved-{export_version}.pkl', 'wb'))
#     # break

# # show(p)
# # save to html


# %%
# 所以这个也不再导入了
# 导出平铺配准图片的
# for combine_name, meta_group in input_df.groupby('combine'):
#     export_version = f'20231203-groupby-inj-no-warp-per{per_image}-d{delta_image_distance}-{combine_name}-raw-merge-42'
#     (assets_path.parent.parent / export_version).mkdir(exist_ok=True, parents=True)

#     info = pickle.load(open(assets_path.parent.parent / f'saved-{export_version}.pkl', 'rb'))
#     imgs = defaultdict(lambda: {
#         'left': [],
#         'right': [],
#     })
    
#     for i in range(len(info['imgs']['url'])):
#         animal_id = info['record_cells']['animal_id'][i]
#         slice_id = info['record_cells']['slice_id'][i]
#         side = info['record_cells']['side'][i]
#         dx = info['dxs']['final'][i]

#         cells = info['record_cells']['cells'][i]
#         cells[:, 0] -= dx
#         cells = cells.astype(int)

#         img = cv2.imread(str(
#             assets_path.parent.parent / info['imgs']['url'][i].replace('./', '')
#         ), -1)
#         cnts, _ = cv2.findContours(
#             img, 
#             cv2.RETR_EXTERNAL, 
#             cv2.CHAIN_APPROX_SIMPLE
#         )
#         if len(cnts) != 1:
#             print(animal_id, slice_id, len(cnts))
#             continue
#         b = cv2.boundingRect(cnts[0])
        
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         for cell in cells:
#             cv2.circle(img, tuple(cell), 1, (0, 0, 255), -1)
#         imgs[animal_id][side].append(img[:, b[0]:b[0]+b[2]])
#         # break
    
#     for animal_id in imgs:
#         for side in imgs[animal_id]:
#             if not imgs[animal_id][side]: continue
#             curr_img = np.hstack(imgs[animal_id][side])
#             cv2.imwrite((assets_path.parent.parent / export_version / f"{animal_id}-{side}-{combine_name}.png").as_posix(), curr_img)
#     # break
# %%

# 导出矢量分区和细胞位置的
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

@memory.cache(ignore=['datasets'])
def get_mask_cnts_and_raw(
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
    return mask, raw_cnt, smooth_cnt


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
per_image = 8
delta_image_distance = 0


p配准时缩放 = 0.1
v展示时缩放 = 0.5
j计算缩放 = int(v展示时缩放 / p配准时缩放)
per_image = 1
delta_image_distance = 110 * j计算缩放
image_size = 128 * j计算缩放

regist_results = pickle.load(open(assets_path.parent / 'regist_results_20240129_connectome_to_C042.pkl', 'rb'))
regist_to_mq179_results = pickle.load(open(assets_path.parent / 'regist_results_20240118_C042_to_Mq179.pkl', 'rb'))

regist_results = [i for i in regist_results if isinstance(i, PariRegistInfo)]
regist_to_mq179_results = [i for i in regist_to_mq179_results if isinstance(i, PariRegistInfo)]
# %%
stereo_masks = pickle.load(open(assets_path.parent / 'stereo_masks_20240118.pkl', 'rb'))
stereo_masks = {v['chip']: v for v in stereo_masks}

# %%
# 此处不配准到 mq179，不加dx，读入C042的矫正文件以平铺
fix_slices = set([])

C042_corr_conf_p = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240306-wml/C042-L/DESKTOP-40EUC53-config.yaml')

for combine_name, meta_group in input_df.groupby('combine'):
    # 如果是 nan 则跳过
    
    # if combine_name != 3: continue
    print(f'{combine_name=} {np.isnan(combine_name)}')
    
    export_version = f'20240306-no-warp-per{per_image}-d{delta_image_distance}-{combine_name}-raw-merge-42'
    export_path = assets_path.parent.parent / f'{export_version}.html'
    # export_path = Path('./output/') / f'{export_version}.html'

    export_path.parent.mkdir(exist_ok=True, parents=True)

    target_info = meta_group.to_dict(orient='list')

    all_animal_ids = [
        i[0] for i in itertools.groupby(regist_results, lambda x: x.mov_sec.animal_id)
        if i[0] in target_info['animal_id']
    ]
    hsv_colors = dict(zip(
        all_animal_ids, 
        [tuple(i[:3]) for i in (plt.cm.hsv(np.linspace(0, 1, len(all_animal_ids))) * 255).astype(int).tolist()]
    ))


    p = figure(width=2400, height=1000)

    dxs = defaultdict(list)
    records = defaultdict(list)

    # for sec, item in fix_regist_datasets:
    #     if sec.slice_id_int % per_image != 0: continue

    #     dx = sec.slice_id_int - fix_regist_datasets[0][0].slice_id_int

    dy_count = 1
    for mov_animal_index, (mov_animal_id_side, group) in enumerate(itertools.groupby(regist_results, lambda x: f'{x.mov_sec.animal_id}-{x.mov_sec.side}')):
        if mov_animal_id_side not in target_info['animal_id_side']: continue
        mov_animal_id = mov_animal_id_side.split('-')[0]

        legal_df = meta_group.animal_id == mov_animal_id
        legal_color = set(meta_group[legal_df].color)
        legal_side = set(meta_group[legal_df].side)
        legal_region = set(meta_group[legal_df].region)

        group = sorted(group, key=lambda x: x.mov_sec.slice_id)
        # group_indexs = split_into_n_parts_slice(group, 21)
        # group = [group[i] for i in group_indexs]

        print(mov_animal_id, len(group))
        print(meta_group[legal_df])
        continue

        regist_datasets = read_regist_datasets(
            mov_animal_id, v展示时缩放, exclude_slices=exclude_slices, 
            target_shape=(128*j计算缩放, 128*j计算缩放)
        )

        for i, pair_regist_info in enumerate(group):
            to_mq179_pair = [
                i for i in regist_to_mq179_results
                if (
                    i.mov_sec.animal_id == pair_regist_info.fix_sec.animal_id
                    and i.mov_sec.slice_id == pair_regist_info.fix_sec.slice_id
                    and i.mov_sec.side == pair_regist_info.fix_sec.side
                )
            ]
            if not to_mq179_pair:
                print(f'missing: {pair_regist_info.mov_sec}')
                continue

            best_to_mq179_pair = max(to_mq179_pair, key=lambda x: x.iou)

            # side = 'right' if pair_regist_info.mov_sec.need_flip else 'left'
            side = pair_regist_info.mov_sec.side
            if pair_regist_info.fix_sec.slice_id_int % per_image != 0: continue
            if f'{mov_animal_id}-{pair_regist_info.mov_sec.side}' not in target_info['animal_id_side']: continue
            fix_slices.add(pair_regist_info.fix_sec.slice_id)

            mov_image    = pair_regist_info.mov_sec.image
            mov_mask, mov_cells_df = regist_datasets[AnimalSection(
                animal_id=pair_regist_info.mov_sec.animal_id,
                slice_id=pair_regist_info.mov_sec.slice_id,
            )]['pad_res'][side]

            raw_cnt, smooth_cnt = get_mask_cnts(
                pair_regist_info.fix_sec.animal_id, 
                pair_regist_info.fix_sec.slice_id, 
                pair_regist_info.fix_sec.side
            )
            raw_cnt = raw_cnt.astype(np.float32)
            smooth_cnt = smooth_cnt.astype(np.float32)


            mov_cells_df = mov_cells_df.filter(
                pc('color').is_in(legal_color)
            )
            # print(f'{mov_animal_id=} {len(mov_cells_df)=} {side=}')


            mov_cells = mov_cells_df.select('x', 'y').to_numpy().copy()
            # mov_cells = mov_cells[
            #     mov_image[mov_cells[:, 1], mov_cells[:, 0]] != 0
            # ]

            if len(mov_cells) < 2: continue
            # print(mov_cells.shape)
            dx = (pair_regist_info.fix_sec.slice_id_int - group[0].fix_sec.slice_id_int)
            if pair_regist_info.mov_sec.need_flip:
                mov_cells[:, 0]  = mov_image.shape[1] * j计算缩放 - mov_cells[:, 0]
                # raw_cnt[:, 0]    = mov_image.shape[1] * j计算缩放 - raw_cnt[:, 0]
                # smooth_cnt[:, 0] = mov_image.shape[1] * j计算缩放 - smooth_cnt[:, 0]

            mov_cells = pair_regist_info.transform_points(mov_cells / j计算缩放)
            # mov_cells_to_mq179 = best_to_mq179_pair.transform_points(mov_cells / j计算缩放) * j计算缩放


            mov_cells_int = (mov_cells).astype(int)
            mov_cells = mov_cells[
                pair_regist_info.warped_image[mov_cells_int[:, 1], mov_cells_int[:, 0]] != 0
            ]
            mov_cells *= j计算缩放

            # smooth_cnt = pair_regist_info.transform_points(smooth_cnt / j计算缩放) * j计算缩放
            # raw_cnt = pair_regist_info.transform_points(raw_cnt / j计算缩放) * j计算缩放

            mov_cells[:, 0] += ((dx * (image_size - delta_image_distance)) / per_image)
            smooth_cnt[:, 0] += ((dx * (image_size - delta_image_distance)) / per_image)
            raw_cnt[:, 0] += ((dx * (image_size - delta_image_distance)) / per_image)

            records['animal_id'].append(mov_animal_id)
            records['slice_id'].append(pair_regist_info.mov_sec.slice_id)
            records['side'].append(side)
            records['cells'].append(mov_cells)
            records['smooth_cnt'].append(smooth_cnt)
            records['raw_cnt'].append(raw_cnt)
            # records['mask'].append(pair_regist_info.warped_image)

            dxs['final'].append((dx * (image_size - delta_image_distance)) / per_image)
            dxs['raw'].append(dx)

        # break
        dy_count += 1

    to_save = {
        # 'cells': cells,
        'record_cells': records,
        'dxs': dxs
    }
    pickle.dump(to_save, open(export_path.parent / f'saved-{export_version}.pkl', 'wb'))
    # break

# show(p)
# save to html

# %%

mov_animal_to_regist_results = defaultdict(list)
for i in regist_results:
    mov_animal_to_regist_results[(i.mov_sec.animal_id, i.mov_sec.side)].append(i)
C042_corr_conf_p = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240306-wml/C042-L/DESKTOP-40EUC53-config.yaml')

corr = get_connectome_corr_para_from_file(C042_corr_conf_p, 'C042', 182, 100)
corr
# %%
# fetch_data_from_group
    
meta_group_item = meta_group.iloc[0] # combine	draw_seq	draw_color	combine_area	animal_id	region	side	tracer	color	animal_id_side	fetch_id
mov_animal_id = meta_group_item['animal_id']
mov_color = meta_group_item['color']
# %%
raw_datasets = read_regist_datasets(
    mov_animal_id, v展示时缩放, exclude_slices=exclude_slices, 
    target_shape=(128*j计算缩放, 128*j计算缩放)
)
# %%
C042_corr_conf_p = Path('/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240306-wml/C042-L/myuan-precision3650tower-config.yaml')

l_regist_results = mov_animal_to_regist_results[(meta_group_item['animal_id'], meta_group_item['side'])]
for pair_regist_info in l_regist_results:
    
    side      = pair_regist_info.mov_sec.side
    mov_image = pair_regist_info.mov_sec.image
    raw_item  = raw_datasets[pair_regist_info.mov_sec.to_sec()]

    mov_mask, mov_cells_df = raw_item['pad_res'][side]
    mov_cells_df = mov_cells_df.filter(pc('color') == mov_color)


    fix_mask, raw_cnt, smooth_cnt = get_mask_cnts_and_raw(
        pair_regist_info.fix_sec.animal_id, 
        pair_regist_info.fix_sec.slice_id, 
        pair_regist_info.fix_sec.side
    )
    cv2.imwrite(
        f'/mnt/97-macaque/projects/cla/connectome-cla-highlight-20240306-wml/C042-L-pad/C042-{pair_regist_info.fix_sec.slice_id}.png',
        cv2.resize((fix_mask * 255).astype(np.uint8), (128, 128))
    )
    
    raw_cnt    = raw_cnt.astype(np.float32)
    smooth_cnt = smooth_cnt.astype(np.float32)

    mov_cells = mov_cells_df.select('x', 'y').to_numpy().copy()
    if len(mov_cells) < 2: continue
    mov_cells = pair_regist_info.transform_points(mov_cells / j计算缩放)
    mov_cells_int = (mov_cells).astype(int)
    mov_cells = mov_cells[
        pair_regist_info.warped_image[mov_cells_int[:, 1], mov_cells_int[:, 0]] != 0
    ]
    mov_cells *= j计算缩放

    corr = get_connectome_corr_para_from_file(
        C042_corr_conf_p, 'C042', 
        pair_regist_info.fix_sec.slice_id_int, 
        int(20) # 20 是分割时的缩放
    )
    assert corr is not None
    print(pair_regist_info.fix_sec.slice_id_int, corr.left)

    mov_cells = corr.warp_point(mov_cells)
    smooth_cnt = corr.warp_point(smooth_cnt)
    # mov_cells[:, 0] += corr.left
    # smooth_cnt[:, 0] += corr.left

    plt.scatter(*mov_cells.T, s=1)
    plt.plot(*smooth_cnt.T)
    # plt.plot(*smooth_cnt_new.T, c='black')
    

    # break

plt.gca().invert_yaxis()
# %%
# 此后不加 dx 并且配准到mq179，用于给朱智勇绘图

group_len = len(input_df['combine_area'].unique())

hsv_colors = np.array([tuple(i[:3]) for i in (plt.cm.hsv(np.linspace(0, 1, group_len)) * 255).astype(int).tolist()])
hsv_colors
# %%

@cache
def corr_select(chip: str):
    return CorrectionPara.select(chip)[0]
# %%
def record_to_df(records):
    dfs = []

    for i in range(len(records['cells'])):
        target_chip = records['target_chip'][i]
        target_chip = f'T{target_chip}'
        corr = corr_select(target_chip)
        rxry = corr.wrap_point(records['cells'][i])

        _df = pl.from_numpy(records['cells'][i], schema=['x', 'y']).with_columns(
            pl.lit(records['animal_id'][i]).alias('animal_id'),
            pl.lit(records['slice_id'][i]).alias('slice_id'),
            pl.lit(records['side'][i]).alias('side'),

            pl.lit(records['target_chip'][i]).alias('target_chip'),
            pl.lit(records['group_name'][i]).alias('group_name'),
            pl.lit(rxry[:, 0]).alias('rx'),
            pl.lit(rxry[:, 1]).alias('ry'),
            pl.lit(records['draw_color_key'][i]).alias('draw_color_key'),
        )
        dfs.append(_df)

    df = pl.concat(dfs)
    return df

all_records = set([tuple(i.values()) for i in input_df[['animal_id', 'tracer', 'combine_area']].to_dict(orient='records')])


for (combine_name, meta_group), group_color in zip(input_df.groupby('combine_area'), hsv_colors):
    # 如果是 nan 则跳过
    
    # if combine_name != '8l': continue
    print(f'{combine_name=}')
    
    export_version = f'20240201-no-warp-per{per_image}-d{delta_image_distance}-raw-merge-42'
    export_path = assets_path.parent.parent / export_version
    export_path.mkdir(exist_ok=True, parents=True)

    target_info = meta_group.to_dict(orient='list')

    # all_animal_ids = [
    #     i[0] for i in itertools.groupby(regist_results, lambda x: x.mov_sec.animal_id)
    #     if i[0] in target_info['animal_id']
    # ]
    # hsv_colors = dict(zip(
    #     all_animal_ids, 
    #     [tuple(i[:3]) for i in (plt.cm.hsv(np.linspace(0, 1, len(all_animal_ids))) * 255).astype(int).tolist()]
    # ))


    dxs = defaultdict(list)
    records = defaultdict(list)

    for mov_animal_index, (mov_animal_id, group) in enumerate(itertools.groupby(regist_results, lambda x: x.mov_sec.animal_id)):
        if mov_animal_id not in target_info['animal_id']: continue

        legal_df = meta_group.animal_id == mov_animal_id
        legal_color = set(meta_group[legal_df].color)
        legal_side = set(meta_group[legal_df].side)
        legal_region = set(meta_group[legal_df].region)
        legal_tracer = set(meta_group[legal_df].tracer)

        # draw_color = meta_group[legal_df].draw_color.iloc[0]
        draw_color = group_color

        group = sorted(group, key=lambda x: x.mov_sec.slice_id)
        print(mov_animal_id, f"{len(group)=} {len(legal_region)=}") # TODO 处理这里同一只动物多个相同注射位点的情况

        regist_datasets = read_regist_datasets(
            mov_animal_id, v展示时缩放, exclude_slices=exclude_slices, 
            target_shape=(128*j计算缩放, 128*j计算缩放)
        )

        for c in legal_tracer:
            record = (mov_animal_id, c, combine_name)
            if record not in all_records: 
                print(f'-----------{record} not in all_records')
            else:
                all_records.remove(record)

        for i, pair_regist_info in enumerate(group):
            to_mq179_pair = [
                i for i in regist_to_mq179_results
                if (
                    i.mov_sec.animal_id == pair_regist_info.fix_sec.animal_id
                    and i.mov_sec.slice_id == pair_regist_info.fix_sec.slice_id
                    and i.mov_sec.side == pair_regist_info.fix_sec.side
                )
            ]
            if not to_mq179_pair:
                print(f'missing: {pair_regist_info.mov_sec}')
                continue

            best_to_mq179_pair = max(to_mq179_pair, key=lambda x: x.iou)

            # side = 'right' if pair_regist_info.mov_sec.need_flip else 'left'
            side = pair_regist_info.mov_sec.side
            if pair_regist_info.fix_sec.slice_id_int % per_image != 0: continue
            if f'{mov_animal_id}-{pair_regist_info.mov_sec.side}' not in target_info['animal_id_side']: continue

            mov_image    = pair_regist_info.mov_sec.image
            mov_mask, mov_cells_df = regist_datasets[AnimalSection(
                animal_id=pair_regist_info.mov_sec.animal_id,
                slice_id=pair_regist_info.mov_sec.slice_id,
            )]['pad_res'][side]

            raw_cnt, smooth_cnt = get_mask_cnts(
                pair_regist_info.fix_sec.animal_id, 
                pair_regist_info.fix_sec.slice_id, 
                pair_regist_info.fix_sec.side
            )
            raw_cnt = raw_cnt.astype(np.float32)
            smooth_cnt = smooth_cnt.astype(np.float32)


            mov_cells_df = mov_cells_df.filter(
                pc('color').is_in(legal_color)
            )
            # print(f'{mov_animal_id=} {len(mov_cells_df)=} {side=}')


            mov_cells = mov_cells_df.select('x', 'y').to_numpy()
            # mov_cells = mov_cells[
            #     mov_image[mov_cells[:, 1], mov_cells[:, 0]] != 0
            # ]

            if len(mov_cells) < 2: continue
            # print(mov_cells.shape)
            dx = (pair_regist_info.fix_sec.slice_id_int - group[0].fix_sec.slice_id_int)
            if pair_regist_info.mov_sec.need_flip:
                mov_cells[:, 0]  = mov_image.shape[1] * j计算缩放 - mov_cells[:, 0]
                # raw_cnt[:, 0]    = mov_image.shape[1] * j计算缩放 - raw_cnt[:, 0]
                # smooth_cnt[:, 0] = mov_image.shape[1] * j计算缩放 - smooth_cnt[:, 0]

            mov_cells = pair_regist_info.transform_points(mov_cells / j计算缩放) * j计算缩放
            mov_cells_to_mq179 = best_to_mq179_pair.transform_points(mov_cells / j计算缩放)


            mov_cells_int = (mov_cells).astype(int)
            outputside_mask = mov_cells_int[:, 1] != 0

            mov_cells_with_ones = np.hstack([mov_cells_to_mq179, np.ones((len(mov_cells), 1))])
            m = stereo_masks[f'T{best_to_mq179_pair.fix_sec.slice_id}']['transformation_matrix']
            m = np.linalg.inv(m)
            mov_cells_in_stereo = (m @ mov_cells_with_ones.T).T[:, :2]

            # mov_cells_in_stereo *= j计算缩放

            # smooth_cnt = pair_regist_info.transform_points(smooth_cnt / j计算缩放) * j计算缩放
            # raw_cnt = pair_regist_info.transform_points(raw_cnt / j计算缩放) * j计算缩放

            # mov_cells[:, 0] += ((dx * (image_size - delta_image_distance)) / per_image)
            # smooth_cnt[:, 0] += ((dx * (image_size - delta_image_distance)) / per_image)
            # raw_cnt[:, 0] += ((dx * (image_size - delta_image_distance)) / per_image)

            records['animal_id'].append(mov_animal_id)
            records['slice_id'].append(pair_regist_info.mov_sec.slice_id)
            records['side'].append(side)
            records['cells'].append(mov_cells_in_stereo)
            records['outputside_mask'].append(outputside_mask)
            records['target_chip'].append(best_to_mq179_pair.fix_sec.slice_id)
            records['group_name'].append(combine_name)
            # pl.col('animal_id') + '_' + pl.col('tracer') + '_' + pl.col('combine_area')
            records['draw_color_key'].append(f'{mov_animal_id}_{next(iter(legal_tracer))}_{combine_name}')
            records['draw_color'].append(
                np.repeat(
                    draw_color.reshape(1, -1), 
                    len(mov_cells_in_stereo), 
                    axis=0
                )
            )

        #     break
        # break
    if len(records['cells']):
        res_df = record_to_df(records)
        res_df.write_parquet(export_path / f'saved-{combine_name}-{export_version}.parquet')
    else:
        print(combine_name, 'no cell data')
    # break
    # pickle.dump(to_save, open(export_path.parent / f'saved-{export_version}.pkl', 'wb'))
# %%
# %%
dfs = []

for i in range(len(records['cells'])):
    target_chip = records['target_chip'][i]
    target_chip = f'T{target_chip}'
    corr = CorrectionPara.select(target_chip)[0]
    rxry = corr.wrap_point(records['cells'][i])

    _df = pl.from_numpy(records['cells'][i], schema=['x', 'y']).with_columns(
        pl.lit(records['animal_id'][i]).alias('animal_id'),
        pl.lit(records['slice_id'][i]).alias('slice_id'),
        pl.lit(records['side'][i]).alias('side'),
        pl.lit(records['target_chip'][i]).alias('target_chip'),
        pl.lit(records['group_name'][i]).alias('group_name'),
        pl.lit(rxry[:, 0]).alias('rx'),
        pl.lit(rxry[:, 1]).alias('ry'),
        pl.lit(records['draw_color'][i]).alias('draw_color'),
    )
    dfs.append(_df)

df = pl.concat(dfs)
# %%
all_src_animals = set(input_df.animal_id)

used_animals = set()

for f in export_path.glob(f'saved-*-{export_version}.parquet'):
    res_df = pl.read_parquet(f)
    for a in res_df['animal_id'].unique().to_list():
        used_animals.add(a)
# %%
all_src_animals - used_animals
# %%
used_animals
# %%
import orjson
from ntp_manager import NTPRegion, from_dict

chip = 'T31'

corr = CorrectionPara.select(chip)[0]
slice_meta_p = Path(f'/data/sdf/ntp/macaque/Mq179-CLA-20230505/region-mask/Mq179-{chip}-P0.json')
slice_meta = orjson.loads(slice_meta_p.read_bytes())
regions: list[NTPRegion] = [
    from_dict(NTPRegion, r) for r in slice_meta['regions']
]

plt.figure(figsize=(10, 10))

for region in regions:
    # if 'cla' not in region.label.name.lower(): continue
    pnts = region.polygon.exterior.coords
    plt.plot(*np.array(pnts).T)

sample = res_df.filter(pc('target_chip') == chip[1:]).select(['x', 'y']).to_numpy() - corr.offset
colors = res_df.filter(pc('target_chip') == chip[1:])['draw_color']
plt.scatter(*sample.T, s=0.1, c=colors)
# %%
df.write_parquet(export_path.parent / f'saved-{export_version}.parquet')
# %%
# 根据上面导出的矢量结果画图的


for p in tqdm(assets_path.parent.parent.glob('saved-20240306*.pkl')):
    records = pickle.load(open(p, 'rb'))['record_cells']

    all_cells = np.vstack(records['cells'])
    min_x, min_y = all_cells.min(0)
    max_x, max_y = all_cells.max(0)

    fixsize = ((max_x - min_x) / 300 * 2, (max_y - min_y) / 300 *2)
    first_draw_cnt = defaultdict(lambda: True)

    plt.figure(figsize=fixsize)
    for i in range(len(records['cells'])):
        animal_id = records['animal_id'][i]
        slice_id = records['slice_id'][i]

        cells = records['cells'][i]
        # cells[:, 0] = -cells[:, 0]
        smooth_cnt = records['smooth_cnt'][i]
        # smooth_cnt[:, 0] = -smooth_cnt[:, 0]

        
        c = input_df[
            (input_df.animal_id == animal_id)
        ].draw_color.dropna().iloc[0]
        if '2.0' in str(p):
            c = 'magenta'

        plt.scatter(*cells.T, s=0.01, alpha=1, c=c)
        if first_draw_cnt[(animal_id, slice_id)]:
            plt.plot(*smooth_cnt.T, linewidth=1, alpha=1, c='black')
            first_draw_cnt[(animal_id, slice_id)] = False

        # print(records['slice_id'][i], records['side'][i])

    # 逆转y轴, xy等比

    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    # 关闭坐标轴
    plt.axis('off')
    plt.savefig(assets_path.parent.parent / f'{p.stem}.svg')
    plt.close()

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
regist_datasets = read_regist_datasets(
    animal_id, v展示时缩放, exclude_slices=exclude_slices, 
    target_shape=(128*j计算缩放, 128*j计算缩放)
)
# %%

fix_datasets = read_regist_datasets(
    'C042', v展示时缩放, exclude_slices=exclude_slices,
    target_shape=(128*j计算缩放, 128*j计算缩放)
)

def points_kde(points: np.ndarray, image_size: int, mesh_size: int=0, bandwidth=0.025, zz_factor=lambda x: x) -> np.ndarray:
    ps = points / image_size # normalize to [0, 1]
    if mesh_size == 0: mesh_size = image_size // 4

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(ps)
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

def mask_points_by_contours(points: np.ndarray, *contours: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(points), dtype=bool)
    for cnt in contours:
        for i, point in enumerate(points):
            mask[i] = cv2.pointPolygonTest(cnt, point, False) >= 0
    return points[mask]


import matplotlib.cm as cm
import matplotlib.colors as colors

original_cmap = cm.get_cmap('RdYlBu')
colors_array = original_cmap(np.linspace(0, 1, original_cmap.N))[::-1, ...]

colors_array[0] = [0, 0, 0, 0]
new_cmap = colors.ListedColormap(colors_array)

combine_to_draws: dict[str, defaultdict[str, list]] = {}


for combine_name, meta_group in input_df.groupby('Combine3'):
    if not isinstance(combine_name, str): continue
    print(f'{combine_name=}')
    export_version = f'20231214-groupby-inj-{combine_name}'
    export_root = assets_path.parent.parent / export_version
    export_root.mkdir(exist_ok=True, parents=True)

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
            mov_cells_filtered = mask_points_by_contours(mov_cells, smooth_cnt)
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
current_combine_result['182'][0]
# %%
combine_to_draws_with_kde = {}
# 合并 cells 并计算 KDE 乘数因子

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
        mov_cells_filtered = np.vstack([i['mov_cells_filtered'] for i in mov_items])
        mean_cell_number = np.median([len(i['mov_cells_filtered']) for i in mov_items])
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

# %%
# 计算 KDE

for current_combine, merged_to_draws in combine_to_draws_with_kde.items():
    # if current_combine != 'posterior': continue

    for merged in tqdm(merged_to_draws, desc=current_combine):
        mov_cells_filtered = merged['mov_cells_filtered']
        if len(mov_cells_filtered) > 1:
            factor = np.sqrt(merged['mean_cell_number'] / merged['max_cells'])
            merged['kde_img'] = points_kde(
                merged['mov_cells_filtered'], image_size, 
                mesh_size=image_size // 4, bandwidth=0.03,
                # zz_factor=lambda x: x * factor
            )
            # print(factor, merged['kde_img'].max())

        else:
            merged['kde_img'] = np.zeros((image_size, image_size))
# %%
import cairo
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


original_cmap = cm.get_cmap('RdYlBu')
original_cmap = cm.get_cmap('magma')

colors_array = original_cmap(np.linspace(0, 1, original_cmap.N))[::-1, ...]
colors_array = (colors_array * 255).astype('uint8')

output_root = Path('/mnt/97-macaque/projects/cla/injections-cells/20231218-kde-draw-reds/')
output_root.mkdir(exist_ok=True, parents=True)
for current_combine in combine_to_draws_with_kde:
    # if current_combine != 'posterior': continue

    pading_px = 10
    total_image_w = 0
    total_image_h = 0
    all_to_draws = []

    merged_to_draws = combine_to_draws_with_kde[current_combine]
    merged_to_draws.sort(key=lambda x: x['fix_slice_id'])
    for to_draw_index in split_into_n_parts_slice(merged_to_draws, 42):
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
        print(factors.shape)
        smooth_n = 3
        factors = np.convolve(factors, np.ones(smooth_n) / smooth_n, mode='same')
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
    plt.figure(figsize=(30, 10))
    plt.imshow(img)
# %%
mov_mask, mov_cells_df = regist_datasets[AnimalSection(
    animal_id = pair.mov_sec.animal_id,
    slice_id  = pair.mov_sec.slice_id,
)]['pad_res'][mov_side]
mov_cells_df = mov_cells_df.filter(
    pc('color') == combine_item['color']
)
if len(mov_cells_df):
    print(mov_cells_df)

raw_cnt, smooth_cnt = get_mask_cnts(
    pair.fix_sec.animal_id, 
    pair.fix_sec.slice_id, 
    pair.fix_sec.side,
    datasets=fix_datasets
)
raw_cnt, smooth_cnt = get_mask_cnts(
    pair.mov_sec.animal_id, 
    pair.mov_sec.slice_id, 
    pair.mov_sec.side,
)

plt.imshow(mov_mask)
plt.scatter(*mov_cells_df.select('x', 'y').to_numpy().T, s=0.1, c='red')
plt.plot(*raw_cnt.T, linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
