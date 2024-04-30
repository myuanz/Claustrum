'''
读取匹配图像，然后配准
'''
# %%
# %load_ext autoreload
# %autoreload 2
# %%
import itertools
import os

os.chdir(os.path.dirname('/home/myuan/projects/cla/pwvy/show_nnunet_result.py'))

import pickle
from collections import defaultdict
from pathlib import Path

import ants
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torchvision.models
from bidict import bidict
from connectome_utils import find_connectome_ntp, get_czi_size
from dataset_utils import (
    ClaDataset,
    SimilarityMatrix,
    get_base_path,
    get_dataset,
    read_regist_datasets,
    stereo_masks_to_cla_dataset,
)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, SSDMetric
from dipy.viz import regtools
from joblib import Memory, Parallel, delayed
from loguru import logger
from ntp_manager import SliceMeta, parcellate
from scipy.spatial.distance import cdist
from sim_network import SimilarityNet, SimilarityNetOutput
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from utils import (
    AnimalSection,
    AnimalSectionWithDistance,
    Literal,
    PariRegistInfo,
    draw_fix_and_mov,
    draw_image_with_spacing,
    dtw,
    dye_to_color,
    read_connectome_masks,
    read_exclude_sections,
    read_stereo_masks,
    to_ants,
    to_numpy,
)

memory = Memory('cache', verbose=0)

pc = pl.col
assets_p = Path('/mnt/97-macaque/projects/cla/injections-cells/assets')

# %%
model_dir = Path('output/runs') / '20231121-1654-lowlr-128-pairtrain-total-16div-firstdropout-d0.25'
# model_dir = Path('output/runs') / '20231123-2317-lowlr-128-pairtrain-16div-firstdropout-d0.25-1dim'
model_dir = Path('output/runs') / '20240126-1728-lowlr-128-pairtrain-16div-firstdropout-d0.25-1dim-circle'


model = SimilarityNet(torchvision.models.resnet18(), output_dim=16, dropout=0).to('cuda')
model.load_model(model_dir)
# %%
PROJECT_NAME = 'CLA'
exclude_slices = read_exclude_sections(PROJECT_NAME)

scale = 0.1
base_shape = np.array((1280, 1280))

target_shapes                 = tuple((base_shape * scale).astype(int))
target_shape: tuple[int, int] = (int(target_shapes[0]), int(target_shapes[1]))

scale, target_shape
# %%
def read_connectome_mask_and_split(
    animal_id: str, scale: float, 
    exclude_slices: list[AnimalSection]=[], 
    force=False, batch_size=256, shuffle=False,
    sides: tuple[Literal['left', 'right'], ...]=('left', 'right'),
    target_shape: tuple[int, int]=(128, 128),
):
    datasets: list[ClaDataset] = []
    raw_cntm_masks = read_connectome_masks(
        get_base_path(animal_id), animal_id, scale,
        exclude_slices=exclude_slices, force=force, 
        min_sum=400
    )
    for side in sides:
        draw_dataset = get_dataset(
            side, raw_cntm_masks, batch_size=batch_size, 
            shuffle=shuffle, 
            animal_id=f'{animal_id}-{side[0].upper()}',
            target_shape=target_shape
        )

        datasets.append(draw_dataset)
    return datasets[0].add(*datasets[1:], shuffle=shuffle)

def calc_dataset_feature(ds: ClaDataset):
    snos = []
    model.eval()

    for d in tqdm(ds.dataloader):
        res = model.calc_feature(d[0])
        snos.append(res)
    res = SimilarityNetOutput.merge_batch(*snos)

    return res

def cosin_dist(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def calc_similarity_matrix(f1: SimilarityNetOutput, f2: SimilarityNetOutput):
    fet1 = f1.feature_np # (m, latent_dim)
    fet2 = f2.feature_np # (n, latent_dim)

    # m = fet1.shape[0]
    # n = fet2.shape[0]

    # similarity_matrix = np.ones((m, n))

    # for i in range(m):
    #     for j in range(n):
    #         similarity_matrix[i, j] = cosin_dist(
    #             fet1[i], 
    #             fet2[j]
    #         )
    cosine_distance = cdist(fet1, fet2, 'cosine')
    similarity_matrix = 1.0 - cosine_distance

    return similarity_matrix

chips_dict = {
    'Mq179': '''T105 T103 T101 T99 T97 T95 T93 T91 T89 T87 T85 T83 T81 T79 T77 T75 T73 T71 T67 T65 T63 T61 T59 T57 T55 T53 T51 T49 T47 T45 T43 T41 T39 T37 T33 T31 T29 T27 T25 T26 T28 T30 T32 T34'''.split()
}

# 在这里切换 stereo 和 connectome

# <connectome>
fix_animal_id = 'C042'
fix_ds = read_connectome_mask_and_split(
    fix_animal_id, scale, exclude_slices=exclude_slices, sides=('left', ), 
    target_shape=target_shape
)
# </connectome>

# <stereo>
# fix_animal_id = 'Mq179'

# stereo_masks = read_stereo_masks(
#     chips_dict[fix_animal_id], ntp_version='Mq179-CLA-20230505', 
#     scale=1/315, min_sum=140
# )
# fix_ds = stereo_masks_to_cla_dataset(fix_animal_id, stereo_masks)
# </stereo>

fix_feature = calc_dataset_feature(fix_ds)

# %%
draw_image_with_spacing(fix_ds.raw_images, 35)
# %%
# s = np.array(fix_ds.raw_slice_ids)
# s = s-s.min()
# plt.scatter(fix_feature.feature_np[:, 0], fix_feature.feature_np[:, 1], s=s)
# %%
src_animal_ids = '''C006
C007
C008
C009
C011
C012
C013
C015
C018
C021
C023
C025
C027
C029
C030
C031
C032
C034
C035
C036
C037
C038
C039
C040
C041
C042
C043
C045
C049
C051
C052
C053
C057
C058
C059
C060
C061
C062
C063
C064
C067
C069
C070
C072
C074
C075
C077
C078
C080
C081
C083
C084
C093
C095
C096
C097
Mq179'''.splitlines()
# src_animal_ids = ['C075', 'C096', 'C077', 'C080']
# src_animal_ids = ['C042']
# src_animal_ids = ['Mq179']
src_animal_ids.sort()
    

for src_animal_id, side in tqdm(
    itertools.product(src_animal_ids, ('left', 'right')), 
    total=len(src_animal_ids) * 2
):
    sim = SimilarityMatrix(fix_animal_id, src_animal_id, side)
    if sim.exists: continue

    try:
        if src_animal_id.startswith('C'):
            mov_ds = read_connectome_mask_and_split(
                src_animal_id, scale, 
                exclude_slices=exclude_slices, sides=(side, ),
                target_shape=target_shape
            )
        else:
            if side == 'right': continue
            stereo_masks = read_stereo_masks(
                chips_dict[src_animal_id], ntp_version='Mq179-CLA-20230505', 
                scale=1/315, min_sum=140
            )
            mov_ds = stereo_masks_to_cla_dataset(src_animal_id, stereo_masks)

    except IndexError:
        logger.error(f'{src_animal_id}-{side} failed')
        continue
    except KeyError:
        logger.error(f'{src_animal_id}-{side} failed')
        continue
    except Exception as e:
        logger.error(f'{src_animal_id}-{side} failed')
        logger.exception(e)
        continue

    mov_feature = calc_dataset_feature(mov_ds)
    similarity_matrix = calc_similarity_matrix(fix_feature, mov_feature)

    sim.save(
        similarity_matrix, 
        row_labels=[f'{fix_animal_id}-{i}' for i in fix_ds.raw_slice_ids], 
        col_labels=[f'{src_animal_id}-{i}' for i in mov_ds.raw_slice_ids]
    )

# %%
sim = SimilarityMatrix('Mq179', 'C042', 'right')
similarity_matrix = sim.load().to_numpy()
plt.imshow(similarity_matrix)
# %%
path = dtw(similarity_matrix)
plt.imshow(((similarity_matrix)), cmap='Blues') 
plt.plot(path[:, 1], path[:, 0], color='orange', linewidth=2)
# %%
# read_regist_datasets('C057', 0.2, exclude_slices=exclude_slices, force=False)
# %%
def get_image_path(animal_id: str, side: Literal['left', 'right'], slice_id: str):
    return assets_p / 'raw_image' / animal_id / f'{animal_id}-{side}-{slice_id}.png'


def save_regist_datasets_image(regist_datasets: dict[AnimalSection, dict]):
    for k, v in (regist_datasets.items()):
        slice_id = k.slice_id
        animal_id = k.animal_id
        base_p = assets_p / 'raw_image' / animal_id
        base_p.mkdir(exist_ok=True, parents=True)

        pad_res = v['pad_res']
        for side in ('left', 'right'):
            p = base_p / f'{animal_id}-{side}-{slice_id}.png'
            if p.exists(): continue

            image = (pad_res[side][0] * 255).astype(np.uint8)
            cv2.imwrite(str(p), image)

# %%
regist_scale = 0.1
target_shape = (128, 128)
# regist_datasets: dict[AnimalSection, dict] = {}

# 在这里切换 fix 是 stereo 还是 connectome

# <connectome>
fix_animal_id = 'C042'
fix_regist_datasets = read_regist_datasets(fix_animal_id, regist_scale, exclude_slices=exclude_slices, force=False, target_shape=target_shape)
save_regist_datasets_image(fix_regist_datasets)
fix_ds = read_connectome_mask_and_split(
    fix_animal_id, scale, exclude_slices=exclude_slices, sides=('left', ), 
    target_shape=target_shape
)

# </connectome>

# <stereo>
# fix_animal_id = 'Mq179'

# stereo_masks = read_stereo_masks(
#     chips_dict[fix_animal_id], ntp_version='Mq179-CLA-20230505', 
#     scale=1/315, min_sum=140
# )
# fix_ds = stereo_masks_to_cla_dataset(fix_animal_id, stereo_masks)
# </stereo>
# %%
mov_regist_datasets = read_regist_datasets(
    'C042', regist_scale, exclude_slices=exclude_slices, 
    force=False, target_shape=target_shape
)
save_regist_datasets_image(mov_regist_datasets)
# %%
stereo_masks = read_stereo_masks(
    chips_dict['Mq179'], ntp_version='Mq179-CLA-20230505', 
    scale=1/315, min_sum=140
)
stereo_dataset = stereo_masks_to_cla_dataset('Mq179', stereo_masks)

animal_id = 'Mq179'
for (slice_id, image) in zip(
    stereo_dataset.raw_slice_ids, 
    stereo_dataset.raw_images,
):
    base_p = assets_p / 'raw_image' / animal_id
    base_p.mkdir(exist_ok=True, parents=True)
    side = 'left'

    p = base_p / f'{animal_id}-{side}-{slice_id}.png'
    if p.exists(): continue

    image = (image * 255).astype(np.uint8)
    cv2.imwrite(str(p), image)

# %%
# %%pyinstrument

def f(
    fix_sec: AnimalSectionWithDistance, mov_sec: AnimalSectionWithDistance, 
    corr_baseline = 0.5, max_reg_count = 3,
):
    fix_img = fix_sec.image[..., 0] if len(fix_sec.image.shape) == 3 else fix_sec.image
    mov_img = mov_sec.image[..., 0] if len(mov_sec.image.shape) == 3 else mov_sec.image

    # if fix_sec.need_flip:
    #     fix_img = np.flip(fix_img, axis=1)
    # if mov_sec.need_flip:
    #     mov_img = np.flip(mov_img, axis=1)

    output_p = Path(fix_sec.image_path).parent / 'warped' / mov_sec.animal_id / f'{fix_sec.animal_id}-{fix_sec.side}-{fix_sec.slice_id}-warped-{mov_sec.animal_id}-{mov_sec.side}-{mov_sec.slice_id}.png'
    output_p.parent.mkdir(exist_ok=True, parents=True)
    mapping_p = output_p.with_name('warped' + output_p.stem + '.pkl')

    reg_count = 0
    corr = 0
    
    dim = fix_img.ndim
    metric = SSDMetric(dim)
    level_iters = [200, 100, 50, 25]

    res = defaultdict(list)

    for i in range(max_reg_count):
        try:
            sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=10)
            mapping = sdr.optimize(fix_img, mov_img)
            warped_img = mapping.transform(mov_img)
        except Exception as e:
            from loguru import logger
            logger.error(f'{fix_sec=}, {mov_sec=}')
            logger.exception(e)
            reg_count += 1
            continue

        corr = np.sum((fix_img * warped_img) != 0) / np.sum((fix_img + warped_img) != 0)
        res['corr'].append(corr)
        res['mapping'].append(mapping)
        res['warped_img'].append(warped_img)
    if not res: return
    
    best_corr_index = np.argmax(res['corr'])
    best_corr = res['corr'][best_corr_index]

    if best_corr < corr_baseline:
        print(f'匹配质量过差 {best_corr=} {fix_sec=}, {mov_sec=}')
        return

    mapping = res['mapping'][best_corr_index]
    warped_img = res['warped_img'][best_corr_index]

    grid_p = output_p.with_stem(f"{output_p.stem}-grid")
    warped_forward, warped_backward = regtools.plot_2d_diffeomorphic_map(mapping, 3, show_figure=False)
    cv2.imwrite(str(grid_p), warped_forward)
    cv2.imwrite(str(output_p), warped_img)
    pickle.dump(mapping, open(mapping_p, 'wb'))

    return PariRegistInfo(
        fix_sec          = fix_sec,
        mov_sec          = mov_sec,
        warped_img_path  = str(output_p),
        warped_grid_path = str(grid_p),
        mapping_path     = str(mapping_p),
        iou              = best_corr,
    )


# fix_sec_dict = fix_ds[('C042', '174')] # type: ignore
# fix_sec_dict = {
#     'animal_id': fix_animal_id,
#     'slice_id': '67',
#     'fix_side': 'left',
# }
# fix_side = 'left'

# fix_sec_m = AnimalSectionWithDistance(
#     fix_sec_dict['animal_id'], fix_sec_dict['slice_id'],
#     image_path=get_image_path(
#         fix_sec_dict['animal_id'], fix_side, fix_sec_dict['slice_id']
#     ).as_posix(),
#     need_flip=fix_side != 'left',
#     side=fix_side,
# )

# mov_sec_dict = mov_regist_datasets[('C042', '185')] # type: ignore
# mov_side = 'right'

# mov_sec_m = AnimalSectionWithDistance(
#     mov_sec_dict['animal_id'], mov_sec_dict['slice_id'],
#     image_path=get_image_path(
#         mov_sec_dict['animal_id'], mov_side, mov_sec_dict['slice_id']
#     ).as_posix(), 
#     need_flip=mov_side != 'left',
#     side=mov_side,
# )

# pri = f(fix_sec_m, mov_sec_m, corr_baseline=0.5)
# assert pri is not None

# draw_fix_and_mov(to_numpy(fix_sec_m.image[:, :, 0]), to_numpy(mov_sec_m.image), to_numpy(pri.warped_image))
# # %%

# plt.subplot(121)
# plt.imshow(cv2.imread(str(pri.warped_img_path)))
# plt.subplot(122)
# plt.imshow(cv2.imread(str(pri.warped_grid_path)))
# # %%
# cnts, _ = cv2.findContours(to_numpy(mov_sec_m.image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# x, y, w, h = cv2.boundingRect(np.concatenate(cnts))

# X, Y = np.meshgrid(
#     np.linspace(x, x+w, w//2),
#     np.linspace(y, y+h, h//2),
# )
# src_pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)


# plt.figure(figsize=(5, 10))
# plt.subplot(211)
# plt.imshow(to_numpy(mov_sec_m.image))
# plt.scatter(src_pts[:, 0], src_pts[:, 1], s=1)

# dst_pts = pri.transform_points(src_pts)


# plt.subplot(212)
# plt.imshow(pri.warped_image)
# # plt.scatter(src_pts[:, 0], src_pts[:, 1], s=2, label='src')
# plt.scatter(dst_pts[:, 0], dst_pts[:, 1], s=1, label='dst')

# plt.legend()

# %%pyinstrument
tasks: list[tuple[
    AnimalSectionWithDistance,
    AnimalSectionWithDistance,
]] = []
exclude_slices = read_exclude_sections(PROJECT_NAME)

for curr_animal_id in tqdm(src_animal_ids):
    # if curr_animal_id != 'Mq179': continue

    if curr_animal_id.startswith('C'):
        mov_regist_datasets = read_regist_datasets(
            curr_animal_id, regist_scale, exclude_slices=exclude_slices, 
            force=False, target_shape=target_shape, 
        )
        save_regist_datasets_image(mov_regist_datasets)


    for side in ('left', 'right'):
        sim = SimilarityMatrix(fix_animal_id, curr_animal_id, side)
        try:
            sim_df = sim.load()
        except FileNotFoundError:
            logger.error(f'{fix_animal_id}-{curr_animal_id}-{side} simdf not found')
            continue
        fix_target_rows = np.linspace(0, sim_df.shape[0]-1, 42, dtype=int)
        sim_df = sim_df.iloc[fix_target_rows]
        sim_mat = sim_df.to_numpy()

        dtw_path = dtw(sim_mat.T)
        # plt.imshow(((sim_mat.T)), cmap='Blues')
        # plt.plot(dtw_path[:, 1], dtw_path[:, 0], color='orange', linewidth=2)

        added_mov_slice_id = set([])

        for path_index, slice_index in enumerate(dtw_path[:, 0]):
            mov_slice_id = sim_df.columns[slice_index].split('-')[1]
            if mov_slice_id in added_mov_slice_id:
                continue
            added_mov_slice_id.add(mov_slice_id)
            if not (get_image_path(curr_animal_id, side, mov_slice_id)).exists():
                print(f'{curr_animal_id}-{side}-{mov_slice_id} not exists')
                continue

            mov_sec_m = AnimalSectionWithDistance(
                curr_animal_id, mov_slice_id, 
                # image_path=f'{os.getcwd()}/output/assets/{curr_animal_id}-{side}-{mov_slice_id}.png', 
                image_path=get_image_path(curr_animal_id, side, mov_slice_id).as_posix(),
                need_flip=side != 'left',
                side=side,
            )

            fix_slice_id = sim_df.index[dtw_path[path_index, 1]].split('-')[1] # type: ignore
            fix_sec_m = AnimalSectionWithDistance(
                fix_animal_id, fix_slice_id, 
                # image_path=f'{os.getcwd()}/output/assets/{fix_animal_id}-left-{fix_slice_id}.png', 
                image_path=get_image_path(fix_animal_id, 'left', fix_slice_id).as_posix(),
                need_flip=False,
                side='left',
            )
            # print(f'{mov_slice_id} <-> {fix_slice_id}')
            tasks.append((fix_sec_m, mov_sec_m))


# %%
with open(assets_p / 'regist_input.pkl', 'wb') as ffff:
    logger.info('dumping regist_input')
    pickle.dump(tasks, ffff)

print(f'{len(tasks)=}')
# %%
res = Parallel(
    n_jobs=32, 
    # backend='multiprocessing',
    verbose=10,
)([delayed(f)(fix_sec, mov_sec) for fix_sec, mov_sec in tasks])
# %%
res
# %%
with open(assets_p / 'regist_results_20240129_connectome_to_C042.pkl', 'wb') as ffff:
    logger.info('dumping regist_results')
    pickle.dump(res, ffff)
# with open(assets_p / 'stereo_masks_20240118.pkl', 'wb') as ffff:

#     pickle.dump(stereo_masks, ffff)
# %%
# pad_res = mov_regist_datasets[('C011', '201')]['pad_res']
# for side in ('left', 'right'):
#     plt.imshow(pad_res[side][0], cmap='gray')
#     plt.scatter(pad_res[side][1][:, 'x'], pad_res[side][1][:, 'y'], s=1)
#     plt.show()
# %%
# item = mov_regist_datasets[('C011', '201')]
# extract_cell_df(item)
# %%

# dst_animal = 'C042-L'
# src_animal = 'C057-L'

# htmls = []

# htmls.append('<html>')
# htmls.append('<head><link rel="stylesheet" href="index.css"></head>')
# htmls.append('<body><div class="root">')

# matches: defaultdict[tuple[str, int], list[AnimalSectionWithDistance]] = defaultdict(list)

# for d, i in zip(distances, indices):
#     htmls.append('<div class="col">')
#     curr_main_animal_id: str = ''
#     curr_main_slice_id  = 0

#     curr_match_count = 0

#     for dd, ii in zip(d, i):
#         curr_animal_id = ds.animal_ids[ii]
#         curr_slice_id: int = ds.raw_slice_ids[ii]
#         curr_image_path = f'assets/{curr_animal_id}-{curr_slice_id}.png'

#         flip_class = ''

#         if curr_main_animal_id == '': 
#             if curr_animal_id != dst_animal:
#                 continue
#             curr_main_animal_id = curr_animal_id
#             curr_main_slice_id  = curr_slice_id
#         else:
#             if curr_animal_id != src_animal or curr_main_animal_id != dst_animal:
#                 continue
#             if curr_main_animal_id[-1] != curr_animal_id[-1]:
#                 flip_class = 'image-w-flip'
#         matches[(curr_main_animal_id, curr_main_slice_id)].append(AnimalSectionWithDistance(
#             animal_id  = curr_animal_id,
#             slice_id   = f'{curr_slice_id:03d}',
#             image_path = f'{os.getcwd()}/output/{curr_image_path}',
#             need_flip  = flip_class != '', 
#             distance   = dd,
#         ))

#         htmls.append(f'''
#             <img src="{curr_image_path}" class="{flip_class}" />
#             <div class="animal_id">{curr_animal_id}</div>
#             <div class="slice_id">{curr_slice_id}</div>
#             <div class="distance">{dd:.4f}</div>
#         ''')


#         curr_match_count += 1
#         if curr_match_count > 3:
#             break

#     htmls.append('\n</div>')

# htmls.append('</div><body><html>')

# (assets_p.parent / 'index.html').write_text('\n'.join(htmls))
# # %%

# secs = matches[('C042-L', 187)]
# fix_sec_m: AnimalSectionWithDistance = secs[0]
# fix_sec = regist_datasets[fix_sec_m.to_sec(with_side=False)]
# fix_sec = AnimalSectionWithDistance(
#     animal_id  = fix_sec_m.animal_id,
#     slice_id   = fix_sec_m.slice_id,
#     image_path = f'{os.getcwd()}/output/assets/{fix_sec_m.animal_id}-{fix_sec_m.slice_id}.png',
#     need_flip  = fix_sec_m.need_flip,
#     distance   = 0,
# )

# mov_sec_m: AnimalSectionWithDistance = secs[1]
# mov_sec = regist_datasets[mov_sec_m.to_sec(with_side=False)]
# mov_sec = AnimalSectionWithDistance(
#     animal_id  = mov_sec_m.animal_id,
#     slice_id   = mov_sec_m.slice_id,
#     image_path = f'{os.getcwd()}/output/assets/{mov_sec_m.animal_id}-{mov_sec_m.slice_id}.png',
#     need_flip  = mov_sec_m.need_flip,
#     distance   = 0,
# )

# # pri = f(secs[0], secs[1])
# pri = f(fix_sec, mov_sec)
# assert pri is not None

# plt.subplot(121)
# plt.imshow(cv2.imread(str(pri.warped_img_path)))
# plt.subplot(122)
# plt.imshow(cv2.imread(str(pri.warped_grid_path)))
# # %%
# cnts, _ = cv2.findContours(to_numpy(mov_sec.image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# x, y, w, h = cv2.boundingRect(np.concatenate(cnts))

# X, Y = np.meshgrid(
#     np.linspace(x, x+w, w//4),
#     np.linspace(y, y+h, h//4),
# )
# src_pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)


# plt.figure(figsize=(5, 10))
# plt.subplot(211)
# plt.imshow(to_numpy(mov_sec.image))
# plt.scatter(src_pts[:, 0], src_pts[:, 1], s=1)

# dst_pts = pri.transform_points(src_pts)


# plt.subplot(212)
# plt.imshow(pri.warped_image)
# # plt.scatter(src_pts[:, 0], src_pts[:, 1], s=2, label='src')
# plt.scatter(dst_pts[:, 0], dst_pts[:, 1], s=1, label='dst')

# plt.legend()

# # %%
# draw_fix_and_mov(to_numpy(fix_sec.image), to_numpy(mov_sec.image), to_numpy(pri.warped_image))
# # %%

# tasks = []

# for (fix_animal_id, fix_slice_id), secs in matches.items():
#     fix_sec_m: AnimalSectionWithDistance = secs[0]
#     fix_sec = regist_datasets[fix_sec_m.to_sec(with_side=False)]
#     fix_sec = AnimalSectionWithDistance(
#         animal_id  = fix_sec_m.animal_id,
#         slice_id   = fix_sec_m.slice_id,
#         image_path = f'{os.getcwd()}/output/assets/{fix_sec_m.animal_id}-{fix_sec_m.slice_id}.png',
#         need_flip  = fix_sec_m.need_flip,
#         distance   = 0,
#     )

#     mov_sec_m: AnimalSectionWithDistance = secs[1]
#     mov_sec = regist_datasets[mov_sec_m.to_sec(with_side=False)]
#     mov_sec = AnimalSectionWithDistance(
#         animal_id  = mov_sec_m.animal_id,
#         slice_id   = mov_sec_m.slice_id,
#         image_path = f'{os.getcwd()}/output/assets/{mov_sec_m.animal_id}-{mov_sec_m.slice_id}.png',
#         need_flip  = mov_sec_m.need_flip,
#         distance   = 0,
#     )


#     tasks.append(delayed(f)((fix_sec), (mov_sec)))

# regist_results = Parallel(
#     n_jobs=32, 
#     # backend='multiprocessing',
#     verbose=10,
# )(tasks)
# # %%
# regist_results[0].transform_points(np.array([[0, 0], [100, 100]]))

# # %%
# def find_cell(sec: AnimalSectionWithDistance):
#     assert sec.animal_id[-1] in 'LR'
#     animal_id = sec.animal_id[:-2]
#     side = {'L': 'left', 'R': 'right'}[sec.animal_id[-1]]
#     slice_id = sec.slice_id

#     return regist_datasets[AnimalSection(
#         animal_id=animal_id,
#         slice_id=slice_id,
#     )]['pad_res'][side][1]


# # %%
# def np_to_svg_points(array: np.ndarray, radius=1):
#     assert array.ndim == 2
#     assert array.shape[1] == 2

#     str_list = array.astype(str).tolist()

#     circles = '\n'.join([f'<circle cx="{point[0]}" cy="{point[1]}" r="{radius}"/>' for point in str_list])

#     svg = f'{circles}\n'
#     return svg
# print(np_to_svg_points(np.random.rand(5, 2)))

# # %%
# htmls = []

# htmls.append('<html>')
# htmls.append('<head><link rel="stylesheet" href="index.css"></head>')
# htmls.append('<body><div class="root">')


# # for (fix_animal_id, fix_slice_id), secs in tqdm(matches.items()):
# for pair in tqdm(regist_results):
#     pair: PariRegistInfo

#     htmls.append('<div class="col">')

#     for sec in (pair.fix_sec, pair.mov_sec):
#         curr_image_path = f'assets/{sec.animal_id}-{sec.slice_id}.png'
#         flip_class = 'w-flip' if sec.need_flip else ''
#         cells = find_cell(sec)
#         svgs = []

#         for color_name, g in cells.group_by('color'):
#             svg_txt = np_to_svg_points(g[['x', 'y']].to_numpy() / 2, radius=1)
#             svgs.append(f'''
#                 <svg class="point-svg point-svg-{color_name} {flip_class}">
#                     {svg_txt}
#                 </svg>
#             ''')

#         htmls.append(f'''
#             <div style="position: relative; display: inline-block;">
#                 <img src="{curr_image_path}" class="{flip_class}"/>
#                 {''.join(svgs)}
#                 <div class="animal_id">{sec.animal_id}</div>
#                 <div class="slice_id">{sec.slice_id}</div>
#             </div>
#         ''')
#     warped_image_path = f'assets/{pair.fix_sec.animal_id}-{pair.fix_sec.slice_id}-warped-{pair.mov_sec.animal_id}-{pair.mov_sec.slice_id}.png'

#     grid_image_path = warped_image_path.replace('.png', '-grid.png')


#     cells = find_cell(pair.mov_sec)
#     cells_arr = cells[['x', 'y']].to_numpy()
#     if len(cells_arr) > 1:
#         cells_arr = pair.transform_points(cells_arr)
#         cells = cells.with_columns(
#             pl.lit(cells_arr[:, 0]).alias('x'),
#             pl.lit(cells_arr[:, 1]).alias('y'),
#         )

#     svgs = []

#     for color_name, g in cells.group_by('color'):
#         svg_txt = np_to_svg_points(g[['x', 'y']].to_numpy() / 2, radius=1)
#         svgs.append(f'''
#             <svg class="point-svg point-svg-{color_name} {flip_class}">
#                 {svg_txt}
#             </svg>
#         ''')


#     htmls.append(f'''
#         <div style="position: relative; display: inline-block;">
#             <img src="{warped_image_path}"/>
#             {''.join(svgs)}
#             <div class="animal_id">{sec.animal_id}</div>
#             <div class="slice_id">{sec.slice_id} warped</div>
#         </div>
#     ''')
#     htmls.append(f'''
#         <img src="{grid_image_path}"/>
#         <div class="animal_id">{sec.animal_id}</div>
#         <div class="slice_id">{sec.slice_id} warped</div>
#     ''')
#     htmls.append('\n</div>')

# htmls.append('</div><body><html>')

# (assets_p.parent / 'index-warped.html').write_text('\n'.join(htmls))

# # %%
# m = {}

# for pair in tqdm([i for i in regist_results if isinstance(i, PariRegistInfo)]):
#     # print(pair)
#     # break
#     if pair.mov_sec.to_sec() in m:
#         print(f'{pair.mov_sec.animal_id}-{pair.mov_sec.slice_id} already in m, {m[pair.mov_sec.to_sec()]=}')
#     else:
#         m[pair.mov_sec.to_sec()] = pair.fix_sec.to_sec()
    
# # %%
# regist_result_output_path = Path('/mnt/97-macaque/projects/cla/regist_result_output') / src_animal
# regist_result_output_path.mkdir(exist_ok=True, parents=True)


# for pair in tqdm([i for i in regist_results if isinstance(i, PariRegistInfo)]):
#     mov_sec = pair.mov_sec
#     warped_image = cv2.cvtColor(pair.warped_image, cv2.COLOR_GRAY2BGR)

#     cells = find_cell(mov_sec)

#     cells_arr = cells[['x', 'y']].to_numpy()
#     if len(cells_arr) > 1:
#         cells_arr = pair.transform_points(cells_arr)
#         for x, y in cells_arr:
#             cv2.circle(warped_image, (int(x), int(y)), 1, (0, 0, 255), -1)

#     else:
#         # print(f'{mov_sec.animal_id}-{mov_sec.slice_id} has no cells')
#         with open(regist_result_output_path / 'output.txt', 'a+') as ffff:
#             ffff.write(f'{mov_sec.animal_id}-{mov_sec.slice_id} has no cells\n')

#     output_p = regist_result_output_path / f'{mov_sec.animal_id}-{mov_sec.slice_id}.png'
#     cv2.imwrite(str(output_p), warped_image)