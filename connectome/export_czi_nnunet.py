# %%
import os
__file__ = '/home/myuan/projects/cla/pwvy/export_czi_nnunet.py'
os.chdir(os.path.dirname(__file__))

from pathlib import Path
from typing import Optional, cast
import cv2
import czi_shader
import joblib
import numpy as np
from loguru import logger
import connectome_utils as cu
from ntp_manager import from_dict, SliceMeta, parcellate
from joblib import Parallel, delayed
import json
from dataclasses import dataclass, asdict
import aicspylibczi
import utils
import polars as pl
from tqdm import tqdm

def export_czi_to_png(
    animal_id: str, slice_id: int | str, current_data_set_base_path: str, scale: float, FORCE_UPDATE: bool
) -> tuple[str, str, str] | None:
    
    ps: list[Path] = []
    if isinstance(slice_id, int):
        slice_id = f'{slice_id:03d}'
    for i in range(4):
        p = Path(current_data_set_base_path).parent / 'Claustrum_infer' / 'imagesTr' / f'{animal_id}_{slice_id}_{i:04d}.tif'
        if not p.exists(): break

        ps.append(p)
    if len(ps) == 4:
        for p in ps:
            target_path = Path(current_data_set_base_path) / 'imagesTr' / p.name
            if target_path.exists(): continue
            try:
                target_path.hardlink_to(p)
            except FileExistsError:
                pass

        return animal_id, slice_id, str(Path(current_data_set_base_path) / 'imagesTr' / ps[0].name)


    try:
        czi_path = cu.find_czi(animal_id, slice_id=slice_id)
        if not czi_path:
            logger.warning(f'{animal_id}-{slice_id} czi not found')
            return None

        czi_path = czi_path[0]
        channels = czi_shader.CZIChannel.from_czi(czi_path)
        channels = sorted(channels, key=lambda ch: ch.dye_name)
        # print([(ch.id, ch.dye_name) for ch in channels])

        target_paths = [
            Path(current_data_set_base_path) / 'imagesTr' / f'{animal_id}_{slice_id}_{i:04d}.tif'
            for i in range(len(channels))
        ]

        if not FORCE_UPDATE and all([p.exists() for p in target_paths]):
            return animal_id, slice_id, str(target_paths[0])

        czi = aicspylibczi.CziFile(czi_path)
        for ch, p in zip(channels, target_paths):
            image = czi.read_mosaic(scale_factor=scale, C=ch.id)[0]
            cv2.imwrite(p.as_posix(), image)
        
        return animal_id, slice_id, str(target_paths[0])
    except Exception as e:
        logger.error(f'{animal_id}-{slice_id} {e}')
        return None

@dataclass
class Dataset_nnUNet:
    '''https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson'''
    channel_names: dict[int, str]
    labels: dict[str, int]
    numTraining: int
    file_ending: str


base_output_path = Path('/mnt/90-connectome/connectome-export/cla-reg-nnunet')

VERSION      = '20231011'
FORCE_UPDATE = False
FOR_INFER    = False
scale        = 1/20
PROJECT_NAME = 'GPe/GPi/VP'

VERSION      = '20230926'
PROJECT_NAME = 'CLA'
FOR_INFER    = True

if FOR_INFER:
    dataset_prefix = 'Claustrum_infer'
    # dataset_prefix = 'Dataset006_CLA_6-15_057'

else:
    dataset_prefix = 'Dataset003_GPe-GPi-VP_C008C057'
    dataset_prefix = 'Dataset006_CLA_6-15_057'
    dataset_prefix = 'Dataset007_CLA_100p'

dataset_prefix = 'Dataset007_CLA_100p'

current_data_set_base_path = base_output_path / VERSION / dataset_prefix
(current_data_set_base_path / 'imagesTr').mkdir(exist_ok=True, parents=True)
(current_data_set_base_path / 'labelsTr').mkdir(exist_ok=True, parents=True)
(current_data_set_base_path / 'ntpCells').mkdir(exist_ok=True, parents=True)

animals = 'C001 C002 C003 C004 C005 C006 C007 C008 C011 C012 C013 C015 C021 C023 C025 C027 C028 C029 C030 C031 C032 C034 C035 C036 C037 C038 C039 C040 C041 C042 C043 C045 C046 C049 C051 C052 C053 C056 C057 C058 C059 C060 C061 C062 C063 C064 C066 C067 C068 C069 C070 C071 C072 C073 C074 C075 C076 C077 C078 C079 C080 C081 C082 C083 C084 C086 C087 C088 C089 C091 C093 C094 C095 C096 C097'.split(' ')

animals = 'C006 C007 C008 C011 C012 C013 C015 C021 C023 C025 C027 C028 C029 C030 C031 C032 C034 C035 C036 C037 C038 C039 C040 C041 C042 C043 C045 C046 C049 C051 C052 C053 C056 C057 C058 C059 C060 C061 C062 C063 C064 C066 C067 C068 C069 C070 C071 C072 C073 C074 C075 C076 C077 C078 C079 C080 C081 C082 C083 C084 C086 C087 C088 C089 C091 C093 C094 C095 C096 C097'.split(' ')
# animals = 'C013 C015 C021 C023 C025 C027 C028 C029 C030 C031 C032 C034 C035 C036 C037 C038 C039 C040 C041 C042 C043 C045 C046 C049 C051 C052 C053 C056 C057 C058 C059 C060 C061 C062 C063 C064 C066 C067 C068 C069 C070 C071 C072 C073 C074 C075 C076 C077 C078 C079 C080 C081 C082 C083 C084 C086 C087 C088 C089 C091 C093 C094 C095 C096 C097'.split(' ')
# animals = 'C006 C007 C008 C057 C011 C012 C013 C015'.split(' ')
# animals = [f'C{i:03d}' for i in range(11, 21)]
# animals = ['C042']
animals = [f'C{i:03d}' for i in range(1, 7)]
animals = [f'C{i:03d}' for i in range(100, 119)]

labeled_slices = utils.read_labeled_sections(PROJECT_NAME)
exclude_slices = utils.read_exclude_sections(PROJECT_NAME)

writed_image: dict[str, Path] = {}

tasks = []

for animal_id in animals:
    print(f'{animal_id=}')

    if FOR_INFER:
        try:
            czi_files = cu.find_czi(animal_id)
        except FileNotFoundError:
            open('./error.log', 'a+').write(f'{animal_id} not found\n')
            continue

        if len(czi_files) == 0: 
            open('./error.log', 'a+').write(f'{animal_id} no czi\n')
            continue
        slice_ids = [int(cu.ntp_p_to_slice_id(i)) for i in czi_files]
    else:
        slice_ids = [
            s.slice_id_int for s in labeled_slices 
            if s.animal_id == animal_id and (s.animal_id, s.slice_id_int) not in exclude_slices
        ]

    tasks.extend([
        delayed(export_czi_to_png)(
            animal_id, slice_id, current_data_set_base_path, scale, FORCE_UPDATE
        )
        for slice_id in slice_ids
        if (animal_id, slice_id) not in exclude_slices
    ])
# %%
for res in tqdm(Parallel(
    n_jobs=32, verbose=10, 
    # return_as='generator_unordered',
    backend='multiprocessing'
)(tasks)):
    if res is None: continue
    animal_id, slice_id, target_path = res
    writed_image[f'{animal_id}-{slice_id}'] = Path(target_path)
exit()
# %%
region_ids = utils.read_region_ids()
base_ntp_path = ('/mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/20231117-collect-fixed-ntp/added-label/')

def export_mask(
    animal_id: str, slice_id: str, writed_image: dict[str, Path], 
    scale: float, target_region_names: list[str], 
) -> Path | None:
    # print(f'{animal_id}-{slice_id}')
    image_path = writed_image.get(f'{animal_id}-{slice_id}')
    if not image_path or not image_path.exists(): 
        logger.warning(f'{animal_id}-{slice_id} image not found')
        return

    ntp_p = cu.find_connectome_ntp(animal_id, slice_id=slice_id, base_path=base_ntp_path)
    # if not ntp_p: 
    #     ntp_p = cu.find_connectome_ntp(animal_id, slice_id=slice_id, base_path='/mnt/90-connectome/finalNTP-layer4-parcellation91-106/')
    #     assert False, f'{animal_id}-{slice_id:03d} ntp not found'
    if not ntp_p:
        logger.warning(f'{animal_id}-{slice_id:03d} ntp not found')
        image_paths = image_path.parent.glob(f"{animal_id}_{slice_id:03d}_*.tif")
        for p in image_paths:
            p.unlink()
        image_path.unlink(missing_ok=True)
        return
    ntp_p = ntp_p[0]

    slice_meta = parcellate(
        ntp_p, background_path=image_path.as_posix(), 
        bin_size=int(1/scale), export_position_policy='none',
        ignore_regions=['region3','region4','mark4','region5','region6']
    )
    assert isinstance(slice_meta, SliceMeta)
    # globals()['slice_meta'] = slice_meta

    w, h = int(slice_meta.w), int(slice_meta.h)
    # print(f'{w=} {h=}')
    region_mask_raw = np.zeros(shape=(h, w), dtype=np.uint8)
    for r in slice_meta.regions:
        if all([
            target_region_name.lower() not in r.label.name.lower()
            for target_region_name in target_region_names
        ]):
            continue

        last_name = "-".join(r.label.name.split('-')[1:])
        last_name = 'Cla-s'
        region_id = region_ids[last_name]

        cv2.fillPoly(region_mask_raw, [
            np.array(r.polygon.exterior.coords).astype(np.int32),
            *[np.array(hole.coords).astype(np.int32) for hole in r.polygon.interiors]
        ], (region_id, ))
    if np.sum(region_mask_raw) != 0:
        mask_path = image_path.parent.parent / 'labelsTr' / f'{animal_id}_{slice_id:03d}.tif'
        cv2.imwrite(
            mask_path.as_posix(), 
            region_mask_raw
        )
        return mask_path
    else:
        logger.warning(f'{animal_id}-{slice_id} mask is empty')
        image_path.unlink()
        # image_path.unlink(missing_ok=True)

def export_cell(
    animal_id: str, slice_id: int, writed_image: dict[str, Path], 
    scale: float, 
) -> Path | None:
    image_path = writed_image.get(f'{animal_id}-{slice_id}')
    if not image_path or not image_path.exists(): return

    ntp_p = cu.find_connectome_ntp(animal_id, slice_id=slice_id, base_path='/mnt/90-connectome/finalNTP/')
    if not ntp_p: 
        logger.warning(f'{animal_id}-{slice_id} ntp not found')
        return
    ntp_p = ntp_p[0]
    # print(animal_id, slice_id, ntp_p)

    slice_meta = parcellate(
        ntp_p, background_path=image_path.as_posix(), 
        bin_size=int(1/scale), export_position_policy='none',
        ignore_regions=['region3','region4','mark4','region5','region6']
    )
    assert isinstance(slice_meta, SliceMeta)

    w, h = int(slice_meta.w), int(slice_meta.h)
    for c in ['red', 'green', 'blue', 'yellow']:
        if slice_meta.cells is None: continue
        cells: np.ndarray | None = getattr(slice_meta.cells, c)
        if cells is None or cells.size == 0: continue
        cells = cells.astype(np.int32)
        # filter all coordinates that are out of bounds
        cells = pl.from_numpy(cells, schema=['x', 'y']).filter(
            (pl.col('x') >= 0) & (pl.col('x') < w) & (pl.col('y') >= 0) & (pl.col('y') < h)
        ).to_numpy()

        cell_mask = np.zeros(shape=(h, w), dtype=np.uint8)

        cell_mask[cells[:, 1], cells[:, 0]] = 255
        cv2.imwrite(
            (image_path.parent.parent / 'ntpCells' / f'{animal_id}_{slice_id:03d}_{c}.tif').as_posix(),
            cell_mask
        )
masks: list[Path] = []

for animal_id in animals:
    if FOR_INFER:
        slice_ids = set(cu.get_all_slice_ids(animal_id))
    else:
        slice_ids = [s.slice_id_int for s in labeled_slices if s.animal_id == animal_id]

    print(animal_id, slice_ids)
    # for slice_id in tqdm(slice_ids):
    #     # res = export_mask(animal_id, slice_id, writed_image, scale, ['GPi', 'GPe', 'VP'])
    #     res = export_mask(animal_id, slice_id, writed_image, scale, ['cla'])

    tasks = [
        delayed(export_mask)(
            animal_id, slice_id, writed_image, scale, ['cla']
        )
        for slice_id in slice_ids
    ]

    for res in tqdm(Parallel(
        n_jobs=32, verbose=10, 
        return_as='generator_unordered',
        backend='threading'
    )(tasks), total=len(tasks)):
        if res is None: continue
        masks.append(res)
    #     break
    # break
# %%pyinstrument
# masks: list[Path] = []

# for animal_id in animals:
#     if FOR_INFER:
#         slice_ids = set(cu.get_all_slice_ids(animal_id))
#     else:
#         slice_ids = [s.slice_id_int for s in labeled_slices if s.animal_id == animal_id]

#     print(animal_id, slice_ids)
#     # for slice_id in slice_ids:
#     #     res = export_mask(animal_id, slice_id, writed_image, scale, ['GPi', 'GPe', 'VP'])
#     tasks = [
#         delayed(export_mask)(
#             animal_id, slice_id, writed_image, scale, ['Cla']
#         )
#         for slice_id in slice_ids
#     ] + [
#         delayed(export_cell)(
#             animal_id, slice_id, writed_image, scale
#         )
#         for slice_id in slice_ids
#     ]

#     for res in tqdm(Parallel(
#         n_jobs=1, verbose=10, 
#         backend='multiprocessing', 
#         return_as='generator_unordered'
#     )(tasks)):
#         if res is None: continue
#         masks.append(res)

# %%
dataset_meta = Dataset_nnUNet(
    channel_names={
        0: 'Cy3',
        1: 'Cy5',
        2: 'DAPI',
        3: 'GFP',
    },
    labels={
        "background": 0,
        "Cla": 1,
        # **{k: v for k, v in utils.read_region_ids().items() if k is not None},
    },
    numTraining=len(masks),
    file_ending='.tif'
)
(current_data_set_base_path / 'dataset.json').write_text(json.dumps(asdict(dataset_meta)))

# %%
masks_names = [m.stem for m in masks]
image_names = [m.replace('-', '_') for m in writed_image]
set(image_names) - set(masks_names), set(masks_names) - set(image_names)