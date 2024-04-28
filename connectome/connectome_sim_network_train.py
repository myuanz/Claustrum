'''
计算图像降维，寻找不同动物的最佳匹配
'''
# %%
# %load_ext autoreload
# %autoreload 2
# %%
import base64
import pickle
import random
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from itertools import groupby, product
from pathlib import Path
from typing import Callable, Literal

import bokeh.plotting as blt
import cv2
import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import polars as pl
import seaborn as sns
import SimpleITK as sitk
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models
import torchvision.transforms.functional
import umap
import umap.plot
from bokeh.models import CategoricalColorMapper, ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_notebook, show
from connectome_utils import find_connectome_ntp, get_czi_size, ntp_p_to_slice_id
from dataset_utils import ClaDataset, get_base_path, get_dataset
from joblib import Parallel, delayed, Memory
from loguru import logger
from nhp_utils.image_correction import CorrectionPara
from ntp_manager import NTPRegion, SliceMeta, from_dict, parcellate
from PIL import Image
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.ops import linemerge, polygonize_full, unary_union
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import ToTensor, v2
from tqdm import tqdm

from utils import (
    AnimalSection,
    AnimalSectionWithDistance,
    draw_fix_and_mov,
    draw_image_with_spacing,
    draw_images,
    dtw,
    merge_draw,
    read_connectome_masks,
    read_exclude_sections,
    split_left_right,
    to_numpy,
    pad_or_crop,
    crop_black
)

warnings.simplefilter("ignore")
sitk.ProcessObject_SetGlobalWarningDisplay(False)
output_notebook()

memory = Memory('cache', verbose=0)

# %%
scale = 0.25
PROJECT_NAME = 'CLA'

connectome_mask_base_path = Path('/mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Claustrum_infer/imagesInfer2/') # 从哪里读原始数据
connectome_mask_base_path = Path('/mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Dataset004_CLA_6.8.57/imagesInfer20231031') # 从哪里读原始数据
connectome_mask_base_path = Path('/mnt/90-connectome/connectome-export/cla-reg-nnunet/20230926/Claustrum_infer/imagesInferResult20231020/')


exclude_slices = read_exclude_sections(PROJECT_NAME)
exclude_slices

# %%
animal_id = 'C042'

scale = 0.1
base_shape = np.array((1280, 1280))
# base_shape = np.array((2240, 2240))

target_shape = tuple((base_shape * scale).astype(int))
target_shape = (target_shape[0], target_shape[1])

raw_cntm_masks = read_connectome_masks(
    get_base_path(animal_id), animal_id, scale, 
    exclude_slices=exclude_slices, force=False, 
    min_sum=400
)
len(raw_cntm_masks)
# %%
item = raw_cntm_masks[33]
animal_id = item['animal_id']
slice_id = item['slice_id']

p = find_connectome_ntp(animal_id, slice_id=slice_id)[0]

w, h = get_czi_size(animal_id, slice_id, bin_size=20)
sm = parcellate(p, bin_size=20, export_position_policy='none', w=w, h=h)
assert isinstance(sm, SliceMeta)
assert sm.cells is not None

plt.imshow(item['mask'], cmap='gray')

for c in sm.cells.colors:
    cells = getattr(sm.cells, c) * scale
    plt.scatter(cells[:, 0], cells[:, 1], color=c, s=1)

for r in sm.regions:
    xy = np.array(r.polygon.exterior.coords.xy) * scale
    plt.plot(xy[0], xy[1], color='red', linewidth=0.5)

plt.title(f"{animal_id}-{slice_id}")
# %%
split_res = split_left_right(item['mask'])
assert split_res is not None

side = 'right'
remove_side = 'left'

spr = split_res[side]
minx = spr.cnt[0]
miny = spr.cnt[1]
maxy = spr.cnt[1] + spr.cnt[3]

pt = (minx, miny)
pb = (minx, maxy)

mask = item['mask'].copy()

mask[
    split_res[remove_side].cnt[1]:split_res[remove_side].cnt[1] + split_res[remove_side].cnt[3],
    split_res[remove_side].cnt[0]:split_res[remove_side].cnt[0] + split_res[remove_side].cnt[2],
] = 0

plt.figure(figsize=(10, 10))
plt.imshow(item['mask'], cmap='gray')

target_cells = {}
new_ps = defaultdict(list)

def cartesian_to_polar(x, y, base_x, base_y):
    return np.array([
        np.sqrt((x - base_x) ** 2 + (y - base_y) ** 2),
        np.arctan2(y - base_y, x - base_x),
    ])

for c in sm.cells.colors:
# for c in ('red', ):
    cells = (getattr(sm.cells, c) * scale).astype(int)
    where = mask[cells[:, 1], cells[:, 0]] > 0
    cells = cells[where]
    target_cells[f'{side}-{c}'] = cells.copy()

    plt.scatter(cells[:, 0], cells[:, 1], color=c, s=1)

    for i, draw_c in zip((pt, pb), ('blue', 'green')):
        for cell in cells:
            plt.axline(i, cell, linewidth=0.5, c=draw_c, alpha=0.2)
        # cells 直角坐标系到极坐标系
        cells_pol = np.array([
            cartesian_to_polar(cell[0] - i[0], cell[1] - i[1], *i)
            for cell in cells
        ])
        new_ps[c].append(cells_pol)

plt.scatter(*pt, color='red', s=10)
plt.scatter(*pb, color='red', s=10)
# %%
cells = np.hstack(new_ps['blue'])
# %%
mapper = umap.UMAP(n_components=2, n_neighbors=64, metric='cosine')
embedding = mapper.fit_transform(cells)
# %%
plt.scatter(embedding[:, 0], embedding[:, 1], s=1)

# %%
draw_image_with_spacing([i['mask'] for i in raw_cntm_masks], 12, shape=(3, 4), 
    titles=[f"{i['animal_id']}-{i['slice_id']}" for i in raw_cntm_masks]
)
# %%
DEVICE = DEV = torch.device('cuda:0')
BATCH_SIZE = 64

# (h, w, n) -> (h, w, 3, n)
# raw_images = np.repeat(side_splited['left'][:, :, np.newaxis, ...], 3, axis=2)
# print(raw_images.shape)
# raw_images = [
#     raw_images[:, :, :, i] for i in range(raw_images.shape[-1])
# ]
# raw_slice_ids = [i['slice_id'] for i in raw_cntm_masks]

curr_dataset = get_dataset('left-right', raw_cntm_masks, 32, shuffle=False, target_shape=target_shape)

len(curr_dataset.raw_images), len(curr_dataset.raw_slice_ids)
# %%
draw_image_with_spacing(curr_dataset.raw_images, 16)
# %%
next(iter(curr_dataset.dataloader))
for i in curr_dataset.dataloader:
    print(i[0].shape, i[1].shape, i[0].max(), i[0].min())
# %%
class PairDataset(Dataset):
    def __init__(self, *datasets: ClaDataset):
        self.datasets = datasets
        self.ranges: dict[tuple[int, int], ClaDataset] = {}

        start_index = 0
        for s in self.datasets:
            end_index = len(s.raw_slice_ids) + start_index
            self.ranges[(start_index, end_index)] = s
            start_index = end_index

    def query_dataset(self, index: int):
        for (start_index, end_index), s in self.ranges.items():
            if start_index <= index < end_index:
                break
        else:
            raise IndexError(f'{index=} out of range')
        return s

    def __getitem__(self, index):
        s = self.query_dataset(index)
        index = index % len(s.raw_slice_ids)

        image1, slice_id1 = s.dataset[index]
        image2, slice_id2 = s.dataset[random.randint(0, len(s.raw_slice_ids) - 1)]
        return image1, slice_id1, image2, slice_id2

    def __len__(self):
        return sum(len(i.raw_slice_ids) for i in self.datasets)
# %%
animal_ids = [
    'C006', 'C007', 'C008', 'C011', 'C012', 'C013', 
    'C015', 'C018', 'C057', 'C075', 'C042'
]
# animal_ids = (['C057', 'C042'])
# animal_ids = ['C042']
animal_ids = ['C075']
animal_ids = '''C006
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
C097'''.splitlines()
animal_ids.sort()

cla_datasets = []
BATCH_SIZE = 480

for current_animal_id in tqdm(animal_ids):
    train_raw_cntm_masks = read_connectome_masks(
        get_base_path(current_animal_id), current_animal_id, scale,
        exclude_slices=exclude_slices, force=False, 
        min_sum=400
    )
    if not train_raw_cntm_masks:
        print(f'{current_animal_id} not mask')
        continue
    for side in ('left', 'right'):
        cla_datasets.append(get_dataset(
            side, train_raw_cntm_masks, 
            batch_size=BATCH_SIZE, shuffle=True, 
            animal_id = current_animal_id
        ))

train_pair_dataset = PairDataset(*cla_datasets)

# %%
for ds in DataLoader(train_pair_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0):
    print(ds[0].shape, ds[1].shape, ds[0].max().item(), ds[0].min().item())


# %%
len(train_pair_dataset)
# %%
img1, sli1, img2, sli2 = train_pair_dataset[100]
plt.subplot(121)
plt.imshow(to_numpy(img1[0]), cmap='gray')
plt.title(f'{sli1.item()}')
plt.subplot(122)
plt.imshow(to_numpy(img2[0]), cmap='gray')
plt.title(f'{sli2.item()}')


# %%
class SimilarityNet(torch.nn.Module):
    def __init__(self, base_model, output_dim=128, dropout=0.5):
        super().__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.base_model.fc.in_features, output_dim)
        self.fc_after = torch.nn.Identity()
        self.fc_after = torch.nn.Tanh()

        self.fc_theta = torch.nn.Linear(self.base_model.fc.in_features, 1)
        self.fc_theta_act = torch.nn.Tanh()

        self.fc_side = torch.nn.Linear(self.base_model.fc.in_features, 2) # 0 朝左，1 朝右
        self.fc_side_act = torch.nn.Softmax(dim=1)

        # self.fc_after = torch.nn.BatchNorm1d(output_dim)
        # self.fc_after = torch.nn.LogSigmoid()
        # self.act = torch.nn.ReLU()
        # self.act = torch.nn.Softmax()

        self.base_model.fc = torch.nn.Identity()

        self.output_dim = output_dim
        self.dropout_ratio = dropout

    def forward(self, image1, image2):
        x1 = self.base_model(image1)
        x2 = self.base_model(image2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1_theta = (self.fc_theta_act(self.fc_theta(x1))) # -1 ~ 1, 乘 180 换算到角度
        x2_theta = (self.fc_theta_act(self.fc_theta(x2)))

        # x1_side = self.fc_side_act(self.fc_side(x1))
        # x2_side = self.fc_side_act(self.fc_side(x2))
        x1_side = (self.fc_side(x1))
        x2_side = (self.fc_side(x2))
        x1 = self.fc(x1)
        x2 = self.fc(x2)


        # print(x1, self.fc_after(x1))
        x1 = self.fc_after(x1)
        x2 = self.fc_after(x2)

        # x1 = self.act(x1)
        # x2 = self.act(x2)
        return x1, x2, x1_theta, x2_theta, x1_side, x2_side

base_model = torchvision.models.resnet18()
model = SimilarityNet(base_model, output_dim=16, dropout=0.25).to(DEVICE)


def embed_dist_fn(embedding1, embedding2):
    embedding_d = (1 - torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1).unsqueeze(1)) / 2
    return embedding_d

def loss_fn3(slice_id1, slice_id2, embedding1, embedding2):
    slice_d = ((slice_id2 - slice_id1) / 16).float().abs()
    # slice_d = torch.clip(slice_d, 0, 1)
    slice_d = (torch.sigmoid(slice_d) - 0.5) * 2

    embedding_d = embed_dist_fn(embedding1, embedding2).reshape(-1)
    res = (slice_d - embedding_d).pow(2)
    # print(f'{slice_d=}', f'{embedding_d=}', f'{res=}')
    # print(f'{slice_d.shape=}', f'{embedding_d.shape=}', f'{res.shape=}')
    return res

def theta_loss_fn(theta_true, theta_pred) -> torch.Tensor:
    if torch.isnan(theta_true).any():
        theta_true = torch.zeros_like(theta_pred)

    return (theta_true - theta_pred).pow(2)

_side_loss = nn.CrossEntropyLoss()

def side_loss_fn(side_pred, target) -> torch.Tensor:
    # side_true: (batch_size, 2), side_pred: (batch_size, )
    return _side_loss(side_pred, target)

study_name = f'{datetime.now().strftime("%Y%m%d-%H%M")}-lowlr-128-pairtrain-16div-firstdropout-d0.25-1dim-circle'
logdir = Path(f'output/runs/{study_name}')

writer = SummaryWriter(logdir=logdir.as_posix())

lr     = 1e-3 # 大于 1e-4不能收敛
min_lr = 1e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
losses = []

cfg = {'config': {
    'lr': lr,
    'study_name': study_name,
    'min_lr': min_lr,
    'optimizer': optimizer.__class__.__name__,
    'model': {
        'name': f'{model.__class__.__name__}-{base_model.__class__.__name__}',
        'output_dim': model.output_dim,
        'dropout': model.dropout_ratio,
    },
    'batch_size': BATCH_SIZE,
    'target_shape': target_shape,
    'scale': scale,
}}
toml.dump(cfg, open(f'{logdir}/cfg.toml', 'w'))

# %%
@dataclass
class RFConfig:
    rotates: torch.Tensor
    flips: torch.Tensor

    def side_target_tensor(self, side: Literal['left', 'right']) -> torch.Tensor:
        ''' 左 & noflip = 0, 左 & flip = 1, 右 & noflip = 1, 右 & flip = 0 '''
        batch_size = self.rotates.shape[0]

        side_t = torch.zeros((batch_size, ), dtype=torch.long).to(DEVICE)
        if side == 'left':
            side_t[self.flips == 1] = 1
        elif side == 'right':
            side_t[self.flips == 0] = 1
        return side_t


def get_affine_mat(degrees: torch.Tensor, flips: torch.Tensor):
    B, *_ = degrees.shape
    flip_mat = torch.zeros((B, 3, 3))
    flip_mat[:, 1, 1] = 1
    flip_mat[:, 0, 0] = 1 - 2*flips
    flip_mat[:, 2, 2] = 1

    affine_mat = torch.zeros((B, 3, 3))
    affine_mat[:, 0, 0] = torch.cos(degrees / 180 * np.pi)
    affine_mat[:, 0, 1] = -torch.sin(degrees / 180 * np.pi)
    affine_mat[:, 1, 0] = torch.sin(degrees / 180 * np.pi)
    affine_mat[:, 1, 1] = torch.cos(degrees / 180 * np.pi)
    affine_mat[:, 2, 2] = 1

    # handle w flip

    affine_mat = torch.matmul(flip_mat, affine_mat)[..., :2, :]
    return affine_mat


def get_random_rotate_flip(
    max_degree: float = 0.0, flip_p: float = 0
) -> Callable[[torch.Tensor], tuple[torch.Tensor, RFConfig]]:
    def _f(image: torch.Tensor) -> tuple[torch.Tensor, RFConfig]:
        '''image: (B, 3, h, w)'''

        B, *_ = image.shape
        degrees = (torch.rand(B) - 0.5) * max_degree * 2
        xflips = torch.rand(B) < flip_p
        affine_mat = get_affine_mat(degrees, xflips)
        affine_mat = affine_mat.to(image.device)
        grid = F.affine_grid(affine_mat, [*image.size()])
        image = F.grid_sample(image, grid)
        return image, RFConfig(degrees.to(image.device), xflips.to(image.device))
    return _f
# # %%
# loader = train_cla_datasets['right'][0].dataloader
# images, _ = next(iter(loader))
# # %%
# to_draw_images = [
#     to_numpy(images[i]) for i in range(len(images))
# ]
# draw_image_with_spacing(to_draw_images, 16, )
# # %%

# rot_images = get_random_rotate_flip(max_degree=45, flip_p=0.5)(images)[0]
# to_draw_images = [
#     to_numpy(rot_images[i]) for i in range(len(rot_images))
# ]
# draw_image_with_spacing(to_draw_images, 16, )

# %%
max_degree = 45
BATCH_SIZE = 512

aug = v2.Compose([
    # v2.RandomHorizontalFlip(),
    # v2.RandomAffine(degrees=30, translate=(0.2, 0.1), scale=(0.666, 1.5), shear=(-30, 30, -30, 30))
    v2.RandomAffine(degrees=45.0 * 0.0, translate=(0.2, 0.1), scale=(0.909, 1.1))
])
model.train()
if lr < (1e-5) / 2:
    lr = (8e-4)
lr = 1e-4
min_lr = 1e-5
save_epoch = 100

rrf = get_random_rotate_flip(max_degree=max_degree, flip_p=0.5)
dataloader = DataLoader(train_pair_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

with tqdm(range(2000)) as pbar:
    for epoch in pbar:
        current_losses = []

        for fix_images, fix_slice_ids, moving_images, moving_slice_ids in dataloader:
            fix_images   , fix_rf    = rrf(fix_images)
            moving_images, moving_rf = rrf(moving_images)
            
            # fix_images    = aug(fix_images)
            # moving_images = aug(moving_images)
            # if side == 'right':
            #     moving_images = torch.flip(moving_images, dims=(3, ))

            e1, e2, theta1, theta2, flip1, flip2 = model(
                fix_images, moving_images
            )

            dist_loss = torch.abs(loss_fn3(
                fix_slice_ids, moving_slice_ids, 
                e1, e2
            )).mean()
            rotate_loss = (theta_loss_fn(
                fix_rf.rotates / max_degree, theta1.reshape(-1)
            ) + theta_loss_fn(
                moving_rf.rotates / max_degree, theta2.reshape(-1)
            )).mean()
            flip_loss = (side_loss_fn(
                flip1, fix_rf.side_target_tensor(side)
            ) + side_loss_fn(
                flip2, moving_rf.side_target_tensor(side)
            ))
            e = torch.concat([e1, e2])

            norm = torch.norm(e, dim=1)
            unit_circle_loss = (norm ** 2 - 1).pow(2).mean()

            loss = dist_loss + rotate_loss * 0 + flip_loss * 0 + unit_circle_loss * 0
            losses.append(dict(
                total_loss  = loss.item(),
                dist_loss   = dist_loss.item(),
                rotate_loss = rotate_loss.item(),
                flip_loss   = flip_loss.item(),
                unit_circle_loss = unit_circle_loss.item(), 
            ))
            current_losses.append(losses[-1])
            # print(losses[-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if torch.isnan(loss): 
                print('break with nan', dist_loss)
                break
            # break
            # print(f'{loss=}')

            if len(current_losses) % 29 == 0 and lr > min_lr:
                # reduce lr
                lr *= 0.95
                writer.add_scalar('lr', lr, len(losses))
                for g in optimizer.param_groups:
                    g['lr'] = lr
            # if max_degree < 90:
            #     max_degree += 0.0

            writer.add_scalar('total_mean_loss', np.mean([i['total_loss'] for i in losses[-100:]]), len(losses))
            writer.add_scalar('max_degree', max_degree, len(losses))

            for key in current_losses[0].keys():
                writer.add_scalar(f'loss/{key}', np.mean([i[key] for i in current_losses]), len(losses))
            mean_curr_loss = np.mean([i["total_loss"] for i in current_losses])
 
            pbar.set_description(f'[{epoch}] @ {lr:.6f} loss: {mean_curr_loss:3.8}')
        if torch.isnan(loss): 
            print('break with nan', dist_loss)
            break

        if epoch % save_epoch == 0:
            torch.save(model.state_dict(), logdir / f'model-{epoch}.pth')
        
for key in losses[0].keys():
    plt.plot(np.log([i[key] for i in losses]), label=key) # best 3e-4
    plt.legend()
torch.save(model.state_dict(), logdir / 'model-final.pth')
# %%
torch.save(model.state_dict(), logdir / 'model-final.pth')

# %%
# plt.plot((losses))
# %%
images = []
slice_ids = []

# cla_dataset = train_cla_datasets['left'][0]
# for i in cla_dataset.dataset:
#     images.append(i[0])
#     slice_ids.append(i[1].item())

to_draw_images = [
    to_numpy(moving_images[i]) for i in range(len(moving_images))
]
draw_image_with_spacing(to_draw_images[:16], 16, )

# %%
# torch.save(model.state_dict(), 'output/runs/with-theta-twice-side2-noarcsin-nostd-more-data-aug-highlr/model.pth')
# %%
base_model = torchvision.models.resnet18()
model = SimilarityNet(base_model, output_dim=2, dropout=0.25).to(DEVICE)
model.load_state_dict(torch.load('output/runs/20231123-2337-lowlr-128-pairtrain-16div-firstdropout-d0.25-1dim/model-final.pth'))
# model.load_state_dict(torch.load('output/model.pth'))


# base_model = torchvision.models.resnet18()
# model = SimilarityNet(base_model, output_dim=16, dropout=0.25).to(DEVICE)
# model.load_state_dict(torch.load('output/runs/20231121-1654-lowlr-128-pairtrain-total-16div-firstdropout-d0.25/model-final.pth'))

# %%

# umap_animal_ids = ['C008', 'C057']
# umap_animal_ids = animal_ids + ['C042']
# umap_animal_ids = animal_ids
umap_animal_ids = ['C042', 'C057', 'C008']
umap_animal_ids = ['C075'] + umap_animal_ids

umap_datasets: list[ClaDataset] = []

for current_animal_id in umap_animal_ids:
    draw_raw_cntm_masks = read_connectome_masks(
        get_base_path(current_animal_id), current_animal_id, scale,
        exclude_slices=exclude_slices, force=False, 
        min_sum=400
    )
    for side in ('left', 'right'):
        draw_dataset = get_dataset(side, draw_raw_cntm_masks, batch_size=232, shuffle=False, animal_id=f'{current_animal_id}-{side[0].upper()}')

        umap_datasets.append(draw_dataset)
# %%
umap_datasets = cla_datasets
# %%
def poly_to_bitmap(poly, w=0, h=0, buffer_x=50, buffer_y=50):
    minx, miny, maxx, maxy = poly.bounds
    if minx < 0 or miny < 0:
        print('Polygon has negative coordinates')

    coords = np.array(poly.exterior.coords, dtype=np.int32)
    coords[:, 0] -= int(minx) - buffer_x
    coords[:, 1] -= int(miny) - buffer_y
    if w == 0: w = int(maxx - minx) + buffer_x * 2
    if h == 0: h = int(maxy - miny) + buffer_y * 2

    bitmap = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(bitmap, [coords], color=(255, ))
    return bitmap


def search_stereo_cla(chip: str, ntp_version='Mq179-CLA-20230505', bin_size=1.0):
    slice_meta_p = next((Path('/data/sde/ntp/macaque') / ntp_version / 'region-mask').glob(f'*{chip}*.json'))

    slice_meta: dict = orjson.loads(slice_meta_p.read_bytes())
    regions: list[NTPRegion] = [
        from_dict(NTPRegion, r) for r in slice_meta['regions']
    ]
    corr = CorrectionPara.select(chip)[0]

    all_cla_list = []

    for r in regions:
        if 'cla' not in r.label.name: continue
        if 'lim' in r.label.name: continue
        
        points = np.array(r.polygon.exterior.xy).T
        points = corr.wrap_point(points) / bin_size
        all_cla_list.append(Polygon(points))
    #     plt.plot(*points.T)
    # plt.axis('equal')
    all_cla: Polygon | MultiPolygon = unary_union(all_cla_list)
    # if multi polygon, choose the largest one
    if isinstance(all_cla, MultiPolygon):
        all_cla = max(all_cla.geoms, key=lambda x: x.area)
    
    img = poly_to_bitmap(all_cla, buffer_x=5, buffer_y=5)
    img = pad_or_crop(img, 128, 128)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return {
        'chip': chip,
        'chip_num': int(chip[1:]),
        'polygon': all_cla,
        'image': img,
    }

# y reverse
# plt.figure(figsize=(10, 10))
# plt.gca().invert_yaxis()

stereo_res = []
for chip_num in tqdm(range(167)):
    try: t = search_stereo_cla(f'T{chip_num}', bin_size=315)
    except: continue

    stereo_res.append(t)

# %%

# for item in stereo_res:
#     img = item['image']
#     plt.imshow(img)
#     plt.title(item['chip'] + f' {(img[:, :, 0] != 0).sum()}')
#     plt.show()
# %%
draw_image_with_spacing([i['image'] for i in stereo_res], 16, titles=[
    (i['image'][..., 0] != 0).sum() for i in stereo_res
])
# %%


# %%
def stereo_res_to_cla_dataset(animal_id: str, res: list[dict], device = 'cuda', shuffle=False, batch_size=128, min_sum=100):
    res = [i for i in res if (i['image'][..., 0] != 0).sum() > min_sum]
    raw_images = [i['image'] / 255 for i in res]
    raw_slice_ids = [i['chip_num'] for i in res]
    

    dataset = TensorDataset(
        (torch.tensor(np.array(raw_images)).float().permute(0, 3, 1, 2) / 1).to(device),
        (torch.tensor(raw_slice_ids).long()).to(device),
    )
    dataloaders = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )

    return ClaDataset(
        animal_id     = animal_id,
        raw_images    = raw_images,
        raw_slice_ids = raw_slice_ids,
        dataset       = dataset,
        dataloader    = dataloaders,
        side          = 'left'
    )
stereo_cla_dataset = stereo_res_to_cla_dataset('Mq179', stereo_res, min_sum=140)
stereo_cla_dataset
# %%
umap_datasets = [*cla_datasets, stereo_cla_dataset]
# %%
umap_datasets = [stereo_cla_dataset]
# %%
results = {
    'es'           : [],
    'slice_ids'    : [],
    'raw_slice_ids': [],
    'images'       : [], 
    'animal_ids'   : [],
}
model.eval()

for umap_dataset in tqdm(umap_datasets):
    curr_es = []
    curr_images = []
    for images, slice_ids in umap_dataset.dataloader:
        current_batch_size = images.shape[0] // 2 * 2
        slice_id1 = slice_ids[:current_batch_size//2].unsqueeze(1)
        slice_id2 = slice_ids[current_batch_size//2:].unsqueeze(1)
        images1 = images[:current_batch_size//2]
        images2 = images[current_batch_size//2:]
        e1, e2, *_ = model(images1, images2)

        es = torch.vstack((e1, e2)).detach().cpu().numpy()
        slice_ids = torch.vstack((slice_id1, slice_id2)).reshape(-1).cpu().numpy()


        curr_es.append(es)
        curr_images.append(images.detach().cpu().numpy()[:, 0, ...])


        results['raw_slice_ids'].extend(slice_ids)
        results['slice_ids'].extend(slice_ids - slice_ids.min())
        results['animal_ids'].extend([umap_dataset.animal_id] * len(slice_ids))
    results['images'].append(np.vstack(curr_images))
    results['es'].append(np.vstack(curr_es))


# %%
color_names = ['Purples', 'Blues', 'Greens', 'YlOrRd', 'Oranges', 'Reds', 'GnBu', 'PuBuGn']
colors_rgb = []
for i in range(len(umap_datasets)):
    curr_len = len(results['es'][i])
    c = color_names[i % len(color_names)]
    mpl_color = mpl.colormaps[c](np.linspace(0, 1, curr_len))
    colors_rgb.extend(mpl_color)
    # print(curr_len, len(mpl_color))

reducer = umap.UMAP(n_components=2, n_neighbors=128, metric='cosine')

res = np.vstack(results['es'])
slice_ids = np.hstack(results['slice_ids'])
# plt.hist(res.reshape(-1), bins=100)

embedding = reducer.fit_transform(res) # type: ignore


get_color = mpl.colormaps['viridis']
draw_slice_ids = slice_ids / slice_ids.max()

if embedding.shape[1] == 2:
    plt.scatter(
        embedding[:, 0], embedding[:, 1], 
        s=(draw_slice_ids * 30 + 5).astype(int), 
        alpha=0.8,
        c=colors_rgb
    )
else:
    plt.scatter(
        draw_slice_ids, embedding, 
        s=[int(i*300) for i in draw_slice_ids], 
        alpha=0.5,
        c=[get_color(i) for i in draw_slice_ids]
    )
plt.colorbar()
# plt.title(f'embedding umap {"|".join([i.animal_id for i in umap_datasets])}')
plt.savefig(logdir / 'umap.png')


# %%
# ax = plt.subplot(111, projection='polar')
ax = plt.subplot(111)
# axie equal
ax.set_aspect('equal', 'box')
rho = np.sqrt(res[:, 0] ** 2 + res[:, 1] ** 2)
theta = np.arctan(res[:, 0], res[:, 0])
draw_slice_ids = slice_ids / slice_ids.max()

ax.scatter(
    res[:, 0], res[:, 1],
    # theta, rho, 
    s=(draw_slice_ids * 300 + 0).astype(int), 
    alpha=0.4,
    c=colors_rgb
)
plt.savefig(logdir / '2d-draw.png')

# plt.plot(theta, 'o', alpha=0.5)
# %%
plt.imshow(res)
# %%
stereos = [i for i, ani in enumerate(results['animal_ids']) if ani == 'Mq179']
all_res = np.vstack(results['es'])

ax = plt.subplot(111)
ax.set_aspect('equal', 'box')

res = all_res[[i for i in range(len(all_res)) if i not in stereos]]
ax.scatter(
    res[:, 0], res[:, 1],
    alpha=0.4, s=1,
    label='connectome cla'
)
res = all_res[stereos]
ax.scatter(
    res[:, 0], res[:, 1],
    alpha=0.3, c='red', s=40,
    label='stereo cla'
)
plt.legend()
# %%
ax = plt.subplot(111)
ax.set_aspect('equal', 'box')

res = all_res[[i for i in range(len(all_res)) if i not in stereos]]
rho = np.sqrt(res[:, 0] ** 2 + res[:, 1] ** 2)
theta = np.arctan(res[:, 0], res[:, 0])

ax.scatter(
    # res[:, 0], res[:, 1],
    theta, rho, 
    alpha=0.4, s=1,
    label='connectome cla'
)
res = all_res[stereos]
rho = np.sqrt(res[:, 0] ** 2 + res[:, 1] ** 2)
theta = np.arctan(res[:, 0], res[:, 0])

ax.scatter(
    # res[:, 0], res[:, 1],
    theta, rho, 
    alpha=0.3, c='red', s=40,
    label='stereo cla'
)
plt.legend()
# 设置x轴标签
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\rho$')
plt.xlim(-1, 1)
plt.ylim(0, 1.5)


# %%
plt.plot(res[:, 0])
# %%

def embeddable_image(data):
    img_data = (255 - 255 * data).astype(np.uint8)
    # print(img_data[0, :, :].max(), img_data[0, :, :].min())
    image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.Resampling.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
# digits_df = pd.DataFrame(res, columns=('x', 'y'))

digits_df['raw_slice_id'] = results['raw_slice_ids']
digits_df['animal_id'] = results['animal_ids']
digits_df['image'] = list(map(embeddable_image, np.vstack(results['images'])))

datasource = ColumnDataSource(digits_df)
# color_mapping = CategoricalColorMapper(
#     factors=1,
#     palette=Spectral10
# )

plot_figure = figure(
    title='UMAP projection of the Digits dataset',
    # outer_width=600,
    # outer_height=600,
    tools=('pan, wheel_zoom, reset')
)

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 25px 25px 25px 25px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>section id:</span>
        <span style='font-size: 18px'>@animal_id-@raw_slice_id</span>
    </div>
</div>
"""))

plot_figure.circle(
    'x',
    'y',
    source=datasource,
    # color=dict(field='animal_id', transform=color_mapping),
    # color=[('red', 'green')[int(a[1:]) % 2] for a in digits_df['animal_id']], 
    line_alpha=0.6,
    fill_alpha=0.6,
    size=6
)
show(plot_figure)
blt.save(plot_figure, logdir / 'umap.html')

# %%
curr_fix = umap_datasets[1]
curr_moving = umap_datasets[2]

# %%
model.eval()

confusion_matrix = np.zeros((max(curr_fix.raw_slice_ids)+1, max(curr_moving.raw_slice_ids)+1))

for current_data_fix in tqdm(curr_fix.dataset):
    fix_image, fix_slice = current_data_fix

    fix_slice_ids    = fix_slice.repeat(BATCH_SIZE, 1)
    current_slice_id = fix_slice.item()
    # print(current_slice_id)

    for moving_mask, moving_slice_ids in curr_moving.dataloader:
        current_batch_size = moving_mask.shape[0]
        fix_images         = fix_image.repeat(current_batch_size, 1, 1, 1)

        e1, e2, *_ = model(fix_images, moving_mask)
        d = embed_dist_fn(
            e1, e2
        )
        confusion_matrix[
            current_slice_id, 
            moving_slice_ids.detach().cpu().numpy()
        ] = 1 - d.detach().cpu().numpy()[:, 0]
        # print(d)
    # break
plt.imshow(confusion_matrix)
# %%
confusion_matrix_nonzero = confusion_matrix.copy()
confusion_matrix_nonzero[confusion_matrix_nonzero == 0] = np.nan
confusion_matrix_nonzero = confusion_matrix_nonzero[~np.isnan(confusion_matrix_nonzero).all(axis=1)]
confusion_matrix_nonzero = confusion_matrix_nonzero[:, ~np.isnan(confusion_matrix_nonzero).all(axis=0)]
plt.imshow(confusion_matrix_nonzero)
paths = dtw(confusion_matrix_nonzero)
plt.plot(paths[:, 1], paths[:, 0], color='red', linewidth=2)

plt.colorbar()
plt.title(f'confusion matrix {curr_fix.animal_id}-{curr_fix.side} vs {curr_moving.animal_id}-{curr_moving.side}')
plt.savefig(logdir / f'confusion_matrix_{curr_fix.animal_id}_vs_{curr_moving.animal_id}.png')
# %%

# %%
