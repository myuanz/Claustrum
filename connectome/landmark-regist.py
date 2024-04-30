# pip install pycairo numpy matplotlib itk --index-url https://pypi.tuna.tsinghua.edu.cn/simple
# pip install ntp-manager --index-url https://ntp.mkyr.fun/pypi/
'''
工作流程：

0. 参考之前导出过的平铺图，找出连接组动物到转录组动物切片的一一对应关系，填入本工具中
1. 生成高亮图，版本填 stereo-cla-highlight-20240327-landmarker。
    连接组数据 bin=20，转录组数据 bin=13
    转录组 ntp 版本为 Mq179-CLA-sh-20240204
    目标分区填 .*(cla)|(DEn)|(shCla)|(Cla).*
    忽略分区空
2. 在 Fiji 中拖入对应的图片，然后 Plugins->BigDataViewer->Big Warp
    fix 永远选转录组
    mov 永远选转录组
3. 在关键点点击以添加 landmarker
4. 在 Landmarks 窗口点 File->Export landmarks, 保存到当前动物连接组目录中，命名为 animal-slice-chip.csv, 例如 C042-179 配准到 T67 上的 landmarks 文件，其命名为 C042-179-T67.csv
5. 点击工具上的检查，检查工具会弹出窗口，依次检查 landmarkers、mov img、fix img 是否一一对应
6. 关闭打开的图片，重复流程 2，直到所有动物应对完成，切换到下一只动物
'''

# %%

from pathlib import Path

import cairo
import connectome_utils as cutils
import itk
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ntp_manager import SliceMeta, parcellate

base_img_path = Path('/mnt/97-macaque/projects/cla/stereo-cla-highlight-20240326-test/')
base_img_scale = 1/20

fix_chip = 'T73'
mov_chip = 'C042-179'

fix_img_path = next(base_img_path.glob(f'**/*L/*{fix_chip}*.png'))
mov_img_path = next(base_img_path.glob(f'**/*L/*{mov_chip}*.png'))
print(f'{fix_img_path=}\n{mov_img_path=}')
landmark_path = base_img_path / 'landmarks.csv'
landmark_df = pl.read_csv(
    landmark_path, has_header=False, 
    new_columns=[
        'p', 'enable', 'mov_x', 'mov_y', 'fix_x', 'fix_y'
    ]
).filter('enable')
landmark_df



# %%
Dimension = 2

def try_to_itk_img(img):
    if isinstance(img, itk.Image):
        return img
    return itk.image_view_from_array(img)

def try_to_np_img(img):
    if isinstance(img, np.ndarray):
        return img
    return itk.array_from_image(img)


class ItkTransformWarp:
    def __init__(
        self, source_landmarks: np.ndarray, target_landmarks: np.ndarray, 
        T: type=itk.ThinPlateSplineKernelTransform[itk.D, Dimension] # type: ignore
    ):
        self.T = T
        self.source_landmarks = source_landmarks
        self.target_landmarks = target_landmarks
        self.source_landmarks_itk = itk.vector_container_from_array(self.source_landmarks.flatten())
        self.target_landmarks_itk = itk.vector_container_from_array(self.target_landmarks.flatten())

        self.transform = T.New()
        self.transform.GetSourceLandmarks().SetPoints(self.source_landmarks_itk)
        self.transform.GetTargetLandmarks().SetPoints(self.target_landmarks_itk)
        self.transform.ComputeWMatrix()

        self.transform_inverse = T.New()
        self.transform_inverse.GetSourceLandmarks().SetPoints(self.target_landmarks_itk)
        self.transform_inverse.GetTargetLandmarks().SetPoints(self.source_landmarks_itk)
        self.transform_inverse.ComputeWMatrix()

    @staticmethod
    def transform_points(transform, points: np.ndarray):
        new_points = []
        for p in points:
            new_p = transform.TransformPoint(p)
            new_points.append(new_p)
        return np.array(new_points, dtype=points.dtype)

    def transform_points_to_fix(self, points: np.ndarray):
        return self.transform_points(self.transform, points)
    
    def transform_points_to_mov(self, points: np.ndarray):
        return self.transform_points(self.transform_inverse, points)
    
    def transform_image_to_fix(self, mov_img, fix_img):
        return itk.resample_image_filter( # type: ignore
            try_to_itk_img(mov_img),
            use_reference_image=True,
            reference_image=try_to_itk_img(fix_img),
            transform=self.transform_inverse,
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(L={len(self.source_landmarks)})'

itw = ItkTransformWarp(
    landmark_df[['mov_x', 'mov_y']].to_numpy(), 
    landmark_df[['fix_x', 'fix_y']].to_numpy(),
)
itw

# %%
mov_animal, mov_slice = mov_chip.split('-')

mov_ntp = cutils.find_connectome_ntp(mov_animal, slice_id='179')[0]
fix_ntp = cutils.find_stereo_ntp(
    'Mq179-CLA-sh-20240204', chip=fix_chip, 
    base_path='/data/sde/ntp/macaque/'
)[0]

mov_resolution = 0.65 # um/px
mov_bin_size = 20

fix_resolution = 1 # um/px
fix_bin_size = 13

mov_w, mov_h = cutils.get_czi_size(mov_animal, mov_slice)
mov_sm = parcellate(
    mov_ntp, um_per_pixel=mov_resolution, bin_size=mov_bin_size, 
    w=mov_w / mov_bin_size, h=mov_h / mov_bin_size, 
    export_position_policy='none'
)

fix_sm = parcellate(
    fix_ntp, um_per_pixel=fix_resolution, bin_size=fix_bin_size, 
    export_position_policy='none'
)
assert isinstance(fix_sm, SliceMeta)
assert isinstance(mov_sm, SliceMeta)

# %%

for r in fix_sm.regions:
    if 'cla' not in r.label.name.lower() and 'DEn' not in r.label.name:
        continue
    exterior = np.array(r.polygon.exterior.xy).T
    plt.plot(*exterior.T, label=r.label.name, c='lightblue')


for r in mov_sm.regions:
    if 'cla' not in r.label.name.lower():
        continue
    if not r.label.name.startswith('L'):
        continue

    exterior = np.array(r.polygon.exterior.xy).T
    exterior = itw.transform_points_to_fix(exterior)
    plt.plot(*exterior.T, label=r.label.name, c='lightcoral', linestyle='--')

mov_landmarks = landmark_df[['mov_x', 'mov_y']].to_numpy()
moved_landmarks = itw.transform_points_to_fix(mov_landmarks)
plt.scatter(*moved_landmarks.T, label='mov landmarks', c='r', alpha=0.5)

fix_landmarks = landmark_df[['fix_x', 'fix_y']].to_numpy()
plt.scatter(*fix_landmarks.T, label='fix landmarks', c='g', alpha=0.5)

plt.imshow(itw.transform_image_to_fix(mov_img, fix_img), alpha=0.5)

# plt.imshow(itk.array_from_image(output_image), alpha=0.5)
# plt.imshow(itk.array_from_image(mov_img), alpha=0.5)
# plt.imshow(itk.array_from_image(fix_img), alpha=0.5)

# plt.gca().invert_yaxis()
plt.axis('equal')
# plt.legend()
