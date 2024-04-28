from pathlib import Path
import itk
import numpy as np
import polars as pl

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

    @staticmethod
    def read_big_warp_df(p: str | Path):
        return pl.read_csv(
            p, has_header=False,
            new_columns=[
                'p', 'enable', 'mov_x', 'mov_y', 'fix_x', 'fix_y'
            ]
        ).filter('enable').cast({
            'mov_x': pl.Float64,
            'mov_y': pl.Float64,
            'fix_x': pl.Float64,
            'fix_y': pl.Float64,
        })

    @staticmethod
    def from_big_warp_df(df: pl.DataFrame | str | Path):
        '''https://imagej.net/plugins/bigwarp'''
        if isinstance(df, str) or isinstance(df, Path):
            df = ItkTransformWarp.read_big_warp_df(df)
        source_landmarks = df[['mov_x', 'mov_y']].to_numpy()
        target_landmarks = df[['fix_x', 'fix_y']].to_numpy()

        return ItkTransformWarp(source_landmarks, target_landmarks)
