# pip install numpy itk pyinstaller nuitka PySide6 --index-url  https://pypi.tuna.tsinghua.edu.cn/simple
# pip install ntp-manager connectome_utils --index-url https://ntp.mkyr.fun/pypi/ --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
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

import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Literal, cast
from itertools import groupby

import connectome_utils as cutils
import itk
import numpy as np
import polars as pl
from correction_para import CorrectionPara
from ntp_manager import SliceMeta, parcellate
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QAction, QColor, QMouseEvent, QPainter, QPolygonF, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsBlurEffect,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QStatusBar,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

curr_os = sys.platform
color_to_tracer = {
    'blue'  : ('FB', (0, 0, 255, 200)),
    'yellow': ('CTB555', (0, 255, 255, 200)),
    'red'   : ('CTB647', (255, 0, 0, 200)),
    'green' : ('CTB488', (0, 255, 0, 200)),
}
mov_resolution = 0.65 # um/px
mov_bin_size = 20

fix_resolution = 1 # um/px
fix_bin_size = 13

def find_connectome_ntp(
    animal_id: str, /, *,
    slice_id: int | str | None = None, 
    base_path: str='/mnt/90-connectome/finalNTP-layer4-parcellation91-106/',
    type = '',
    skip_pre_pro: bool = True,
) -> list[Path]:
    pa_paths = [i for i in Path(base_path).glob(f'{animal_id}*') if i.is_dir()]
    if skip_pre_pro:
        pa_paths = [p for p in pa_paths if 'Pre' not in p.name and 'Pro' not in p.name] # Pre Data 是待处理的，要跳过
    else:
        pa_paths = [p for p in pa_paths]

    if not pa_paths:
        return []

    pa_path = pa_paths[0] / type

    match slice_id:
        case None:
            mode = f'{animal_id}-*.ntp'
        case int(slice_id):
            mode = f'{animal_id}-{slice_id:03d}*.ntp'
        case str(slice_id):
            mode = f'{animal_id}-{slice_id}*.ntp'
        case _:
            raise ValueError(f'Invalid slice_id: {animal_id}-{slice_id}')

    ntp_path = pa_path.glob(mode)
    return list(ntp_path)


@lru_cache(maxsize=256)
def parcellate_with_cache(
    ntp_path: str | Path, 
    /, *, 
    um_per_pixel: float = 0.65,
    bin_size: int = 1, 
    verbose: bool = False, ignore_regions: list[str] = ['region3','region4','mark4'],
    # region3 是是血管，region4 是检查，mark4是 layer4 但是在时空芯片里应该全都选
    background_path: str = '', 
    export_position_policy: Literal['default', 'none'] | Path = 'default',
    w: int=0, h: int=0, 
    return_bytes: bool = False, 
    transpose: bool = False,
):
    return parcellate(
        ntp_path, um_per_pixel=um_per_pixel, bin_size=bin_size, 
        verbose=verbose, ignore_regions=ignore_regions,
        background_path=background_path, export_position_policy=export_position_policy,
        w=w, h=h, return_bytes=return_bytes, transpose=transpose,
    )

@cache
def match_regex(regex: str, text: str):
    return re.search(regex, text)

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

    @cache
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
        

@dataclass
class Layer:
    name: str
    data: list[np.ndarray] | np.ndarray
    '''对于多边形，数据格式为list[np.ndarray]，不同的arr是不同的多边形。对于点，数据格式为np.ndarray'''
    type: Literal['point', 'polygon']
    colors: list[tuple[int, int, int, int]]
    enabled: bool = True


    @staticmethod
    def merge(*layers: 'Layer'):
        sorted_layers = sorted(layers, key=lambda x: x.name)
        for name, g in groupby(sorted_layers, lambda x: x.name):
            g = list(g)
            data = []
            types = []
            colors = []
            enableds = []
            for gi in g:
                if len(gi.data) == 0:
                    continue
                if isinstance(gi.data, list):
                    data.extend(gi.data)
                else:
                    data.append(gi.data)
                types.append(gi.type)
                colors.extend(gi.colors)
                enableds.append(gi.enabled)
            if len(data) == 0: continue
            assert len(set(types)) <= 1, f'{types=}, must be same'
            if types[0] == 'point':
                data = np.concatenate(data, axis=0)

            # print(data)
            yield Layer(
                name=name,
                data=data,
                type=types[0],
                colors=colors,
                enabled=all(enableds)
            )
        

fix_chips = '''T105
T101
T97
T93
T89
T85
T81
T77
T73
T67
T63
T59
T55
T49
T47
T43
T39
T33
T27
T28
T32'''.splitlines()
fix_ntp_version = 'Mq179-CLA-sh-20240204'

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setMouseTracking(True)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            zoom_factor = 1.25
            if event.angleDelta().y() < 0:
                zoom_factor = 1 / zoom_factor
            self.scale(zoom_factor, zoom_factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setInteractive(False)
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setInteractive(True)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("cla landmark checker")
        self.resize(1200, 900)

        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()

        self.module_tab = QTabWidget()
        self.module_tab.setMaximumWidth(300)

        regist_tab_layout = QVBoxLayout()
        form_layout = QVBoxLayout()

        form_layout.addWidget(QLabel("inner-data"))
        self.inner_data_input = QLineEdit()
        self.inner_data_input.setText('A:/')
        if curr_os == 'linux':
            self.inner_data_input.setText('/data/sdf/')
        form_layout.addWidget(self.inner_data_input)
        
        form_layout.addWidget(QLabel("ntp 基本路径"))
        self.connectome_ntp_base_path_input = QLineEdit()
        self.connectome_ntp_base_path_input.setText('Z:/finalNTP-layer4-parcellation91-106')
        if curr_os == 'linux':
            self.connectome_ntp_base_path_input.setText(
                '/mnt/90-connectome/finalNTP-layer4-parcellation91-106'
            )
        form_layout.addWidget(self.connectome_ntp_base_path_input)
        horizontal_line = QFrame()
        horizontal_line.setFrameShape(QFrame.Shape.HLine)
        horizontal_line.setFrameShadow(QFrame.Shadow.Sunken)
        form_layout.addWidget(horizontal_line)

        form_layout.addWidget(QLabel("动物编号"))
        self.animal_id_input = QLineEdit()
        if curr_os == 'linux':
            self.animal_id_input.setText('C025')
        self.animal_id_input.textChanged.connect(self.read_match_file_to_widget)
        form_layout.addWidget(self.animal_id_input)
        form_layout.addWidget(QLabel("左、右侧"))
        self.brain_direction_input = QComboBox()
        self.brain_direction_input.addItems(["L", "R"])
        self.brain_direction_input.setCurrentIndex(1)
        self.brain_direction_input.currentIndexChanged.connect(self.read_match_file_to_widget)
        form_layout.addWidget(self.brain_direction_input)
        self.base_path_input = QLineEdit()
        self.base_path_input.setText(r'M:\Macaque\projects\cla\stereo-cla-highlight-20240327-landmarker')
        if curr_os == 'linux':
            self.base_path_input.setText(
                r'/mnt/97-macaque/projects/cla/stereo-cla-highlight-20240327-landmarker/'
            )
        self.base_path_input.textChanged.connect(self.read_match_file_to_widget)
        form_layout.addWidget(QLabel("高亮目录"))
        form_layout.addWidget(self.base_path_input)

        form_layout.addWidget(QLabel("经办人"))
        self.operator_input = QLineEdit()
        self.operator_input.textChanged.connect(self.save_match_file)
        form_layout.addWidget(self.operator_input)

        regist_tab_layout.addLayout(form_layout)

        self.slice_table = QTableWidget()
        self.slice_table.setColumnCount(2)
        self.slice_table.setHorizontalHeaderLabels(["Fix", "Mov"])
        self.slice_table.setRowCount(len(fix_chips))
        # self.slice_table.cellChanged.connect(self.save_match_file)

        for i, chip in enumerate(fix_chips):
            self.slice_table.setItem(i, 0, QTableWidgetItem(chip))

        self.slice_table.cellPressed.connect(self.check_and_generate_layers)
        regist_tab_layout.addWidget(self.slice_table)

        regist_tab_widget = QWidget()
        regist_tab_widget.setLayout(regist_tab_layout)
        regist_tab_widget.setMaximumWidth(300)

        self.module_tab.addTab(regist_tab_widget, "landmark 配准检查")


        self.arrange_tab_layout = QVBoxLayout()
        self.arrange_tab_layout.addWidget(QLabel("转录组校正配置"))
        self.stereo_corr_config_input = QLineEdit()
        self.arrange_tab_layout.addWidget(self.stereo_corr_config_input)
        self.stereo_corr_config_input.setText('/mnt/97-macaque/projects/cla/stereo-cla-highlight-20240330/Mq179-CLA-sh-20240204-L/DESKTOP-NLB3J50-config.yaml')

        self.arrange_tab_layout.addWidget(QLabel("转录组 NTP 版本"))
        self.stereo_ntp_version = QLineEdit()
        self.stereo_ntp_version.setText(fix_ntp_version)
        self.arrange_tab_layout.addWidget(self.stereo_ntp_version)

        self.arrange_target_chip = QLineEdit()
        self.arrange_target_chip.setText(r'.*(cla)|(DEn)|(shCla)|(Cla).*')
        self.arrange_tab_layout.addWidget(QLabel("目标分区"))
        self.arrange_tab_layout.addWidget(self.arrange_target_chip)

        self.arrange_button = QPushButton("开摆")
        self.arrange_button.clicked.connect(self.arrange)
        self.arrange_tab_layout.addWidget(self.arrange_button)
        

        self.arrange_tab_layout.addStretch()  # Add a stretchable space at the end

        arrange_tab_widget = QWidget()
        arrange_tab_widget.setLayout(self.arrange_tab_layout)

        self.module_tab.addTab(arrange_tab_widget, "排列")


        self.main_layout.addWidget(self.module_tab)
        self.right_layout = QVBoxLayout()
        self.graphics_view = ZoomableGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        # QGraphicsView.setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform
        self.graphics_view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.right_layout.addWidget(self.graphics_view)

        self.right_bottom_layout = QHBoxLayout()
        self.layer_checkboxs_layout = QVBoxLayout()
        self.right_bottom_layout.addLayout(self.layer_checkboxs_layout)

        self.check_output = QTextEdit()
        self.check_output.setMaximumHeight(200)
        self.right_bottom_layout.addWidget(self.check_output)

        self.right_layout.addLayout(self.right_bottom_layout)

        self.main_layout.addLayout(self.right_layout)
        
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        self.data_layers = []
        self.data_layers_items = []

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')

        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_match_file)
        file_menu.addAction(save_action)

        export_svg_action = QAction('Export SVG', self)
        # export_svg_action.triggered.connect(self.export_svg)
        file_menu.addAction(export_svg_action)

        self.read_match_file_to_widget()

    def _read_match_file(self):
        if not self.match_file_p.exists():
            mapping = {}
        else:
            with open(self.match_file_p, 'r') as f:
                mapping = json.load(f)
        return mapping

    def read_match_file_to_widget(self):
        mapping = self._read_match_file()
        for i in range(self.slice_table.rowCount()):
            fix_item = self.slice_table.item(i, 0)
            if not fix_item:
                continue
            map_v = mapping.get(fix_item.text(), '')
            self.slice_table.setItem(i, 1, QTableWidgetItem(map_v))

        self.operator_input.setText(mapping.get('operator', ''))

    def save_match_file(self):
        if not self.mov_p.exists():
            return
        mapping = {
            'operator': self.operator_input.text() or '',
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        for i in range(self.slice_table.rowCount()):
            fix_item = self.slice_table.item(i, 0)
            if fix_item is None:
                continue

            mov_item = self.slice_table.item(i, 1)
            mapping[fix_item.text()] = mov_item.text() if mov_item else ''

        with open(self.match_file_p, 'w') as f:
            json.dump(mapping, f)

    def append_to_output(self, text: str):
        self.check_output.append(
            f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}'
        )
        QApplication.processEvents()

    @property
    def base_p(self):
        return Path(self.base_path_input.text())

    @property
    def mov_p(self):
        return self.base_p / f'{self.animal_id}-{self.brain_direction}'

    @property
    def match_file_p(self):
        return self.mov_p / 'match.json'

    @property
    def animal_id(self):
        return self.animal_id_input.text() or ''

    @property
    def brain_direction(self):
        return self.brain_direction_input.currentText()

    def landmarks_p(self, fix_slice_id: str, mov_slice_id: str, mov_animal_id: str|None = None):
        if not mov_animal_id:
            mov_animal_id = self.animal_id

        return self.mov_p / f'{mov_animal_id}-{mov_slice_id}-{fix_slice_id}.csv'

    def fix_img_p(self, fix_slice_id: str):
        return self.base_p / f'{fix_ntp_version}-L' / f'{fix_ntp_version}-{fix_slice_id}.png'

    def mov_img_p(self, mov_slice_id: str):
        return self.mov_p / f'{self.animal_id}-{mov_slice_id}.png'

    def check_and_generate_layers(self, idx: int):
        # print(f'{idx=}')
        fix_item = self.slice_table.item(idx, 0)
        mov_item = self.slice_table.item(idx, 1)
        self.check_output.setText("")
        self.append_to_output("start")
        if not fix_item:
            self.append_to_output('fix_item is None')
            return
        if not mov_item or not mov_item.text():
            self.append_to_output('mov_item is None')
            return
        if not self.base_p.exists():
            self.append_to_output(f'base_path {self.base_p} 不存在，是否未挂载或选错位置？')
            return

        animal_id = self.animal_id_input.text()
        brain_direction = self.brain_direction_input.currentText()
        mov_slice = mov_item.text()
        fix_slice = fix_item.text()

        if not animal_id:
            self.append_to_output('填下动物呀')
            return
        if not self.landmarks_p(fix_slice, mov_slice).exists():
            self.append_to_output(
                f'landmarks 文件 {self.landmarks_p(fix_slice, mov_slice)} 不存在')
            return

        if not self.fix_img_p(fix_slice).exists():
            self.append_to_output(f'fix 图片 {self.fix_img_p(fix_slice)} 不存在')
            return
        if not self.mov_img_p(mov_slice).exists():
            self.append_to_output(f'mov 图片 {self.mov_img_p(mov_slice)} 不存在')
            return

        self.append_to_output(f'使用 {animal_id}-{brain_direction} {mov_slice}->{fix_slice}')
        self.append_to_output("处理中")

        self.data_layers = self.process_data(animal_id, brain_direction, fix_slice, mov_slice)
        self.append_to_output('已读入数据')
        self.draw_layers()
        self.module_tab.setDisabled(False)


    def process_data(self, mov_animal, brain_direction, fix_slice, mov_slice):
        self.module_tab.setDisabled(True)

        landmark_df = ItkTransformWarp.read_big_warp_df(
            self.landmarks_p(fix_slice, mov_slice)
        )
        self.append_to_output(f'使用 landmark: {landmark_df}')

        itw = ItkTransformWarp.from_big_warp_df(landmark_df)

        mov_ntps = cutils.find_connectome_ntp(
            mov_animal, slice_id=mov_slice, 
            base_path=self.connectome_ntp_base_path_input.text()
        )
        if not mov_ntps:
            self.append_to_output(f'找不到 mov ntp {mov_slice}')
            return []
        mov_ntp = mov_ntps[0]

        fix_ntps = cutils.find_stereo_ntp(
            fix_ntp_version, chip=fix_slice, 
            base_path=Path(self.inner_data_input.text()) / 'ntp/macaque',
        )
        if not fix_ntps:
            self.append_to_output(f'找不到 fix ntp {fix_slice}')
            return []
        fix_ntp = fix_ntps[0]

        self.append_to_output(f'{mov_ntp=}, {fix_ntp=}')

        mov_w, mov_h = cutils.get_czi_size(
            mov_animal, mov_slice, 
            size_path=str(Path(self.connectome_ntp_base_path_input.text()).parent / 'czi-size')
        )

        self.append_to_output(f'{mov_w=}, {mov_h=}')
        mov_sm = parcellate_with_cache(
            mov_ntp, um_per_pixel=mov_resolution, bin_size=mov_bin_size, 
            w=mov_w / mov_bin_size, h=mov_h / mov_bin_size, # type: ignore
            export_position_policy='none'
        )
        fix_sm = parcellate_with_cache(
            fix_ntp, um_per_pixel=fix_resolution, bin_size=fix_bin_size, 
            export_position_policy='none'
        )
        assert isinstance(fix_sm, SliceMeta)
        assert isinstance(mov_sm, SliceMeta)

        res: list[Layer] = []

        for color in ('green', 'red', 'blue', 'yellow'):
            if fix_sm.cells is None: continue
            tracer, show_color = color_to_tracer[color]

            res.append(Layer(
                name=f'fix-cells-{tracer}',
                data=fix_sm.cells[color],
                type='point',
                colors=[show_color for _ in fix_sm.cells[color]]
            ))
        for color in ('green', 'red', 'blue', 'yellow'):
            if mov_sm.cells is None: continue
            tracer, show_color = color_to_tracer[color]

            res.append(Layer(
                name=f'mov-cells-{tracer}',
                data=itw.transform_points_to_fix(mov_sm.cells[color]),
                type='point',
                colors=[show_color for _ in mov_sm.cells[color]]
            ))


        fix_region_exterior = []
        fix_region_colors = []
        mov_region_exterior = []
        mov_region_colors = []

        for r in fix_sm.regions:
            # if 'cla' not in r.label.name.lower() and 'DEn' not in r.label.name:
            #     continue
            exterior = np.array(r.polygon.exterior.xy).T
            fix_region_exterior.append(exterior)
            c = (127, 127, 127, 100)
            if 'cla' in r.label.name.lower() or 'DEn' in r.label.name:
                c = (0, 0, 255, 100)

            fix_region_colors.append(c)


        res.append(Layer(**{
            'type': 'polygon',
            'name': 'fix regions',
            'data': fix_region_exterior,
            'colors': fix_region_colors,
        }))
        for r in mov_sm.regions:
            # if 'cla' not in r.label.name.lower():
            #     continue
            c = (255, 0, 0, 100) if 'cla' in r.label.name.lower() else (127, 127, 127, 50)

            if not r.label.name.startswith(brain_direction):
                c = (60, 60, 60, 50)

            exterior = np.array(r.polygon.exterior.xy).T
            exterior = itw.transform_points_to_fix(exterior)
            mov_region_exterior.append(exterior)
            mov_region_colors.append(c)

        res.append(Layer(**{
            'type': 'polygon',
            'name': 'moved regions',
            'data': mov_region_exterior,
            'colors': mov_region_colors,
        }))

        mov_landmarks = landmark_df[['mov_x', 'mov_y']].to_numpy()
        moved_landmarks = itw.transform_points_to_fix(mov_landmarks)
        fix_landmarks = landmark_df[['fix_x', 'fix_y']].to_numpy()
        res.append(Layer(**{
            'type': 'point',
            'name': 'fix landmarks',
            'data': fix_landmarks,
            'colors': [(200, 200, 200, 255) for _ in fix_landmarks]
        }))
        res.append(Layer(**{
            'type': 'point',
            'name': 'moved landmarks',
            'data': moved_landmarks,
            'colors': [(255, 0, 0, 255) for _ in moved_landmarks]
        }))

        for i, layer in enumerate(res):
            self.append_to_output(f'F{i+1} -> {layer.name}')

        self.module_tab.setDisabled(False)
        return res

    def draw_layer(self, idx: int, layer: Layer):
        if self.layer_checkboxs_layout.count() > idx:
            checkbox = self.layer_checkboxs_layout.itemAt(idx).widget()
            checkbox = cast(QCheckBox, checkbox)
        else:
            checkbox = QCheckBox()
            checkbox.setDisabled(True)
            self.layer_checkboxs_layout.addWidget(checkbox)
        checkbox.setText(f"F{idx+1} {layer.name}")
        checkbox.setChecked(layer.enabled)

        if layer.type == 'polygon':
            for data, color in zip(layer.data, layer.colors):
                item = QGraphicsPolygonItem(QPolygonF([QPointF(*p) for p in data]))
                item.setBrush(QColor(*color))
                self.graphics_scene.addItem(item)
        elif layer.type == 'point':
            for data, color in zip(layer.data, layer.colors):
                if len(data) == 0: continue
                item = self.graphics_scene.addEllipse(0, 0, 8, 8)
                item.setBrush(QColor(*color))
                item.setPos(*data)


    def draw_layers(self):
        self.graphics_scene.clear()
        self.data_layers_items.clear()

        for i, layer in enumerate(self.data_layers):
            if self.layer_checkboxs_layout.count() > i:
                checkbox = self.layer_checkboxs_layout.itemAt(i).widget()
                checkbox = cast(QCheckBox, checkbox)
            else:
                checkbox = QCheckBox(f"F{i+1} {layer.name}", self)
                checkbox.setDisabled(True)
                self.layer_checkboxs_layout.addWidget(checkbox)
            checkbox.setChecked(layer.enabled)

            if not layer.enabled: continue
            if layer.type == 'polygon':
                for data, color in zip(layer.data, layer.colors):
                    # print(data)
                    item = QGraphicsPolygonItem(QPolygonF([QPointF(*p) for p in data]))
                    item.setBrush(QColor(*color))

                    self.graphics_scene.addItem(item)
            elif layer.type == 'point':
                for data, color in zip(layer.data, layer.colors):
                    if len(data) == 0: continue
                    # print(layer, data)
                    item = self.graphics_scene.addEllipse(0, 0, 8, 8)
                    item.setBrush(QColor(*color))
                    item.setPos(*data)

    def change_layer_visibility(self, enables: list[bool]):
        for i, enable in enumerate(enables):
            self.graphics_scene.items()[i].setVisible(enable)
            
    def keyPressEvent(self, event):
        idx = event.key() - Qt.Key.Key_F1
        # print(f'{idx=}')
        if idx < 0 or idx > 12: return
        if idx >= len(self.data_layers): return
        self.append_to_output(f'切换层：{idx}')

        self.data_layers[idx].enabled = not self.data_layers[idx].enabled
        self.graphics_scene.items()[idx].setVisible(self.data_layers[idx].enabled)
        self.layer_checkboxs_layout.itemAt(idx).widget().setChecked(self.data_layers[idx].enabled)
        print(self.data_layers[idx].enabled, self.graphics_scene.items()[idx].isVisible())

        self.graphics_scene.update()
        self.graphics_view.update()

    def arrange(self):
        self.module_tab.setDisabled(True)
        regist_mapping = self._read_match_file()

        mov_points_layers = []
        fix_points_layers = []
        mov_polygon_layers = []
        fix_polygon_layers = []

        fix_ntp_base_path = Path(self.inner_data_input.text()) / 'ntp/macaque'
        target_regex = self.arrange_target_chip.text()

        for fix_chip in fix_chips[:10]:
            try:
                arrange_corr = CorrectionPara.from_yaml(
                    self.stereo_corr_config_input.text(), target_chip=fix_chip
                ).with_scale(100).with_scale(1/13)
                # print(arrange_corr)
            except ValueError as e:
                self.append_to_output(f'{fix_chip} {e}')
                continue

            mov_chip = regist_mapping.get(fix_chip, '')
            if not mov_chip:
                self.append_to_output(f'{fix_chip} 没有对应的 mov_chip')
                continue
            landmarks_p = self.landmarks_p(fix_chip, mov_chip)
            if not landmarks_p.exists():
                self.append_to_output(f'{landmarks_p} 不存在')
                continue
            landmark_df = ItkTransformWarp.read_big_warp_df(landmarks_p)
            itw = ItkTransformWarp.from_big_warp_df(landmark_df)

            fix_ntps = cutils.find_stereo_ntp(
                fix_ntp_version, chip=fix_chip, 
                base_path=fix_ntp_base_path
            )
            if not fix_ntps:
                self.append_to_output(f'找不到 fix ntp {fix_chip} @ {fix_ntp_base_path}')
                continue
            fix_ntp = fix_ntps[0]

            mov_ntps = cutils.find_connectome_ntp(
                self.animal_id, slice_id=mov_chip, 
                base_path=self.connectome_ntp_base_path_input.text()
            )
            if not mov_ntps:
                self.append_to_output(f'找不到 mov ntp {mov_chip} @ {self.connectome_ntp_base_path_input.text()}')
                continue
            mov_ntp = mov_ntps[0]

            self.append_to_output(f'{fix_ntp=}, {mov_ntp=}')

            mov_w, mov_h = cutils.get_czi_size(
                self.animal_id, mov_chip, 
                size_path=str(Path(self.connectome_ntp_base_path_input.text()).parent / 'czi-size')
            )
            self.append_to_output(f'{mov_w=}, {mov_h=}')

            mov_sm = parcellate_with_cache(
                mov_ntp, um_per_pixel=mov_resolution, bin_size=mov_bin_size, 
                w=mov_w / mov_bin_size, h=mov_h / mov_bin_size, # type: ignore
                export_position_policy='none'
            )
            fix_sm = parcellate_with_cache(
                fix_ntp, um_per_pixel=fix_resolution, bin_size=fix_bin_size, 
                export_position_policy='none', 
            )
            assert isinstance(fix_sm, SliceMeta)
            assert isinstance(mov_sm, SliceMeta)

            for r in fix_sm.regions:
                if not match_regex(target_regex, r.label.name):
                    continue
                exterior = np.array(r.polygon.exterior.xy).T
                exterior = arrange_corr.warp_point(exterior)
                c = (0, 0, 255, 100)

                fix_polygon_layers.append(Layer(
                    name=f'fix-{r.label.name}',
                    data=[exterior],
                    type='polygon',
                    colors=[c]
                ))

            for r in mov_sm.regions:
                if not match_regex(target_regex, r.label.name):
                    continue
                if not r.label.name.startswith(self.brain_direction):
                    continue
                c = (255, 0, 0, 100)

                exterior = np.array(r.polygon.exterior.xy).T
                exterior = itw.transform_points_to_fix(exterior)
                exterior = arrange_corr.warp_point(exterior)
                mov_polygon_layers.append(Layer(
                    name=f'mov-{r.label.name}',
                    data=[exterior],
                    type='polygon',
                    colors=[c]
                ))

            for color in ('green', 'red', 'blue', 'yellow'):
                if fix_sm.cells is None: continue
                tracer, show_color = color_to_tracer[color]
                data = fix_sm.cells[color]
                data = arrange_corr.warp_point(data)

                fix_points_layers.append(Layer(
                    name=f'fix-cells-{tracer}',
                    data=data,
                    type='point',
                    colors=[show_color for _ in fix_sm.cells[color]]
                ))

            for color in ('green', 'red', 'blue', 'yellow'):
                if mov_sm.cells is None: continue
                tracer, show_color = color_to_tracer[color]
                data = mov_sm.cells[color]
                data = itw.transform_points_to_fix(data)
                data = arrange_corr.warp_point(data)

                mov_points_layers.append(Layer(
                    name=f'mov-cells-{tracer}',
                    data=data,
                    type='point',
                    colors=[show_color for _ in mov_sm.cells[color]]
                ))

        self.data_layers = (
            list(Layer.merge(*fix_polygon_layers)) + list(Layer.merge(*mov_polygon_layers)) + 
            list(Layer.merge(*fix_points_layers)) + list(Layer.merge(*mov_points_layers))
        )
        self.draw_layers()
        self.module_tab.setDisabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
