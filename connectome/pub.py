'''
Copy code file to publish
'''
from pathlib import Path
import shutil

files = {
    'pub.py': 'pub.py',
    'pwvy/utils.py': 'utils.py',
    'pwvy/regist_utils.py': 'resigt_utils.py',
    'pwvy/dataset_utils.py': 'dataset_utils.py',
    'pwvy/sim_network.py': 'sim_network.py',
    'pwvy/cairo_utils.py': 'cairo_utils.py',
    'pwvy/match_region.py': 'match_region.py',

    'pwvy/cluster_df.csv': 'cluster_df.csv',

    'pwvy/20240131-select-color-from-svg.py': 'select-color-from-svg.py',
    'pwvy/20240205-export-stereo-and-read-from-ai.py': 'export-stereo-svg-calc-projection-zone-iou-export-cell.py',
    'pwvy/20240223-calc-injection-zone.py': 'calc-projection-zone-HDBSCAN.py', 
    'pwvy/20240308-create-region-svg.py': 'calc-projection-zone-KDE-registed.py',
    'pwvy/20240326-landmark-regist.py': 'landmark-regist.py',
    'pwvy/20240328-landmark-regist-checker.py': 'landmark-regist-checker-gui.py',
    'pwvy/20240403-stat-ntp-cells.py': 'stat-ntp-cells.py',
    'pwvy/20240412-cla-foot-registe.py': 'cla-foot-registe.py',
    'pwvy/20240422-global-kde.py': 'calc-projection-zone-KDE.py',

    'pwvy/draw_regist_results.py': 'draw_regist_results.py',
    'pwvy/draw_regist_results_kde.py': 'draw_regist_results_kde.py',

    'pwvy/export_czi_nnunet.py': 'export_czi_nnunet.py',
    'pwvy/test_nn_similarity_register.py': 'connectome_regist.py',
    'pwvy/test_nn_similarity.py': 'connectome_sim_network_train.py',
    'pyproject.toml': 'pyproject.toml',
}

output_path = 'dist/connectome'
Path(output_path).mkdir(exist_ok=True, parents=True)

for src, dst in files.items():
    assert Path(src).exists(), f'{src} not exists'
    shutil.copyfile(src, Path(output_path) / dst)
    print(f'{src:<48}\t->\t{dst}')
