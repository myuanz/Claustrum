'''
Copy code file to publish
'''
from pathlib import Path
import shutil
from ftplib import FTP
import os
from tqdm import tqdm

# code

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

output_path = '~/projects/Claustrum/connectome/'
output_path = Path(output_path).expanduser()
shutil.rmtree(output_path, ignore_errors=True)
output_path.mkdir(parents=True, exist_ok=True)

for src, dst in files.items():
    assert Path(src).exists(), f'{src} not exists'
    shutil.copyfile(src, Path(output_path) / dst)
    print(f'{src:<48}\t->\t{dst}')

# data


ftp_server = "172.16.102.74"
ftp_port = 2121
ftp_user = "shen_macaque_project_dataset1"
ftp_password = "shen_macaque_project_dataset1"
remote_directory = "/"
local_directory = "/data/sdbd/connectome/project-cla-zip/"

ftp = FTP()
ftp.connect(ftp_server, ftp_port)
ftp.login(user=ftp_user, passwd=ftp_password)


ftp.cwd(remote_directory)
def create_directory_structure(ftp, remote_dir):
    dirs = remote_dir.split("/")
    for dir in dirs:
        if dir:
            try:
                ftp.mkd(dir)
            except:
                pass
            ftp.cwd(dir)
    ftp.cwd("/")

def file_exists_with_same_size(ftp, remote_file_path, local_file_size):
    try:
        remote_file_size = ftp.size(remote_file_path)
        return remote_file_size == local_file_size
    except:
        return False

all_files = []
for root, dirs, files in (pbar := tqdm(os.walk(local_directory))):
    for dir in dirs:
        remote_dir = os.path.join(remote_directory, os.path.relpath(os.path.join(root, dir), local_directory))
        create_directory_structure(ftp, remote_dir)
    
    for file in files:
        all_files.append((root, file))



for root, file in (pbar := tqdm(all_files)):
    local_file_path = os.path.join(root, file)
    remote_file_path = os.path.join(remote_directory, os.path.relpath(local_file_path, local_directory))
    
    local_file_size = os.path.getsize(local_file_path)
    
    if file_exists_with_same_size(ftp, remote_file_path, local_file_size):
        pbar.set_description(f"Skipping {remote_file_path} (already exists with the same size)")
    else:
        pbar.set_description(f"Uploading {remote_file_path}")
        with open(local_file_path, "rb") as file:
            ftp.storbinary(f"STOR {remote_file_path}", file)

ftp.quit()
