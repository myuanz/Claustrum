# %%
from typing import Literal
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import base64
import cv2
from PIL import Image
import numpy as np
import re
import polars as pl
import pandas as pd

p = "/mnt/90-connectome/A-Temporary/INJ Color/Inj-color-zero-.svg"
# p = './INJ Color.svg'

def get_scale(tag: ET.Element):
    transform = tag.attrib['transform']
    scale = re.search(r'scale\((.+?)\)', transform).group(1)
    scales = list(map(float, scale.split(' ')))
    sx = scales[0]
    sy = scales[1] if len(scales) > 1 else sx
    return sx, sy

def find_image(root_tag: ET.ElementTree, id: Literal['RGB', 'CMYK']):
    img_tag = root_tag.find(
        f'.//{{http://www.w3.org/2000/svg}}g[@id="{id}"]/{{http://www.w3.org/2000/svg}}image'
    )

    assert img_tag is not None
    img_scale_x, img_scale_y = get_scale(img_tag)

    base64_img = img_tag.attrib['{http://www.w3.org/1999/xlink}href']
    imgb = base64.b64decode(base64_img[len('data:image/png;base64,'):])
    img = cv2.imdecode(np.frombuffer(imgb, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img, img_scale_x, img_scale_y

def get_text(tag: ET.Element):
    children = tag.findall('*')
    return "".join([c.text or '' for c in children])

def get_pos(tag: ET.Element):
    transform = tag.attrib['transform']
    translate = re.search(r'translate\((.+?)\)', transform).group(1)
    x, y = map(float, translate.split(' '))
    return x, y

labels = []
draw = True


tree = ET.parse(p)
text_tags = tree.findall('.//{http://www.w3.org/2000/svg}text')

for img_id in ('RGB', 'CMYK'):
    if draw:
        plt.figure()
    img, img_scale_x, img_scale_y = find_image(tree, id=img_id)
    for text_tag in text_tags:
        label = get_text(text_tag).replace('/', '-')
        x, y = get_pos(text_tag)

        x, y = x / img_scale_x, y / img_scale_y
        if draw:
            plt.scatter(x, y, s=1)
            plt.text(x, y, label)

        x, y = int(x), int(y)
        labels.append((x, y, label, tuple(img[y, x]), img_id))

    if draw:
        plt.imshow(img)
    
# %%
def hsl_to_rgb(h, s, l):
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)

labels_ = [*labels]
for color_type in ('RGB', 'CMYK'):
    labels.append((0, 0, 'CLA_C075_left', hsl_to_rgb(0.0, 0.5, 0.5), color_type))
    labels.append((0, 0, 'CLA_C077_right', hsl_to_rgb(0.25, 0.5, 0.5), color_type))
    labels.append((0, 0, 'CLA_C080_right', hsl_to_rgb(0.5, 0.5, 0.5), color_type))
    labels.append((0, 0, 'CLA_C096_left', hsl_to_rgb(0.75, 0.5, 0.5), color_type))

exported_df = pl.DataFrame(labels, schema=['x', 'y', 'label', 'color', 'color_type'])
exported_df
# %%
input_df = pd.read_excel('/mnt/97-macaque/projects/cla/injections-cells/Combine-20240202.xlsx').rename(columns={
    'Combine3': 'combine',
    'Combine_area': 'combine_area', 
    'Animal': 'animal_id',
    'injectionSites': 'region',
    'hemisphere': 'side',
    'Dye': 'tracer',
    'draw_color': 'draw_color', 
}).sort_values(['combine', 'animal_id', 'region'], ignore_index=True).fillna(False).convert_dtypes()
input_df['registed'] = input_df['registed'].astype(bool)
input_df = pl.DataFrame(input_df.to_dict('records'))
input_df
# %%
cla_rows = input_df['combine_area'].str.contains('CLA')
input_df = input_df.with_columns(
    pl.when(cla_rows).then(
        pl.col('combine_area') + '_' + pl.col('animal_id') + '_' + pl.col('side').map_dict({'L': 'left', 'R': 'right'})
    ).otherwise(pl.col('combine_area')).alias('combine_area')
)


final_df = input_df.join(
    exported_df, right_on='label', left_on='combine_area', how='left'
).sort('combine_area').with_columns(
    (
        pl.col('animal_id') + '_' + pl.col('tracer') + '_' + pl.col('combine_area')
    ).alias('draw_color_key')
)

# pl.col('animal_id') + '_' + pl.col('tracer') + '_' + pl.col('combine_area')


# df.write_excel('/mnt/90-connectome/A-Temporary/INJ Color/Combine-error.xlsx')
# final_df
cla_rows = final_df['combine_area'].str.contains('CLA')
final_df.filter(cla_rows)
# %%
for t, g in final_df.group_by('color_type'):
    if t is None: continue
    print(t)
    g.write_excel(f'/mnt/97-macaque/projects/cla/injections-cells/20240201-no-warp-per1-d550-raw-merge-42/palette-{t}.xlsx')
    g.with_columns(
        pl.col('color').cast(pl.List(pl.String)).list.join(',')
    ).write_csv(f'/mnt/97-macaque/projects/cla/injections-cells/20240201-no-warp-per1-d550-raw-merge-42/palette-{t}.csv')

# %%
