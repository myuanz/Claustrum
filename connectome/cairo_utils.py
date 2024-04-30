import cairo
import numpy as np


def draw_cnt(ctx: cairo.Context, cnt: np.ndarray, fill=False):
    ctx.move_to(*cnt[0])
    for i in cnt[1:]:
        ctx.line_to(*i)
    ctx.move_to(*cnt[0])
    if fill:
        ctx.fill()
    ctx.close_path()
    ctx.stroke()
