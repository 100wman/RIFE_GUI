# encoding=utf-8
import numpy as np
from line_profiler_pycharm import profile

from Utils.utils import Tools

a = np.random.rand(1920, 1080, 3) * 255
a = a.astype(np.uint8)
b = a.copy()


@profile
def test_speed():
    a1, b1 = a.copy(), b.copy()
    t = Tools.get_norm_img_diff(a1, b1)
    a1, b1 = a.copy(), b.copy()
    t = Tools.get_norm_img_diff(a1, b1, risk=False)
    return t


for i in range(int(1e3)):
    test_speed()
