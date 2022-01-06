# encoding=utf-8
import warnings

import cv2

from Utils.StaticParameters import RGB_TYPE
from Utils.utils import TransitionDetection_ST
warnings.filterwarnings("ignore")
from skvideo.io import FFmpegReader

if __name__ == "__main__":
    RGB_TYPE.change_8bit(True)
    detector = TransitionDetection_ST(scene_queue_length=10, scdet_threshold=16,
                                      project_dir=r"D:\60-fps-Project\input or ref\Test\output\test_scene",
                                      scdet_output=True)
    test_video = r"D:\60-fps-Project\input or ref\Test\【4】4.0测试-虚渊玄× 荒木哲郎× 小畑健× 泽野弘之！动画《泡泡》前导预告发布！.mp4"
    reader = FFmpegReader(filename=test_video, outputdict={"-vf": "minterpolate=fps=60:mi_mode=dup"}).nextFrame()
    last_img = next(reader)
    for i, img in enumerate(reader):
        r = detector.check_scene(last_img, img)
        last_img = img
        if r:
            print(f"at {i} frame, find scene")
    pass
