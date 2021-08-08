# encoding=utf-8
class VideoFrameInterpolation:
    def __init__(self):
        pass

    def initiate_algorithm(self):
        raise NotImplementedError()

    def generate_n_interp(self, img0, img1, n, scale, debug=False):
        raise NotImplementedError()
