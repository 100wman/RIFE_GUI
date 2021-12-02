import enum
import os

abspath = os.path.abspath(__file__)
appDir = os.path.dirname(os.path.dirname(abspath))

INVALID_CHARACTERS = ["'", '"']

class TASKBAR_STATE(enum.Enum):
    TBPF_NOPROGRESS = 0x00000000
    TBPF_INDETERMINATE = 0x00000001
    TBPF_NORMAL = 0x00000002
    TBPF_ERROR = 0x00000004
    TBPF_PAUSED = 0x00000008


class HDR_STATE(enum.Enum):
    AUTO = -2
    NOT_CHECKED = -1
    NONE = 0
    CUSTOM_HDR = 1
    HDR10 = 2
    HDR10_PLUS = 3
    DOLBY_VISION = 4
    HLG = 5


class SupportFormat:
    img_inputs = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    img_outputs = ['.png', '.tiff', '.jpg']
    vid_outputs = ['.mp4', '.mkv', '.mov']


class EncodePresetAssemply:
    encoder = {
        "CPU": {
            "H264,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow", "placebo", ],
            "H264,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "AV1,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "AV1,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "ProRes,422": ["hq", "4444", "4444xq"],
            "ProRes,444": ["hq", "4444", "4444xq"],
        },
        "NVENC":
            {"H264,8bit": ["slow", "fast", "hq", "bd", "llhq", "loseless", "p7"],
             "H265,8bit": ["slow", "fast", "hq", "bd", "llhq", "loseless", "p7"],
             "H265,10bit": ["slow", "fast", "hq", "bd", "llhq", "loseless", "p7"], },
        "NVENCC":
            {"H264,8bit": ["default", "performance", "quality"],
             "H265,8bit": ["default", "performance", "quality"],
             "H265,10bit": ["default", "performance", "quality"], },
        "QSVENCC":
            {"H264,8bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"],
             "H265,8bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"],
             "H265,10bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"], },
        "QSV":
            {"H264,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,10bit": ["slow", "fast", "medium", "veryslow", ], },
        "SVT":
            {"VP9,8bit": ["slowest", "slow", "fast", "faster"],
             "H265,8bit": ["slowest", "slow", "fast", "faster"],
             "AV1,8bit": ["slowest", "slow", "fast", "faster"],
             },

    }
