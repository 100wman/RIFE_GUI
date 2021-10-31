from skvideo.io import FFmpegReader, EnccWriter, FFmpegWriter, SVTWriter

if __name__ == "__main__":
    # _input_file = r"D:\60-fps-Project\input_or_ref\Test\【3】批处理1.mp4"
    _input_file = r"D:\60-fps-Project\input_or_ref\Test\【2】4KUHD测试【转场】【暗场】.mkv"
    _output_file = r"D:\60-fps-Project\input_or_ref\Test\svt_output.mp4"
    _reader = FFmpegReader(filename=_input_file, inputdict={"-to": "2"}, outputdict={"-vframes":"10000000000"})
    # _writer = EnccWriter(filename=_output_file, inputdict={"--codec": "hevc", "--fps": "23.976", "--vbr":"0", "--vbr-quality": "16", 'encc': 'NVENCC', }, outputdict={"--output-depth": "10"})
    _writer = EnccWriter(filename=_output_file, inputdict={"--fps": "23.976",'encc': 'QSVENCC', '--codec': "hevc"}, outputdict={"--output-depth": "10", "--quality": "best"})
    # _writer = EnccWriter(filename=_output_file, inputdict={}, outputdict={})
    # """"encc": "hevc", '-brr': '1', '-sharp': '1', '-b':"", '-bit-depth': '10', '-q': '16',"""
    # _writer = SVTWriter(filename=_output_file, inputdict={"-fps": "25", "-n": "10"}, outputdict={"encc": "hevc", "-brr": "1", "-sharp": "1", "-q": "16", '-bit-depth': '10',})
    # _writer = SVTWriter(filename=_output_file, inputdict={"-fps": "25", "-n": "241"}, outputdict={"encc": "vp9", "-q": "16",'-bit-depth': '10', "-tune": "0", })
    # _writer = FFmpegWriter(filename=_output_file)
    cnt = 1
    for _i in _reader.nextFrame():
        _writer.writeFrame(_i)
        print(f"cnt: {cnt}")
        cnt+=1
    _reader.close()
    _writer.close()