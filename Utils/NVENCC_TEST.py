from skvideo.io import FFmpegReader, EnccWriter, FFmpegWriter, SVTWriter

if __name__ == "__main__":
    # _input_file = r"D:\60-fps-Project\input_or_ref\Test\【3】批处理1.mp4"
    _input_file = r"D:\60-fps-Project\input_or_ref\Test\【4】暗场+黑边裁切+时长片段+字幕轨合并.mkv"
    _output_file = r"D:\60-fps-Project\input_or_ref\Test\svt_output.mp4"
    _reader = FFmpegReader(filename=_input_file, outputdict={"-vframes":"10000000000"})
    # _writer = NVenccWriter(filename=_output_file, inputdict={"-c": "h264", "--cqp": "16"}, outputdict={})
    # _writer = EnccWriter(filename=_output_file, inputdict={}, outputdict={})
    """"encc": "hevc", '-brr': '1', '-sharp': '1', '-b':"", '-bit-depth': '10', '-q': '16',"""
    _writer = SVTWriter(filename=_output_file, inputdict={"-fps": "25", "-n": "241"}, outputdict={"encc": "hevc", "-brr": "1", "-sharp": "1", "-q": "16", '-bit-depth': '10',})
    # _writer = SVTWriter(filename=_output_file, inputdict={"-fps": "25", "-n": "241"}, outputdict={"encc": "vp9", "-q": "16",'-bit-depth': '10', "-tune": "0", })
    # _writer = FFmpegWriter(filename=_output_file)
    cnt = 1
    for _i in _reader.nextFrame():
        _writer.writeFrame(_i)
        print(f"cnt: {cnt}")
        cnt+=1
    _reader.close()
    _writer.close()