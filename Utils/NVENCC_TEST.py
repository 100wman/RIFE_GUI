from skvideo.io import FFmpegReader, NVenccWriter, FFmpegWriter

if __name__ == "__main__":
    _input_file = r"D:\60-fps-Project\input_or_ref\Test\【3】批处理1.mp4"
    _output_file = r"D:\60-fps-Project\input_or_ref\Test\nvencc_output.264"
    _reader = FFmpegReader(filename=_input_file)
    # _writer = NVenccWriter(filename=_output_file, inputdict={"-c": "h264", "--cqp": "16"}, outputdict={})
    _writer = NVenccWriter(filename=_output_file, inputdict={}, outputdict={})
    # _writer = FFmpegWriter(filename=_output_file)
    cnt = 1
    for _i in _reader.nextFrame():
        _writer.writeFrame(_i)
        print(f"cnt: {cnt}")
        cnt+=1