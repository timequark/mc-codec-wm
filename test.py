import util as util

def test():
    file_in     = "images/mkking-01.png"
    file_out    = "images/mkking-01-resized.png"

    util.resize_by_width(file_in, file_out, target_width=1400, dpi=600, resample=util.Image.Resampling.LANCZOS)

# 使用示例
if __name__ == "__main__":
    test()