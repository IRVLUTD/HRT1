# my_ros_package/scripts/frame_processor.py
import cv2
import numpy as np
from bundlesdf import YcbineoatReader

class FrameProcessor:
    def __init__(self, video_dir, shorter_side, use_segmenter, erode_mask_size):
        self.reader = YcbineoatReader(video_dir=video_dir, shorter_side=shorter_side)
        self.use_segmenter = use_segmenter
        self.erode_mask_size = erode_mask_size
        self.frame_count = 0

    def get_intrinsics(self):
        return self.reader.K.copy()

    def get_total_frames(self):
        return len(self.reader.color_files)

    def process_frame(self, index, stride):
        if index >= len(self.reader.color_files):
            return None, None, None, None

        color_file = self.reader.color_files[index]
        color = cv2.imread(color_file)
        H0, W0 = color.shape[:2]
        depth = self.reader.get_depth(index)
        H, W = depth.shape[:2]
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        if index == 0:
            mask = self.reader.get_mask(0)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            if self.use_segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
        else:
            if self.use_segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
            else:
                mask = self.reader.get_mask(index)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.erode_mask_size > 0:
            kernel = np.ones((self.erode_mask_size, self.erode_mask_size), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)

        id_str = self.reader.id_strs[index]
        self.frame_count += stride
        return color, depth, mask, id_str