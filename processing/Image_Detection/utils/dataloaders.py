# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Dataloaders and dataset utils."""

import glob
import os
from pathlib import Path

import numpy as np

from utils.augmentations import (
    letterbox,
)
from utils.general import (
    cv2,
)

# Parameters
HELP_URL = "See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders

# Digunakan
# class LoadScreenshots:
#     # YOLOv5 screenshot dataloader, i.e. `python detect.py --source "screen 0 100 100 512 256"`
#     def __init__(self, source, img_size=640, stride=32, auto=True, transforms=None):
#         """
#         Initializes a screenshot dataloader for YOLOv5 with specified source region, image size, stride, auto, and
#         transforms.

#         Source = [screen_number left top width height] (pixels)
#         """
#         check_requirements("mss")
#         import mss

#         source, *params = source.split()
#         self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
#         if len(params) == 1:
#             self.screen = int(params[0])
#         elif len(params) == 4:
#             left, top, width, height = (int(x) for x in params)
#         elif len(params) == 5:
#             self.screen, left, top, width, height = (int(x) for x in params)
#         self.img_size = img_size
#         self.stride = stride
#         self.transforms = transforms
#         self.auto = auto
#         self.mode = "stream"
#         self.frame = 0
#         self.sct = mss.mss()

#         # Parse monitor shape
#         monitor = self.sct.monitors[self.screen]
#         self.top = monitor["top"] if top is None else (monitor["top"] + top)
#         self.left = monitor["left"] if left is None else (monitor["left"] + left)
#         self.width = width or monitor["width"]
#         self.height = height or monitor["height"]
#         self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

#     def __iter__(self):
#         """Iterates over itself, enabling use in loops and iterable contexts."""
#         return self

#     def __next__(self):
#         """Captures and returns the next screen frame as a BGR numpy array, cropping to only the first three channels
#         from BGRA.
#         """
#         im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
#         s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

#         if self.transforms:
#             im = self.transforms(im0)  # transforms
#         else:
#             im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
#             im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#             im = np.ascontiguousarray(im)  # contiguous
#         self.frame += 1
#         return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s

# Digunakan
class LoadImages:
    """YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`"""

    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initializes YOLOv5 loader for images/videos, supporting glob patterns, directories, and lists of paths."""
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        """Initializes iterator by resetting count and returns the iterator object itself."""
        self.count = 0
        return self

    def __next__(self):
        """Advances to the next file in the dataset, raising StopIteration if at the end."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """Initializes a new video capture object with path, frame count adjusted by stride, and orientation
        metadata.
        """
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        """Rotates a cv2 image based on its orientation; supports 0, 90, and 180 degrees rotations."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.nf  # number of files

# Digunakan
# class LoadStreams:
#     # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
#     def __init__(self, sources="file.streams", img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
#         """Initializes a stream loader for processing video streams with YOLOv5, supporting various sources including
#         YouTube.
#         """
#         torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
#         self.mode = "stream"
#         self.img_size = img_size
#         self.stride = stride
#         self.vid_stride = vid_stride  # video frame-rate stride
#         sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
#         n = len(sources)
#         self.sources = [clean_str(x) for x in sources]  # clean source names for later
#         self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
#         for i, s in enumerate(sources):  # index, source
#             # Start thread to read frames from video stream
#             st = f"{i + 1}/{n}: {s}... "
#             if urlparse(s).hostname in ("www.youtube.com", "youtube.com", "youtu.be"):  # if source is YouTube video
#                 # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
#                 check_requirements(("pafy", "youtube_dl==2020.12.2"))
#                 import pafy

#                 s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
#             s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
#             if s == 0:
#                 assert not is_colab(), "--source 0 webcam unsupported on Colab. Rerun command in a local environment."
#                 assert not is_kaggle(), "--source 0 webcam unsupported on Kaggle. Rerun command in a local environment."
#             cap = cv2.VideoCapture(s)
#             assert cap.isOpened(), f"{st}Failed to open {s}"
#             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
#             self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")  # infinite stream fallback
#             self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

#             _, self.imgs[i] = cap.read()  # guarantee first frame
#             self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
#             LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
#             self.threads[i].start()
#         LOGGER.info("")  # newline

#         # check for common shapes
#         s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
#         self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
#         self.auto = auto and self.rect
#         self.transforms = transforms  # optional
#         if not self.rect:
#             LOGGER.warning("WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.")

#     def update(self, i, cap, stream):
#         """Reads frames from stream `i`, updating imgs array; handles stream reopening on signal loss."""
#         n, f = 0, self.frames[i]  # frame number, frame array
#         while cap.isOpened() and n < f:
#             n += 1
#             cap.grab()  # .read() = .grab() followed by .retrieve()
#             if n % self.vid_stride == 0:
#                 success, im = cap.retrieve()
#                 if success:
#                     self.imgs[i] = im
#                 else:
#                     LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
#                     self.imgs[i] = np.zeros_like(self.imgs[i])
#                     cap.open(stream)  # re-open stream if signal was lost
#             time.sleep(0.0)  # wait time

#     def __iter__(self):
#         """Resets and returns the iterator for iterating over video frames or images in a dataset."""
#         self.count = -1
#         return self

#     def __next__(self):
#         """Iterates over video frames or images, halting on thread stop or 'q' key press, raising `StopIteration` when
#         done.
#         """
#         self.count += 1
#         if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):  # q to quit
#             cv2.destroyAllWindows()
#             raise StopIteration

#         im0 = self.imgs.copy()
#         if self.transforms:
#             im = np.stack([self.transforms(x) for x in im0])  # transforms
#         else:
#             im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
#             im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
#             im = np.ascontiguousarray(im)  # contiguous

#         return self.sources, im, im0, None, ""

#     def __len__(self):
#         """Returns the number of sources in the dataset, supporting up to 32 streams at 30 FPS over 30 years."""
#         return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years