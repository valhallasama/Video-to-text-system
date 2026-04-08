"""Frame extraction module with adaptive sampling."""

from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np


@dataclass
class Frame:
    """Container for extracted frame data."""
    index: int
    timestamp_sec: float
    image: np.ndarray
    metadata: dict


class FrameExtractor:
    """Extract frames from video with configurable sampling strategy."""
    
    def __init__(
        self,
        fps: float = 5.0,
        adaptive: bool = False,
        skip_similar: bool = False,
        similarity_threshold: float = 0.95
    ):
        self.fps = fps
        self.adaptive = adaptive
        self.skip_similar = skip_similar
        self.similarity_threshold = similarity_threshold
        self._prev_frame = None
    
    def extract(self, video_path: str) -> List[Frame]:
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        step = max(int(round(video_fps / self.fps)), 1)
        
        frames: List[Frame] = []
        frame_idx = 0
        sample_idx = 0
        
        while True:
            ret, image = cap.read()
            if not ret:
                break
            
            if frame_idx % step == 0:
                if self.skip_similar and self._is_similar(image):
                    frame_idx += 1
                    continue
                
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                frames.append(Frame(
                    index=sample_idx,
                    timestamp_sec=timestamp,
                    image=image.copy(),
                    metadata={
                        "original_frame_idx": frame_idx,
                        "total_frames": total_frames,
                        "video_fps": video_fps
                    }
                ))
                
                self._prev_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                sample_idx += 1
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def _is_similar(self, frame: np.ndarray) -> bool:
        """Check if frame is similar to previous frame."""
        if self._prev_frame is None:
            return False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if gray.shape != self._prev_frame.shape:
            return False
        
        diff = cv2.absdiff(gray, self._prev_frame)
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity >= self.similarity_threshold
