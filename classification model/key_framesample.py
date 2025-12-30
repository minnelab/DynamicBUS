import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def keyframe_sampling(video_path, num_frames=16):
    """
    Extract keyframes from a video.
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample.
    
    Returns:
        keyframes (list): List of keyframes (numpy arrays).
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    
    video_manager.set_downscale_factor()
    video_manager.start()
    
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    keyframes = []
    for scene in scene_list:
        start_frame, end_frame = scene
        mid_frame = (start_frame + end_frame) // 2
        video_manager.seek(mid_frame)
        ret, frame = video_manager.read()
        if ret:
            keyframes.append(frame)
    
    video_manager.release()
    # Pad or truncate to get exactly num_frames keyframes
    if len(keyframes) > num_frames:
        keyframes = keyframes[:num_frames]
    elif len(keyframes) < num_frames:
        keyframes += [keyframes[-1]] * (num_frames - len(keyframes))
    return keyframes