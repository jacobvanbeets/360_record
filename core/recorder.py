# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Video recording functionality for camera track animations."""

import os
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

import numpy as np
import lichtfeld as lf

from .camera_track import CameraTrack


@dataclass
class RecordingSettings:
    """Settings for video recording.
    
    Attributes:
        output_path: Path to save the output video (MP4)
        resolution: Video resolution (width, height)
        fps: Frames per second
        quality: Video quality (0-100, higher is better)
    """
    output_path: str = ""
    resolution: Tuple[int, int] = (1920, 1080)
    fps: float = 30.0
    quality: int = 85


def get_default_output_path() -> str:
    """Get default output path based on current scene/model.
    
    Returns:
        Default path for the output video
    """
    scene = lf.get_scene()
    if scene is not None:
        # Try to get the scene file path
        scene_path = scene.file_path if hasattr(scene, 'file_path') else None
        if scene_path:
            base_dir = os.path.dirname(scene_path)
            base_name = os.path.splitext(os.path.basename(scene_path))[0]
            return os.path.join(base_dir, f"{base_name}_360.mp4")
    
    # Default to user's Videos folder
    videos_dir = os.path.expanduser("~/Videos")
    if not os.path.exists(videos_dir):
        videos_dir = os.path.expanduser("~")
    return os.path.join(videos_dir, "360_record.mp4")


def record_circular_video(
    track: CameraTrack,
    settings: RecordingSettings,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """Record a video following the camera track.
    
    Args:
        track: The camera track to follow
        settings: Recording settings
        progress_callback: Optional callback(progress: float, message: str)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Validate output path
        if not settings.output_path:
            return False, "No output path specified"
        
        output_dir = os.path.dirname(settings.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Get total frames
        total_frames = track.get_total_frames(settings.fps)
        if total_frames <= 0:
            return False, "Invalid frame count"
        
        if progress_callback:
            progress_callback(0.0, "Initializing recorder...")
        
        # Get the scene and view
        scene = lf.get_scene()
        if not scene:
            return False, "No scene loaded"
        
        view = lf.get_current_view()
        if not view:
            return False, "No active view"
        
        # Store original camera state to restore later
        original_translation = view.translation.numpy().copy()
        original_rotation = view.rotation.numpy().copy() if hasattr(view, 'rotation') else None
        
        # Start video recording using LichtFeld's recorder API
        recorder = lf.get_recorder()
        if recorder is None:
            return False, "Recorder not available"
        
        recorder.start(
            settings.output_path,
            width=settings.resolution[0],
            height=settings.resolution[1],
            fps=settings.fps,
            quality=settings.quality
        )
        
        if progress_callback:
            progress_callback(0.05, "Recording frames...")
        
        # Record each frame
        frame_duration = 1.0 / settings.fps
        for frame_idx in range(total_frames):
            t = frame_idx * frame_duration
            
            # Get camera transform for this time
            pos, rotation = track.get_camera_transform(t)
            
            # Apply camera transform
            view.set_translation(lf.Tensor.from_numpy(pos.astype(np.float32)))
            if hasattr(view, 'set_rotation_matrix'):
                view.set_rotation_matrix(lf.Tensor.from_numpy(rotation.astype(np.float32)))
            elif hasattr(view, 'look_at'):
                # Alternative: use look_at if available
                target = np.array(track.get_camera_look_at())
                view.look_at(
                    lf.Tensor.from_numpy(pos.astype(np.float32)),
                    lf.Tensor.from_numpy(target.astype(np.float32))
                )
            
            # Render and capture frame
            lf.ui.request_redraw()
            recorder.capture_frame()
            
            # Update progress
            if progress_callback:
                progress = 0.05 + 0.90 * (frame_idx + 1) / total_frames
                progress_callback(progress, f"Frame {frame_idx + 1}/{total_frames}")
        
        # Finalize recording
        if progress_callback:
            progress_callback(0.95, "Finalizing video...")
        
        recorder.stop()
        
        # Restore original camera state
        view.set_translation(lf.Tensor.from_numpy(original_translation.astype(np.float32)))
        if original_rotation is not None and hasattr(view, 'set_rotation_matrix'):
            view.set_rotation_matrix(lf.Tensor.from_numpy(original_rotation.astype(np.float32)))
        
        lf.ui.request_redraw()
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return True, f"Saved video to {settings.output_path}"
        
    except Exception as e:
        return False, f"Recording failed: {str(e)}"


def preview_camera_position(track: CameraTrack, angle_degrees: float) -> bool:
    """Move the camera to preview a position on the track.
    
    Args:
        track: The camera track
        angle_degrees: Angle to preview (in degrees)
        
    Returns:
        True if successful
    """
    try:
        view = lf.get_current_view()
        if not view:
            return False
        
        # Calculate time for this angle
        angle_offset = angle_degrees - track.starting_angle
        t = (angle_offset / 360.0) * track.speed
        
        # Get camera transform
        pos, rotation = track.get_camera_transform(t)
        
        # Apply to view
        view.set_translation(lf.Tensor.from_numpy(pos.astype(np.float32)))
        if hasattr(view, 'set_rotation_matrix'):
            view.set_rotation_matrix(lf.Tensor.from_numpy(rotation.astype(np.float32)))
        elif hasattr(view, 'look_at'):
            target = np.array(track.get_camera_look_at())
            view.look_at(
                lf.Tensor.from_numpy(pos.astype(np.float32)),
                lf.Tensor.from_numpy(target.astype(np.float32))
            )
        
        lf.ui.request_redraw()
        return True
        
    except Exception as e:
        lf.log.error(f"Preview failed: {e}")
        return False
