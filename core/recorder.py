# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Video recording functionality for camera track animations."""

import os
import tempfile
import shutil
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

import numpy as np
import lichtfeld as lf
import lichtfeld.io as lf_io

from .camera_track import CameraTrack
from .linear_path import LinearPath


def compute_up_vector(track: CameraTrack, t: float) -> tuple:
    """Compute the proper up vector to prevent camera roll.
    
    The up vector should be along the orbit axis to keep the camera level.
    """
    axis = track.orbit_axis
    
    if axis == "z":
        # Orbit around Z - up is Z
        return (0.0, 0.0, 1.0)
    elif axis == "y":
        # Orbit around Y - up is Y
        return (0.0, 1.0, 0.0)
    elif axis == "x":
        # Orbit around X - up is X
        return (1.0, 0.0, 0.0)
    else:
        return (0.0, 0.0, 1.0)

# Try to import imageio for video encoding
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio as iio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False


@dataclass
class RecordingSettings:
    """Settings for video recording.
    
    Attributes:
        output_path: Path to save the output video (MP4)
        resolution: Video resolution (width, height)
        fps: Frames per second
        quality: Video quality (0-100, higher is better)
        fov: Field of view in degrees
    """
    output_path: str = ""
    resolution: Tuple[int, int] = (1920, 1080)
    fps: float = 30.0
    quality: int = 85
    fov: float = 60.0


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
    """Record a video following the camera track using lf.render_at.
    
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
        
        # Check if we have imageio for video encoding
        if not HAS_IMAGEIO:
            return False, "imageio not installed. Run: pip install imageio[ffmpeg]"
        
        if progress_callback:
            progress_callback(0.0, "Initializing...")
        
        # Check scene
        scene = lf.get_scene()
        if not scene:
            return False, "No scene loaded"
        
        # Get target (POI)
        target = track.get_camera_look_at()
        width, height = settings.resolution
        
        if progress_callback:
            progress_callback(0.02, "Starting video encoding...")
        
        # Open video writer
        # Use ffmpeg backend with h264 codec
        writer = iio.imopen(
            settings.output_path,
            "w",
            plugin="pyav",
        )
        writer.init_video_stream("h264", fps=settings.fps)
        
        # Record each frame
        frame_duration = 1.0 / settings.fps
        for frame_idx in range(total_frames):
            t = frame_idx * frame_duration
            
            # Get camera position for this time
            eye = track.get_camera_position(t)
            
            # Compute proper up vector to prevent camera roll
            up = compute_up_vector(track, t)
            
            # Render frame using lf.render_at
            frame_tensor = lf.render_at(
                eye=eye,
                target=target,
                width=width,
                height=height,
                fov=settings.fov,
                up=up
            )
            
            if frame_tensor is None:
                return False, f"Render failed at frame {frame_idx}"
            
            # Convert tensor to numpy array (0-255 uint8)
            frame_np = frame_tensor.numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
            
            # Write frame to video
            writer.write_frame(frame_np)
            
            # Update progress
            if progress_callback:
                progress = 0.02 + 0.93 * (frame_idx + 1) / total_frames
                progress_callback(progress, f"Frame {frame_idx + 1}/{total_frames}")
        
        # Close video writer
        if progress_callback:
            progress_callback(0.95, "Finalizing video...")
        
        writer.close()
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return True, f"Saved video to {settings.output_path}"
        
    except Exception as e:
        return False, f"Recording failed: {str(e)}"


def record_linear_video(
    path: LinearPath,
    settings: RecordingSettings,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """Record a video following a linear camera path.
    
    Args:
        path: The linear camera path to follow
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
        
        # Validate path
        if not path.segments:
            return False, "No path segments defined"
        
        # Get total frames
        total_frames = path.get_total_frames(settings.fps)
        if total_frames <= 0:
            return False, "Invalid frame count"
        
        # Check if we have imageio for video encoding
        if not HAS_IMAGEIO:
            return False, "imageio not installed. Run: pip install imageio[pyav]"
        
        if progress_callback:
            progress_callback(0.0, "Initializing...")
        
        # Check scene
        scene = lf.get_scene()
        if not scene:
            return False, "No scene loaded"
        
        width, height = settings.resolution
        total_duration = path.get_total_duration()
        
        if progress_callback:
            progress_callback(0.02, "Starting video encoding...")
        
        # Open video writer
        writer = iio.imopen(
            settings.output_path,
            "w",
            plugin="pyav",
        )
        writer.init_video_stream("h264", fps=settings.fps)
        
        # Record each frame
        frame_duration = 1.0 / settings.fps
        for frame_idx in range(total_frames):
            t = frame_idx * frame_duration
            
            # Get camera position, target, and up vector
            eye = path.get_camera_position(t)
            target = path.get_camera_target(t)
            up = path.get_up_vector(t)
            
            # Render frame using lf.render_at
            frame_tensor = lf.render_at(
                eye=eye,
                target=target,
                width=width,
                height=height,
                fov=settings.fov,
                up=up
            )
            
            if frame_tensor is None:
                return False, f"Render failed at frame {frame_idx}"
            
            # Convert tensor to numpy array (0-255 uint8)
            frame_np = frame_tensor.numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
            
            # Write frame to video
            writer.write_frame(frame_np)
            
            # Update progress
            if progress_callback:
                progress = 0.02 + 0.93 * (frame_idx + 1) / total_frames
                progress_callback(progress, f"Frame {frame_idx + 1}/{total_frames}")
        
        # Close video writer
        if progress_callback:
            progress_callback(0.95, "Finalizing video...")
        
        writer.close()
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return True, f"Saved video to {settings.output_path}"
        
    except Exception as e:
        return False, f"Recording failed: {str(e)}"


def record_linear_frames_to_folder(
    path: LinearPath,
    settings: RecordingSettings,
    output_folder: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """Record frames from a linear path as individual images to a folder.
    
    Args:
        path: The linear camera path to follow
        settings: Recording settings
        output_folder: Folder to save frames
        progress_callback: Optional callback(progress: float, message: str)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Validate path
        if not path.segments:
            return False, "No path segments defined"
        
        # Get total frames
        total_frames = path.get_total_frames(settings.fps)
        if total_frames <= 0:
            return False, "Invalid frame count"
        
        if progress_callback:
            progress_callback(0.0, "Initializing...")
        
        # Check scene
        scene = lf.get_scene()
        if not scene:
            return False, "No scene loaded"
        
        width, height = settings.resolution
        
        # Record each frame
        frame_duration = 1.0 / settings.fps
        for frame_idx in range(total_frames):
            t = frame_idx * frame_duration
            
            # Get camera position, target, and up vector
            eye = path.get_camera_position(t)
            target = path.get_camera_target(t)
            up = path.get_up_vector(t)
            
            # Render frame using lf.render_at
            frame_tensor = lf.render_at(
                eye=eye,
                target=target,
                width=width,
                height=height,
                fov=settings.fov,
                up=up
            )
            
            if frame_tensor is None:
                return False, f"Render failed at frame {frame_idx}"
            
            # Save frame using lf.io.save_image
            frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.png")
            lf_io.save_image(frame_path, frame_tensor)
            
            # Update progress
            if progress_callback:
                progress = 0.05 + 0.90 * (frame_idx + 1) / total_frames
                progress_callback(progress, f"Frame {frame_idx + 1}/{total_frames}")
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return True, f"Saved {total_frames} frames to {output_folder}"
        
    except Exception as e:
        return False, f"Recording failed: {str(e)}"


def record_frames_to_folder(
    track: CameraTrack,
    settings: RecordingSettings,
    output_folder: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[bool, str]:
    """Record frames as individual images to a folder.
    
    Args:
        track: The camera track to follow
        settings: Recording settings
        output_folder: Folder to save frames
        progress_callback: Optional callback(progress: float, message: str)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get total frames
        total_frames = track.get_total_frames(settings.fps)
        if total_frames <= 0:
            return False, "Invalid frame count"
        
        if progress_callback:
            progress_callback(0.0, "Initializing...")
        
        # Check scene
        scene = lf.get_scene()
        if not scene:
            return False, "No scene loaded"
        
        # Get target (POI)
        target = track.get_camera_look_at()
        width, height = settings.resolution
        
        # Record each frame
        frame_duration = 1.0 / settings.fps
        for frame_idx in range(total_frames):
            t = frame_idx * frame_duration
            
            # Get camera position for this time
            eye = track.get_camera_position(t)
            
            # Compute proper up vector to prevent camera roll
            up = compute_up_vector(track, t)
            
            # Render frame using lf.render_at
            frame_tensor = lf.render_at(
                eye=eye,
                target=target,
                width=width,
                height=height,
                fov=settings.fov,
                up=up
            )
            
            if frame_tensor is None:
                return False, f"Render failed at frame {frame_idx}"
            
            # Save frame using lf.io.save_image
            frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.png")
            lf_io.save_image(frame_path, frame_tensor)
            
            # Update progress
            if progress_callback:
                progress = 0.05 + 0.90 * (frame_idx + 1) / total_frames
                progress_callback(progress, f"Frame {frame_idx + 1}/{total_frames}")
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return True, f"Saved {total_frames} frames to {output_folder}"
        
    except Exception as e:
        return False, f"Recording failed: {str(e)}"
