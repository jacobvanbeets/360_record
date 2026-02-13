# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera track computation for circular orbits around a point of interest."""

import math
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class CameraTrack:
    """Defines a circular camera track around a point of interest.
    
    Attributes:
        poi: Point of interest (x, y, z) - the center point the camera looks at
        elevation: Offset from POI along the orbit axis (in scene units)
        radius: Radius of the circular track (in scene units)
        speed: Time in seconds for one complete revolution
        starting_angle: Starting angle in degrees
        orbit_axis: Axis to orbit around ('z', 'x', or 'y')
    """
    poi: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    elevation: float = 1.5
    radius: float = 5.0
    speed: float = 30.0  # seconds per circle
    starting_angle: float = 0.0  # degrees
    orbit_axis: str = "z"  # 'z' = horizontal orbit, 'x' or 'y' = vertical orbits
    invert_direction: bool = False  # Invert the offset direction
    
    def get_camera_position(self, t: float) -> Tuple[float, float, float]:
        """Get camera position at time t.
        
        Args:
            t: Time in seconds from start
            
        Returns:
            Camera position (x, y, z)
        """
        # Calculate angle based on time and speed
        # Full circle = 2*pi radians over 'speed' seconds
        angle_rad = math.radians(self.starting_angle) + (2.0 * math.pi * t / self.speed)
        
        # Apply inversion to elevation offset
        offset = -self.elevation if self.invert_direction else self.elevation
        
        # Camera position depends on orbit axis
        if self.orbit_axis == "z":
            # Orbit around Z axis (horizontal circle in XY plane)
            x = self.poi[0] + self.radius * math.cos(angle_rad)
            y = self.poi[1] + self.radius * math.sin(angle_rad)
            z = self.poi[2] + offset
        elif self.orbit_axis == "x":
            # Orbit around X axis (vertical circle in YZ plane)
            x = self.poi[0] + offset
            y = self.poi[1] + self.radius * math.cos(angle_rad)
            z = self.poi[2] + self.radius * math.sin(angle_rad)
        elif self.orbit_axis == "y":
            # Orbit around Y axis (vertical circle in XZ plane)
            x = self.poi[0] + self.radius * math.cos(angle_rad)
            y = self.poi[1] + offset
            z = self.poi[2] + self.radius * math.sin(angle_rad)
        else:
            # Default to Z axis
            x = self.poi[0] + self.radius * math.cos(angle_rad)
            y = self.poi[1] + self.radius * math.sin(angle_rad)
            z = self.poi[2] + offset
        
        return (x, y, z)
    
    def get_camera_look_at(self) -> Tuple[float, float, float]:
        """Get the point the camera should look at (the POI).
        
        Returns:
            Look-at position (x, y, z)
        """
        return self.poi
    
    def get_total_frames(self, fps: float = 30.0) -> int:
        """Get total number of frames for one complete revolution.
        
        Args:
            fps: Frames per second
            
        Returns:
            Total frame count
        """
        return int(self.speed * fps)
    
    def get_camera_transform(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera position and rotation at time t.
        
        Args:
            t: Time in seconds from start
            
        Returns:
            Tuple of (position [3], rotation_matrix [3, 3])
        """
        pos = np.array(self.get_camera_position(t))
        target = np.array(self.get_camera_look_at())
        
        # Compute look-at rotation matrix
        forward = target - pos
        forward = forward / np.linalg.norm(forward)
        
        # Up vector (world Z)
        up = np.array([0.0, 0.0, 1.0])
        
        # Right vector
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            # Camera looking straight up/down, use different up
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Rotation matrix (camera looks along -Z in its local space)
        rotation = np.array([
            right,
            up,
            -forward
        ]).T
        
        return pos, rotation


def compute_camera_position(
    poi: Tuple[float, float, float],
    elevation: float,
    radius: float,
    angle_degrees: float,
    orbit_axis: str = "z",
    invert_direction: bool = False
) -> Tuple[float, float, float]:
    """Compute camera position for a given angle on the circular track.
    
    Args:
        poi: Point of interest (x, y, z)
        elevation: Offset along orbit axis
        radius: Radius of the circle
        angle_degrees: Angle in degrees
        orbit_axis: Axis to orbit around ('z', 'x', or 'y')
        invert_direction: Invert the offset direction
        
    Returns:
        Camera position (x, y, z)
    """
    angle_rad = math.radians(angle_degrees)
    offset = -elevation if invert_direction else elevation
    
    if orbit_axis == "z":
        # Orbit around Z axis (horizontal circle in XY plane)
        x = poi[0] + radius * math.cos(angle_rad)
        y = poi[1] + radius * math.sin(angle_rad)
        z = poi[2] + offset
    elif orbit_axis == "x":
        # Orbit around X axis (vertical circle in YZ plane)
        x = poi[0] + offset
        y = poi[1] + radius * math.cos(angle_rad)
        z = poi[2] + radius * math.sin(angle_rad)
    elif orbit_axis == "y":
        # Orbit around Y axis (vertical circle in XZ plane)
        x = poi[0] + radius * math.cos(angle_rad)
        y = poi[1] + offset
        z = poi[2] + radius * math.sin(angle_rad)
    else:
        x = poi[0] + radius * math.cos(angle_rad)
        y = poi[1] + radius * math.sin(angle_rad)
        z = poi[2] + offset
    
    return (x, y, z)
