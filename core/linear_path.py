# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Linear camera path with multi-segment support and smooth transitions."""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class LineSegment:
    """A single line segment in the camera path.
    
    Attributes:
        start_point: Starting position (x, y, z)
        end_point: Ending position (x, y, z)
        look_mode: "forward" to look along travel direction, "poi" to look at a point
        poi: Point of interest when look_mode="poi"
    """
    start_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end_point: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    look_mode: str = "forward"  # "forward" or "poi"
    poi: Optional[Tuple[float, float, float]] = None
    
    def get_length(self) -> float:
        """Get the length of this segment."""
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        dz = self.end_point[2] - self.start_point[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_direction(self) -> Tuple[float, float, float]:
        """Get normalized direction vector from start to end."""
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        dz = self.end_point[2] - self.start_point[2]
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 1e-6:
            return (1.0, 0.0, 0.0)
        return (dx / length, dy / length, dz / length)
    
    def get_position_at(self, t: float) -> Tuple[float, float, float]:
        """Get position at parameter t (0=start, 1=end)."""
        t = max(0.0, min(1.0, t))
        return (
            self.start_point[0] + t * (self.end_point[0] - self.start_point[0]),
            self.start_point[1] + t * (self.end_point[1] - self.start_point[1]),
            self.start_point[2] + t * (self.end_point[2] - self.start_point[2]),
        )


def _catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    """Catmull-Rom spline interpolation between p1 and p2.
    
    Args:
        p0, p1, p2, p3: Control points (p1 and p2 are the segment endpoints)
        t: Parameter in [0, 1] for position between p1 and p2
    
    Returns:
        Interpolated position
    """
    t2 = t * t
    t3 = t2 * t
    
    return 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two direction vectors.
    
    Args:
        v0, v1: Unit direction vectors
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated unit direction vector
    """
    # Normalize inputs
    v0 = v0 / (np.linalg.norm(v0) + 1e-8)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    
    # If vectors are nearly parallel, use linear interpolation
    if dot > 0.9995:
        result = v0 + t * (v1 - v0)
        return result / (np.linalg.norm(result) + 1e-8)
    
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    
    if abs(sin_theta) < 1e-6:
        return v0
    
    a = math.sin((1.0 - t) * theta) / sin_theta
    b = math.sin(t * theta) / sin_theta
    
    result = a * v0 + b * v1
    return result / (np.linalg.norm(result) + 1e-8)


@dataclass
class LinearPath:
    """A camera path composed of multiple connected line segments.
    
    Attributes:
        segments: List of line segments in order
        speed: Travel speed in units per second
        smooth_factor: Smoothing intensity for transitions (0=sharp, 1=maximum smooth)
        elevation: Offset distance perpendicular to the path
        up_axis: Axis for camera level ('z', 'x', or 'y')
        invert_elevation: Invert the elevation direction
    """
    segments: List[LineSegment] = field(default_factory=list)
    speed: float = 1.0  # units per second
    smooth_factor: float = 0.5  # 0-1, higher = smoother transitions
    elevation: float = 0.0  # offset perpendicular to path
    up_axis: str = "z"  # 'z', 'x', or 'y'
    invert_elevation: bool = False
    
    def get_total_distance(self) -> float:
        """Get the total path distance including transitions."""
        if not self.segments:
            return 0.0
        
        total = 0.0
        for i, segment in enumerate(self.segments):
            total += segment.get_length()
            
            # Add transition distance to next segment
            if i < len(self.segments) - 1:
                next_seg = self.segments[i + 1]
                # Transition distance is the gap between end and next start
                dx = next_seg.start_point[0] - segment.end_point[0]
                dy = next_seg.start_point[1] - segment.end_point[1]
                dz = next_seg.start_point[2] - segment.end_point[2]
                total += math.sqrt(dx*dx + dy*dy + dz*dz)
        
        return total
    
    def get_total_duration(self) -> float:
        """Get total path duration in seconds."""
        if self.speed <= 0:
            return 0.0
        return self.get_total_distance() / self.speed
    
    def get_total_frames(self, fps: float = 30.0) -> int:
        """Get total frame count for recording."""
        return max(1, int(self.get_total_duration() * fps))
    
    def _get_all_points(self) -> List[np.ndarray]:
        """Get all waypoints including segment endpoints."""
        if not self.segments:
            return []
        
        points = []
        for i, segment in enumerate(self.segments):
            points.append(np.array(segment.start_point))
            if i == len(self.segments) - 1:
                points.append(np.array(segment.end_point))
        
        return points
    
    def _get_cumulative_distances(self) -> List[float]:
        """Get cumulative distances at each waypoint."""
        points = self._get_all_points()
        if len(points) < 2:
            return [0.0]
        
        distances = [0.0]
        for i in range(1, len(points)):
            d = np.linalg.norm(points[i] - points[i-1])
            distances.append(distances[-1] + d)
        
        return distances
    
    def _get_elevation_offset(self) -> np.ndarray:
        """Get the elevation offset vector based on up_axis and invert setting."""
        offset = self.elevation if not self.invert_elevation else -self.elevation
        
        if self.up_axis == "z":
            return np.array([0.0, 0.0, offset])
        elif self.up_axis == "y":
            return np.array([0.0, offset, 0.0])
        elif self.up_axis == "x":
            return np.array([offset, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, offset])
    
    def get_camera_position(self, t: float) -> Tuple[float, float, float]:
        """Get camera position at time t (seconds from start).
        
        Uses Catmull-Rom spline interpolation for smooth transitions.
        
        Args:
            t: Time in seconds from path start
            
        Returns:
            Camera position (x, y, z)
        """
        if not self.segments:
            return (0.0, 0.0, 0.0)
        
        points = self._get_all_points()
        if len(points) < 2:
            return tuple(points[0]) if points else (0.0, 0.0, 0.0)
        
        # Convert time to distance
        distance = t * self.speed
        total_distance = self.get_total_distance()
        distance = max(0.0, min(distance, total_distance))
        
        # Find which segment we're on
        cumulative = self._get_cumulative_distances()
        
        # Find segment index
        seg_idx = 0
        for i in range(1, len(cumulative)):
            if distance <= cumulative[i]:
                seg_idx = i - 1
                break
        else:
            seg_idx = len(cumulative) - 2
        
        # Local parameter within segment
        seg_start_dist = cumulative[seg_idx]
        seg_end_dist = cumulative[seg_idx + 1]
        seg_length = seg_end_dist - seg_start_dist
        
        if seg_length < 1e-6:
            local_t = 0.0
        else:
            local_t = (distance - seg_start_dist) / seg_length
        
        local_t = max(0.0, min(1.0, local_t))
        
        # Apply Catmull-Rom smoothing if smooth_factor > 0
        if self.smooth_factor > 0 and len(points) >= 2:
            # Get control points for Catmull-Rom
            p1 = points[seg_idx]
            p2 = points[seg_idx + 1]
            
            # Extrapolate p0 and p3 if at boundaries
            if seg_idx == 0:
                p0 = p1 - (p2 - p1)  # Mirror
            else:
                p0 = points[seg_idx - 1]
            
            if seg_idx + 2 >= len(points):
                p3 = p2 + (p2 - p1)  # Mirror
            else:
                p3 = points[seg_idx + 2]
            
            # Blend between linear and Catmull-Rom based on smooth_factor
            linear_pos = p1 + local_t * (p2 - p1)
            smooth_pos = _catmull_rom(p0, p1, p2, p3, local_t)
            
            base_pos = linear_pos * (1.0 - self.smooth_factor) + smooth_pos * self.smooth_factor
        else:
            # Linear interpolation
            p1 = points[seg_idx]
            p2 = points[seg_idx + 1]
            base_pos = p1 + local_t * (p2 - p1)
        
        # Apply elevation offset
        result = base_pos + self._get_elevation_offset()
        return tuple(result)
    
    def _get_segment_look_direction(self, seg_idx: int, pos: np.ndarray) -> np.ndarray:
        """Get the look direction for a segment.
        
        Args:
            seg_idx: Segment index
            pos: Current camera position
            
        Returns:
            Normalized look direction vector
        """
        if seg_idx < 0 or seg_idx >= len(self.segments):
            return np.array([1.0, 0.0, 0.0])
        
        segment = self.segments[seg_idx]
        
        if segment.look_mode == "poi" and segment.poi is not None:
            direction = np.array(segment.poi) - pos
        else:
            direction = np.array(segment.get_direction())
        
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return direction / norm
    
    def _smooth_step(self, t: float) -> float:
        """Smooth step function for easing (hermite interpolation)."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)
    
    def get_camera_target(self, t: float) -> Tuple[float, float, float]:
        """Get the camera look-at target at time t.
        
        Uses extended blending at segment transitions for smooth look direction.
        
        Args:
            t: Time in seconds from path start
            
        Returns:
            Look-at position (x, y, z)
        """
        if not self.segments:
            return (1.0, 0.0, 0.0)
        
        # Find which segment we're on
        distance = t * self.speed
        total_distance = self.get_total_distance()
        distance = max(0.0, min(distance, total_distance))
        
        cumulative = self._get_cumulative_distances()
        
        # Find segment index
        seg_idx = 0
        for i in range(1, len(cumulative)):
            if distance <= cumulative[i]:
                seg_idx = i - 1
                break
        else:
            seg_idx = max(0, len(self.segments) - 1)
        
        # Clamp segment index
        seg_idx = min(seg_idx, len(self.segments) - 1)
        
        # Local parameter within segment for transition blending
        seg_start_dist = cumulative[seg_idx]
        seg_end_dist = cumulative[min(seg_idx + 1, len(cumulative) - 1)]
        seg_length = seg_end_dist - seg_start_dist
        
        if seg_length < 1e-6:
            local_t = 0.5
        else:
            local_t = (distance - seg_start_dist) / seg_length
        local_t = max(0.0, min(1.0, local_t))
        
        pos = np.array(self.get_camera_position(t))
        
        # For smooth transitions, blend between previous, current, and next segment directions
        # The key is that at segment boundaries, both sides must compute the SAME direction
        
        has_prev = seg_idx > 0
        has_next = seg_idx < len(self.segments) - 1
        
        # Get directions for relevant segments
        current_dir = self._get_segment_look_direction(seg_idx, pos)
        prev_dir = self._get_segment_look_direction(seg_idx - 1, pos) if has_prev else current_dir
        next_dir = self._get_segment_look_direction(seg_idx + 1, pos) if has_next else current_dir
        
        # Blend zone as fraction of segment (15% base + up to 35% from smooth_factor)
        blend_zone = 0.15 + 0.35 * self.smooth_factor
        
        # Compute a continuous blend parameter that goes:
        # - From 0 (start of blend-in from prev) to 0.5 (middle of segment) to 1 (end of blend-out to next)
        # This ensures both sides of a boundary compute the same blended direction
        
        if local_t < blend_zone and has_prev:
            # In the blend-in zone from previous segment
            # At local_t=0 (boundary): blend is 50% prev, 50% current
            # At local_t=blend_zone: blend is 100% current
            # zone_t goes from 0.5 to 1.0
            zone_t = 0.5 + 0.5 * (local_t / blend_zone)
            zone_t = self._smooth_step(zone_t * 2.0 - 1.0) * 0.5 + 0.5  # Smooth easing
            # Blend from prev to current
            result_dir = _slerp(prev_dir, current_dir, zone_t)
            
        elif local_t > (1.0 - blend_zone) and has_next:
            # In the blend-out zone to next segment  
            # At local_t=1-blend_zone: blend is 100% current (zone_t=0)
            # At local_t=1 (boundary): blend is 50% current, 50% next (zone_t=0.5)
            zone_t = 0.5 * (local_t - (1.0 - blend_zone)) / blend_zone
            zone_t = self._smooth_step(zone_t * 2.0) * 0.5  # Smooth easing
            # Blend from current to next
            result_dir = _slerp(current_dir, next_dir, zone_t)
            
        else:
            # In the middle of the segment, use current direction
            result_dir = current_dir
        
        # Return a point along the look direction
        return tuple(pos + result_dir * 10.0)
    
    def get_up_vector(self, t: float) -> Tuple[float, float, float]:
        """Get the camera up vector at time t.
        
        Uses the configured up_axis to keep camera level.
        
        Args:
            t: Time in seconds from path start
            
        Returns:
            Up vector (x, y, z)
        """
        pos = np.array(self.get_camera_position(t))
        target = np.array(self.get_camera_target(t))
        
        forward = target - pos
        forward_len = np.linalg.norm(forward)
        if forward_len < 1e-6:
            if self.up_axis == "z":
                return (0.0, 0.0, 1.0)
            elif self.up_axis == "y":
                return (0.0, 1.0, 0.0)
            elif self.up_axis == "x":
                return (1.0, 0.0, 0.0)
            return (0.0, 0.0, 1.0)
        forward = forward / forward_len
        
        # World up based on configured axis
        if self.up_axis == "z":
            world_up = np.array([0.0, 0.0, 1.0])
        elif self.up_axis == "y":
            world_up = np.array([0.0, 1.0, 0.0])
        elif self.up_axis == "x":
            world_up = np.array([1.0, 0.0, 0.0])
        else:
            world_up = np.array([0.0, 0.0, 1.0])
        
        # Check if looking nearly along the up axis
        dot = abs(np.dot(forward, world_up))
        if dot > 0.99:
            # Use alternate axis when looking along the up axis
            if self.up_axis == "z":
                world_up = np.array([0.0, 1.0, 0.0])
            elif self.up_axis == "y":
                world_up = np.array([0.0, 0.0, 1.0])
            elif self.up_axis == "x":
                world_up = np.array([0.0, 0.0, 1.0])
        
        # Compute right and recompute up
        right = np.cross(forward, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-6:
            if self.up_axis == "z":
                return (0.0, 0.0, 1.0)
            elif self.up_axis == "y":
                return (0.0, 1.0, 0.0)
            elif self.up_axis == "x":
                return (1.0, 0.0, 0.0)
            return (0.0, 0.0, 1.0)
        right = right / right_len
        
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        return tuple(up)


def compute_linear_camera_position(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    t: float
) -> Tuple[float, float, float]:
    """Simple linear interpolation between two points.
    
    Args:
        start: Start position
        end: End position
        t: Parameter [0, 1]
        
    Returns:
        Interpolated position
    """
    t = max(0.0, min(1.0, t))
    return (
        start[0] + t * (end[0] - start[0]),
        start[1] + t * (end[1] - start[1]),
        start[2] + t * (end[2] - start[2]),
    )
