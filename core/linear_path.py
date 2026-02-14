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
    segment_type: str = "linear"  # Always "linear" for LineSegment
    
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
    
    def get_start_point(self) -> Tuple[float, float, float]:
        """Get the starting position of this segment."""
        return self.start_point
    
    def get_end_point(self) -> Tuple[float, float, float]:
        """Get the ending position of this segment."""
        return self.end_point
    
    def get_duration(self, speed: float) -> float:
        """Get duration in seconds at given speed (units/sec)."""
        if speed <= 0:
            return 0.0
        return self.get_length() / speed


@dataclass
class OrbitSegment:
    """A circular orbit segment around a point of interest.
    
    Attributes:
        poi: Center point of the orbit (x, y, z)
        radius: Distance from POI to camera
        elevation: Offset along the orbit axis
        orbit_axis: Axis to orbit around ('z', 'x', or 'y')
        start_angle: Starting angle in degrees
        arc_degrees: How much to orbit (e.g., 360 for full circle)
        duration: Time in seconds for this orbit arc
        invert_direction: Orbit in reverse direction
    """
    poi: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 5.0
    elevation: float = 1.5
    orbit_axis: str = "z"
    start_angle: float = 0.0  # degrees
    arc_degrees: float = 360.0  # degrees
    duration: float = 30.0  # seconds
    invert_direction: bool = False
    segment_type: str = "orbit"  # Always "orbit" for OrbitSegment
    
    def get_length(self) -> float:
        """Get the arc length of this orbit segment."""
        arc_radians = math.radians(abs(self.arc_degrees))
        return self.radius * arc_radians
    
    def get_duration(self, speed: float = None) -> float:
        """Get duration in seconds. For orbits, duration is explicit, not speed-based."""
        return self.duration
    
    def _compute_position(self, angle_deg: float) -> Tuple[float, float, float]:
        """Compute camera position at a given angle."""
        angle_rad = math.radians(angle_deg)
        offset = -self.elevation if self.invert_direction else self.elevation
        
        if self.orbit_axis == "z":
            x = self.poi[0] + self.radius * math.cos(angle_rad)
            y = self.poi[1] + self.radius * math.sin(angle_rad)
            z = self.poi[2] + offset
        elif self.orbit_axis == "x":
            x = self.poi[0] + offset
            y = self.poi[1] + self.radius * math.cos(angle_rad)
            z = self.poi[2] + self.radius * math.sin(angle_rad)
        elif self.orbit_axis == "y":
            x = self.poi[0] + self.radius * math.cos(angle_rad)
            y = self.poi[1] + offset
            z = self.poi[2] + self.radius * math.sin(angle_rad)
        else:
            x = self.poi[0] + self.radius * math.cos(angle_rad)
            y = self.poi[1] + self.radius * math.sin(angle_rad)
            z = self.poi[2] + offset
        
        return (x, y, z)
    
    def get_start_point(self) -> Tuple[float, float, float]:
        """Get the starting position of this orbit segment."""
        return self._compute_position(self.start_angle)
    
    def get_end_point(self) -> Tuple[float, float, float]:
        """Get the ending position of this orbit segment."""
        end_angle = self.start_angle + self.arc_degrees
        return self._compute_position(end_angle)
    
    def get_position_at(self, t: float) -> Tuple[float, float, float]:
        """Get position at parameter t (0=start, 1=end)."""
        t = max(0.0, min(1.0, t))
        current_angle = self.start_angle + t * self.arc_degrees
        return self._compute_position(current_angle)
    
    def get_direction(self, t: float = 0.5) -> Tuple[float, float, float]:
        """Get tangent direction at parameter t (direction of travel)."""
        # For orbits, the tangent is perpendicular to the radius
        current_angle = self.start_angle + t * self.arc_degrees
        angle_rad = math.radians(current_angle)
        
        # Direction of increasing angle (tangent to circle)
        if self.orbit_axis == "z":
            dx = -math.sin(angle_rad)
            dy = math.cos(angle_rad)
            dz = 0.0
        elif self.orbit_axis == "x":
            dx = 0.0
            dy = -math.sin(angle_rad)
            dz = math.cos(angle_rad)
        elif self.orbit_axis == "y":
            dx = -math.sin(angle_rad)
            dy = 0.0
            dz = math.cos(angle_rad)
        else:
            dx = -math.sin(angle_rad)
            dy = math.cos(angle_rad)
            dz = 0.0
        
        # Normalize and account for arc direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 1e-6:
            return (1.0, 0.0, 0.0)
        
        sign = 1.0 if self.arc_degrees >= 0 else -1.0
        return (sign * dx / length, sign * dy / length, sign * dz / length)
    
    def get_look_at(self) -> Tuple[float, float, float]:
        """Get the point the camera should look at (the POI)."""
        return self.poi
    
    def get_up_vector(self) -> Tuple[float, float, float]:
        """Get the up vector based on orbit axis."""
        if self.orbit_axis == "z":
            return (0.0, 0.0, 1.0)
        elif self.orbit_axis == "y":
            return (0.0, 1.0, 0.0)
        elif self.orbit_axis == "x":
            return (1.0, 0.0, 0.0)
        return (0.0, 0.0, 1.0)


# Union type for segments
Segment = LineSegment | OrbitSegment


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
    """A camera path composed of multiple connected segments (linear or orbit).
    
    Attributes:
        segments: List of segments (LineSegment or OrbitSegment) in order
        speed: Travel speed in units per second (for linear segments)
        smooth_factor: Smoothing intensity for transitions (0=sharp, 1=maximum smooth)
        elevation: Offset distance perpendicular to path (for linear segments)
        up_axis: Axis for camera level ('z', 'x', or 'y')
        invert_elevation: Invert the elevation direction
    """
    segments: List[Segment] = field(default_factory=list)
    speed: float = 1.0  # units per second (for linear segments)
    smooth_factor: float = 0.5  # 0-1, higher = smoother transitions
    elevation: float = 0.0  # offset perpendicular to path (linear segments)
    up_axis: str = "z"  # 'z', 'x', or 'y'
    invert_elevation: bool = False
    _timeline_cache: List[Tuple[float, str, int]] = field(default_factory=list, repr=False)
    _timeline_cache_seg_count: int = field(default=0, repr=False)
    
    def _get_segment_duration(self, segment: Segment) -> float:
        """Get duration for a segment in seconds."""
        if isinstance(segment, OrbitSegment):
            return segment.duration
        else:
            # LineSegment - duration based on length and speed
            if self.speed <= 0:
                return 0.0
            return segment.get_length() / self.speed
    
    def _get_transition_duration(self, from_idx: int, to_idx: int) -> float:
        """Get duration to travel from end of one segment to start of next."""
        if from_idx < 0 or to_idx >= len(self.segments) or self.speed <= 0:
            return 0.0
        
        from_seg = self.segments[from_idx]
        to_seg = self.segments[to_idx]
        
        from_end = np.array(from_seg.get_end_point())
        to_start = np.array(to_seg.get_start_point())
        
        distance = np.linalg.norm(to_start - from_end)
        return distance / self.speed
    
    def _get_cumulative_durations(self) -> List[Tuple[float, str, int]]:
        """Get cumulative durations with segment/transition markers (cached).
        
        Returns list of (cumulative_time, type, index) where:
        - type is 'segment' or 'transition'
        - index is segment index (for 'segment') or from-segment index (for 'transition')
        """
        # Return cached if valid (invalidate if segment count changed)
        if self._timeline_cache and self._timeline_cache_seg_count == len(self.segments):
            return self._timeline_cache
        
        if not self.segments:
            return [(0.0, 'segment', 0)]
        
        timeline = [(0.0, 'segment', 0)]  # Start of first segment
        cumulative = 0.0
        
        for i, segment in enumerate(self.segments):
            # Add segment duration
            cumulative += self._get_segment_duration(segment)
            
            # Add transition to next segment if there is one
            if i < len(self.segments) - 1:
                timeline.append((cumulative, 'transition', i))
                trans_dur = self._get_transition_duration(i, i + 1)
                cumulative += trans_dur
                timeline.append((cumulative, 'segment', i + 1))
        
        timeline.append((cumulative, 'end', len(self.segments) - 1))
        self._timeline_cache = timeline
        self._timeline_cache_seg_count = len(self.segments)
        return timeline
    
    def get_total_distance(self) -> float:
        """Get the total path distance (arc length) including transitions."""
        if not self.segments:
            return 0.0
        
        total = 0.0
        for i, segment in enumerate(self.segments):
            total += segment.get_length()
            # Add transition distance
            if i < len(self.segments) - 1:
                from_end = np.array(segment.get_end_point())
                to_start = np.array(self.segments[i + 1].get_start_point())
                total += np.linalg.norm(to_start - from_end)
        
        return total
    
    def get_total_duration(self) -> float:
        """Get total path duration in seconds (including transitions)."""
        if not self.segments:
            return 0.0
        
        timeline = self._get_cumulative_durations()
        # Last entry is the 'end' marker with total time
        return timeline[-1][0]
    
    def get_total_frames(self, fps: float = 30.0) -> int:
        """Get total frame count for recording."""
        return max(1, int(self.get_total_duration() * fps))
    
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
    
    def _find_position_at_time(self, t: float) -> Tuple[str, int, float]:
        """Find what's happening at time t.
        
        Returns:
            (type, index, local_t) where:
            - type is 'segment' or 'transition'
            - index is segment index or from-segment index for transition
            - local_t is 0-1 progress within that segment/transition
        """
        if not self.segments:
            return ('segment', 0, 0.0)
        
        timeline = self._get_cumulative_durations()
        total_duration = timeline[-1][0]
        t = max(0.0, min(t, total_duration))
        
        # Find where we are in the timeline
        prev_entry = timeline[0]
        for i in range(1, len(timeline)):
            curr_entry = timeline[i]
            if t <= curr_entry[0]:
                # We're between prev_entry and curr_entry
                entry_type = prev_entry[1]
                entry_idx = prev_entry[2]
                start_time = prev_entry[0]
                end_time = curr_entry[0]
                duration = end_time - start_time
                
                if duration < 1e-6:
                    local_t = 0.0
                else:
                    local_t = (t - start_time) / duration
                
                return (entry_type, entry_idx, max(0.0, min(1.0, local_t)))
            prev_entry = curr_entry
        
        # At the end
        return ('segment', len(self.segments) - 1, 1.0)
    
    def _is_linear_segment(self, idx: int) -> bool:
        """Check if segment at index is a linear segment."""
        if idx < 0 or idx >= len(self.segments):
            return False
        return isinstance(self.segments[idx], LineSegment)
    
    def _get_segment_end_position(self, idx: int) -> np.ndarray:
        """Get the end position of a segment (with elevation for linear)."""
        segment = self.segments[idx]
        pos = np.array(segment.get_end_point())
        if isinstance(segment, LineSegment):
            pos = pos + self._get_elevation_offset()
        return pos
    
    def _get_segment_start_position(self, idx: int, apply_elevation: bool = True) -> np.ndarray:
        """Get the start position of a segment.
        
        Args:
            idx: Segment index
            apply_elevation: If True, apply elevation offset for linear segments.
                             Set False when transitioning from orbit to linear.
        """
        segment = self.segments[idx]
        pos = np.array(segment.get_start_point())
        if apply_elevation and isinstance(segment, LineSegment):
            pos = pos + self._get_elevation_offset()
        return pos
    
    def get_camera_position(self, t: float) -> Tuple[float, float, float]:
        """Get camera position at time t (seconds from start).
        
        For linear segments, uses Catmull-Rom spline interpolation (only between consecutive linears).
        For orbit segments, uses circular interpolation.
        For transitions, interpolates between segment end and next segment start.
        
        Args:
            t: Time in seconds from path start
            
        Returns:
            Camera position (x, y, z)
        """
        if not self.segments:
            return (0.0, 0.0, 0.0)
        
        pos_type, idx, local_t = self._find_position_at_time(t)
        
        # Handle transition between segments
        if pos_type == 'transition':
            from_seg = self.segments[idx]
            to_seg = self.segments[idx + 1]
            
            from_end = self._get_segment_end_position(idx)
            # Don't apply elevation to linear target if coming from orbit
            # (the linear segment will blend it in gradually)
            apply_elev = not isinstance(from_seg, OrbitSegment)
            to_start = self._get_segment_start_position(idx + 1, apply_elevation=apply_elev)
            
            # Smooth interpolation
            smooth_t = local_t * local_t * (3 - 2 * local_t)
            result = from_end * (1 - smooth_t) + to_start * smooth_t
            return tuple(result)
        
        # Handle segment
        segment = self.segments[idx]
        
        # Orbit segments handle their own position calculation
        if isinstance(segment, OrbitSegment):
            return segment.get_position_at(local_t)
        
        # Linear segment
        p1 = np.array(segment.get_start_point())
        p2 = np.array(segment.get_end_point())
        
        # Only apply Catmull-Rom smoothing between consecutive LINEAR segments
        prev_is_linear = self._is_linear_segment(idx - 1)
        next_is_linear = self._is_linear_segment(idx + 1)
        
        if self.smooth_factor > 0 and (prev_is_linear or next_is_linear):
            # Get control points for Catmull-Rom
            if prev_is_linear:
                p0 = np.array(self.segments[idx - 1].get_start_point())
            else:
                p0 = p1 - (p2 - p1)  # Mirror
            
            if next_is_linear:
                p3 = np.array(self.segments[idx + 1].get_end_point())
            else:
                p3 = p2 + (p2 - p1)  # Mirror
            
            linear_pos = p1 + local_t * (p2 - p1)
            smooth_pos = _catmull_rom(p0, p1, p2, p3, local_t)
            base_pos = linear_pos * (1.0 - self.smooth_factor) + smooth_pos * self.smooth_factor
        else:
            base_pos = p1 + local_t * (p2 - p1)
        
        # Apply elevation offset for linear segments
        # If coming from an orbit, gradually blend in the elevation over the segment
        elevation_offset = self._get_elevation_offset()
        prev_is_orbit = idx > 0 and isinstance(self.segments[idx - 1], OrbitSegment)
        
        if prev_is_orbit:
            # Blend elevation from 0 to full over the segment
            # Use smooth step for natural easing
            elevation_blend = local_t * local_t * (3 - 2 * local_t)
            elevation_offset = elevation_offset * elevation_blend
        
        result = base_pos + elevation_offset
        return tuple(result)
    
    def _get_segment_look_direction(self, seg_idx: int, pos: np.ndarray, local_t: float = 0.5) -> np.ndarray:
        """Get the look direction for a segment.
        
        Args:
            seg_idx: Segment index
            pos: Current camera position
            local_t: Position within segment (0-1)
            
        Returns:
            Normalized look direction vector
        """
        if seg_idx < 0 or seg_idx >= len(self.segments):
            return np.array([1.0, 0.0, 0.0])
        
        segment = self.segments[seg_idx]
        
        # Orbit segments always look at POI
        if isinstance(segment, OrbitSegment):
            direction = np.array(segment.poi) - pos
        elif hasattr(segment, 'look_mode') and segment.look_mode == "poi" and segment.poi is not None:
            # Linear segment with POI look mode
            direction = np.array(segment.poi) - pos
        else:
            # Linear segment looking forward
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
        
        For orbit segments, always looks at the POI.
        For linear segments, uses extended blending at transitions (only between linear segments).
        For transitions, blends look direction.
        
        Args:
            t: Time in seconds from path start
            
        Returns:
            Look-at position (x, y, z)
        """
        if not self.segments:
            return (1.0, 0.0, 0.0)
        
        pos_type, idx, local_t = self._find_position_at_time(t)
        pos = np.array(self.get_camera_position(t))
        
        # Handle transition - blend look directions
        if pos_type == 'transition':
            from_seg = self.segments[idx]
            to_seg = self.segments[idx + 1]
            
            from_dir = self._get_segment_look_direction(idx, pos, 1.0)
            to_dir = self._get_segment_look_direction(idx + 1, pos, 0.0)
            
            smooth_t = local_t * local_t * (3 - 2 * local_t)
            result_dir = _slerp(from_dir, to_dir, smooth_t)
            return tuple(pos + result_dir * 10.0)
        
        segment = self.segments[idx]
        
        # Orbit segments always look at POI
        if isinstance(segment, OrbitSegment):
            return segment.poi
        
        # Check previous/next segment types
        prev_is_linear = self._is_linear_segment(idx - 1)
        prev_is_orbit = idx > 0 and isinstance(self.segments[idx - 1], OrbitSegment)
        next_is_linear = self._is_linear_segment(idx + 1)
        
        current_dir = self._get_segment_look_direction(idx, pos, local_t)
        
        # If coming from orbit, blend look direction over the entire segment
        if prev_is_orbit:
            prev_orbit = self.segments[idx - 1]
            # Direction from current pos toward the orbit's POI (what we were looking at)
            prev_dir = np.array(prev_orbit.poi) - pos
            prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-8)
            
            # Blend over the full segment
            blend_t = local_t * local_t * (3 - 2 * local_t)
            result_dir = _slerp(prev_dir, current_dir, blend_t)
            return tuple(pos + result_dir * 10.0)
        
        blend_zone = 0.15 + 0.35 * self.smooth_factor
        
        if local_t < blend_zone and prev_is_linear:
            prev_dir = self._get_segment_look_direction(idx - 1, pos, 1.0)
            zone_t = 0.5 + 0.5 * (local_t / blend_zone)
            zone_t = self._smooth_step(zone_t * 2.0 - 1.0) * 0.5 + 0.5
            result_dir = _slerp(prev_dir, current_dir, zone_t)
            
        elif local_t > (1.0 - blend_zone) and next_is_linear:
            next_dir = self._get_segment_look_direction(idx + 1, pos, 0.0)
            zone_t = 0.5 * (local_t - (1.0 - blend_zone)) / blend_zone
            zone_t = self._smooth_step(zone_t * 2.0) * 0.5
            result_dir = _slerp(current_dir, next_dir, zone_t)
            
        else:
            result_dir = current_dir
        
        return tuple(pos + result_dir * 10.0)
    
    def get_up_vector(self, t: float) -> Tuple[float, float, float]:
        """Get the camera up vector at time t.
        
        For orbit segments, uses the orbit's axis.
        For linear segments and transitions, uses the configured up_axis.
        
        Args:
            t: Time in seconds from path start
            
        Returns:
            Up vector (x, y, z)
        """
        if not self.segments:
            return (0.0, 0.0, 1.0)
        
        pos_type, idx, local_t = self._find_position_at_time(t)
        
        # For transitions, blend up vectors if transitioning to/from orbit
        if pos_type == 'transition':
            from_seg = self.segments[idx]
            to_seg = self.segments[idx + 1]
            
            if isinstance(from_seg, OrbitSegment):
                from_up = np.array(from_seg.get_up_vector())
            else:
                from_up = self._get_world_up()
            
            if isinstance(to_seg, OrbitSegment):
                to_up = np.array(to_seg.get_up_vector())
            else:
                to_up = self._get_world_up()
            
            smooth_t = local_t * local_t * (3 - 2 * local_t)
            result = from_up * (1 - smooth_t) + to_up * smooth_t
            result = result / (np.linalg.norm(result) + 1e-8)
            return tuple(result)
        
        segment = self.segments[idx]
        
        # Orbit segments use their own up vector
        if isinstance(segment, OrbitSegment):
            return segment.get_up_vector()
        
        # Linear segments compute up from forward direction
        pos = np.array(self.get_camera_position(t))
        target = np.array(self.get_camera_target(t))
        
        forward = target - pos
        forward_len = np.linalg.norm(forward)
        if forward_len < 1e-6:
            return tuple(self._get_world_up())
        forward = forward / forward_len
        
        world_up = self._get_world_up()
        
        # Check if looking nearly along the up axis
        dot = abs(np.dot(forward, world_up))
        if dot > 0.99:
            # Use alternate axis
            if self.up_axis == "z":
                world_up = np.array([0.0, 1.0, 0.0])
            elif self.up_axis == "y":
                world_up = np.array([0.0, 0.0, 1.0])
            else:
                world_up = np.array([0.0, 0.0, 1.0])
        
        right = np.cross(forward, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-6:
            return tuple(self._get_world_up())
        right = right / right_len
        
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        return tuple(up)
    
    def _get_world_up(self) -> np.ndarray:
        """Get the world up vector based on configured axis."""
        if self.up_axis == "z":
            return np.array([0.0, 0.0, 1.0])
        elif self.up_axis == "y":
            return np.array([0.0, 1.0, 0.0])
        elif self.up_axis == "x":
            return np.array([1.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 1.0])


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
