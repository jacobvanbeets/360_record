# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Linear Path Panel for multi-segment camera path recording."""

import json
import os
import threading
from typing import Optional, List, Dict, Any

import numpy as np
import lichtfeld as lf
from lfs_plugins.types import Panel

from ..core.linear_path import LinearPath, LineSegment, OrbitSegment, compute_linear_camera_position
from ..core.recorder import RecordingSettings, get_default_output_path, record_linear_video, record_linear_frames_to_folder
from ..operators.point_picker import set_point_callback, clear_point_callback, was_point_cancelled
from ..operators.path_preview import start_preview, stop_preview, is_preview_active, get_preview_progress


# Module-level state for draw handler
_linear_draw_handler_registered = False
_linear_path_state = {
    'segments': [],  # List of segment dicts
    'show_preview': True,
    'smooth_factor': 0.5,
    'elevation': 1.5,
    'up_axis': 'z',
    'invert_elevation': False,
    'speed': 1.0,  # Travel speed for linear segments
    'highlighted_segment': -1,  # Index of segment to highlight, -1 = none
    'highlight_time': 0.0,  # Time when highlight started
}

# Pending point pick result
_pending_point = None
_pending_point_target = None  # ("segment", idx, "start"|"end"|"poi")


def _on_point_picked(world_pos):
    """Callback when a point is picked."""
    global _pending_point, _pending_point_target
    _pending_point = world_pos
    lf.ui.request_redraw()


import math as _math
import time as _time

def _linear_path_draw_handler(ctx):
    """Draw handler for visualizing the linear path in the viewport."""
    segments = _linear_path_state['segments']
    if not segments or not _linear_path_state['show_preview']:
        return
    
    # Highlight state
    highlighted_idx = _linear_path_state.get('highlighted_segment', -1)
    highlight_start = _linear_path_state.get('highlight_time', 0.0)
    highlight_duration = 1.5  # seconds
    
    # Calculate highlight pulse (fades out over time)
    highlight_alpha = 0.0
    if highlighted_idx >= 0:
        elapsed = _time.time() - highlight_start
        if elapsed < highlight_duration:
            # Pulsing effect that fades out
            pulse = _math.sin(elapsed * 8.0) * 0.5 + 0.5
            fade = 1.0 - (elapsed / highlight_duration)
            highlight_alpha = pulse * fade
            lf.ui.request_redraw()  # Keep animating
        else:
            # Clear highlight after duration
            _linear_path_state['highlighted_segment'] = -1
    
    # Colors
    line_color = (0.2, 0.8, 1.0, 0.8)
    start_color = (0.2, 1.0, 0.2, 1.0)
    end_color = (1.0, 0.4, 0.2, 1.0)
    poi_color = (1.0, 0.2, 0.8, 1.0)
    orbit_color = (0.8, 0.4, 1.0, 0.8)
    transition_color = (0.5, 0.5, 1.0, 0.5)
    
    # Cache for orbit data to avoid recreating objects
    orbit_cache = {}
    
    def get_orbit_data(idx, seg_data):
        """Get cached orbit start/end points."""
        if idx not in orbit_cache:
            poi = seg_data.get('poi')
            if poi:
                orbit = OrbitSegment(
                    poi=poi,
                    radius=seg_data.get('radius', 5.0),
                    elevation=seg_data.get('elevation', 1.5),
                    orbit_axis=seg_data.get('orbit_axis', 'z'),
                    start_angle=seg_data.get('start_angle', 0.0),
                    arc_degrees=seg_data.get('arc_degrees', 360.0),
                    invert_direction=seg_data.get('invert_direction', False)
                )
                # Pre-calculate arc points
                num_points = max(24, int(abs(orbit.arc_degrees) / 6))
                arc_points = [orbit.get_position_at(j / num_points) for j in range(num_points + 1)]
                orbit_cache[idx] = {
                    'start': orbit.get_start_point(),
                    'end': orbit.get_end_point(),
                    'arc_points': arc_points
                }
            else:
                orbit_cache[idx] = None
        return orbit_cache[idx]
    
    # Draw each segment
    for i, seg_data in enumerate(segments):
        seg_type = seg_data.get('type', 'linear')
        is_highlighted = (i == highlighted_idx and highlight_alpha > 0)
        
        if seg_type == 'orbit':
            poi = seg_data.get('poi')
            if poi:
                orbit_data = get_orbit_data(i, seg_data)
                if not orbit_data:
                    continue
                
                # Draw POI center
                point_size = 22.0 + (10.0 * highlight_alpha) if is_highlighted else 22.0
                color = (1.0, 1.0, 0.0, highlight_alpha + 0.5) if is_highlighted else poi_color
                ctx.draw_point_3d(poi, color, point_size)
                screen_poi = ctx.world_to_screen(poi)
                if screen_poi:
                    ctx.draw_text_2d((screen_poi[0] + 10, screen_poi[1] - 8), f"O{i+1}", poi_color)
                
                # Draw arc from cached points
                arc_color = (1.0, 1.0, 0.0, 0.5 + highlight_alpha * 0.5) if is_highlighted else orbit_color
                arc_width = 2.5 + (3.0 * highlight_alpha) if is_highlighted else 2.5
                arc_points = orbit_data['arc_points']
                for j in range(len(arc_points) - 1):
                    ctx.draw_line_3d(arc_points[j], arc_points[j + 1], arc_color, arc_width)
                
                # Draw start/end points
                ctx.draw_point_3d(orbit_data['start'], start_color, 16.0)
                ctx.draw_point_3d(orbit_data['end'], end_color, 16.0)
                
                # Draw radius line
                ctx.draw_line_3d(poi, orbit_data['start'], (poi_color[0], poi_color[1], poi_color[2], 0.3), 1.0)
        else:
            start = seg_data.get('start')
            end = seg_data.get('end')
            
            if start:
                point_size = 18.0 + (10.0 * highlight_alpha) if is_highlighted else 18.0
                color = (1.0, 1.0, 0.0, highlight_alpha + 0.5) if is_highlighted else start_color
                ctx.draw_point_3d(start, color, point_size)
                screen_start = ctx.world_to_screen(start)
                if screen_start:
                    ctx.draw_text_2d((screen_start[0] + 10, screen_start[1] - 8), f"S{i+1}", start_color)
            
            if end:
                point_size = 18.0 + (10.0 * highlight_alpha) if is_highlighted else 18.0
                color = (1.0, 1.0, 0.0, highlight_alpha + 0.5) if is_highlighted else end_color
                ctx.draw_point_3d(end, color, point_size)
                screen_end = ctx.world_to_screen(end)
                if screen_end:
                    ctx.draw_text_2d((screen_end[0] + 10, screen_end[1] - 8), f"E{i+1}", end_color)
            
            if start and end:
                seg_color = (1.0, 1.0, 0.0, 0.5 + highlight_alpha * 0.5) if is_highlighted else line_color
                seg_width = 2.0 + (3.0 * highlight_alpha) if is_highlighted else 2.0
                ctx.draw_line_3d(start, end, seg_color, seg_width)
                
                # Draw midpoint indicator
                mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2)
                ctx.draw_point_3d(mid, line_color, 8.0)
            
            # Draw POI if set
            poi = seg_data.get('poi')
            if seg_data.get('look_mode') == "poi" and poi:
                ctx.draw_point_3d(poi, poi_color, 20.0)
                screen_poi = ctx.world_to_screen(poi)
                if screen_poi:
                    ctx.draw_text_2d((screen_poi[0] + 10, screen_poi[1] - 8), f"POI{i+1}", poi_color)
                if start and end:
                    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2)
                    ctx.draw_line_3d(mid, poi, (poi_color[0], poi_color[1], poi_color[2], 0.4), 1.0)
            
            # Note: Angled look direction is drawn from the smooth camera path below
        
        # Draw transition to next segment
        if i < len(segments) - 1:
            if seg_type == 'orbit':
                orbit_data = get_orbit_data(i, seg_data)
                curr_end = orbit_data['end'] if orbit_data else None
            else:
                curr_end = seg_data.get('end')
            
            next_seg = segments[i + 1]
            if next_seg.get('type') == 'orbit':
                next_orbit_data = get_orbit_data(i + 1, next_seg)
                next_start = next_orbit_data['start'] if next_orbit_data else None
            else:
                next_start = next_seg.get('start')
            
            if curr_end and next_start:
                ctx.draw_line_3d(curr_end, next_start, transition_color, 1.5)
    
    # Draw smoothed camera path overlay
    smooth_color = (1.0, 0.8, 0.2, 0.6)
    elevation = _linear_path_state.get('elevation', 0.0)
    up_axis = _linear_path_state.get('up_axis', 'z')
    invert_elevation = _linear_path_state.get('invert_elevation', False)
    smooth_factor = _linear_path_state.get('smooth_factor', 0.5)
    speed = _linear_path_state.get('speed', 1.0)
    
    # Build path for smooth visualization
    path = LinearPath(
        speed=speed,
        smooth_factor=smooth_factor,
        elevation=elevation,
        up_axis=up_axis,
        invert_elevation=invert_elevation
    )
    
    for seg_data in segments:
        seg_type = seg_data.get('type', 'linear')
        if seg_type == 'orbit' and seg_data.get('poi'):
            path.segments.append(OrbitSegment(
                poi=seg_data['poi'],
                radius=seg_data.get('radius', 5.0),
                elevation=seg_data.get('elevation', 1.5),
                orbit_axis=seg_data.get('orbit_axis', 'z'),
                start_angle=seg_data.get('start_angle', 0.0),
                arc_degrees=seg_data.get('arc_degrees', 360.0),
                duration=seg_data.get('duration', 30.0),
                invert_direction=seg_data.get('invert_direction', False)
            ))
        elif seg_data.get('start') and seg_data.get('end'):
            path.segments.append(LineSegment(
                start_point=seg_data['start'],
                end_point=seg_data['end'],
                look_mode=seg_data.get('look_mode', 'forward'),
                poi=seg_data.get('poi'),
                look_angle_h=seg_data.get('look_angle_h', 0.0),
                look_angle_v=seg_data.get('look_angle_v', 0.0)
            ))
    
    if path.segments:
        total_dur = path.get_total_duration()
        if total_dur > 0:
            # Reduced samples for performance (was 100+, now ~50)
            num_samples = min(50, max(20, int(total_dur * 10)))
            prev_pos = None
            for i in range(num_samples + 1):
                t = (i / num_samples) * total_dur
                pos = path.get_camera_position(t)
                if prev_pos is not None:
                    ctx.draw_line_3d(prev_pos, pos, smooth_color, 1.5)
                prev_pos = pos
            
            # Draw angled look direction indicators from actual camera path
            for seg_idx, seg_data in enumerate(segments):
                if seg_data.get('type', 'linear') == 'linear' and seg_data.get('look_mode') == 'angled':
                    angle_h = seg_data.get('look_angle_h', 0.0)
                    angle_v = seg_data.get('look_angle_v', 0.0)
                    start = seg_data.get('start')
                    end = seg_data.get('end')
                    
                    if start and end:
                        # Get segment length for line scaling
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]
                        dz = end[2] - start[2]
                        seg_length = _math.sqrt(dx*dx + dy*dy + dz*dz)
                        
                        # Sample a point in the middle of this segment's time
                        # Find approximate time for this segment
                        seg_start_t = 0.0
                        for j in range(seg_idx):
                            s = segments[j]
                            if s.get('type') == 'orbit' and s.get('poi'):
                                seg_start_t += s.get('duration', 30.0)
                            elif s.get('start') and s.get('end'):
                                sx, sy, sz = s['start']
                                ex, ey, ez = s['end']
                                seg_start_t += _math.sqrt((ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2) / speed
                        
                        seg_dur = seg_length / speed
                        mid_t = seg_start_t + seg_dur * 0.5
                        
                        if mid_t <= total_dur:
                            # Get actual camera position on smooth path
                            cam_pos = path.get_camera_position(mid_t)
                            
                            # Compute forward direction from segment
                            forward = (dx/seg_length, dy/seg_length, dz/seg_length) if seg_length > 1e-6 else (1,0,0)
                            
                            # Get up vector
                            if up_axis == 'z':
                                world_up = (0.0, 0.0, 1.0)
                            elif up_axis == 'y':
                                world_up = (0.0, 1.0, 0.0)
                            else:
                                world_up = (1.0, 0.0, 0.0)
                            
                            # Compute right vector (cross product)
                            right = (
                                forward[1] * world_up[2] - forward[2] * world_up[1],
                                forward[2] * world_up[0] - forward[0] * world_up[2],
                                forward[0] * world_up[1] - forward[1] * world_up[0]
                            )
                            right_len = _math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
                            if right_len > 1e-6:
                                right = (right[0]/right_len, right[1]/right_len, right[2]/right_len)
                                
                                # Compute proper up
                                up_vec = (
                                    right[1] * forward[2] - right[2] * forward[1],
                                    right[2] * forward[0] - right[0] * forward[2],
                                    right[0] * forward[1] - right[1] * forward[0]
                                )
                                
                                # Rotate forward by angles
                                angle_h_rad = _math.radians(angle_h)
                                angle_v_rad = _math.radians(angle_v)
                                
                                # Horizontal rotation around up axis
                                cos_h = _math.cos(angle_h_rad)
                                sin_h = _math.sin(angle_h_rad)
                                look_dir = (
                                    forward[0] * cos_h + right[0] * sin_h,
                                    forward[1] * cos_h + right[1] * sin_h,
                                    forward[2] * cos_h + right[2] * sin_h
                                )
                                
                                # Also rotate right axis by horizontal angle
                                rotated_right = (
                                    right[0] * cos_h + forward[0] * (-sin_h),
                                    right[1] * cos_h + forward[1] * (-sin_h),
                                    right[2] * cos_h + forward[2] * (-sin_h)
                                )
                                
                                # Vertical rotation around the rotated right axis
                                # Using Rodrigues' formula: v' = v*cos + (axis x v)*sin + axis*(axis.v)*(1-cos)
                                cos_v = _math.cos(angle_v_rad)
                                sin_v = _math.sin(angle_v_rad)
                                # Cross product: rotated_right x look_dir
                                cross = (
                                    rotated_right[1] * look_dir[2] - rotated_right[2] * look_dir[1],
                                    rotated_right[2] * look_dir[0] - rotated_right[0] * look_dir[2],
                                    rotated_right[0] * look_dir[1] - rotated_right[1] * look_dir[0]
                                )
                                # Dot product: rotated_right . look_dir
                                dot = rotated_right[0]*look_dir[0] + rotated_right[1]*look_dir[1] + rotated_right[2]*look_dir[2]
                                look_dir = (
                                    look_dir[0] * cos_v + cross[0] * sin_v + rotated_right[0] * dot * (1 - cos_v),
                                    look_dir[1] * cos_v + cross[1] * sin_v + rotated_right[1] * dot * (1 - cos_v),
                                    look_dir[2] * cos_v + cross[2] * sin_v + rotated_right[2] * dot * (1 - cos_v)
                                )
                                
                                # Draw look direction line from camera position
                                line_length = max(seg_length * 0.5, 3.0)
                                look_end = (
                                    cam_pos[0] + look_dir[0] * line_length,
                                    cam_pos[1] + look_dir[1] * line_length,
                                    cam_pos[2] + look_dir[2] * line_length
                                )
                                
                                # Yellow line for look direction
                                look_color = (1.0, 0.9, 0.2, 0.9)
                                ctx.draw_line_3d(cam_pos, look_end, look_color, 2.5)
                                ctx.draw_point_3d(look_end, look_color, 10.0)


def _ensure_linear_draw_handler():
    """Ensure the draw handler is registered."""
    global _linear_draw_handler_registered
    if not _linear_draw_handler_registered:
        try:
            lf.remove_draw_handler("linear_path_preview")
        except:
            pass
        lf.add_draw_handler("linear_path_preview", _linear_path_draw_handler, "POST_VIEW")
        _linear_draw_handler_registered = True


class LinearPathPanel(Panel):
    """Panel for configuring and recording camera path videos (linear + orbit segments)."""
    
    label = "Camera Path"
    space = "MAIN_PANEL_TAB"
    order = 30
    
    RESOLUTION_ITEMS = [
        ((1920, 1080), "1080p (1920x1080)"),
        ((2560, 1440), "1440p (2560x1440)"),
        ((3840, 2160), "4K (3840x2160)"),
        ((1280, 720), "720p (1280x720)"),
        ((1080, 1080), "Square (1080x1080)"),
    ]
    
    LOOK_MODE_ITEMS = [
        ("forward", "Look Forward"),
        ("angled", "Look Angled"),
        ("poi", "Look at POI"),
    ]
    
    UP_AXIS_ITEMS = [
        ("z", "Z-Axis (Up)"),
        ("y", "Y-Axis (Up)"),
        ("x", "X-Axis (Up)"),
    ]
    
    SEGMENT_TYPE_ITEMS = [
        ("linear", "Linear"),
        ("orbit", "Orbit"),
    ]
    
    ORBIT_AXIS_ITEMS = [
        ("z", "Z-Axis"),
        ("y", "Y-Axis"),
        ("x", "X-Axis"),
    ]
    
    # Walking speed in meters/second (average ~1.4 m/s)
    WALKING_SPEED = 1.4
    
    def __init__(self):
        # Segments list - each segment is a dict with start, end, look_mode, poi
        self._segments: List[Dict[str, Any]] = []
        self._expanded_segments: Dict[int, bool] = {}  # Track which segments are expanded
        
        # Picking state
        self._picking = False
        self._pick_target = None  # (segment_idx, "start"|"end"|"poi")
        
        # Path settings
        self._speed = 1.0  # units per second
        self._smooth_factor = 0.5  # 0-1
        self._elevation = 1.5  # offset above path
        self._up_axis_idx = 0  # 0=Z, 1=Y, 2=X
        self._invert_elevation = False
        
        # Recording settings
        self._resolution_idx = 0
        self._fps = 30.0
        self._fov = 60.0
        self._output_path = ""
        self._export_frames = False
        self._use_hardware_encoding = True
        
        # Preview
        self._show_preview = True
        self._preview_speed = 1.0  # Preview speed multiplier
        
        # Track file
        self._track_path = ""
        
        # State
        self._status_msg = ""
        self._status_is_error = False
        self._recording = False
        self._record_progress = 0.0
        self._record_thread: Optional[threading.Thread] = None
        self._record_result: Optional[tuple] = None  # (success, message)
    
    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()
    
    def _process_pending_point(self) -> bool:
        """Process any pending point pick."""
        global _pending_point, _pending_point_target
        if _pending_point is None:
            return False
        
        if self._pick_target:
            seg_idx, point_type = self._pick_target
            
            if seg_idx < len(self._segments):
                current_seg = self._segments[seg_idx]
                
                if point_type == "start":
                    current_seg['start'] = _pending_point
                    # Update previous segment's end point if it's a linear segment
                    if seg_idx > 0:
                        prev_seg = self._segments[seg_idx - 1]
                        if prev_seg.get('type', 'linear') == 'linear':
                            prev_seg['end'] = _pending_point
                            
                elif point_type == "end":
                    current_seg['end'] = _pending_point
                    # Update next segment's start point if it's a linear segment
                    if seg_idx < len(self._segments) - 1:
                        next_seg = self._segments[seg_idx + 1]
                        if next_seg.get('type', 'linear') == 'linear':
                            next_seg['start'] = _pending_point
                            
                elif point_type == "poi":
                    current_seg['poi'] = _pending_point
                
                self._status_msg = f"Point set: ({_pending_point[0]:.2f}, {_pending_point[1]:.2f}, {_pending_point[2]:.2f})"
                self._status_is_error = False
        
        _pending_point = None
        self._picking = False
        self._pick_target = None
        
        self._update_draw_state()
        lf.ui.request_redraw()
        return True
    
    def _start_picking(self, segment_idx: int, point_type: str):
        """Start picking mode for a point."""
        global _pending_point
        _pending_point = None
        
        self._picking = True
        self._pick_target = (segment_idx, point_type)
        
        label_map = {"start": "start point", "end": "end point", "poi": "POI"}
        self._status_msg = f"Click to set {label_map.get(point_type, point_type)}..."
        self._status_is_error = False
        
        set_point_callback(_on_point_picked, point_type)
        op_id = "lfs_plugins.360_record.operators.point_picker.LINEARPATH_OT_pick_point"
        lf.ui.ops.invoke(op_id)
        lf.ui.request_redraw()
    
    def _cancel_picking(self):
        """Cancel picking mode."""
        self._picking = False
        self._pick_target = None
        clear_point_callback()
        lf.ui.ops.cancel_modal()
        self._status_msg = "Picking cancelled"
        self._status_is_error = False
        lf.ui.request_redraw()
    
    def _add_segment(self, segment_type: str = "linear"):
        """Add a new segment.
        
        Args:
            segment_type: "linear" or "orbit"
        """
        if segment_type == "orbit":
            new_segment = {
                'type': 'orbit',
                'poi': None,
                'radius': 5.0,
                'elevation': 1.5,
                'orbit_axis': 'z',
                'start_angle': 0.0,
                'arc_degrees': 360.0,
                'duration': 30.0,
                'invert_direction': False,
            }
        else:
            new_segment = {
                'type': 'linear',
                'start': None,
                'end': None,
                'look_mode': 'forward',
                'poi': None,
            }
            # If we have segments, default start to previous end
            if self._segments:
                prev_seg = self._segments[-1]
                if prev_seg.get('type') == 'orbit' and prev_seg.get('poi'):
                    # Connect to orbit's exit point
                    orbit = OrbitSegment(
                        poi=prev_seg['poi'],
                        radius=prev_seg.get('radius', 5.0),
                        elevation=prev_seg.get('elevation', 1.5),
                        orbit_axis=prev_seg.get('orbit_axis', 'z'),
                        start_angle=prev_seg.get('start_angle', 0.0),
                        arc_degrees=prev_seg.get('arc_degrees', 360.0),
                        invert_direction=prev_seg.get('invert_direction', False)
                    )
                    new_segment['start'] = orbit.get_end_point()
                elif prev_seg.get('end'):
                    new_segment['start'] = prev_seg['end']
        
        self._segments.append(new_segment)
        self._expanded_segments[len(self._segments) - 1] = True  # Expand new segment
        self._update_draw_state()
        lf.ui.request_redraw()
    
    def _remove_segment(self, idx: int):
        """Remove a segment and reconnect remaining segments if needed."""
        if not (0 <= idx < len(self._segments)):
            return
        
        # If removing a middle segment, reconnect the next segment to the previous
        if idx > 0 and idx < len(self._segments) - 1:
            prev_seg = self._segments[idx - 1]
            next_seg = self._segments[idx + 1]
            
            # Get the end point of the previous segment
            if prev_seg.get('type') == 'orbit' and prev_seg.get('poi'):
                # Get orbit's end point
                orbit = OrbitSegment(
                    poi=prev_seg['poi'],
                    radius=prev_seg.get('radius', 5.0),
                    elevation=prev_seg.get('elevation', 1.5),
                    orbit_axis=prev_seg.get('orbit_axis', 'z'),
                    start_angle=prev_seg.get('start_angle', 0.0),
                    arc_degrees=prev_seg.get('arc_degrees', 360.0),
                    invert_direction=prev_seg.get('invert_direction', False)
                )
                connect_point = orbit.get_end_point()
            else:
                connect_point = prev_seg.get('end')
            
            # Connect the next segment's start to the previous segment's end
            if connect_point and next_seg.get('type') == 'linear':
                next_seg['start'] = connect_point
        
        # Remove the segment
        del self._segments[idx]
        
        # Update expanded state indices
        new_expanded = {}
        for old_idx, expanded in self._expanded_segments.items():
            if old_idx < idx:
                new_expanded[old_idx] = expanded
            elif old_idx > idx:
                new_expanded[old_idx - 1] = expanded
        self._expanded_segments = new_expanded
        self._update_draw_state()
        lf.ui.request_redraw()
    
    def _move_segment_up(self, idx: int):
        """Move a segment up in the list."""
        if idx > 0:
            self._segments[idx], self._segments[idx - 1] = self._segments[idx - 1], self._segments[idx]
            # Swap expanded states
            exp_curr = self._expanded_segments.get(idx, False)
            exp_prev = self._expanded_segments.get(idx - 1, False)
            self._expanded_segments[idx] = exp_prev
            self._expanded_segments[idx - 1] = exp_curr
            self._update_draw_state()
            lf.ui.request_redraw()
    
    def _move_segment_down(self, idx: int):
        """Move a segment down in the list."""
        if idx < len(self._segments) - 1:
            self._segments[idx], self._segments[idx + 1] = self._segments[idx + 1], self._segments[idx]
            # Swap expanded states
            exp_curr = self._expanded_segments.get(idx, False)
            exp_next = self._expanded_segments.get(idx + 1, False)
            self._expanded_segments[idx] = exp_next
            self._expanded_segments[idx + 1] = exp_curr
            self._update_draw_state()
            lf.ui.request_redraw()
    
    def _update_draw_state(self):
        """Update the draw handler state."""
        _linear_path_state['segments'] = self._segments
        _linear_path_state['show_preview'] = self._show_preview
        _linear_path_state['smooth_factor'] = self._smooth_factor
        _linear_path_state['elevation'] = self._elevation
        _linear_path_state['up_axis'] = self.UP_AXIS_ITEMS[self._up_axis_idx][0]
        _linear_path_state['invert_elevation'] = self._invert_elevation
        _linear_path_state['speed'] = self._speed
    
    def _highlight_segment(self, idx: int):
        """Highlight a segment in the 3D view with a pulsing effect."""
        import time
        _linear_path_state['highlighted_segment'] = idx
        _linear_path_state['highlight_time'] = time.time()
        lf.ui.request_redraw()
    
    def _browse_save_file(self, title: str, default_path: str, filetypes: list) -> Optional[str]:
        """Open a save file dialog. Cross-platform (Windows/Linux).
        
        Args:
            title: Dialog title
            default_path: Default file path
            filetypes: List of (description, pattern) tuples, e.g. [("JSON", "*.json")]
        
        Returns:
            Selected path or None if cancelled
        """
        import sys
        import subprocess
        
        try:
            if sys.platform == 'win32':
                # Windows: use PowerShell for native dialog
                initial_dir = os.path.dirname(default_path) if default_path else os.path.expanduser("~")
                initial_file = os.path.basename(default_path) if default_path else ""
                
                # Build filter string for PowerShell
                filter_parts = []
                for desc, pattern in filetypes:
                    filter_parts.append(f"{desc} ({pattern})|{pattern}")
                filter_str = "|".join(filter_parts)
                
                ps_script = f'''
                Add-Type -AssemblyName System.Windows.Forms
                $dialog = New-Object System.Windows.Forms.SaveFileDialog
                $dialog.Title = "{title}"
                $dialog.InitialDirectory = "{initial_dir.replace(chr(92), chr(92)+chr(92))}"
                $dialog.FileName = "{initial_file}"
                $dialog.Filter = "{filter_str}"
                if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
                    Write-Output $dialog.FileName
                }}
                '''
                
                result = subprocess.run(
                    ['powershell', '-NoProfile', '-Command', ps_script],
                    capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
                )
                path = result.stdout.strip()
                return path if path else None
            else:
                # Linux: try zenity or kdialog
                initial_dir = os.path.dirname(default_path) if default_path else os.path.expanduser("~")
                initial_file = os.path.basename(default_path) if default_path else "untitled"
                
                # Try zenity first
                try:
                    result = subprocess.run(
                        ['zenity', '--file-selection', '--save', '--confirm-overwrite',
                         '--title', title, '--filename', os.path.join(initial_dir, initial_file)],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                    return None
                except FileNotFoundError:
                    pass
                
                # Try kdialog
                try:
                    result = subprocess.run(
                        ['kdialog', '--getsavefilename', os.path.join(initial_dir, initial_file), filetypes[0][1] if filetypes else '*'],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                    return None
                except FileNotFoundError:
                    self._status_msg = "No file dialog available (install zenity or kdialog)"
                    self._status_is_error = True
                    return None
                    
        except Exception as e:
            self._status_msg = f"File dialog error: {str(e)}"
            self._status_is_error = True
            return None
    
    def _browse_open_file(self, title: str, default_path: str, filetypes: list) -> Optional[str]:
        """Open a file open dialog. Cross-platform (Windows/Linux).
        
        Args:
            title: Dialog title
            default_path: Default file path
            filetypes: List of (description, pattern) tuples, e.g. [("JSON", "*.json")]
        
        Returns:
            Selected path or None if cancelled
        """
        import sys
        import subprocess
        
        try:
            if sys.platform == 'win32':
                # Windows: use PowerShell for native dialog
                initial_dir = os.path.dirname(default_path) if default_path else os.path.expanduser("~")
                
                # Build filter string for PowerShell
                filter_parts = []
                for desc, pattern in filetypes:
                    filter_parts.append(f"{desc} ({pattern})|{pattern}")
                filter_str = "|".join(filter_parts)
                
                ps_script = f'''
                Add-Type -AssemblyName System.Windows.Forms
                $dialog = New-Object System.Windows.Forms.OpenFileDialog
                $dialog.Title = "{title}"
                $dialog.InitialDirectory = "{initial_dir.replace(chr(92), chr(92)+chr(92))}"
                $dialog.Filter = "{filter_str}"
                if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
                    Write-Output $dialog.FileName
                }}
                '''
                
                result = subprocess.run(
                    ['powershell', '-NoProfile', '-Command', ps_script],
                    capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
                )
                path = result.stdout.strip()
                return path if path else None
            else:
                # Linux: try zenity or kdialog
                initial_dir = os.path.dirname(default_path) if default_path else os.path.expanduser("~")
                
                # Build file filter for zenity
                file_filter = []
                for desc, pattern in filetypes:
                    file_filter.extend(['--file-filter', f"{desc} | {pattern}"])
                
                # Try zenity first
                try:
                    cmd = ['zenity', '--file-selection', '--title', title, '--filename', initial_dir + '/'] + file_filter
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        return result.stdout.strip()
                    return None
                except FileNotFoundError:
                    pass
                
                # Try kdialog
                try:
                    result = subprocess.run(
                        ['kdialog', '--getopenfilename', initial_dir, filetypes[0][1] if filetypes else '*'],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                    return None
                except FileNotFoundError:
                    self._status_msg = "No file dialog available (install zenity or kdialog)"
                    self._status_is_error = True
                    return None
                    
        except Exception as e:
            self._status_msg = f"File dialog error: {str(e)}"
            self._status_is_error = True
            return None
    
    def _get_default_track_path(self) -> str:
        """Get default track save path."""
        try:
            scene_path = lf.get_scene_path()
            if scene_path:
                base = os.path.splitext(scene_path)[0]
                return base + "_track.json"
        except:
            pass
        return os.path.join(os.path.expanduser("~"), "camera_track.json")
    
    def _save_track(self, path: str) -> tuple:
        """Save current track to JSON file.
        
        Returns:
            (success, message) tuple
        """
        if not self._segments:
            return False, "No segments to save"
        
        try:
            data = {
                'version': 1,
                'segments': self._segments,
                'settings': {
                    'speed': self._speed,
                    'smooth_factor': self._smooth_factor,
                    'elevation': self._elevation,
                    'up_axis_idx': self._up_axis_idx,
                    'invert_elevation': self._invert_elevation,
                    'resolution_idx': self._resolution_idx,
                    'fps': self._fps,
                    'fov': self._fov,
                    'preview_speed': self._preview_speed,
                }
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True, f"Track saved to {os.path.basename(path)}"
        except Exception as e:
            return False, f"Save failed: {str(e)}"
    
    def _load_track(self, path: str) -> tuple:
        """Load track from JSON file.
        
        Returns:
            (success, message) tuple
        """
        if not os.path.exists(path):
            return False, "File not found"
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Load segments
            self._segments = data.get('segments', [])
            self._expanded_segments = {}  # Reset expanded state
            
            # Load settings
            settings = data.get('settings', {})
            self._speed = settings.get('speed', 1.0)
            self._smooth_factor = settings.get('smooth_factor', 0.5)
            self._elevation = settings.get('elevation', 1.5)
            self._up_axis_idx = settings.get('up_axis_idx', 0)
            self._invert_elevation = settings.get('invert_elevation', False)
            self._resolution_idx = settings.get('resolution_idx', 0)
            self._fps = settings.get('fps', 30.0)
            self._fov = settings.get('fov', 60.0)
            self._preview_speed = settings.get('preview_speed', 1.0)
            
            self._update_draw_state()
            lf.ui.request_redraw()
            
            return True, f"Loaded {len(self._segments)} segments"
        except json.JSONDecodeError:
            return False, "Invalid JSON file"
        except Exception as e:
            return False, f"Load failed: {str(e)}"
    
    def _get_path(self) -> Optional[LinearPath]:
        """Get the current linear path configuration."""
        if not self._segments:
            return None
        
        up_axis = self.UP_AXIS_ITEMS[self._up_axis_idx][0]
        
        path = LinearPath(
            speed=self._speed,
            smooth_factor=self._smooth_factor,
            elevation=self._elevation,
            up_axis=up_axis,
            invert_elevation=self._invert_elevation
        )
        
        for seg_data in self._segments:
            seg_type = seg_data.get('type', 'linear')
            
            if seg_type == 'orbit':
                # Orbit segment - needs POI
                if seg_data.get('poi'):
                    segment = OrbitSegment(
                        poi=seg_data['poi'],
                        radius=seg_data.get('radius', 5.0),
                        elevation=seg_data.get('elevation', 1.5),
                        orbit_axis=seg_data.get('orbit_axis', 'z'),
                        start_angle=seg_data.get('start_angle', 0.0),
                        arc_degrees=seg_data.get('arc_degrees', 360.0),
                        duration=seg_data.get('duration', 30.0),
                        invert_direction=seg_data.get('invert_direction', False)
                    )
                    path.segments.append(segment)
            else:
                # Linear segment - needs start and end
                if seg_data.get('start') and seg_data.get('end'):
                    segment = LineSegment(
                        start_point=seg_data['start'],
                        end_point=seg_data['end'],
                        look_mode=seg_data.get('look_mode', 'forward'),
                        poi=seg_data.get('poi'),
                        look_angle_h=seg_data.get('look_angle_h', 0.0),
                        look_angle_v=seg_data.get('look_angle_v', 0.0)
                    )
                    path.segments.append(segment)
        
        return path if path.segments else None
    
    def _get_recording_settings(self) -> RecordingSettings:
        """Get the current recording settings."""
        resolution = self.RESOLUTION_ITEMS[self._resolution_idx][0]
        
        if not self._output_path:
            base_path = get_default_output_path()
            self._output_path = base_path.replace("_360.mp4", "_linear.mp4")
        
        return RecordingSettings(
            output_path=self._output_path,
            resolution=resolution,
            fps=self._fps,
            fov=self._fov,
            use_hardware_encoding=self._use_hardware_encoding
        )
    
    def _on_record_progress(self, progress: float, message: str):
        """Progress callback for recording."""
        self._record_progress = progress
        self._status_msg = message
        self._status_is_error = False
        lf.ui.request_redraw()
    
    def _record_thread_func(self, path: LinearPath, settings: RecordingSettings, export_frames: bool):
        """Recording function that runs in a background thread."""
        try:
            if export_frames:
                folder = os.path.splitext(settings.output_path)[0] + "_frames"
                success, msg = record_linear_frames_to_folder(path, settings, folder, self._on_record_progress)
            else:
                success, msg = record_linear_video(path, settings, self._on_record_progress)
            self._record_result = (success, msg)
        except Exception as e:
            self._record_result = (False, f"Recording failed: {str(e)}")
        lf.ui.request_redraw()
    
    def _start_recording(self):
        """Start video recording in a background thread."""
        path = self._get_path()
        if path is None:
            self._status_msg = "Add at least one complete segment first"
            self._status_is_error = True
            return
        
        settings = self._get_recording_settings()
        self._recording = True
        self._record_progress = 0.0
        self._record_result = None
        self._status_msg = "Starting recording..."
        self._status_is_error = False
        
        # Start recording in a background thread
        self._record_thread = threading.Thread(
            target=self._record_thread_func,
            args=(path, settings, self._export_frames),
            daemon=True
        )
        self._record_thread.start()
    
    def _check_recording_complete(self):
        """Check if recording thread has completed."""
        if self._recording and self._record_thread is not None:
            if not self._record_thread.is_alive():
                # Thread completed
                self._recording = False
                self._record_thread = None
                if self._record_result:
                    success, msg = self._record_result
                    self._status_msg = msg
                    self._status_is_error = not success
                    self._record_result = None
    
    def _format_point(self, point: Optional[tuple]) -> str:
        """Format a point for display."""
        if point is None:
            return "Not set"
        return f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
    
    def draw(self, layout):
        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()
        
        # Check if recording thread completed
        self._check_recording_complete()
        
        # Process pending point pick
        self._process_pending_point()
        
        # Check if picking was cancelled
        if was_point_cancelled() and self._picking:
            self._picking = False
            self._pick_target = None
            self._status_msg = "Picking cancelled"
            self._status_is_error = False
        
        # Ensure draw handler
        _ensure_linear_draw_handler()
        
        # Update draw state
        self._update_draw_state()
        
        # === Segments Section ===
        if layout.collapsing_header("Path Segments", default_open=True):
            if not self._segments:
                layout.text_colored("No segments. Add a Linear or Orbit segment to start.", theme.palette.text_dim)
            
            # Draw each segment
            for i, seg_data in enumerate(self._segments):
                is_expanded = self._expanded_segments.get(i, True)
                seg_type = seg_data.get('type', 'linear')
                
                # Determine if segment is complete
                if seg_type == 'orbit':
                    is_complete = seg_data.get('poi') is not None
                    type_label = "Orbit"
                else:
                    is_complete = seg_data.get('start') is not None and seg_data.get('end') is not None
                    type_label = "Linear"
                
                # Segment header
                header_text = f"{i + 1}. {type_label}"
                if not is_complete:
                    header_text += " (incomplete)"
                
                # Expand/collapse button
                expand_label = "v" if is_expanded else ">"
                if layout.button(f"{expand_label}##{i}_expand", (20 * scale, 0)):
                    self._expanded_segments[i] = not is_expanded
                layout.same_line()
                
                # Clickable header label - highlights segment in 3D view
                if layout.button(f"{header_text}##{i}_header", (0, 0)):
                    self._highlight_segment(i)
                layout.same_line()
                
                # Delete button
                if layout.button(f"Del##{i}_del", (32 * scale, 0)):
                    self._remove_segment(i)
                    continue  # Skip rest of this segment
                
                # Segment details (if expanded)
                if is_expanded:
                    layout.indent(15 * scale)
                    
                    if seg_type == 'orbit':
                        # === ORBIT SEGMENT UI ===
                        
                        # POI (required)
                        layout.label("Center POI:")
                        layout.same_line()
                        layout.text_colored(self._format_point(seg_data.get('poi')),
                                           (1.0, 0.2, 0.8, 1.0) if seg_data.get('poi') else theme.palette.text_dim)
                        layout.same_line()
                        
                        picking_poi = self._picking and self._pick_target == (i, "poi")
                        if picking_poi:
                            if layout.button_styled(f"[Cancel]##{i}_poi_cancel", "error", (70 * scale, 0)):
                                self._cancel_picking()
                        else:
                            if layout.button(f"Pick##{i}_poi", (50 * scale, 0)):
                                self._start_picking(i, "poi")
                        
                        # Orbit Axis
                        layout.label("Orbit Axis:")
                        orbit_axis_labels = [item[1] for item in self.ORBIT_AXIS_ITEMS]
                        current_axis_idx = 0
                        for idx, item in enumerate(self.ORBIT_AXIS_ITEMS):
                            if item[0] == seg_data.get('orbit_axis', 'z'):
                                current_axis_idx = idx
                                break
                        layout.push_item_width(100 * scale)
                        changed, new_axis_idx = layout.combo(f"##orbit_axis_{i}", current_axis_idx, orbit_axis_labels)
                        if changed:
                            seg_data['orbit_axis'] = self.ORBIT_AXIS_ITEMS[new_axis_idx][0]
                            self._update_draw_state()
                        layout.pop_item_width()
                        
                        # Radius
                        layout.label("Radius:")
                        layout.push_item_width(-1)
                        changed, new_radius = layout.drag_float(f"##radius_{i}", seg_data.get('radius', 5.0), 0.1, 0.1, 500.0)
                        if changed:
                            seg_data['radius'] = max(0.1, new_radius)
                            self._update_draw_state()
                        layout.pop_item_width()
                        if layout.is_item_hovered():
                            layout.set_tooltip("Drag or Ctrl+Click to enter value manually")
                        
                        # Elevation (offset along orbit axis)
                        layout.label("Elevation:")
                        layout.push_item_width(-1)
                        changed, new_elev = layout.drag_float(f"##orb_elev_{i}", seg_data.get('elevation', 1.5), 0.1, -500.0, 500.0)
                        if changed:
                            seg_data['elevation'] = new_elev
                            self._update_draw_state()
                        layout.pop_item_width()
                        if layout.is_item_hovered():
                            layout.set_tooltip("Drag or Ctrl+Click to enter value manually")
                        
                        # Start Angle
                        layout.label("Start Angle:")
                        layout.push_item_width(-1)
                        changed, new_start = layout.slider_float(f"##start_angle_{i}", seg_data.get('start_angle', 0.0), 0.0, 360.0)
                        if changed:
                            seg_data['start_angle'] = new_start
                            self._update_draw_state()
                        layout.pop_item_width()
                        
                        # Arc Degrees
                        layout.label("Arc Amount (°):")
                        layout.push_item_width(-1)
                        changed, new_arc = layout.slider_float(f"##arc_deg_{i}", seg_data.get('arc_degrees', 360.0), -720.0, 720.0)
                        if changed:
                            seg_data['arc_degrees'] = new_arc
                            self._update_draw_state()
                        layout.pop_item_width()
                        if layout.is_item_hovered():
                            layout.set_tooltip("Positive = counter-clockwise, Negative = clockwise. 360 = full circle.")
                        
                        # Duration
                        layout.label("Duration (s):")
                        layout.push_item_width(-1)
                        changed, new_dur = layout.slider_float(f"##duration_{i}", seg_data.get('duration', 30.0), 1.0, 120.0)
                        if changed:
                            seg_data['duration'] = new_dur
                            self._update_draw_state()
                        layout.pop_item_width()
                        
                        # Invert direction
                        changed, new_invert = layout.checkbox(f"Invert Elevation##{i}_invert", seg_data.get('invert_direction', False))
                        if changed:
                            seg_data['invert_direction'] = new_invert
                            self._update_draw_state()
                        
                    else:
                        # === LINEAR SEGMENT UI ===
                        
                        # Check if this segment's start is connected to previous segment
                        is_connected = i > 0 and seg_data.get('start') is not None
                        
                        # Start point
                        layout.label("Start:")
                        layout.same_line()
                        
                        if is_connected:
                            # Show as connected (linked to previous segment)
                            layout.text_colored(self._format_point(seg_data.get('start')), (0.5, 0.8, 1.0, 1.0))
                            layout.same_line()
                            layout.text_colored("(linked)", theme.palette.text_dim)
                        else:
                            # First segment or not connected - show picker
                            layout.text_colored(self._format_point(seg_data.get('start')), 
                                               (0.2, 1.0, 0.2, 1.0) if seg_data.get('start') else theme.palette.text_dim)
                            layout.same_line()
                            
                            picking_start = self._picking and self._pick_target == (i, "start")
                            if picking_start:
                                if layout.button_styled(f"[Cancel]##{i}_start_cancel", "error", (70 * scale, 0)):
                                    self._cancel_picking()
                            else:
                                if layout.button(f"Pick##{i}_start", (50 * scale, 0)):
                                    self._start_picking(i, "start")
                        
                        # End point
                        layout.label("End:")
                        layout.same_line()
                        layout.text_colored(self._format_point(seg_data.get('end')),
                                           (1.0, 0.4, 0.2, 1.0) if seg_data.get('end') else theme.palette.text_dim)
                        layout.same_line()
                        
                        picking_end = self._picking and self._pick_target == (i, "end")
                        if picking_end:
                            if layout.button_styled(f"[Cancel]##{i}_end_cancel", "error", (70 * scale, 0)):
                                self._cancel_picking()
                        else:
                            if layout.button(f"Pick##{i}_end", (50 * scale, 0)):
                                self._start_picking(i, "end")
                        
                        # Look mode
                        layout.label("Look Mode:")
                        look_mode_labels = [item[1] for item in self.LOOK_MODE_ITEMS]
                        current_look_idx = 0
                        for idx, item in enumerate(self.LOOK_MODE_ITEMS):
                            if item[0] == seg_data.get('look_mode', 'forward'):
                                current_look_idx = idx
                                break
                        
                        layout.push_item_width(150 * scale)
                        changed, new_look_idx = layout.combo(f"##look_mode_{i}", current_look_idx, look_mode_labels)
                        if changed:
                            seg_data['look_mode'] = self.LOOK_MODE_ITEMS[new_look_idx][0]
                            self._update_draw_state()
                        layout.pop_item_width()
                        
                        # Angle controls (only if look_mode is "angled")
                        if seg_data.get('look_mode') == 'angled':
                            # Horizontal angle (left/right)
                            layout.label("Horizontal (L/R):")
                            layout.push_item_width(-1)
                            changed, new_angle_h = layout.slider_float(
                                f"##look_angle_h_{i}", 
                                seg_data.get('look_angle_h', 0.0), 
                                -180.0, 180.0
                            )
                            if changed:
                                seg_data['look_angle_h'] = new_angle_h
                                self._update_draw_state()
                            layout.pop_item_width()
                            if layout.is_item_hovered():
                                layout.set_tooltip("Negative = look left, Positive = look right")
                            
                            # Vertical angle (up/down)
                            layout.label("Vertical (Up/Dn):")
                            layout.push_item_width(-1)
                            changed, new_angle_v = layout.slider_float(
                                f"##look_angle_v_{i}", 
                                seg_data.get('look_angle_v', 0.0), 
                                -90.0, 90.0
                            )
                            if changed:
                                seg_data['look_angle_v'] = new_angle_v
                                self._update_draw_state()
                            layout.pop_item_width()
                            if layout.is_item_hovered():
                                layout.set_tooltip("Negative = look down, Positive = look up")
                        
                        # POI (only if look_mode is "poi")
                        if seg_data.get('look_mode') == 'poi':
                            layout.label("POI:")
                            layout.same_line()
                            layout.text_colored(self._format_point(seg_data.get('poi')),
                                               (1.0, 0.2, 0.8, 1.0) if seg_data.get('poi') else theme.palette.text_dim)
                            layout.same_line()
                            
                            picking_poi = self._picking and self._pick_target == (i, "poi")
                            if picking_poi:
                                if layout.button_styled(f"[Cancel]##{i}_poi_cancel", "error", (70 * scale, 0)):
                                    self._cancel_picking()
                            else:
                                if layout.button(f"Pick##{i}_poi", (50 * scale, 0)):
                                    self._start_picking(i, "poi")
                    
                    layout.unindent(15 * scale)
                    layout.spacing()
            
            # Add segment buttons
            layout.spacing()
            layout.label("Add Segment:")
            if layout.button("+ Linear##add_linear", (100 * scale, 28 * scale)):
                self._add_segment("linear")
            layout.same_line()
            if layout.button("+ Orbit##add_orbit", (100 * scale, 28 * scale)):
                self._add_segment("orbit")
        
        layout.separator()
        
        # === Path Settings Section ===
        if layout.collapsing_header("Path Settings", default_open=True):
            btn_w = 45 * scale
            
            # Up Axis (camera level)
            layout.label("Camera Up Axis:")
            up_axis_labels = [item[1] for item in self.UP_AXIS_ITEMS]
            changed, self._up_axis_idx = layout.combo("##upaxis", self._up_axis_idx, up_axis_labels)
            if changed:
                self._update_draw_state()
            if layout.is_item_hovered():
                layout.set_tooltip("Sets which axis points up for the camera")
            
            layout.spacing()
            
            # Elevation offset
            axis_name = self.UP_AXIS_ITEMS[self._up_axis_idx][0].upper()
            layout.label(f"Elevation (offset along {axis_name}):")
            
            layout.push_item_width(-80 * scale)
            changed, self._elevation = layout.slider_float("##elev_slider", self._elevation, 0.0, 50.0)
            if changed:
                self._update_draw_state()
            layout.pop_item_width()
            layout.same_line()
            layout.push_item_width(70 * scale)
            changed, self._elevation = layout.input_float("##elev_input", self._elevation, 0.0, 0.0)
            if changed:
                self._elevation = max(0.0, self._elevation)
                self._update_draw_state()
            layout.pop_item_width()
            
            # Elevation adjustment buttons
            if layout.button("-0.5##elevsub05", (btn_w, 0)):
                self._elevation = max(0.0, self._elevation - 0.5)
                self._update_draw_state()
            layout.same_line()
            if layout.button("+0.5##elevadd05", (btn_w, 0)):
                self._elevation += 0.5
                self._update_draw_state()
            layout.same_line()
            if layout.button("-1##elevsub1", (btn_w, 0)):
                self._elevation = max(0.0, self._elevation - 1.0)
                self._update_draw_state()
            layout.same_line()
            if layout.button("+1##elevadd1", (btn_w, 0)):
                self._elevation += 1.0
                self._update_draw_state()
            
            # Invert elevation checkbox
            changed, self._invert_elevation = layout.checkbox("Invert Elevation##invert_elev", self._invert_elevation)
            if changed:
                self._update_draw_state()
            if layout.is_item_hovered():
                layout.set_tooltip("Flip offset direction (e.g., below instead of above)")
            
            layout.spacing()
            
            # Speed (units per second)
            layout.label("Speed (units/second):")
            
            layout.push_item_width(-80 * scale)
            changed, self._speed = layout.slider_float("##speed_slider", self._speed, 0.1, 10.0)
            layout.pop_item_width()
            layout.same_line()
            layout.push_item_width(70 * scale)
            changed, self._speed = layout.input_float("##speed_input", self._speed, 0.0, 0.0)
            if changed:
                self._speed = max(0.01, self._speed)
            layout.pop_item_width()
            
            # Speed adjustment buttons + Walking speed
            if layout.button("-0.1##spdsub01", (btn_w, 0)):
                self._speed = max(0.01, self._speed - 0.1)
            layout.same_line()
            if layout.button("+0.1##spdadd01", (btn_w, 0)):
                self._speed += 0.1
            layout.same_line()
            if layout.button("-1##spdsub1", (btn_w, 0)):
                self._speed = max(0.01, self._speed - 1.0)
            layout.same_line()
            if layout.button("+1##spdadd1", (btn_w, 0)):
                self._speed += 1.0
            
            # Walking speed button
            if layout.button("Walking Speed (1.4 m/s)##walkspeed", (-1, 0)):
                self._speed = self.WALKING_SPEED
            if layout.is_item_hovered():
                layout.set_tooltip("Set to average human walking speed (assumes units are meters)")
            
            # Show estimated duration based on current speed
            path = self._get_path()
            if path:
                total_dur = path.get_total_duration()
                mins = int(total_dur // 60)
                secs = total_dur % 60
                if mins > 0:
                    dur_str = f"{mins}m {secs:.1f}s"
                else:
                    dur_str = f"{secs:.1f}s"
                layout.text_colored(
                    f"Estimated video duration: {dur_str}",
                    (0.4, 1.0, 0.4, 1.0)
                )
            
            layout.spacing()
            
            # Smooth factor
            layout.label("Smoothing:")
            layout.push_item_width(-1)
            changed, self._smooth_factor = layout.slider_float("##smooth", self._smooth_factor, 0.0, 1.0)
            if changed:
                self._update_draw_state()
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip("0 = Sharp corners, 1 = Maximum smoothing at transitions")
        
        layout.separator()
        
        # === Preview Section ===
        if layout.collapsing_header("Preview", default_open=True):
            changed, self._show_preview = layout.checkbox("Show Path Preview##showpreview", self._show_preview)
            if changed:
                self._update_draw_state()
                lf.ui.request_redraw()
            
            # Live preview controls
            path = self._get_path()
            can_preview = path is not None and len(path.segments) > 0 and not self._picking and not self._recording
            
            layout.spacing()
            
            if is_preview_active():
                # Show progress and stop button
                progress = get_preview_progress()
                layout.progress_bar(progress, f"Preview: {progress * 100:.0f}%", -1, 24 * scale)
                
                if layout.button_styled("STOP PREVIEW (Esc)##stoppreview", "error", (-1, 32 * scale)):
                    stop_preview()
                
                layout.text_colored("Press ESC or Right-click to cancel", theme.palette.text_dim)
                lf.ui.request_redraw()  # Keep updating progress
            else:
                # Preview speed
                layout.label("Preview Speed:")
                layout.same_line()
                speed_labels = ["0.5x", "1x", "2x", "4x"]
                speed_values = [0.5, 1.0, 2.0, 4.0]
                current_speed_idx = 1  # Default 1x
                for idx, val in enumerate(speed_values):
                    if abs(self._preview_speed - val) < 0.01:
                        current_speed_idx = idx
                        break
                
                layout.push_item_width(80 * scale)
                changed, new_speed_idx = layout.combo("##previewspeed", current_speed_idx, speed_labels)
                if changed:
                    self._preview_speed = speed_values[new_speed_idx]
                layout.pop_item_width()
                
                if can_preview:
                    if layout.button_styled("PREVIEW PATH##startpreview", "secondary", (-1, 32 * scale)):
                        start_preview(path, self._fov, self._preview_speed)
                else:
                    layout.button("PREVIEW PATH##preview_disabled", (-1, 32 * scale))
                    if not path or len(path.segments) == 0:
                        layout.text_colored("Add complete segments to preview", theme.palette.text_dim)
            
            layout.spacing()
            
            # Show path info
            if path:
                total_dist = path.get_total_distance()
                total_dur = path.get_total_duration()
                layout.text_colored(
                    f"Distance: {total_dist:.2f} units | Duration: {total_dur:.1f}s",
                    theme.palette.text_dim
                )
        
        layout.separator()
        
        # === Recording Settings Section ===
        if layout.collapsing_header("Recording Settings", default_open=True):
            # Resolution
            resolution_labels = [item[1] for item in self.RESOLUTION_ITEMS]
            changed, self._resolution_idx = layout.combo("Resolution##res", self._resolution_idx, resolution_labels)
            
            # FPS
            layout.push_item_width(100 * scale)
            changed, self._fps = layout.input_float("FPS##fps", self._fps, 0.0, 0.0)
            self._fps = max(1.0, min(120.0, self._fps))
            layout.pop_item_width()
            
            # FOV
            layout.push_item_width(100 * scale)
            changed, self._fov = layout.slider_float("FOV##fov", self._fov, 30.0, 120.0)
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip("Field of view in degrees")
            
            layout.spacing()
            
            # Output path
            layout.label("Output Path:")
            if not self._output_path:
                base_path = get_default_output_path()
                self._output_path = base_path.replace("_360.mp4", "_linear.mp4")
            
            layout.push_item_width(-60 * scale)
            changed, self._output_path = layout.input_text("##outpath", self._output_path)
            layout.pop_item_width()
            layout.same_line()
            if layout.button("...##browse", (50 * scale, 0)):
                filetypes = [("MP4 Video", "*.mp4"), ("All Files", "*.*")]
                path = self._browse_save_file("Save Video", self._output_path, filetypes)
                if path:
                    self._output_path = path
            
            # Export as frames
            changed, self._export_frames = layout.checkbox("Export as PNG frames##exportframes", self._export_frames)
            if layout.is_item_hovered():
                layout.set_tooltip("Export individual PNG frames instead of MP4 video")
            
            # Hardware encoding
            changed, self._use_hardware_encoding = layout.checkbox("Hardware Encoding (GPU)##hwenc", self._use_hardware_encoding)
            if layout.is_item_hovered():
                layout.set_tooltip("Use GPU for faster video encoding (NVIDIA/AMD/Intel). Falls back to software if unavailable.")
            
            # Display estimated info
            path = self._get_path()
            if path:
                total_frames = path.get_total_frames(self._fps)
                duration = path.get_total_duration()
                layout.text_colored(
                    f"Duration: {duration:.1f}s | Frames: {total_frames}",
                    theme.palette.text_dim
                )
        
        layout.separator()
        
        # === Save/Load Track Section ===
        if layout.collapsing_header("Save/Load Track", default_open=False):
            # Track file path
            if not self._track_path:
                self._track_path = self._get_default_track_path()
            
            layout.label("Track File:")
            layout.push_item_width(-60 * scale)
            changed, self._track_path = layout.input_text("##trackpath", self._track_path)
            layout.pop_item_width()
            layout.same_line()
            if layout.button("...##browsetrack", (50 * scale, 0)):
                # Show open dialog to browse for existing track files
                filetypes = [("Camera Track", "*.json"), ("All Files", "*.*")]
                path = self._browse_open_file("Open Camera Track", self._track_path, filetypes)
                if path:
                    self._track_path = path
            
            layout.spacing()
            
            # Save/Load buttons side by side
            btn_width = (layout.get_content_region_avail()[0] - 8 * scale) / 2
            
            has_segments = len(self._segments) > 0
            if has_segments:
                if layout.button_styled("Save##save", "secondary", (btn_width, 32 * scale)):
                    # Allow saving to new location via dialog
                    filetypes = [("Camera Track", "*.json"), ("All Files", "*.*")]
                    path = self._browse_save_file("Save Camera Track", self._track_path, filetypes)
                    if path:
                        self._track_path = path
                        success, msg = self._save_track(self._track_path)
                        self._status_msg = msg
                        self._status_is_error = not success
            else:
                layout.button("Save##save_disabled", (btn_width, 32 * scale))
            
            layout.same_line()
            
            if layout.button_styled("Load##load", "secondary", (btn_width, 32 * scale)):
                # Always show dialog to pick file to load
                filetypes = [("Camera Track", "*.json"), ("All Files", "*.*")]
                path = self._browse_open_file("Load Camera Track", self._track_path, filetypes)
                if path:
                    self._track_path = path
                    success, msg = self._load_track(self._track_path)
                    self._status_msg = msg
                    self._status_is_error = not success
            
            if not has_segments:
                layout.text_colored("Add segments to enable Save", theme.palette.text_dim)
        
        layout.separator()
        
        # === Record Button ===
        if self._recording:
            # Show progress bar with status
            # progress_bar(fraction, overlay, width, height)
            layout.progress_bar(self._record_progress, f"{self._record_progress * 100:.0f}%", -1, 28 * scale)
            layout.text_colored(self._status_msg, theme.palette.text_dim)
        else:
            path = self._get_path()
            can_record = path is not None and len(path.segments) > 0 and not self._picking
            
            if not can_record:
                layout.text_colored("Add complete segments to enable recording", theme.palette.text_dim)
            
            button_label = "EXPORT FRAMES" if self._export_frames else "RECORD VIDEO"
            if can_record:
                if layout.button_styled(f"{button_label}##record", "primary", (-1, 40 * scale)):
                    self._start_recording()
            else:
                layout.button(f"{button_label}##record_disabled", (-1, 40 * scale))
        
        # Status message
        if self._status_msg:
            layout.spacing()
            color = (1.0, 0.4, 0.4, 1.0) if self._status_is_error else (0.4, 1.0, 0.4, 1.0)
            layout.text_colored(self._status_msg, color)
