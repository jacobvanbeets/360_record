# Camera Control API Request for LichtFeld Studio

## Purpose
Enable plugins to programmatically control camera position for automated video recording (360° orbits, flythroughs, etc.)

## Minimum APIs Needed

### 1. Camera Look-At
```python
lf.set_camera_look_at(eye: tuple, target: tuple, up: tuple = (0,0,1))
```
- `eye`: Camera position (x, y, z)
- `target`: Point to look at (x, y, z)

### 2. Frame Capture
```python
frame = lf.capture_frame() -> numpy.ndarray
```
- Returns current viewport as RGB image array

## Optional (Nice to Have)
- `lf.render_frame()` - Force synchronous render
- `lf.set_render_resolution(w, h)` - Set output size independent of window
- Built-in video recorder API

## Use Case
Plugin moves camera along path → captures each frame → encodes to MP4 using Python (opencv/imageio).
