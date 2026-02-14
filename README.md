# 360 Record - LichtFeld Studio Plugin

A plugin for [LichtFeld Studio](https://lichtfeld.io) that enables recording camera path videos around 3D Gaussian Splat scenes.

[![Watch Demo Video](https://img.youtube.com/vi/eFl7kilU9ts/maxresdefault.jpg)](https://www.youtube.com/watch?v=eFl7kilU9ts)

**▶️ Click the image above to watch the demo video**

## Features

Create complex camera paths by combining **Linear** and **Orbit** segments in any order.

### Linear Segments
- Pick **start and end points** on your model
- Per-segment **look mode**:
  - **Look Forward** - Camera looks along travel direction
  - **Look at POI** - Camera looks at a specific point of interest
- **Smooth transitions** with Catmull-Rom spline interpolation
- **Walking Speed** preset (1.4 m/s for realistic walkthroughs)

### Orbit Segments (360° Camera)
- Pick a **Point of Interest (POI)** as the orbit center
- Set **radius** (distance from POI)
- Set **elevation** (offset along orbit axis)
- Choose **orbit axis** (X, Y, or Z)
- Adjustable **start angle** and **arc amount** (partial or full circles)
- Configurable **duration** (seconds for the orbit)

### Path Settings
- **Elevation offset** above the path (for linear segments)
- **Travel speed** (units per second for linear segments)
- **Smoothing factor** for corner transitions
- **Camera up axis** selection (X, Y, or Z)

### Recording Options
- **Resolution presets**: 720p, 1080p, 1440p, 4K, Square
- Adjustable **FPS** (1-120)
- Configurable **Field of View**
- Export as **MP4 video** or **PNG frame sequence**
- **Hardware encoding** (GPU acceleration) - auto-detects NVIDIA NVENC, AMD AMF, Intel QuickSync
- **Live progress bar** during recording

### Viewport Preview
- Real-time visualization of camera paths
- Start/end point markers for each segment
- Orbit arc visualization
- POI markers
- Smoothed path overlay
- Segment transition indicators
- **Click segment headers** to highlight in 3D view

## Installation

1. Copy the repository URL: `https://github.com/jacobvanbeets/360_record`
2. Open **LichtFeld Studio**
3. Go to the **Plugin Manager**
4. Paste the URL and click **Install**

That's it! LFS will handle all dependencies automatically.

## Usage

1. Open the **Camera Path** panel
2. Click **+ Linear** or **+ Orbit** to add a segment
3. Configure the segment:
   - **Linear**: Pick start and end points, choose look mode
   - **Orbit**: Pick POI, set radius, elevation, arc amount, and duration
4. Add more segments as needed (mix linear and orbit freely)
5. Adjust path settings (speed, smoothing, elevation)
6. Set recording options (resolution, FPS, output path)
7. Click **RECORD VIDEO** or **EXPORT FRAMES**

## Visualization Colors

- **Green markers (S1, S2...)**: Segment start points
- **Orange markers (E1, E2...)**: Segment end points
- **Pink markers (O1, O2...)**: Orbit center POIs
- **Cyan lines**: Linear segment paths
- **Purple arcs**: Orbit segment paths
- **Yellow curve**: Smoothed camera path overlay
- **Blue dashed**: Segment transitions

## API Requirements

This plugin uses the following LichtFeld Studio APIs:
- `lf.render_at(eye, target, width, height, fov, up)` - Render from camera position
- `lf.io.save_image(path, tensor)` - Save image tensor to file
- `lf.add_draw_handler()` - Viewport visualization
- `lf.selection.pick_at_screen()` - Point picking

## License

GPL-3.0-or-later

---

Co-Authored-By: Warp <agent@warp.dev>
