# 360 Record - LichtFeld Studio Plugin

A plugin for [LichtFeld Studio](https://lichtfeld.io) that enables recording camera path videos around 3D Gaussian Splat scenes.

[![Watch Demo Video](https://img.youtube.com/vi/eFl7kilU9ts/maxresdefault.jpg)](https://www.youtube.com/watch?v=eFl7kilU9ts)

**▶️ Click the image above to watch the demo video**

## Features

### Circular Camera Track (360° Recording)
- Pick a **Point of Interest (POI)** on your model
- Set **elevation** (offset above/below POI)
- Set **radius** (distance from POI)
- Choose **orbit axis** (X, Y, or Z)
- Adjustable **starting angle**
- Configurable **speed** (seconds per full circle)
- Invert direction option

### Linear Camera Path
- Create **multi-segment paths** with connected waypoints
- Pick **start and end points** for each segment
- Per-segment **look mode**:
  - **Look Forward** - Camera looks along travel direction
  - **Look at POI** - Camera looks at a specific point of interest
- **Smooth transitions** with Catmull-Rom spline interpolation
- Adjustable **smoothing factor** for corner transitions
- **Elevation offset** above the path
- **Global travel speed** (units per second)
- **Walking Speed** preset (1.4 m/s for realistic walkthroughs)

### Recording Options
- **Resolution presets**: 720p, 1080p, 1440p, 4K, Square
- Adjustable **FPS** (1-120)
- Configurable **Field of View**
- Export as **MP4 video** or **PNG frame sequence**
- **Live progress bar** during recording

### Viewport Preview
- Real-time visualization of camera paths
- Start/end point markers
- POI markers
- Smoothed path overlay
- Transition indicators

## Installation

1. Clone or download this repository
2. Create a junction/symlink to the LichtFeld plugins directory:
   ```powershell
   # Windows (PowerShell as Admin)
   New-Item -ItemType Junction -Path "$env:USERPROFILE\.lichtfeld\plugins\360_record" -Target "C:\path\to\360_record"
   ```
   ```bash
   # macOS/Linux
   ln -s /path/to/360_record ~/.lichtfeld/plugins/360_record
   ```
3. Restart LichtFeld Studio or reload plugins

## Dependencies

- **numpy** - Array operations
- **imageio[pyav]** - MP4 video encoding

Install with:
```bash
pip install numpy imageio[pyav]
```

## Usage

### Circular Track (360° Video)
1. Open the **Camera Track** panel
2. Click **Pick Point of Interest** and click on your model
3. Adjust elevation, radius, and speed
4. Choose orbit axis and starting angle
5. Set recording options (resolution, FPS, output path)
6. Click **RECORD VIDEO** or **EXPORT FRAMES**

### Linear Path
1. Open the **Linear Path** panel
2. Click **+ Add Segment**
3. Pick the **start point** (click on model)
4. Pick the **end point**
5. Choose look mode (Forward or POI)
6. Add more segments as needed (they auto-connect)
7. Adjust speed, smoothing, and elevation
8. Set recording options
9. Click **RECORD VIDEO** or **EXPORT FRAMES**

## Track Visualization

**Circular Track:**
- **Red marker**: Point of Interest
- **Cyan circle**: Camera orbit path
- **Green marker**: Starting position

**Linear Path:**
- **Green markers**: Segment start points (S1, S2, ...)
- **Orange markers**: Segment end points (E1, E2, ...)
- **Cyan lines**: Segment paths
- **Yellow curve**: Smoothed camera path
- **Pink markers**: POI points

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
