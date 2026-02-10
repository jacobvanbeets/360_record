# 360 Record Plugin for LichtFeld Studio

Record smooth circular camera track videos around a point of interest in your 3D scene.

## Features

- **Point of Interest Selection**: Click on your model to set the center point for the camera orbit
- **Customizable Camera Track**:
  - Elevation: Height above the point of interest
  - Radius: Distance from the center point
  - Speed: Duration of one complete revolution
  - Starting Angle: Where on the circle the camera begins
- **Live Preview**: Visualize the camera track in the viewport before recording
- **Video Recording**: Export smooth MP4 videos with configurable resolution, FPS, and quality

## Installation

1. Copy the `360_record` folder to your LichtFeld Studio plugins directory
2. Restart LichtFeld Studio or enable the plugin from the Plugins menu

## Usage

1. Open a scene with a Gaussian Splat or point cloud model
2. Open the **Camera Track** panel from the side panel
3. Click **Pick Point of Interest** and click on your model to set the center point
4. Adjust the track settings:
   - **Elevation**: How high above the POI the camera orbits
   - **Radius**: How far from the POI the camera orbits
   - **Speed**: How many seconds for one full rotation
   - **Starting Angle**: Where the camera starts on the circle
5. Use the **Preview** section to see where the camera will be at different angles
6. Configure recording settings (resolution, FPS, quality, output path)
7. Click **RECORD CIRCULAR VIDEO** to export

## Track Visualization

When a POI is set, the plugin displays:
- **Red marker**: Point of Interest location
- **Cyan circle**: Camera track path
- **Green marker**: Starting position
- **Green line**: Camera look direction

## Requirements

- LichtFeld Studio 0.5.0 or later
- NumPy

## License

GPL-3.0-or-later
