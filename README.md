[![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# Self-Drive Airism

An autonomous driving simulation project using **AirSim** and **YOLO** for vehicle detection and navigation. This project demonstrates self-driving capabilities with real-time object detection in a simulated environment.

- **Autonomous Navigation**: Self-driving vehicle control with target waypoint navigation - **Real-time Object Detection**: YOLO-based car and object detection - **Camera Simulation**: Multiple camera perspectives (frontal, driver, reverse, follow) - **GPS/NED Coordinates**: Navigation using geodetic coordinates - **Segmentation**: Vehicle and environment segmentation analysis - **CUDA Support**: GPU acceleration for faster inference

## Overview

A robust lane detection and tracking pipeline for autonomous vehicles in Microsoft AirSim. Utilizing OpenCV to process real-time video feeds from Unreal Engine, this project implements color filtering, perspective transforms (bird's-eye view), and polynomial fitting to identify lane boundaries and calculate curvature for steering control.

## Repository Structure

| Path | Description |
| --- | --- |
| `assets/` | Images, figures, or other supporting media used by the project. |
| `scripts/` | Top-level project directory containing repository-specific resources. |
| `utils/` | Reusable helper modules and shared utility functions. |
| `.gitignore` | Top-level file included in the repository. |
| `README.md` | Primary project documentation. |
| `requirements.txt` | Python dependency specification for local setup. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/Self-Drive-Airism.git
   cd Self-Drive-Airism
   ```

2. Prepare the local environment.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run or inspect the project entry point.

Use the project-specific scripts or notebooks in the repository root to run the workflow.

## Features

- **Autonomous Navigation**: Self-driving vehicle control with target waypoint navigation
- **Real-time Object Detection**: YOLO-based car and object detection
- **Camera Simulation**: Multiple camera perspectives (frontal, driver, reverse, follow)
- **GPS/NED Coordinates**: Navigation using geodetic coordinates
- **Segmentation**: Vehicle and environment segmentation analysis
- **CUDA Support**: GPU acceleration for faster inference

## Installation Troubleshooting

If you encounter issues with `airsim` installation:

1. Ensure Visual C++ redistributables are installed (Windows)
2. Update pip: `python -m pip install --upgrade pip`
3. Try installing with verbose output: `pip install -v airsim`
4. Check system compatibility with: `python utils/check_cuda.py`

## Usage Examples

### Basic Self-Driving
```python
python scripts/selfdrivefinal.py
```

### With Arguments
```python
python scripts/selfdrivefinal.py --camera frontal --vehicle car_0
```

## Project Structure

```
Self-Drive-Airism/
├── scripts/                    # Main autonomous driving scripts
│   ├── selfdrivefinal.py      # Final self-drive implementation
│   ├── selfdrivev*.py         # Versioned development iterations
│   ├── selfdrive.py           # Base self-drive module
│   └── ...
├── utils/                      # Utility and helper functions
│   ├── geo.py                 # Geographic coordinate utilities
│   ├── segmentator.py         # Segmentation utilities
│   ├── check_cuda.py          # CUDA availability checker
│   ├── see.py                 # Visualization utilities
│   ├── seefilters.py          # Filter visualization
│   ├── listcars.py            # Vehicle listing utilities
│   ├── list_objects.py        # Object detection helpers
│   └── move_geo.py            # Geographic movement utilities
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

### Prerequisites
- Python 3.8+
- NVIDIA GPU (for CUDA support)
- AirSim simulator

### Dependencies

Install dependencies in this order to avoid build issues:

```bash
# 1. Install core dependencies first
pip install numpy msgpack-rpc-python opencv-python pillow requests

# 2. Install AirSim
pip install airsim

# 3. Install remaining requirements
pip install -r requirements.txt
```

**Key packages:**
- `airsim` - AirSim Python client
- `ultralytics` - YOLO object detection
- `opencv-python` - Computer vision
- `numpy` - Numerical computing
- `tqdm` - Progress bars
- `yacs` - Configuration management

## Quick Start

1. **Setup Environment**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Check CUDA availability
   python utils/check_cuda.py
   ```

2. **Run Autonomous Driving**
   ```bash
   python scripts/selfdrivefinal.py
   ```

3. **Visualization**
   ```bash
   python utils/see.py           # View simulation output
   python utils/seefilters.py    # View filtered results
   ```

## Script Descriptions

### Main Scripts (`scripts/`)
- **selfdrivefinal.py**: Production-ready self-driving implementation with object detection
- **selfdrivev*.py**: Development versions showing iteration progress

### Utilities (`utils/`)
- **geo.py**: Geographic coordinate transformations and utilities
- **segmentator.py**: Semantic segmentation of environment
- **check_cuda.py**: Verify CUDA GPU availability
- **see.py**: Real-time visualization of simulation
- **listcars.py**: List available vehicles in simulation
- **list_objects.py**: Detect and list objects in scene

## ️ Configuration

Main configuration in `selfdrivefinal.py`:
```python
# Camera mapping
CAMERA_MAP = {
    "frontal": "0", 
    "driver": "3", 
    "reverse": "4", 
    "follow": "follow_cam"
}

# Target waypoint (NED coordinates in meters)
TARGET_X = 595.63   # North
TARGET_Y = -258.19  # East
TARGET_Z = -0.68    # Down
```
