# Isaac Sim Environment Requirements

This document describes the Python package requirements for NVIDIA Isaac Sim used in the SAGE-3D project.

## Files

- **`isaac_sim_requirements.txt`**: Complete list of Python packages installed in Isaac Sim 5.0+ environment

## Usage

These requirements are provided for **reference purposes only** to help reproduce our experimental environment. 

### Important Notes

⚠️ **Do NOT install these packages directly into a standard Python environment!**

Isaac Sim comes with its own bundled Python environment that includes:
- Custom builds of PyTorch with CUDA support
- USD (Universal Scene Description) libraries
- Warp language for GPU-accelerated physics
- NVIDIA robotics libraries (IsaacLab, etc.)
- Other proprietary NVIDIA tools

### Recommended Setup

Instead of manually installing these packages, please:

1. **Build Isaac Sim from source** following the [official instructions](https://github.com/isaac-sim/IsaacSim)
2. **Use Isaac Sim's Python interpreter** for running SAGE-3D scripts:

```bash
# Linux
/path/to/IsaacSim/_build/linux-x86_64/release/python.sh your_script.py

# Windows
/path/to/IsaacSim/_build/windows-x86_64/release/python.bat your_script.py
```

3. If you need additional packages beyond what Isaac Sim provides, install them using:

```bash
# Linux
/path/to/IsaacSim/_build/linux-x86_64/release/python.sh -m pip install package_name

# Windows
/path/to/IsaacSim/_build/windows-x86_64/release/python.bat -m pip install package_name
```

## Key Packages Included

The Isaac Sim environment includes (but is not limited to):

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.5.1 | Deep learning framework |
| `numpy` | 1.26.0 | Numerical computing |
| `opencv-python` | 4.11.0.86 | Computer vision |
| `pillow` | 11.0.0 | Image processing |
| `usd-core` | 25.5.1 | Universal Scene Description |
| `warp-lang` | 1.6.1 | GPU-accelerated physics |
| `transformers` | 4.49.0 | NLP models |
| `scipy` | 1.11.2 | Scientific computing |

## Version Information

- **Isaac Sim Version**: 5.0+
- **Python Version**: 3.11
- **CUDA Version**: 12.4+
- **PyTorch Version**: 2.5.1

## Troubleshooting

If you encounter package conflicts or missing dependencies when running SAGE-3D scripts:

1. Verify Isaac Sim is built correctly and runs successfully
2. Check that you're using Isaac Sim's Python interpreter (not system Python)
3. Refer to the [Isaac Sim documentation](https://github.com/isaac-sim/IsaacSim) for environment setup

---

For more information on setting up Isaac Sim for SAGE-3D, see the main [README](../README.md#isaac-sim-setup).

