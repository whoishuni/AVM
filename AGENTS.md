# Agent Instructions for YC_VesselTracer

This document provides instructions for agents and developers working on the YC_VesselTracer project.

## How to Fix the 3D View in the Packaged Application

**Problem:**

The 3D view, which relies on `PyQt6` and `PyQt6-WebEngine`, may not work in the application after it has been packaged (e.g., with PyInstaller).

This is a known issue caused by the packager failing to include all the necessary files for the `QtWebEngineProcess`, which is essential for rendering web-based content like Plotly graphs.

**Solution:**

This repository now includes a pre-configured hook for PyInstaller to resolve this issue. The hook ensures that all required `PyQt6-WebEngine` resources are correctly bundled.

### How to Use the Provided Hook with PyInstaller:

The necessary hook file is located at `hooks/hook-PyQt6.QtWebEngineWidgets.py`.

To use it, you must tell PyInstaller where to find this hook by using the `--additional-hooks-dir` command-line option.

For example, if you are packaging `main.py`, your command should look like this:

```bash
pyinstaller main.py --noconsole --additional-hooks-dir ./hooks
```

*   `--noconsole`: This is recommended for GUI applications to prevent a console window from appearing.
*   `--additional-hooks-dir ./hooks`: This tells PyInstaller to look for hooks in the `hooks` directory, which is now part of this repository.

By using this command, PyInstaller will correctly bundle the `QtWebEngine` dependencies, and the 3D view will function as expected in the packaged application.