# Doomscroll Skyrim Edition

**A CV productivity tool that plays Skyrim Skeleton mode whenever you doomscroll or lose focus.**

![Skyrim Skeleton](https://github.com/user-attachments/assets/d06ccf2f-0b9d-4fdf-8a95-6117c0d77c15)

## Introduction

This is a **Linux port** of the [original version](https://github.com/reinesana/Doomscroll-Skyrim-Edition) by Shana Nursoo, which was only available for macOS.

**Doomscrolling: Skyrim Edition** is a CV productivity tool inspired by the **Skeleton** trend on **TikTok** and my previous doomscrolling tool: **Charlie Kirkification**. Designed for laptop-based work only, the program tracks your eye and iris movement in real time to detect when you’re looking down at your phone (aka doomscrolling).

**Note:** This tool does not work for activities like writing, reading books, or other offline tasks since it uses iris movement to detect doomscrolling.


## How it works

1. Your webcam feed is processed in real time using **MediaPipe FaceLandmarker** for face and iris tracking.
2. The program checks whether your iris movement suggests you’re looking at your phone.
3. If you doomscroll for longer than a set threshold:
   - The Skyrim Skeleton video is played to interrupt your doomscrolling.
   - A “doomscrolling alarm” overlay appears on the webcam feed.

## System Requirements

- **Python:** >=3.9
- **External Dependencies:**
  - `mpv` media player for playing the video file

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/realcharmer/Doomscroll-Skyrim-Edition-Linux.git
cd Doomscroll-Skyrim-Edition-Linux
```

### 2. Create Python venv

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the program

```bash
python main.py
```

## Configuration options

The following configuration options can be passed during execution.

### `--timer`

The minimum amount of time the user must be “looking down” before triggering the program. This acts as a grace period.

- Lower value: triggers faster
- Higher value: longer grace period before triggering

### `--looking_threshold`

The minimum iris position required to consider the user as looking down.

- Lower value: more strict and requires stronger downward gaze
- Higher value: more sensitive

### `--debounce_threshold`

The minimum threshold required for the system to exit the “looking down” state before the program resets.

- Lower value: video stops more easily (more strict while playing)
- Higher value: video stays on longer (more forgiving while playing)
