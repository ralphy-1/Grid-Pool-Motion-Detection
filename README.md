# Grid-Pool-Motion-Detection
Overhead Camera Motion Detection System An overhead camera divides its field of view into a uniform grid. Each cell is analyzed per frame against a background reference — when pixel changes are detected, the cell is flagged, mapping motion in real time across the monitored area.

PoolGrid — Overhead Motion Detection System for Commercial Pools
A real-time, grid-based motion detection system designed for large commercial swimming pools. The system uses an overhead camera to divide the pool surface into a structured grid, monitors each cell for panic-pattern motion, and delivers directional alerts to a lifeguard's wearable device (watch/band).

 Purpose
Standard pool alarms are built for residential use and alert a parent's smartphone. PoolGrid is built for lifeguards. In a large commercial pool, a lifeguard cannot afford to check a phone — they need an instant, directional signal that tells them exactly where to look.
PoolGrid detects sustained rapid motion within a grid zone, flags it as a potential panic event, and sends a vibration alert to a wearable device that identifies the specific grid location.

Key Features

Overhead camera with full pool surface coverage
Uniform grid overlay divides the pool into independently monitored zones
Panic detection via sustained motion threshold within a grid cell
Directional wearable alert (smartwatch/band) pointing to the flagged zone
Designed and optimized for large commercial pools
Lightweight, real-time processing

How It Works

Camera captures a live overhead feed of the pool
Feed is divided into a grid of cells
Each cell is analyzed per frame using background subtraction
If a cell detects sustained rapid motion beyond a set time threshold, it is flagged as a panic event
The flagged grid zone is transmitted to the lifeguard's wearable device
Lifeguard receives a directional alert and can immediately locate the situation

Tech Stack

Language: Python
Computer Vision: OpenCV
Hardware: Overhead camera, wearable device (TBD)
Platform: TBD

-------------
Authors:
Aaron Mihidri
Eildvin Logrono
-------------

This project is licensed under the MIT License. See LICENSE for details.
