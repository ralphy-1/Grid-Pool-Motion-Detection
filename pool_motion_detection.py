"""
PoolGrid - Overhead Motion Detection System
-------------------------------------------
A grid-based motion detection system for commercial pools.
Detects sustained motion (possible panic) within grid zones and
would trigger an alert to a lifeguard's wearable device.

Requirements:
    pip install opencv-python numpy

Collaborators:
    Eildvin Logrono
    Aaron Mihidri
"""

import time
import cv2
import numpy as np
from collections import deque

# ============================================================
# CONFIGURATION - Tune these values for your pool setup
# ============================================================

# Camera source (0 = default webcam, or path to video file for testing)
CAMERA_SOURCE = 0

# Grid settings - how finely to divide the pool view.
# Smaller cells catch localised panic motion that would otherwise be
# averaged out inside a large cell (a swimmer treading in place fills a
# bigger fraction of a small cell, so MIN_MOTION_AREA trips reliably).
GRID_ROWS = 20
GRID_COLS = 20

# Motion sensitivity - higher = less sensitive (fewer false positives).
# Lowered for overhead pool cameras where swimmers are far from the lens
# and their on-screen movement is small.
MOTION_THRESHOLD = 16

# Panic detection - two signals combined, both pointing at "erratic/rapid",
# not "sustained presence" (a calm lap-swimmer should NOT trip these).
#
# 1) Per-cell flip rate: how many times a cell toggles active/inactive in
#    the short window. Thrashing / splashing flickers rapidly; a swimmer
#    treading in place has steady activity and flips rarely.
FLIP_WINDOW_FRAMES = 30       # ~1s at 30 FPS
PANIC_FLIP_COUNT = 4          # >=4 flips in window => cell is panicking
MIN_MOTION_AREA = 0.05        # Fraction of cell that must show motion to count as "active"

# 2) Whole-frame motion-centroid velocity: when the bulk of motion lurches
#    across the frame fast, that's rapid lateral movement between cells.
#    Expressed as a fraction of the frame diagonal per frame so it's
#    resolution-independent.
CENTROID_VELOCITY_THRESHOLD = 0.04
# Mask must have at least this many motion pixels before we trust the
# centroid (otherwise a lone noise speck produces huge fake "velocity").
MIN_MOTION_PIXELS_FOR_CENTROID = 200

# Seconds a panic state must persist before the lifeguard is notified
PANIC_CONFIRM_SECONDS = 5.0

# Background learning rate (lower = slower adaptation to changes)
BG_LEARNING_RATE = 0.001

# Max consecutive failed frame reads before giving up on the camera
MAX_READ_FAILURES = 30


# ============================================================
# GRID CELL TRACKER
# Tracks motion history per cell to detect panic patterns
# ============================================================

def _col_label(col):
    """Excel-style column label: 0->A, 25->Z, 26->AA, ..."""
    s = ""
    n = col
    while True:
        s = chr(ord('A') + (n % 26)) + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


class GridCell:
    """Represents a single cell in the pool grid."""

    def __init__(self, row, col):
        self.row = row
        self.col = col
        # Recent activity (True/False per frame) over the flip window
        self.activity_history = deque(maxlen=FLIP_WINDOW_FRAMES)
        # Per-frame flip markers aligned with activity_history
        self.flip_history = deque(maxlen=FLIP_WINDOW_FRAMES)
        self.flip_count = 0
        self.is_panic = False

    def update(self, is_active, high_velocity_frame=False):
        """Record motion for this frame and recompute panic state.

        Panic fires if the cell is thrashing on its own (flip rate high),
        OR the whole-frame motion centroid is sweeping fast and this cell
        is currently part of it (rapid translation across cells).
        """
        prev = self.activity_history[-1] if self.activity_history else None
        self.activity_history.append(is_active)
        flipped = prev is not None and prev != is_active
        self.flip_history.append(flipped)
        self.flip_count = sum(self.flip_history)

        self.is_panic = (
            self.flip_count >= PANIC_FLIP_COUNT
            or (high_velocity_frame and is_active)
        )

    def label(self):
        """Return grid coordinate label (e.g. 'C4')."""
        return f"{_col_label(self.col)}{self.row + 1}"


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_grid():
    """Create a 2D array of GridCell objects."""
    return [[GridCell(r, c) for c in range(GRID_COLS)] for r in range(GRID_ROWS)]


def compute_cell_bounds(frame_width, frame_height):
    """Return (row_edges, col_edges) arrays that cover the full frame evenly.

    Using linspace-derived edges guarantees the last row/column reaches the
    frame edge even when dimensions don't divide evenly by the grid size.
    """
    row_edges = np.linspace(0, frame_height, GRID_ROWS + 1, dtype=int)
    col_edges = np.linspace(0, frame_width, GRID_COLS + 1, dtype=int)
    return row_edges, col_edges


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

def run_detection():
    # Open camera feed
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Create background subtractor - learns the static pool background
    # and highlights anything that changes (swimmers, splashes)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=MOTION_THRESHOLD,
        detectShadows=False
    )

    # Pre-allocate morphology kernels once (hot path)
    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((5, 5), np.uint8)

    # Initialize grid + cached geometry (populated on first frame)
    grid = initialize_grid()
    row_edges = col_edges = None
    frame_diagonal = 1.0
    last_frame_shape = None

    # Motion-centroid tracker for whole-frame velocity
    prev_centroid = None
    centroid_velocity = 0.0    # fraction of diagonal per frame

    # Panic confirmation timer state
    panic_started_at = None   # monotonic timestamp, or None if no panic active
    alert_sent = False        # True once the lifeguard has been notified this episode

    # FPS tracking
    fps_window = deque(maxlen=30)
    prev_time = time.monotonic()

    read_failures = 0

    print("PoolGrid running. Press 'q' to quit.")

    while True:
        # ----------------------------------------
        # STEP 1: Grab a frame from the camera
        # ----------------------------------------
        ret, frame = cap.read()
        if not ret or frame is None:
            read_failures += 1
            if read_failures >= MAX_READ_FAILURES:
                print("ERROR: Frame capture failed repeatedly — stopping.")
                break
            continue
        read_failures = 0

        # Recompute geometry if the frame size changed (or first frame)
        if frame.shape[:2] != last_frame_shape:
            last_frame_shape = frame.shape[:2]
            row_edges, col_edges = compute_cell_bounds(
                frame.shape[1], frame.shape[0]
            )
            frame_diagonal = float(np.hypot(frame.shape[1], frame.shape[0]))
            prev_centroid = None  # reset so first frame at new size doesn't spike

        # ----------------------------------------
        # STEP 2: Apply background subtraction
        # Produces a mask where white = motion, black = static
        # ----------------------------------------
        motion_mask = bg_subtractor.apply(frame, learningRate=BG_LEARNING_RATE)

        # Clean up the mask - remove small noise, fill gaps
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, open_kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, close_kernel)

        # ----------------------------------------
        # STEP 2b: Motion centroid velocity
        # Track where the bulk of the motion is and how fast it's moving.
        # High velocity = rapid lateral motion across cells (panic-like).
        # ----------------------------------------
        moments = cv2.moments(motion_mask, binaryImage=True)
        if moments["m00"] >= MIN_MOTION_PIXELS_FOR_CENTROID * 255:
            # m00 on a binary mask of 0/255 pixels is 255 * (pixel count)
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            if prev_centroid is not None:
                dx = cx - prev_centroid[0]
                dy = cy - prev_centroid[1]
                centroid_velocity = float(np.hypot(dx, dy) / frame_diagonal)
            else:
                centroid_velocity = 0.0
            prev_centroid = (cx, cy)
        else:
            centroid_velocity = 0.0
            prev_centroid = None

        high_velocity = centroid_velocity >= CENTROID_VELOCITY_THRESHOLD

        # ----------------------------------------
        # STEP 3: Analyze each grid cell for motion
        # ----------------------------------------
        for row in range(GRID_ROWS):
            y1, y2 = row_edges[row], row_edges[row + 1]
            for col in range(GRID_COLS):
                x1, x2 = col_edges[col], col_edges[col + 1]
                cell_region = motion_mask[y1:y2, x1:x2]

                # Calculate % of pixels showing motion in this cell
                if cell_region.size == 0:
                    is_active = False
                else:
                    motion_ratio = np.count_nonzero(cell_region) / cell_region.size
                    is_active = motion_ratio >= MIN_MOTION_AREA

                # Update cell's flip history and panic status
                grid[row][col].update(is_active, high_velocity)

        # ----------------------------------------
        # STEP 4: Draw grid overlay + alerts
        # ----------------------------------------
        display_frame = frame.copy()

        for row in range(GRID_ROWS):
            y1, y2 = row_edges[row], row_edges[row + 1]
            for col in range(GRID_COLS):
                x1, x2 = col_edges[col], col_edges[col + 1]
                cell = grid[row][col]

                # Determine cell color based on status
                if cell.is_panic:
                    color = (0, 0, 255)       # RED - panic detected
                    thickness = 3
                elif cell.activity_history and cell.activity_history[-1]:
                    color = (0, 255, 255)     # YELLOW - motion present
                    thickness = 1
                else:
                    color = (100, 100, 100)   # GRAY - inactive
                    thickness = 1

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                # Label panic cells prominently. Font scaled small so the
                # label stays readable when the grid is fine (e.g. 20x20).
                if cell.is_panic:
                    cv2.putText(
                        display_frame, f"!{cell.label()}",
                        (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
                    )

        # ----------------------------------------
        # STEP 5: Panic confirmation timer
        # A cell flagging panic starts a 5-second counter. If panic stays
        # active the whole window, notify the lifeguard (stub for now).
        # If it clears before 5s, reset — assume false positive.
        # ----------------------------------------
        panic_cells = [
            grid[r][c].label()
            for r in range(GRID_ROWS)
            for c in range(GRID_COLS)
            if grid[r][c].is_panic
        ]
        now = time.monotonic()

        if panic_cells:
            if panic_started_at is None:
                panic_started_at = now
                alert_sent = False
                print(f"Panic suspected at: {', '.join(sorted(panic_cells))} — confirming for {PANIC_CONFIRM_SECONDS:.0f}s")
            elapsed = now - panic_started_at

            if elapsed >= PANIC_CONFIRM_SECONDS and not alert_sent:
                print(
                    f"!!! LIFEGUARD ALERT: sustained panic after "
                    f"{elapsed:.1f}s at {', '.join(sorted(panic_cells))}"
                )
                alert_sent = True
                # TODO: send_to_wearable(panic_cells)

            banner = f"ALERT - Panic motion at: {', '.join(sorted(panic_cells))}"
            cv2.putText(
                display_frame, banner, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        else:
            if panic_started_at is not None and not alert_sent:
                print("Panic cleared before 5s — counter reset")
            panic_started_at = None
            alert_sent = False
            elapsed = 0.0

        # ----------------------------------------
        # STEP 6: FPS + panic counter + output windows
        # ----------------------------------------
        fps_window.append(now - prev_time)
        prev_time = now
        avg_dt = sum(fps_window) / len(fps_window)
        fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        cv2.putText(
            display_frame, f"{fps:5.1f} FPS",
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        vel_color = (0, 0, 255) if high_velocity else (255, 255, 255)
        cv2.putText(
            display_frame, f"VEL {centroid_velocity:.3f}",
            (130, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_color, 2
        )

        # Panic counter at bottom-right. Only shown while a panic is active.
        if panic_started_at is not None:
            shown = min(elapsed, PANIC_CONFIRM_SECONDS)
            counter_text = f"PANIC {shown:4.1f}s / {PANIC_CONFIRM_SECONDS:.0f}s"
            counter_color = (0, 0, 255) if alert_sent else (0, 215, 255)
            (tw, th), _ = cv2.getTextSize(
                counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )
            x = display_frame.shape[1] - tw - 15
            y = display_frame.shape[0] - 15
            cv2.putText(
                display_frame, counter_text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, counter_color, 2
            )

        cv2.imshow("PoolGrid - Live Feed", display_frame)
        cv2.imshow("Motion Mask", motion_mask)

        # Quit on 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_detection()
