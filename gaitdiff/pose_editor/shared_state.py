"""
Unified Application State for GaitDiff
Single source of truth for video registry, pose data, and playback state.

Design:
  - Videos are registered by letter (A-Z).  Each letter owns its pose data,
    file path, and frame count.
  - Two *slots* ('A' and 'B') represent the left/right comparison panes.
    Each slot points to a video letter.
  - All slot-based accessors (used by VideoPlayer, PoseEditor, Analyzer)
    resolve  slot → letter  internally, so callers don't need to know which
    video is currently loaded.
"""

from PySide6.QtCore import QObject, Signal
import pandas as pd
import numpy as np
import os
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List


@dataclass
class VideoEntry:
    """A registered video."""
    letter_id: str          # A, B, C, …
    original_name: str      # e.g. "IMG_7290.MOV"
    file_path: str          # absolute path on disk
    display_name: str       # e.g. "A: IMG_7290.MOV"


class SharedPoseState(QObject):
    """
    Singleton application state shared across the main window, video players,
    pose editor, and analyzer.

    Data is keyed by *video letter* (persistent across slot switches).
    Slot-based methods resolve slot → letter transparently.
    """

    # ── Signals ──
    pose_data_changed = Signal(str)               # slot id ('A' or 'B')
    frame_changed = Signal(str, int)               # slot id, frame_index
    keypoint_edited = Signal(str, int, int)        # slot id, frame, keypoint
    video_list_changed = Signal()                  # video added / removed
    slot_video_changed = Signal(str)               # slot id whose video changed

    MAX_VIDEOS = 26

    def __init__(self):
        super().__init__()

        # ── Video registry  (letter → entry) ──
        self._videos: Dict[str, VideoEntry] = {}

        # ── Per-video data  (letter → data) ──
        self._pose_data: Dict[str, Optional[pd.DataFrame]] = {}
        self._video_paths: Dict[str, Optional[str]] = {}
        self._frame_counts: Dict[str, int] = {}

        # ── Slot → video-letter mapping ──
        self._slot_video: Dict[str, Optional[str]] = {'A': None, 'B': None}

        # ── Per-slot state ──
        self._current_frames: Dict[str, int] = {'A': 0, 'B': 0}
        self._selected_keypoints: Dict[str, Optional[int]] = {'A': None, 'B': None}

        # Keypoint names (shared – always RR21)
        from .pose_format_utils import SUPPORTED_FORMATS
        self.keypoint_names = SUPPORTED_FORMATS["rr21"]

    # ─────────────────────────── helpers ───────────────────────────

    @staticmethod
    def _letter(index: int) -> Optional[str]:
        return string.ascii_uppercase[index] if 0 <= index < 26 else None

    @staticmethod
    def _truncate(name: str, max_len: int = 28) -> str:
        stem = Path(name).stem
        ext = Path(name).suffix
        return f"{stem[:max_len]}..{ext}" if len(stem) > max_len else name

    def _resolve(self, slot: str) -> Optional[str]:
        """Resolve a slot name ('A'/'B') to a video letter."""
        return self._slot_video.get(slot)

    # ═══════════════════════ Video Registry ═══════════════════════

    def add_video(self, file_path: str) -> Optional[str]:
        """Register a video.  Returns its letter or ``None``."""
        if len(self._videos) >= self.MAX_VIDEOS:
            return None
        file_path = os.path.normpath(file_path)
        original_name = os.path.basename(file_path)

        # Duplicate check – return existing letter
        for letter, e in self._videos.items():
            if os.path.normpath(e.file_path) == file_path:
                return letter
        if not os.path.isfile(file_path):
            return None

        letter = self._letter(len(self._videos))
        if letter is None:
            return None

        self._videos[letter] = VideoEntry(
            letter_id=letter,
            original_name=original_name,
            file_path=file_path,
            display_name=f"{letter}: {self._truncate(original_name)}",
        )
        self._video_paths[letter] = file_path
        self.video_list_changed.emit()
        return letter

    def get_video(self, letter: str) -> Optional[VideoEntry]:
        return self._videos.get(letter)

    def get_all_videos(self) -> Dict[str, VideoEntry]:
        return dict(self._videos)

    def video_count(self) -> int:
        return len(self._videos)

    def combo_items(self) -> List[tuple]:
        """Return ``[(display_name, letter), …]`` for QComboBox population."""
        return [(e.display_name, k) for k, e in self._videos.items()]

    def get_path_by_letter(self, letter: str) -> Optional[str]:
        """Get the file path for a video letter directly."""
        e = self._videos.get(letter)
        return e.file_path if e else None

    # ═══════════════════════ Slot Mapping ═══════════════════════

    def get_slot_video(self, slot: str) -> Optional[str]:
        """Which video letter is loaded in *slot*?"""
        return self._slot_video.get(slot)

    def set_slot_video(self, slot: str, letter: Optional[str]):
        """Assign video *letter* to *slot*.  Resets per-slot transients.
        
        Passing None explicitly clears the slot (unassigns any video).
        """
        old = self._slot_video.get(slot)
        if old == letter:
            return  # No change, skip signal emissions
        
        self._slot_video[slot] = letter
        self._current_frames[slot] = 0
        self._selected_keypoints[slot] = None
        
        # Emit signals to notify listeners
        self.slot_video_changed.emit(slot)
        # Only emit pose_data_changed if there's actually a video assigned
        if letter is not None:
            self.pose_data_changed.emit(slot)

    # ═══════════════════════ Pose Data ═══════════════════════

    def get_pose_data(self, slot: str) -> Optional[pd.DataFrame]:
        """Get pose data for a *slot* (resolves slot → letter)."""
        letter = self._resolve(slot)
        if letter is None:
            return None
        return self._pose_data.get(letter)

    def set_pose_data(self, slot: str, pose_data: Optional[pd.DataFrame]):
        """Set pose data via *slot* (resolves slot → letter)."""
        letter = self._resolve(slot)
        if letter is None:
            return
        self._pose_data[letter] = pose_data
        self.pose_data_changed.emit(slot)
        # Also notify the other slot if it shows the same video
        for s, l in self._slot_video.items():
            if l == letter and s != slot:
                self.pose_data_changed.emit(s)

    def set_pose_data_for_video(self, letter: str, pose_data: Optional[pd.DataFrame]):
        """Store pose data directly by video *letter* (not slot)."""
        self._pose_data[letter] = pose_data
        # Notify every slot that currently shows this letter
        for slot, sletter in self._slot_video.items():
            if sletter == letter:
                self.pose_data_changed.emit(slot)

    def get_pose_data_for_video(self, letter: str) -> Optional[pd.DataFrame]:
        """Retrieve pose data directly by video *letter*."""
        return self._pose_data.get(letter)

    def get_pose_for_frame(self, slot: str, frame_idx: int) -> Optional[np.ndarray]:
        """Get pose coordinates for a frame as ``(N, 2)`` array."""
        pose_data = self.get_pose_data(slot)
        if pose_data is None or frame_idx < 0 or frame_idx >= len(pose_data):
            return None
        return pose_data.iloc[frame_idx].values.reshape(-1, 2)

    def update_keypoint(self, slot: str, frame_idx: int, keypoint_idx: int,
                        x: float, y: float):
        """Update a single keypoint (resolves slot → letter)."""
        letter = self._resolve(slot)
        if letter is None:
            return
        pose_data = self._pose_data.get(letter)
        if pose_data is None:
            return
        pose_data.iloc[frame_idx, keypoint_idx * 2] = x
        pose_data.iloc[frame_idx, keypoint_idx * 2 + 1] = y
        self.keypoint_edited.emit(slot, frame_idx, keypoint_idx)

    # ═══════════════════════ Video Path ═══════════════════════

    def get_video_path(self, slot: str) -> Optional[str]:
        """Get file path for the video in *slot*."""
        letter = self._resolve(slot)
        if letter is None:
            return None
        return self._video_paths.get(letter)

    def set_video_path(self, slot: str, path: Optional[str]):
        """Set file path via *slot*.  (Usually redundant after ``add_video``.)"""
        letter = self._resolve(slot)
        if letter is not None:
            self._video_paths[letter] = path

    # ═══════════════════════ Frame State ═══════════════════════

    def get_current_frame(self, slot: str) -> int:
        return self._current_frames.get(slot, 0)

    def set_current_frame(self, slot: str, frame_idx: int):
        self._current_frames[slot] = frame_idx
        self.frame_changed.emit(slot, frame_idx)

    def get_frame_count(self, slot: str) -> int:
        letter = self._resolve(slot)
        if letter is not None:
            return self._frame_counts.get(letter, 0)
        return 0

    def set_frame_count(self, slot: str, count: int):
        """Set frame count for the video in this slot."""
        letter = self._resolve(slot)
        if letter is not None:
            self._frame_counts[letter] = count

    # ═══════════════════════ Selected Keypoint ═══════════════════════

    def get_selected_keypoint(self, slot: str) -> Optional[int]:
        return self._selected_keypoints.get(slot)

    def set_selected_keypoint(self, slot: str, keypoint_idx: Optional[int]):
        self._selected_keypoints[slot] = keypoint_idx

    # ═══════════════════════ Utility ═══════════════════════

    def has_pose_data(self, slot: str) -> bool:
        return self.get_pose_data(slot) is not None

    def clear_slot(self, slot: str):
        """Unassign a slot (does NOT delete the video's cached data)."""
        self._slot_video[slot] = None
        self._current_frames[slot] = 0
        self._selected_keypoints[slot] = None

    # Backward-compat alias
    clear_video = clear_slot


# ── Global singleton ──

_shared_state: Optional[SharedPoseState] = None


def get_shared_state() -> SharedPoseState:
    """Get or create the global shared pose state."""
    global _shared_state
    if _shared_state is None:
        _shared_state = SharedPoseState()
    return _shared_state
