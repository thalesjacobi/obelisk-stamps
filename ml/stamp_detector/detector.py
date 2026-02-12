"""
StampDetector - YOLO-based stamp detection and extraction.

This module provides a reusable class for detecting individual stamps
in images that may contain multiple stamps (2-up, 3-up, grids, etc.)

Detection uses a trained YOLOv8 model to identify stamp bounding boxes.
The model was trained on labeled stamp images and achieves high accuracy
(mAP50: 99.5%) for detecting both single and multiple stamps.
"""

import os
import statistics
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Module paths
MODULE_DIR = Path(__file__).parent
WEIGHTS_DIR = MODULE_DIR / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "stamp_detector.pt"


# ─────────────────────────────────────────────────────────
# Valley-based detection helpers
# ─────────────────────────────────────────────────────────

def _ink_profile(img: Image.Image, axis: str, ignore_bottom_frac: float = 0.18) -> List[int]:
    """
    Sum of "ink" (dark) pixels per column (axis='x') or per row (axis='y').

    Ignores a bottom slice because PostBeeld watermarks often sit there.
    """
    gray = ImageOps.grayscale(img)
    w, h = gray.size
    px = gray.load()

    y_max = int(h * (1.0 - ignore_bottom_frac))
    y_max = max(1, min(h, y_max))

    if axis == "x":
        prof = [0] * w
        for x in range(w):
            s = 0
            for y in range(y_max):
                if px[x, y] < 240:
                    s += 1
            prof[x] = s
        return prof

    prof = [0] * y_max
    for y in range(y_max):
        s = 0
        for x in range(w):
            if px[x, y] < 240:
                s += 1
        prof[y] = s
    return prof


def _brightness_profile(img: Image.Image, axis: str, ignore_bottom_frac: float = 0.18) -> List[float]:
    """
    Average brightness per column (axis='x') or per row (axis='y').

    Higher values = brighter = more likely to be a gap between stamps.
    This is complementary to _ink_profile: it detects light-colored gaps
    even when stamps have light backgrounds (not pure white).
    """
    gray = ImageOps.grayscale(img)
    w, h = gray.size
    px = gray.load()

    y_max = int(h * (1.0 - ignore_bottom_frac))
    y_max = max(1, min(h, y_max))

    if axis == "x":
        prof = [0.0] * w
        for x in range(w):
            s = 0
            for y in range(y_max):
                s += px[x, y]
            prof[x] = s / y_max if y_max > 0 else 0
        return prof

    prof = [0.0] * y_max
    for y in range(y_max):
        s = 0
        for x in range(w):
            s += px[x, y]
        prof[y] = s / w if w > 0 else 0
    return prof


def _gradient_profile(img: Image.Image, axis: str) -> List[float]:
    """
    Compute the vertical or horizontal gradient magnitude profile.

    For axis='x': at each column, compute the average absolute horizontal gradient.
    High gradient columns correspond to vertical edges (like stamp borders).

    For axis='y': at each row, compute the average absolute vertical gradient.
    High gradient rows correspond to horizontal edges (like stamp borders).
    """
    gray = ImageOps.grayscale(img)
    # Apply slight blur to reduce noise
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(gray, dtype=np.float32)
    h, w = arr.shape

    if axis == "x":
        # Horizontal gradient (detects vertical edges)
        if w < 3:
            return [0.0] * w
        grad = np.abs(np.diff(arr, axis=1))
        # Average gradient per column
        profile = [0.0] + [float(grad[:, x].mean()) for x in range(grad.shape[1])]
        return profile

    # Vertical gradient (detects horizontal edges)
    if h < 3:
        return [0.0] * h
    grad = np.abs(np.diff(arr, axis=0))
    profile = [0.0] + [float(grad[y, :].mean()) for y in range(grad.shape[0])]
    return profile


def _find_valleys(profile: List[int], min_run: int, low_quantile: float = 0.12) -> List[Tuple[int, int]]:
    """Find contiguous low-ink runs (valleys) in a pixel profile."""
    if not profile:
        return []
    sorted_vals = sorted(profile)
    q_idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * low_quantile)))
    thresh = sorted_vals[q_idx]

    valleys = []
    in_run = False
    start = 0
    for i, v in enumerate(profile):
        if v <= thresh:
            if not in_run:
                in_run = True
                start = i
        else:
            if in_run:
                end = i - 1
                if (end - start + 1) >= min_run:
                    valleys.append((start, end))
                in_run = False
    if in_run:
        end = len(profile) - 1
        if (end - start + 1) >= min_run:
            valleys.append((start, end))
    return valleys


def _find_bright_valleys(profile: List[float], min_run: int, high_quantile: float = 0.88) -> List[Tuple[int, int]]:
    """Find contiguous high-brightness runs (gaps) in a brightness profile."""
    if not profile:
        return []
    sorted_vals = sorted(profile)
    q_idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * high_quantile)))
    thresh = sorted_vals[q_idx]

    valleys = []
    in_run = False
    start = 0
    for i, v in enumerate(profile):
        if v >= thresh:
            if not in_run:
                in_run = True
                start = i
        else:
            if in_run:
                end = i - 1
                if (end - start + 1) >= min_run:
                    valleys.append((start, end))
                in_run = False
    if in_run:
        end = len(profile) - 1
        if (end - start + 1) >= min_run:
            valleys.append((start, end))
    return valleys


def _find_gradient_peaks(profile: List[float], size: int, min_edge_frac: float = 0.15) -> List[int]:
    """
    Find positions of high gradient (edges) that could be stamp borders.

    Returns cut positions where gradient peaks suggest stamp boundaries.
    Only considers positions in the interior (min_edge_frac..1-min_edge_frac).
    """
    if len(profile) < 10:
        return []

    avg = statistics.mean(profile)
    if avg <= 0:
        return []

    # Find positions where gradient is significantly above average
    threshold = avg * 2.0
    lo = int(size * min_edge_frac)
    hi = int(size * (1.0 - min_edge_frac))

    # Group consecutive high-gradient pixels into peaks
    peaks = []
    in_peak = False
    peak_start = 0
    peak_max_val = 0
    peak_max_pos = 0

    for i in range(lo, hi):
        if i >= len(profile):
            break
        if profile[i] >= threshold:
            if not in_peak:
                in_peak = True
                peak_start = i
                peak_max_val = profile[i]
                peak_max_pos = i
            elif profile[i] > peak_max_val:
                peak_max_val = profile[i]
                peak_max_pos = i
        else:
            if in_peak:
                peaks.append(peak_max_pos)
                in_peak = False
    if in_peak:
        peaks.append(peak_max_pos)

    return peaks


def _valid_cuts(cuts: List[int], size: int, edge_frac: float = 0.18) -> List[int]:
    """Reject cuts too close to edges."""
    return [c for c in cuts if int(size * edge_frac) <= c <= int(size * (1.0 - edge_frac))]


def _pick_up_to_n(cuts: List[int], size: int, max_cuts: int = 2, min_spacing: Optional[int] = None) -> List[int]:
    """
    Select up to N well-spaced cuts, preferring evenly-distributed positions.

    For max_cuts=1: prefers center (1/2)
    For max_cuts=2: prefers 1/3 and 2/3
    For max_cuts=3: prefers 1/4, 1/2, 3/4
    """
    if not cuts:
        return []

    if min_spacing is None:
        min_spacing = max(6, size // 12)

    # Generate ideal target positions based on number of desired cuts
    if max_cuts == 1:
        targets = [size // 2]
    elif max_cuts == 2:
        targets = [size // 3, (2 * size) // 3, size // 2]
    else:
        targets = [int(size * (i + 1) / (max_cuts + 1)) for i in range(max_cuts)]
        targets.append(size // 2)

    scored = sorted([(min(abs(c - t) for t in targets), c) for c in cuts])
    chosen = []
    for _, c in scored:
        if all(abs(c - x) > min_spacing for x in chosen):
            chosen.append(c)
        if len(chosen) == max_cuts:
            break
    return sorted(chosen)


def _is_reasonable_stamp_crop(x1: int, y1: int, x2: int, y2: int, max_ratio: float = 3.0) -> bool:
    """
    Check if a bounding box has reasonable proportions for a stamp.

    Stamps are typically roughly square or mildly rectangular.
    Very elongated strips (ratio > max_ratio) are likely false positives.
    """
    cw = x2 - x1
    ch = y2 - y1
    if cw < 40 or ch < 40:
        return False
    ratio = max(cw, ch) / max(min(cw, ch), 1)
    return ratio <= max_ratio


def _valley_widths(valleys: List[Tuple[int, int]]) -> List[int]:
    """Return the width of each valley."""
    return [(b - a + 1) for a, b in valleys]


def _best_valley_cut(valleys: List[Tuple[int, int]], size: int) -> Optional[int]:
    """
    Find the best single valley cut position near the center of the image.

    Only considers valleys within the middle 60% of the image (20%-80%).
    Returns the center of the widest qualifying valley, or None.
    """
    candidates = []
    for a, b in valleys:
        mid = (a + b) // 2
        if int(size * 0.20) <= mid <= int(size * 0.80):
            candidates.append((b - a + 1, mid))  # (width, center)
    if not candidates:
        return None
    # Return the cut from the widest valley
    candidates.sort(reverse=True)
    return candidates[0][1]


def _multi_pass_valleys(profile, size, axis_label="", desired_cuts: int = 0):
    """
    Run valley detection with multiple parameter settings, from strict to lenient.

    If desired_cuts > 0, will continue to more lenient passes until the desired
    number of cuts is found (or all passes are exhausted). Otherwise, returns
    the best result from any pass (preferring more cuts from stricter passes).
    """
    passes = [
        # (min_run_divisor, low_quantile, edge_frac, max_cuts)
        (60, 0.12, 0.15, 3),    # Strict pass
        (80, 0.14, 0.15, 3),    # Slightly lenient
        (100, 0.16, 0.13, 4),   # More lenient
        (130, 0.18, 0.12, 5),   # Even more lenient, allow more cuts
        (180, 0.22, 0.10, 5),   # Very lenient for touching stamps
    ]

    # Minimum spacing between cuts (prevent two cuts in the same gap)
    min_spacing = max(8, size // 8)

    best_cuts = []

    for divisor, quantile, edge_frac, max_cuts in passes:
        min_run = max(2, size // divisor)
        valleys = _find_valleys(profile, min_run=min_run, low_quantile=quantile)
        cut_positions = sorted({(a + b) // 2 for a, b in valleys})
        valid = _valid_cuts(cut_positions, size, edge_frac=edge_frac)
        picks = _pick_up_to_n(valid, size, max_cuts=max_cuts, min_spacing=min_spacing)

        # Keep the result with the most cuts
        if len(picks) > len(best_cuts):
            best_cuts = picks

        # If we've reached the desired number, stop
        if desired_cuts > 0 and len(best_cuts) >= desired_cuts:
            return best_cuts

    return best_cuts


def _brightness_valley_cuts(img: Image.Image, axis: str, size: int) -> List[int]:
    """
    Find cut positions using brightness profiling (complementary to ink profiling).

    Looks for bright columns/rows that indicate gaps between stamps.
    """
    prof = _brightness_profile(img, axis)

    passes = [
        (60, 0.88, 0.15, 2),
        (90, 0.85, 0.15, 3),
        (120, 0.82, 0.12, 3),
    ]

    for divisor, quantile, edge_frac, max_cuts in passes:
        min_run = max(2, size // divisor)
        valleys = _find_bright_valleys(prof, min_run=min_run, high_quantile=quantile)
        cut_positions = sorted({(a + b) // 2 for a, b in valleys})
        valid = _valid_cuts(cut_positions, size, edge_frac=edge_frac)
        picks = _pick_up_to_n(valid, size, max_cuts=max_cuts)
        if picks:
            return picks

    return []


def _gradient_cuts(img: Image.Image, axis: str, size: int) -> List[int]:
    """
    Find cut positions using gradient analysis (detects edges/borders).
    """
    try:
        prof = _gradient_profile(img, axis)
        peaks = _find_gradient_peaks(prof, size, min_edge_frac=0.15)
        if peaks:
            return _pick_up_to_n(peaks, size, max_cuts=2)
    except Exception:
        pass
    return []


def _make_grid_bboxes(v_cuts: List[int], h_cuts: List[int], w: int, h: int) -> List[Tuple[int, int, int, int]]:
    """Create grid bounding boxes from vertical and horizontal cut positions."""
    xs = [0] + sorted(v_cuts) + [w]
    ys = [0] + sorted(h_cuts) + [h]

    bboxes = []
    for row in range(len(ys) - 1):
        for col in range(len(xs) - 1):
            bbox = (xs[col], ys[row], xs[col + 1], ys[row + 1])
            bboxes.append(bbox)
    return bboxes


def _try_grid(v_cuts, h_cuts, w, h, expected_cols, expected_rows, max_ratio=2.5):
    """
    Try to form a grid with the given cuts and validate all tiles.

    Returns list of valid bboxes or empty list.
    """
    if len(v_cuts) < expected_cols - 1 or len(h_cuts) < expected_rows - 1:
        return []

    # Pick the best cuts for the expected grid
    use_v = _pick_up_to_n(v_cuts, w, max_cuts=expected_cols - 1)
    use_h = _pick_up_to_n(h_cuts, h, max_cuts=expected_rows - 1)

    if len(use_v) != expected_cols - 1 or len(use_h) != expected_rows - 1:
        return []

    bboxes = _make_grid_bboxes(use_v, use_h, w, h)
    valid = [b for b in bboxes if _is_reasonable_stamp_crop(*b, max_ratio=max_ratio)]

    expected_total = expected_cols * expected_rows
    # Accept if we got at least 75% of expected tiles
    if len(valid) >= max(expected_total * 3 // 4, 2):
        return valid

    return []


def _try_strips(cuts, w, h, axis="vertical", max_ratio=2.5):
    """
    Try to form strip bboxes from cuts.

    axis='vertical': cuts split image into side-by-side vertical strips
    axis='horizontal': cuts split image into stacked horizontal strips
    """
    if not cuts:
        return []

    if axis == "vertical":
        positions = [0] + sorted(cuts) + [w]
        bboxes = []
        for i in range(len(positions) - 1):
            bbox = (positions[i], 0, positions[i + 1], h)
            if _is_reasonable_stamp_crop(*bbox, max_ratio=max_ratio):
                bboxes.append(bbox)
        if len(bboxes) >= 2:
            return bboxes
    else:
        positions = [0] + sorted(cuts) + [h]
        bboxes = []
        for i in range(len(positions) - 1):
            bbox = (0, positions[i], w, positions[i + 1])
            if _is_reasonable_stamp_crop(*bbox, max_ratio=max_ratio):
                bboxes.append(bbox)
        if len(bboxes) >= 2:
            return bboxes

    return []


def _collect_all_cuts(img: Image.Image, axis: str, size: int, desired_cuts: int = 0) -> List[int]:
    """
    Collect cut candidates from ALL detection methods and merge them.

    Combines ink-profile valleys, brightness valleys, and gradient peaks
    to get the best possible set of cut positions.
    """
    # 1. Ink-profile multi-pass valleys
    ink_prof = _ink_profile(img, axis)
    ink_cuts = _multi_pass_valleys(ink_prof, size, axis, desired_cuts=desired_cuts)

    # 2. Brightness valleys
    bright_cuts = _brightness_valley_cuts(img, axis, size)

    # 3. Gradient peaks
    grad_cuts = []
    try:
        grad_cuts = _gradient_cuts(img, axis, size)
    except Exception:
        pass

    # Merge all unique candidates
    all_candidates = sorted(set(ink_cuts + bright_cuts + grad_cuts))

    if not all_candidates:
        return []

    # Cluster nearby candidates (within size//15 pixels) and take the median
    min_gap = max(5, size // 15)
    clusters = []
    current_cluster = [all_candidates[0]]

    for c in all_candidates[1:]:
        if c - current_cluster[-1] <= min_gap:
            current_cluster.append(c)
        else:
            clusters.append(current_cluster)
            current_cluster = [c]
    clusters.append(current_cluster)

    # Take median of each cluster
    merged = [sorted(cl)[len(cl) // 2] for cl in clusters]

    # Filter edges
    merged = _valid_cuts(merged, size, edge_frac=0.10)

    return merged


def _valley_detect(img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detect stamp bounding boxes using multi-strategy analysis.

    Strategies (tried in order of confidence):
    1. NxM grid detection (2x2, 3x3, 3x2, 2x3) for roughly square images
    2. Vertical strip split (2-up, 3-up, 4-up) for wide images
    3. Horizontal strip split for tall images
    4. Equal division by aspect ratio (last resort)

    Each detection method combines ink-profile, brightness, and gradient analysis.
    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    w, h = img.size

    if w < 80 or h < 50:
        return []

    aspect_ratio = w / h if h > 0 else 999

    # Estimate expected stamps per axis based on aspect ratio
    # This guides how many cuts to look for
    if aspect_ratio >= 3.5:
        expected_v = 3  # 4 strips need 3 cuts
    elif aspect_ratio >= 2.2:
        expected_v = 2  # 3 strips need 2 cuts
    elif aspect_ratio >= 1.4:
        expected_v = 1  # 2 strips need 1 cut
    else:
        expected_v = 2  # square-ish: could be 3x3 grid

    if aspect_ratio <= 0.3:
        expected_h = 3
    elif aspect_ratio <= 0.5:
        expected_h = 2
    elif aspect_ratio <= 0.7:
        expected_h = 1
    else:
        expected_h = 2  # square-ish: could be 3x3 grid

    # ── Collect cuts from all methods ──
    v_cuts = _collect_all_cuts(img, "x", w, desired_cuts=expected_v)
    h_cuts = _collect_all_cuts(img, "y", h, desired_cuts=expected_h)

    # Minimum spacing for picks
    min_v_spacing = max(8, w // 8)
    min_h_spacing = max(8, h // 8)

    # ── Strategy 1: NxM grid detection ──
    # For images that are roughly square-ish, try grid layouts.
    # Narrowed range to 0.65-1.35 to avoid false grids on clearly rectangular images.
    if 0.65 <= aspect_ratio <= 1.35 and w >= 180 and h >= 140:
        # Try 3x3 grid first (needs 2 v-cuts and 2 h-cuts)
        if len(v_cuts) >= 2 and len(h_cuts) >= 2 and 0.75 <= aspect_ratio <= 1.25:
            v_picked = _pick_up_to_n(v_cuts, w, max_cuts=2, min_spacing=min_v_spacing)
            h_picked = _pick_up_to_n(h_cuts, h, max_cuts=2, min_spacing=min_h_spacing)
            if len(v_picked) == 2 and len(h_picked) == 2:
                result = _try_grid(v_picked, h_picked, w, h, 3, 3, max_ratio=2.0)
                if result:
                    return result

        # Try 2x2 grid - only for truly square-ish images (0.75-1.25)
        if len(v_cuts) >= 1 and len(h_cuts) >= 1 and 0.75 <= aspect_ratio <= 1.25:
            v_center = [c for c in v_cuts if int(w * 0.30) <= c <= int(w * 0.70)]
            h_center = [c for c in h_cuts if int(h * 0.30) <= c <= int(h * 0.70)]
            if v_center and h_center:
                v_picked = _pick_up_to_n(v_center, w, max_cuts=1, min_spacing=min_v_spacing)
                h_picked = _pick_up_to_n(h_center, h, max_cuts=1, min_spacing=min_h_spacing)
                if v_picked and h_picked:
                    result = _try_grid(v_picked, h_picked, w, h, 2, 2, max_ratio=2.5)
                    if result:
                        return result

        # Try 3x2 (3 columns, 2 rows) for slightly wider images
        if 1.25 <= aspect_ratio <= 1.35 and len(v_cuts) >= 2 and len(h_cuts) >= 1:
            v_picked = _pick_up_to_n(v_cuts, w, max_cuts=2, min_spacing=min_v_spacing)
            h_picked = _pick_up_to_n(h_cuts, h, max_cuts=1, min_spacing=min_h_spacing)
            if len(v_picked) == 2 and len(h_picked) == 1:
                result = _try_grid(v_picked, h_picked, w, h, 3, 2, max_ratio=2.5)
                if result:
                    return result

        # Try 2x3 (2 columns, 3 rows) for slightly taller images
        if 0.65 <= aspect_ratio <= 0.75 and len(v_cuts) >= 1 and len(h_cuts) >= 2:
            v_picked = _pick_up_to_n(v_cuts, w, max_cuts=1, min_spacing=min_v_spacing)
            h_picked = _pick_up_to_n(h_cuts, h, max_cuts=2, min_spacing=min_h_spacing)
            if len(v_picked) == 1 and len(h_picked) == 2:
                result = _try_grid(v_picked, h_picked, w, h, 2, 3, max_ratio=2.5)
                if result:
                    return result

    # ── Strategy 2: Vertical strip split (2-up, 3-up, 4-up) ──
    # For wide images with stamps side-by-side.
    # Lowered threshold from 1.4 to 1.35 to catch borderline cases like Saint Helena (1.45)
    if v_cuts and aspect_ratio >= 1.35:
        # Determine how many strips based on aspect ratio
        if aspect_ratio >= 3.5:
            target_cuts = 3  # 4 stamps
        elif aspect_ratio >= 2.2:
            target_cuts = 2  # 3 stamps
        else:
            target_cuts = 1  # 2 stamps

        v_picked = _pick_up_to_n(v_cuts, w, max_cuts=target_cuts, min_spacing=min_v_spacing)
        if v_picked:
            result = _try_strips(v_picked, w, h, axis="vertical", max_ratio=2.8)
            if result:
                return result

        # If we didn't get enough cuts for the target, try with fewer
        if not v_picked or len(v_picked) < target_cuts:
            for tc in range(target_cuts - 1, 0, -1):
                v_picked = _pick_up_to_n(v_cuts, w, max_cuts=tc, min_spacing=min_v_spacing)
                if v_picked:
                    result = _try_strips(v_picked, w, h, axis="vertical", max_ratio=2.8)
                    if result:
                        return result

    # ── Strategy 3: Horizontal strip split (stacked stamps) ──
    if h_cuts and aspect_ratio <= 0.75:
        if aspect_ratio <= 0.3:
            target_cuts = 3
        elif aspect_ratio <= 0.5:
            target_cuts = 2
        else:
            target_cuts = 1

        h_picked = _pick_up_to_n(h_cuts, h, max_cuts=target_cuts, min_spacing=min_h_spacing)
        if h_picked:
            result = _try_strips(h_picked, w, h, axis="horizontal", max_ratio=2.8)
            if result:
                return result

        if not h_picked or len(h_picked) < target_cuts:
            for tc in range(target_cuts - 1, 0, -1):
                h_picked = _pick_up_to_n(h_cuts, h, max_cuts=tc, min_spacing=min_h_spacing)
                if h_picked:
                    result = _try_strips(h_picked, w, h, axis="horizontal", max_ratio=2.8)
                    if result:
                        return result

    # ── Strategy 4: Aspect-ratio equal division (last resort) ──
    # For wide/tall images where all detection methods fail (stamps touching
    # with no detectable gap). Divide equally assuming stamps are roughly square.

    # Wide images: divide into vertical strips
    if aspect_ratio >= 1.9 and w >= 150 and h >= 50:
        n_stamps = max(2, min(6, round(aspect_ratio)))
        strip_w = w / n_stamps
        if strip_w >= 40 and (max(strip_w, h) / max(min(strip_w, h), 1)) <= 3.0:
            bboxes = []
            for i in range(n_stamps):
                x1 = int(i * strip_w)
                x2 = int((i + 1) * strip_w) if i < n_stamps - 1 else w
                bboxes.append((x1, 0, x2, h))
            return bboxes

    # Tall images: divide into horizontal strips
    if aspect_ratio <= 0.5 and h >= 150 and w >= 50:
        inv_ratio = h / w if w > 0 else 999
        n_stamps = max(2, min(6, round(inv_ratio)))
        strip_h = h / n_stamps
        if strip_h >= 40 and (max(w, strip_h) / max(min(w, strip_h), 1)) <= 3.0:
            bboxes = []
            for i in range(n_stamps):
                y1 = int(i * strip_h)
                y2 = int((i + 1) * strip_h) if i < n_stamps - 1 else h
                bboxes.append((0, y1, w, y2))
            return bboxes

    return []


class StampDetector:
    """
    YOLO-based stamp detector for finding and extracting individual stamps.

    Uses a trained YOLOv8 model to detect stamps in images. The model can
    identify both single stamps and multi-stamp images (2-up, 3-up, grids, etc.)

    Example:
        detector = StampDetector()

        # Get cropped images
        crops = detector.detect_and_crop("multi_stamp.jpg")

        # Or save to files
        paths = detector.detect_and_save("multi_stamp.jpg", output_dir="output/")
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        use_valley_fallback: bool = False,  # Deprecated, kept for API compatibility
    ):
        """
        Initialize the stamp detector.

        Args:
            weights_path: Path to YOLO weights file. If None, uses default weights.
            confidence_threshold: Minimum confidence for detection (0.0-1.0).
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto).
            use_valley_fallback: Deprecated parameter, ignored. Kept for API compatibility.
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.use_valley_fallback = False  # Always disabled

        # Determine weights path
        if weights_path:
            self.weights_path = Path(weights_path)
        else:
            self.weights_path = DEFAULT_WEIGHTS

        # Check if model is available
        self._model_available = self.weights_path.exists()

        if self._model_available:
            self._load_model()
        else:
            print(f"[StampDetector] YOLO model not found at {self.weights_path}")
            print("[StampDetector] Run training first or provide weights path.")

    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
            if self.device:
                self.model.to(self.device)
            print(f"[StampDetector] YOLO model loaded from {self.weights_path}")
        except ImportError:
            print("[StampDetector] ultralytics not installed. Run: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"[StampDetector] Error loading YOLO model: {e}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        """Check if the detector is ready to use (YOLO model loaded)."""
        return self.model is not None

    def _detect_yolo(
        self,
        image: Union[str, Path, Image.Image],
    ) -> List[dict]:
        """Run YOLO-based detection. Returns list of detection dicts."""
        if self.model is None:
            return []

        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = self.model.names[cls]

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': class_name,
                    'method': 'yolo',
                })

        # Sort by y-row then x-position (left-to-right, top-to-bottom)
        detections.sort(key=lambda d: (d['bbox'][1] // 50, d['bbox'][0]))
        return detections

    def _detect_valley(
        self,
        image: Union[str, Path, Image.Image],
    ) -> List[dict]:
        """Run valley-based detection. Returns list of detection dicts."""
        if isinstance(image, (str, Path)):
            try:
                img = Image.open(image).convert("RGB")
            except Exception:
                return []
        else:
            img = image

        bboxes = _valley_detect(img)

        detections = []
        for bbox in bboxes:
            detections.append({
                'bbox': bbox,
                'confidence': 0.80,  # Assigned confidence for valley-based detections
                'class': 'stamp',
                'method': 'valley',
            })

        return detections

    def detect(
        self,
        image: Union[str, Path, Image.Image],
    ) -> List[dict]:
        """
        Detect stamps in an image using YOLO model.

        Args:
            image: Path to image file or PIL Image.

        Returns:
            List of detection dictionaries with keys:
                - 'bbox': (x1, y1, x2, y2) bounding box coordinates
                - 'confidence': detection confidence score
                - 'class': class name ('stamp')
                - 'method': 'yolo'
        """
        return self._detect_yolo(image)

    def detect_and_crop(
        self,
        image: Union[str, Path, Image.Image],
        padding: int = 0,
    ) -> List[Image.Image]:
        """
        Detect stamps and return cropped images.

        Args:
            image: Path to image file or PIL Image.
            padding: Extra pixels to add around each crop.

        Returns:
            List of cropped PIL Images for each detected stamp.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
            image_path = image
        else:
            img = image
            image_path = None

        detections = self.detect(image_path or img)

        if not detections:
            return []

        crops = []
        w, h = img.size

        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            # Ensure coordinates are in correct order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Apply padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            # Skip invalid crops
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crops.append(crop)

        return crops

    def detect_and_save(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        padding: int = 0,
        quality: int = 95,
    ) -> List[str]:
        """
        Detect stamps and save cropped images to files.

        Args:
            image_path: Path to input image.
            output_dir: Directory to save crops. If None, saves alongside original.
            padding: Extra pixels to add around each crop.
            quality: JPEG quality for saved images.

        Returns:
            List of paths to saved crop files.
        """
        image_path = Path(image_path)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = image_path.parent

        crops = self.detect_and_crop(image_path, padding=padding)

        if not crops:
            return []

        saved_paths = []
        stem = image_path.stem
        suffix = image_path.suffix or '.jpg'

        for idx, crop in enumerate(crops, start=1):
            crop_name = f"{stem}_stamp{idx}{suffix}"
            crop_path = output_dir / crop_name

            crop.save(str(crop_path), quality=quality)
            saved_paths.append(str(crop_path))

        return saved_paths

    def count_stamps(self, image: Union[str, Path, Image.Image]) -> int:
        """
        Count the number of stamps in an image.

        Args:
            image: Path to image file or PIL Image.

        Returns:
            Number of stamps detected.
        """
        detections = self.detect(image)
        return len(detections)

    def is_multi_stamp(self, image: Union[str, Path, Image.Image]) -> bool:
        """
        Check if an image contains multiple stamps.

        Args:
            image: Path to image file or PIL Image.

        Returns:
            True if more than one stamp detected.
        """
        return self.count_stamps(image) > 1


# Convenience function for quick detection without instantiating
_default_detector: Optional[StampDetector] = None


def detect_stamps(image_path: str, **kwargs) -> List[str]:
    """
    Convenience function to detect and save stamps from an image.

    Args:
        image_path: Path to input image.
        **kwargs: Additional arguments passed to detect_and_save().

    Returns:
        List of paths to saved stamp images, or empty list if no model/detections.
    """
    global _default_detector

    if _default_detector is None:
        _default_detector = StampDetector()

    if not _default_detector.is_ready:
        return []

    return _default_detector.detect_and_save(image_path, **kwargs)
