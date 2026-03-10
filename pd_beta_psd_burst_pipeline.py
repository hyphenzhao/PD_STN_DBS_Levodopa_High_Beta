"""
Reusable PSD and beta-burst pipeline extracted and cleaned from the notebooks:
- Gait_Epochs_LFP_v3_Gait-Copy1.ipynb
- Gait_Epochs_LFP_v3_DBS-Copy1.ipynb

What this file captures
-----------------------
1. PSD calculation for levodopa and DBS comparisons during continuous sitting/standing:
   - Welch PSD on up to 60 s of continuous data
   - band summary from z-scored spectra (within 1-200 Hz)
   - low beta = 12-20 Hz, high beta = 20-30 Hz

2. Beta burst occupancy for levodopa and DBS comparisons during continuous sitting/standing:
   - band-pass each segment in the beta sub-band of interest
   - z-score each segment across time
   - compute Hilbert amplitude envelope
   - define one pooled, condition-agnostic threshold per subject x hemisphere x band
     from the concatenated envelopes across the compared conditions/postures
   - threshold = 75th percentile of the pooled envelope distribution
   - keep bursts lasting >= 100 ms
   - burst occupancy = valid burst samples / total samples * 100

The script is generic: you provide broadband 1D signals per condition and posture.
That keeps the methods portable while still matching the notebook logic.

Typical use
-----------
segment_dict = {
    "Med-Off": {"Sit": sit_off_signal, "Stand": stand_off_signal},
    "Med-On":  {"Sit": sit_on_signal,  "Stand": stand_on_signal},
}
med_df = compute_levodopa_metrics(segment_dict, sfreq=512.0)

segment_dict = {
    "Med-Off": {"Sit": sit_off_signal, "Stand": stand_off_signal},
    "DBS":     {"Sit": sit_dbs_signal, "Stand": stand_dbs_signal},
}
dbs_df = compute_dbs_metrics(segment_dict, sfreq=512.0)

You can also use the MNE helper functions below to extract the 1D signals from FIF raws.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import hilbert, welch
from scipy.stats import zscore

try:
    import mne
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script expects MNE-Python to be installed because the notebooks used MNE filtering."
    ) from exc


BETA_BANDS: Dict[str, Tuple[float, float]] = {
    "Low-Beta": (12.0, 20.0),
    "High-Beta": (20.0, 30.0),
    "Beta": (12.0, 30.0),
}


@dataclass
class BurstMetrics:
    band: str
    threshold: float
    occupancy_pct: float
    total_burst_s: float
    n_valid_bursts: int
    avg_duration_valid_s: float
    avg_duration_all_s: float


@dataclass
class SegmentMetrics:
    condition: str
    gesture: str
    band: str
    psd_mean_z: float
    burst_threshold: float
    burst_occupancy_pct: float
    total_burst_s: float
    n_valid_bursts: int
    avg_burst_duration_valid_s: float
    avg_burst_duration_all_s: float


def rename_stn_channels(
    raw: "mne.io.BaseRaw",
    lfp_channels_csv: str,
    locations_csv: str,
    *,
    source_suffix: str = "_b",
    left_name: str = "STN-L",
    right_name: str = "STN-R",
) -> "mne.io.BaseRaw":
    """
    Rename STN bipolar channels using the same notebook convention.

    Parameters
    ----------
    raw
        Preprocessed raw MNE object.
    lfp_channels_csv
        Comma-separated channel IDs from the metadata table.
    locations_csv
        Comma-separated locations from the metadata table.
    source_suffix
        Notebook convention used '<channel>_b' for the bipolar LFP channel.
    """
    lfp_channels = [x.strip() for x in lfp_channels_csv.split(",")]
    locations = [x.strip() for x in locations_csv.split("/")]
    stn_channels = [ch for ch, loc in zip(lfp_channels, locations) if loc == "STN"]

    rename_map: Dict[str, str] = {}
    for ch in stn_channels:
        src = f"{ch}{source_suffix}"
        if src not in raw.ch_names:
            continue
        if "L" in ch:
            rename_map[src] = left_name
        elif "R" in ch:
            rename_map[src] = right_name

    if rename_map:
        raw = raw.copy().rename_channels(rename_map)
    return raw


def extract_segment_from_raw(
    raw: "mne.io.BaseRaw",
    pick: str,
    tmin: float,
    tmax: float,
    *,
    max_duration_s: float = 60.0,
) -> np.ndarray:
    """Extract one 1D continuous segment from raw, capped at 60 s like the notebooks."""
    seg_tmax = min(float(tmax), float(tmin) + float(max_duration_s))
    data = raw.copy().crop(float(tmin), seg_tmax).get_data(picks=pick)
    if data.size == 0:
        raise ValueError(f"No data returned for pick={pick!r}, tmin={tmin}, tmax={seg_tmax}.")
    return np.asarray(data).squeeze().astype(float)




def parse_time_window(window: str | Sequence[float]) -> Tuple[float, float]:
    """Parse a metadata-style time window such as '12.5-85.0'."""
    if isinstance(window, str):
        left, right = window.split('-')
        return float(left), float(right)
    vals = list(window)
    if len(vals) != 2:
        raise ValueError(f"Expected 2 values for a time window, got {window!r}.")
    return float(vals[0]), float(vals[1])


def build_segment_dict_from_raws(
    raws_by_condition: Mapping[str, "mne.io.BaseRaw"],
    *,
    pick: str,
    gesture_windows: Mapping[str, str | Sequence[float]],
    gestures: Sequence[str] = ("Sit", "Stand"),
    max_duration_s: float = 60.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convenience helper to build the nested input expected by compute_*_metrics.

    Example
    -------
    raws_by_condition = {"Med-Off": raw_off, "Med-On": raw_on}
    gesture_windows = {"Sit": row["Sit Time"], "Stand": row["Stand Time"]}
    segment_dict = build_segment_dict_from_raws(
        raws_by_condition,
        pick="STN-L",
        gesture_windows=gesture_windows,
    )
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for condition, raw in raws_by_condition.items():
        out[condition] = {}
        for gesture in gestures:
            if gesture not in gesture_windows:
                continue
            tmin, tmax = parse_time_window(gesture_windows[gesture])
            out[condition][gesture] = extract_segment_from_raw(
                raw,
                pick=pick,
                tmin=tmin,
                tmax=tmax,
                max_duration_s=max_duration_s,
            )
    return out

def apply_line_notches(
    raw: "mne.io.BaseRaw",
    *,
    line_harmonics: Sequence[float] = (50.0, 100.0, 150.0, 200.0, 250.0),
    dbs_hz: Optional[float] = None,
    emg_only_for_line: bool = False,
) -> "mne.io.BaseRaw":
    """
    Optional helper mirroring notebook notch choices.

    - levodopa notebook: 50 Hz harmonics on EMG channels
    - DBS notebook: 130 Hz on raw + 50 Hz harmonics on EMG channels
    """
    out = raw.copy()
    if dbs_hz is not None:
        out = out.notch_filter([float(dbs_hz)], notch_widths=4.0, verbose=False)
    if line_harmonics:
        picks = "emg" if emg_only_for_line else None
        out = out.notch_filter(np.asarray(line_harmonics, dtype=float), notch_widths=4.0, picks=picks, verbose=False)
    return out


def _zscore_1d(signal_1d: np.ndarray) -> np.ndarray:
    signal_1d = np.asarray(signal_1d, dtype=float).squeeze()
    if signal_1d.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal_1d.shape}.")
    out = zscore(signal_1d, nan_policy="omit")
    if np.isnan(out).all():
        raise ValueError("Signal z-score became all-NaN. Check for constant/empty input.")
    return np.nan_to_num(out)


def bandpass_signal(
    signal_1d: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Band-pass a 1D signal with the same MNE filtering family used in the notebooks."""
    signal_1d = np.asarray(signal_1d, dtype=float).squeeze()
    return mne.filter.filter_data(
        signal_1d,
        sfreq=float(sfreq),
        l_freq=float(band[0]),
        h_freq=float(band[1]),
        verbose=False,
    )


def compute_welch_psd(
    signal_1d: np.ndarray,
    sfreq: float,
    *,
    n_fft: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD corresponding to the notebook use of Raw.compute_psd(..., n_fft=2048).

    Returns
    -------
    freqs, psd
    """
    signal_1d = np.asarray(signal_1d, dtype=float).squeeze()
    freqs, psd = welch(
        signal_1d,
        fs=float(sfreq),
        nperseg=min(int(n_fft), signal_1d.size),
        nfft=int(n_fft),
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def summarize_band_psd(
    signal_1d: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    *,
    n_fft: int = 2048,
    standardize: bool = True,
    standardize_range: Tuple[float, float] = (1.0, 200.0),
) -> float:
    """
    Replicates the band-summary logic used for statistics:
    z-score each spectrum across 1-200 Hz, then average within the band.
    """
    freqs, psd = compute_welch_psd(signal_1d, sfreq=sfreq, n_fft=n_fft)

    norm_mask = (freqs >= standardize_range[0]) & (freqs <= standardize_range[1])
    if not np.any(norm_mask):
        raise ValueError("No PSD bins found inside the requested standardization range.")

    freqs_sel = freqs[norm_mask]
    psd_sel = psd[norm_mask]
    if standardize:
        psd_sel = zscore(psd_sel, nan_policy="omit")
        psd_sel = np.nan_to_num(psd_sel)

    band_mask = (freqs_sel >= band[0]) & (freqs_sel <= band[1])
    if not np.any(band_mask):
        raise ValueError(f"No PSD bins found for band {band}.")
    return float(np.nanmean(psd_sel[band_mask]))


def _find_burst_runs(binary_mask: np.ndarray) -> Sequence[Tuple[int, int]]:
    """Return half-open [start, stop) runs of True values."""
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if binary_mask.ndim != 1:
        raise ValueError("Burst mask must be 1D.")

    padded = np.pad(binary_mask.astype(int), (1, 1), mode="constant")
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]
    return list(zip(starts, stops))


def get_above_threshold_duration(
    env: np.ndarray,
    thresh: float,
    fs: float,
    *,
    min_t: float = 0.1,
) -> Tuple[Sequence[Tuple[int, int]], float, float, Sequence[Tuple[int, int]]]:
    """
    Cleaned version of the helper in the notebooks.

    Returns
    -------
    all_bursts
        All threshold crossings.
    avg_duration_all_s
        Mean duration over all threshold crossings.
    total_valid_burst_s
        Sum of burst durations after applying the >= min_t filter.
    valid_bursts
        Threshold crossings surviving the duration filter.
    """
    env = np.asarray(env, dtype=float).squeeze()
    min_samples = int(round(float(min_t) * float(fs)))
    all_bursts = _find_burst_runs(env > float(thresh))
    valid_bursts = [(start, stop) for start, stop in all_bursts if (stop - start) >= min_samples]

    if all_bursts:
        avg_duration_all_s = float(np.mean([(stop - start) / float(fs) for start, stop in all_bursts]))
    else:
        avg_duration_all_s = 0.0

    total_valid_burst_s = float(sum((stop - start) for start, stop in valid_bursts) / float(fs))
    return all_bursts, avg_duration_all_s, total_valid_burst_s, valid_bursts


def compute_burst_metrics(
    signal_1d: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    pooled_reference_signals: Sequence[np.ndarray],
    *,
    threshold_percentile: float = 75.0,
    min_burst_s: float = 0.1,
) -> BurstMetrics:
    """
    Compute burst occupancy using the pooled-threshold logic from the notebooks.

    Important notebook detail preserved here:
    - each segment is z-scored over time before the Hilbert envelope is computed
    - the threshold is pooled across the compared conditions/postures for the same
      subject x hemisphere x band
    """
    if len(pooled_reference_signals) == 0:
        raise ValueError("pooled_reference_signals is empty.")

    bandpassed = bandpass_signal(signal_1d, sfreq=sfreq, band=band)
    bandpassed_z = _zscore_1d(bandpassed)
    env = np.abs(hilbert(bandpassed_z))

    pooled_envs = []
    for ref in pooled_reference_signals:
        ref_band = bandpass_signal(ref, sfreq=sfreq, band=band)
        ref_band_z = _zscore_1d(ref_band)
        pooled_envs.append(np.abs(hilbert(ref_band_z)))
    pooled_env = np.concatenate(pooled_envs, axis=0)

    threshold = float(np.percentile(pooled_env, float(threshold_percentile)))
    all_bursts, avg_duration_all_s, total_valid_burst_s, valid_bursts = get_above_threshold_duration(
        env,
        thresh=threshold,
        fs=sfreq,
        min_t=min_burst_s,
    )

    if valid_bursts:
        avg_duration_valid_s = float(
            np.mean([(stop - start) / float(sfreq) for start, stop in valid_bursts])
        )
    else:
        avg_duration_valid_s = 0.0

    occupancy_pct = 100.0 * total_valid_burst_s * float(sfreq) / float(len(signal_1d))
    return BurstMetrics(
        band=f"{band[0]}-{band[1]} Hz",
        threshold=threshold,
        occupancy_pct=float(occupancy_pct),
        total_burst_s=total_valid_burst_s,
        n_valid_bursts=len(valid_bursts),
        avg_duration_valid_s=avg_duration_valid_s,
        avg_duration_all_s=avg_duration_all_s,
    )


def compute_comparison_metrics(
    segment_dict: Mapping[str, Mapping[str, np.ndarray]],
    sfreq: float,
    *,
    comparison_conditions: Sequence[str],
    gestures: Sequence[str] = ("Sit", "Stand"),
    bands: Mapping[str, Tuple[float, float]] = BETA_BANDS,
    band_order: Sequence[str] = ("Low-Beta", "High-Beta"),
    n_fft: int = 2048,
    threshold_percentile: float = 75.0,
    min_burst_s: float = 0.1,
    standardize_psd: bool = True,
    standardize_range: Tuple[float, float] = (1.0, 200.0),
) -> pd.DataFrame:
    """
    Core method for both levodopa and DBS comparisons.

    Parameters
    ----------
    segment_dict
        Nested dict like:
        {
            "Med-Off": {"Sit": arr, "Stand": arr},
            "Med-On":  {"Sit": arr, "Stand": arr},
        }

    Returns
    -------
    DataFrame with one row per condition x gesture x band.
    """
    rows = []

    for band_name in band_order:
        if band_name not in bands:
            raise KeyError(f"Band {band_name!r} not present in bands mapping.")
        band_limits = bands[band_name]

        pooled_reference_signals = []
        for condition in comparison_conditions:
            for gesture in gestures:
                signal = segment_dict.get(condition, {}).get(gesture, None)
                if signal is not None:
                    pooled_reference_signals.append(np.asarray(signal, dtype=float).squeeze())

        if len(pooled_reference_signals) == 0:
            continue

        for condition in comparison_conditions:
            for gesture in gestures:
                signal = segment_dict.get(condition, {}).get(gesture, None)
                if signal is None:
                    continue
                signal = np.asarray(signal, dtype=float).squeeze()

                psd_mean_z = summarize_band_psd(
                    signal,
                    sfreq=sfreq,
                    band=band_limits,
                    n_fft=n_fft,
                    standardize=standardize_psd,
                    standardize_range=standardize_range,
                )
                burst = compute_burst_metrics(
                    signal,
                    sfreq=sfreq,
                    band=band_limits,
                    pooled_reference_signals=pooled_reference_signals,
                    threshold_percentile=threshold_percentile,
                    min_burst_s=min_burst_s,
                )

                rows.append(
                    asdict(
                        SegmentMetrics(
                            condition=condition,
                            gesture=gesture,
                            band=band_name,
                            psd_mean_z=float(psd_mean_z),
                            burst_threshold=float(burst.threshold),
                            burst_occupancy_pct=float(burst.occupancy_pct),
                            total_burst_s=float(burst.total_burst_s),
                            n_valid_bursts=int(burst.n_valid_bursts),
                            avg_burst_duration_valid_s=float(burst.avg_duration_valid_s),
                            avg_burst_duration_all_s=float(burst.avg_duration_all_s),
                        )
                    )
                )

    return pd.DataFrame(rows)


def compute_levodopa_metrics(
    segment_dict: Mapping[str, Mapping[str, np.ndarray]],
    sfreq: float,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for Med-Off vs Med-On."""
    return compute_comparison_metrics(
        segment_dict,
        sfreq=sfreq,
        comparison_conditions=("Med-Off", "Med-On"),
        **kwargs,
    )


def compute_dbs_metrics(
    segment_dict: Mapping[str, Mapping[str, np.ndarray]],
    sfreq: float,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for Med-Off vs DBS."""
    return compute_comparison_metrics(
        segment_dict,
        sfreq=sfreq,
        comparison_conditions=("Med-Off", "DBS"),
        **kwargs,
    )


def metrics_to_wide_table(
    metrics_df: pd.DataFrame,
    *,
    prefix_dbs_psd: bool = True,
) -> pd.DataFrame:
    """
    Optional helper to reshape the long output into manuscript-style columns.

    Example columns:
    - PSD Sit STN Low-Beta
    - PSD Stand STN High-Beta
    - DBS PSD Sit STN Low-Beta
    - Low-Beta Burst Med-Off STN Sit Time
    """
    if metrics_df.empty:
        return pd.DataFrame()

    out: MutableMapping[str, float] = {}
    for _, row in metrics_df.iterrows():
        condition = str(row["condition"])
        gesture = str(row["gesture"])
        band = str(row["band"])

        if condition == "DBS" and prefix_dbs_psd:
            psd_col = f"DBS PSD {gesture} STN {band}"
        else:
            psd_col = f"PSD {gesture} STN {band}"
        burst_col = f"{band} Burst {condition} STN {gesture} Time"

        out[psd_col] = float(row["psd_mean_z"])
        out[burst_col] = float(row["burst_occupancy_pct"])

    return pd.DataFrame([out])


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(7)
    sfreq = 512.0
    n = int(60 * sfreq)

    # Tiny synthetic example showing the expected input format.
    levodopa_segments = {
        "Med-Off": {
            "Sit": rng.standard_normal(n),
            "Stand": rng.standard_normal(n),
        },
        "Med-On": {
            "Sit": rng.standard_normal(n),
            "Stand": rng.standard_normal(n),
        },
    }

    dbs_segments = {
        "Med-Off": {
            "Sit": rng.standard_normal(n),
            "Stand": rng.standard_normal(n),
        },
        "DBS": {
            "Sit": rng.standard_normal(n),
            "Stand": rng.standard_normal(n),
        },
    }

    print("Levodopa metrics example")
    print(compute_levodopa_metrics(levodopa_segments, sfreq=sfreq))
    print()
    print("DBS metrics example")
    print(compute_dbs_metrics(dbs_segments, sfreq=sfreq))
