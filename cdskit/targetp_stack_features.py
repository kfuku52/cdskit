"""Sequence-only specialist features for TargetP stack training and inference."""

import numpy as np
from cdskit.localize_model import (
    AA_ACIDIC,
    AA_AROMATIC,
    AA_BASIC,
    AA_HYDROPHOBIC,
    AA_SER_THR,
    AA_SMALL,
    fraction_in_set,
    longest_hydrophobic_run,
    mean_hydropathy,
    to_canonical_aa_sequence,
)
from cdskit.targetp_feature_ensemble import build_targetp_feature_matrix


def _delayed_signal_peptide_scan_features(seq, cut_min=30, cut_max=120):
    seq = to_canonical_aa_sequence(seq)
    best_score = -99.0
    best_cut = 0
    best_parts = [0.0, 0.0, 0.0]
    last_cut = min(int(cut_max), len(seq) - 1)
    for cut in range(int(cut_min), last_cut):
        h_region = seq[max(0, cut - 22) : max(0, cut - 7)]
        c_region = seq[max(0, cut - 7) : cut + 2]
        m3 = seq[cut - 3] if cut >= 3 else "X"
        m1 = seq[cut - 1] if cut >= 1 else "X"
        hydrophobic_frac = fraction_in_set(h_region, AA_HYDROPHOBIC)
        hydrophobic_run = longest_hydrophobic_run(h_region)
        small_region_frac = fraction_in_set(c_region, AA_SMALL)
        has_proline_near_cut = "P" in seq[max(0, cut - 3) : cut + 1]
        score = (
            (2.2 * hydrophobic_frac)
            + (0.15 * hydrophobic_run)
            + (0.8 if m3 in "AVSGTC" else 0.0)
            + (1.0 if m1 in "ASGTC" else 0.0)
            + (0.5 * small_region_frac)
            - (0.9 if has_proline_near_cut else 0.0)
        )
        if score > best_score:
            best_score = float(score)
            best_cut = int(cut)
            best_parts = [
                float(hydrophobic_run),
                float(hydrophobic_frac),
                float(small_region_frac),
            ]
    return [
        float(best_score),
        float(best_cut),
        float(best_cut) / float(max(1, len(seq))),
        *best_parts,
    ]


def _targetp_ltp_signal_features(seq):
    seq = to_canonical_aa_sequence(seq)
    n_terminal = seq[:140]
    out = list()
    for start, stop in [(20, 80), (30, 100), (40, 120), (50, 140)]:
        window = seq[start:stop]
        out.extend(
            [
                mean_hydropathy(window),
                longest_hydrophobic_run(window),
                fraction_in_set(window, AA_HYDROPHOBIC),
                fraction_in_set(window, AA_BASIC),
                fraction_in_set(window, AA_ACIDIC),
                fraction_in_set(window, AA_SER_THR),
                fraction_in_set(window, AA_SMALL),
                fraction_in_set(window, AA_AROMATIC),
            ]
        )
    rr_positions = [
        pos
        for pos in range(max(0, len(n_terminal) - 1))
        if n_terminal[pos : pos + 2] == "RR"
    ]
    out.extend(
        [
            float(len(rr_positions)),
            float(rr_positions[0] if len(rr_positions) > 0 else 999),
            1.0 if any(20 <= pos < 90 for pos in rr_positions) else 0.0,
        ]
    )
    best_after_rr = [0.0, 0.0, 0.0]
    for pos in rr_positions:
        after = n_terminal[pos + 2 : pos + 42]
        values = [
            longest_hydrophobic_run(after),
            mean_hydropathy(after),
            fraction_in_set(after, AA_HYDROPHOBIC),
        ]
        if values[0] > best_after_rr[0]:
            best_after_rr = values
    out.extend([float(value) for value in best_after_rr])
    out.extend(_delayed_signal_peptide_scan_features(seq, cut_min=30, cut_max=120))
    out.extend(_delayed_signal_peptide_scan_features(seq, cut_min=45, cut_max=140))
    return np.asarray(out, dtype=np.float32)


def build_ltp_ctp_specialist_feature_matrix(rows):
    base = build_targetp_feature_matrix(rows=rows).astype(np.float32)
    extra = [_targetp_ltp_signal_features(row.get("sequence", "")) for row in rows]
    if len(extra) == 0:
        return base
    return np.hstack([base, np.vstack(extra).astype(np.float32)]).astype(np.float32)
