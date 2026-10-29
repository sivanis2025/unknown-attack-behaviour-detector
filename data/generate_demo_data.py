"""
Generate a synthetic demonstration dataset for the Unknown Attack Behaviour Detector.

THIS DATASET IS ENTIRELY SYNTHETIC. It is NOT real network traffic.
It exists only to allow the application to run immediately without requiring
an external dataset.

Run this script to regenerate the demo data:
    python data/generate_demo_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RANDOM_SEED = 42
N_NORMAL = 400
N_ATTACK = 100  # 20% anomaly rate

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "demo_dataset.csv")


def generate():
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Normal traffic (benign web / email / DNS flows) ---
    normal = pd.DataFrame(
        {
            "flow_duration": rng.integers(50_000, 2_000_000, N_NORMAL),
            "total_fwd_packets": rng.integers(2, 40, N_NORMAL),
            "total_bwd_packets": rng.integers(2, 35, N_NORMAL),
            "fwd_bytes": rng.integers(100, 20_000, N_NORMAL),
            "bwd_bytes": rng.integers(100, 18_000, N_NORMAL),
            "fwd_pkt_len_mean": rng.uniform(50, 1200, N_NORMAL),
            "bwd_pkt_len_mean": rng.uniform(50, 1200, N_NORMAL),
            "flow_bytes_per_sec": rng.uniform(500, 50_000, N_NORMAL),
            "flow_pkts_per_sec": rng.uniform(1, 200, N_NORMAL),
            "fwd_iat_mean": rng.integers(1_000, 500_000, N_NORMAL),
            "bwd_iat_mean": rng.integers(1_000, 500_000, N_NORMAL),
            "syn_flag": rng.integers(0, 2, N_NORMAL),
            "ack_flag": rng.integers(0, 2, N_NORMAL),
            "psh_flag": rng.integers(0, 2, N_NORMAL),
            "rst_flag": rng.integers(0, 1, N_NORMAL),   # rarely reset
            "fin_flag": rng.integers(0, 2, N_NORMAL),
            "Label": "BENIGN",
        }
    )

    # --- Anomalous / attack-like traffic ---
    # Mix of port-scan-like, flood-like, and exfiltration-like synthetic patterns.

    # Port-scan-like (very short flows, many SYN, minimal data)
    n_scan = N_ATTACK // 3
    scans = pd.DataFrame(
        {
            "flow_duration": rng.integers(100, 5_000, n_scan),
            "total_fwd_packets": rng.integers(1, 3, n_scan),
            "total_bwd_packets": rng.integers(0, 1, n_scan),
            "fwd_bytes": rng.integers(40, 120, n_scan),
            "bwd_bytes": rng.integers(0, 60, n_scan),
            "fwd_pkt_len_mean": rng.uniform(40, 80, n_scan),
            "bwd_pkt_len_mean": rng.uniform(0, 60, n_scan),
            "flow_bytes_per_sec": rng.uniform(8_000, 120_000, n_scan),
            "flow_pkts_per_sec": rng.uniform(200, 2_000, n_scan),
            "fwd_iat_mean": rng.integers(50, 2_000, n_scan),
            "bwd_iat_mean": rng.integers(0, 1_000, n_scan),
            "syn_flag": 1,
            "ack_flag": rng.integers(0, 1, n_scan),
            "psh_flag": 0,
            "rst_flag": rng.integers(0, 2, n_scan),
            "fin_flag": 0,
            "Label": "Anomaly-ScanLike",
        }
    )

    # Flood-like (huge packet counts, high rates, tiny payloads)
    n_flood = N_ATTACK // 3
    floods = pd.DataFrame(
        {
            "flow_duration": rng.integers(100_000, 500_000, n_flood),
            "total_fwd_packets": rng.integers(500, 5_000, n_flood),
            "total_bwd_packets": rng.integers(0, 10, n_flood),
            "fwd_bytes": rng.integers(20_000, 200_000, n_flood),
            "bwd_bytes": rng.integers(0, 500, n_flood),
            "fwd_pkt_len_mean": rng.uniform(40, 80, n_flood),
            "bwd_pkt_len_mean": rng.uniform(0, 40, n_flood),
            "flow_bytes_per_sec": rng.uniform(500_000, 5_000_000, n_flood),
            "flow_pkts_per_sec": rng.uniform(5_000, 50_000, n_flood),
            "fwd_iat_mean": rng.integers(10, 500, n_flood),
            "bwd_iat_mean": rng.integers(0, 200, n_flood),
            "syn_flag": rng.integers(0, 2, n_flood),
            "ack_flag": rng.integers(0, 2, n_flood),
            "psh_flag": rng.integers(0, 2, n_flood),
            "rst_flag": 0,
            "fin_flag": 0,
            "Label": "Anomaly-FloodLike",
        }
    )

    # Exfiltration-like (long duration, large outgoing bytes, few packets)
    n_exfil = N_ATTACK - n_scan - n_flood
    exfils = pd.DataFrame(
        {
            "flow_duration": rng.integers(5_000_000, 30_000_000, n_exfil),
            "total_fwd_packets": rng.integers(5, 20, n_exfil),
            "total_bwd_packets": rng.integers(1, 5, n_exfil),
            "fwd_bytes": rng.integers(50_000, 500_000, n_exfil),
            "bwd_bytes": rng.integers(100, 1_000, n_exfil),
            "fwd_pkt_len_mean": rng.uniform(5_000, 32_000, n_exfil),
            "bwd_pkt_len_mean": rng.uniform(50, 200, n_exfil),
            "flow_bytes_per_sec": rng.uniform(1_000, 10_000, n_exfil),
            "flow_pkts_per_sec": rng.uniform(0.1, 5, n_exfil),
            "fwd_iat_mean": rng.integers(500_000, 5_000_000, n_exfil),
            "bwd_iat_mean": rng.integers(1_000_000, 8_000_000, n_exfil),
            "syn_flag": 0,
            "ack_flag": 1,
            "psh_flag": rng.integers(0, 2, n_exfil),
            "rst_flag": 0,
            "fin_flag": rng.integers(0, 2, n_exfil),
            "Label": "Anomaly-ExfilLike",
        }
    )

    # Combine and shuffle
    df = pd.concat([normal, scans, floods, exfils], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Add a comment column marking data as synthetic
    df.insert(0, "record_id", df.index + 1)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Synthetic demo dataset written to: {OUTPUT_PATH}")
    print(f"       {N_NORMAL} normal + {N_ATTACK} anomaly-like = {N_NORMAL + N_ATTACK} total records.")
    print("       IMPORTANT: This is SYNTHETIC DATA — not real network traffic.")
    return df


if __name__ == "__main__":
    generate()
