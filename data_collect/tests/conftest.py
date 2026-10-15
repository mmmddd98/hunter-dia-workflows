from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def identifications() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": ["inputs/report.tsv"] * 4,
            "raw_file_name": ["run_a", "run_a", "run_b", "run_b"],
            "run_id": ["run_a", "run_a", "run_b", "run_b"],
            "sequence": ["PEPTIDE", "PEPTIDE", "SEQUENCE", "SEQUENCE"],
            "modified_sequence": ["PEPTIDE", "PEPTIDE", "SEQUENCE", "SEQUENCE"],
            "mods": ["", "", "Acetyl", "Acetyl"],
            "mod_sites": ["", "", "N-term", "N-term"],
            "charge": [2, 2, 3, 3],
            "observed_rt": [10.0, 12.0, 30.0, 32.0],
            "q_value": [0.001, 0.02, 0.002, 0.003],
            "pep": [0.001, 0.1, 0.002, 0.003],
            "intensity": [100.0, 10.0, 200.0, 100.0],
            "is_decoy": [False, False, False, True],
            "scan_number": [1, 2, 3, 4],
            "hunter_status": ["HUNTER", "HUNTER", "PRE_HUNTER", "PRE_HUNTER"],
        }
    )
