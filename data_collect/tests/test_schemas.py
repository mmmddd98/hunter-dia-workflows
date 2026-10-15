import pandas as pd

from rt_dataset.schemas import assign_stable_ids, harmonize_schema


def test_deterministic_ids_ignore_input_order(identifications):
    first = assign_stable_ids(harmonize_schema(identifications))
    second = assign_stable_ids(harmonize_schema(identifications.sample(frac=1, random_state=4)))
    assert first["dataset_row_hash"].tolist() == second["dataset_row_hash"].tolist()
    assert first["precursor_id"].tolist() == second["precursor_id"].tolist()


def test_unknown_columns_are_preserved():
    frame = pd.DataFrame({"sequence": ["PEPTIDE"], "unexpected": ["value"]})
    result = harmonize_schema(frame)
    assert '"unexpected": "value"' in result.loc[0, "source_metadata"]
