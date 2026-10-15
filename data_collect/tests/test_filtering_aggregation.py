from rt_dataset.aggregation import AggregationConfig, aggregate_identifications
from rt_dataset.config import FilterConfig
from rt_dataset.filtering import RTDatasetFilter
from rt_dataset.schemas import assign_stable_ids, harmonize_schema


def test_qvalue_and_decoy_filter(identifications):
    data = assign_stable_ids(harmonize_schema(identifications))
    filtered, report = RTDatasetFilter().apply(
        data,
        FilterConfig(maximum_q_value=0.01, exclude_decoys=True),
    )
    assert len(filtered) == 2
    assert report["summary"]["rows"] == 2


def test_two_stage_median_aggregation(identifications):
    data = assign_stable_ids(harmonize_schema(identifications))
    result = aggregate_identifications(
        data,
        AggregationConfig(level="peptidoform", method="median", within_raw_file_first=True),
    )
    peptide = result.loc[result["sequence"] == "PEPTIDE"].iloc[0]
    assert peptide["observed_rt"] == 11.0
    assert peptide["n_psms"] == 2
