from pathlib import Path
import sys

import pandas as pd
import pytest

from rt_dataset.rt_conversion import ChronologerHITransformer, IRTTransformer


def test_linear_irt_calibration():
    calibration = pd.DataFrame(
        {
            "raw_file_id": ["run"] * 4,
            "observed_rt": [10.0, 20.0, 30.0, 40.0],
            "reference_irt": [-50.0, 0.0, 50.0, 100.0],
        }
    )
    transformer = IRTTransformer(minimum_anchors=3).fit(calibration)
    result = transformer.transform(calibration)
    assert result["observed_irt"].tolist() == pytest.approx(calibration["reference_irt"].tolist())
    inverse = transformer.inverse_transform(result)
    assert inverse["reconstructed_rt"].tolist() == pytest.approx(calibration["observed_rt"].tolist())


def test_chronologer_subprocess_wrapper(tmp_path: Path):
    script = tmp_path / "convert.py"
    script.write_text(
        "import argparse, pandas as pd\n"
        "p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output'); a=p.parse_args()\n"
        "d=pd.read_csv(a.input); d['observed_hi']=d['observed_rt']*2; d.to_csv(a.output,index=False)\n"
    )
    data = pd.DataFrame(
        {"dataset_row_hash": ["a", "b", "c"], "observed_rt": [1.0, 2.0, 3.0]}
    )
    result = ChronologerHITransformer(script, minimum_rows=3, python_executable=sys.executable).fit(data).transform(data)
    assert result["observed_hi"].tolist() == [2.0, 4.0, 6.0]
    assert result["observed_rt"].tolist() == [1.0, 2.0, 3.0]
