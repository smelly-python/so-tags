# test_with_pytest.py

import os
import pandas as pd
import tensorflow_data_validation as tfdv


def test_one_plus_one():
    assert (1 + 1 == 2)


def test_data_validation():
    for filename in [os.path.dirname(__file__) + "/../data/train.tsv", os.path.dirname(__file__) + "/../data/test.tsv"]:
        data = pd.read_csv(filename, sep='\t')
        stats = tfdv.generate_statistics_from_dataframe(data)
        schema = tfdv.infer_schema(stats)
        anomalies = tfdv.validate_statistics(stats, schema=schema)
        assert len(anomalies.anomaly_info) == 0
