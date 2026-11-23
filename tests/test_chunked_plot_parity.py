"""Static plot entry points render equivalent eager and chunked inputs."""

import pytest

from benchmarks import check_chunked_plot_parity as parity


def test_plot_parity_inventory():
    parity.suite.check_inventory()


@pytest.mark.parametrize("case", parity.CASES)
def test_static_plot_chunked_parity(case):
    pytest.importorskip("dask.dataframe")
    result = parity.compare(case)
    expected = "no-data-input" if case in parity.NO_DATA_INPUT else "ok"
    assert result["status"] == expected, result
