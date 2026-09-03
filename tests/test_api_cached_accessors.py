"""Read-only high-level API helpers must reuse an existing analysis."""

from fastpisa.api import PISAInterfaceAnalyzer

from conftest import KTZ


def test_read_only_accessors_do_not_recompute(monkeypatch):
    analyzer = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
    analyzer.analyze()

    def unexpected_recompute(*args, **kwargs):
        if kwargs.get("recompute") is False:
            return analyzer.result
        raise AssertionError("analysis was recomputed")

    monkeypatch.setattr(analyzer, "analyze", unexpected_recompute)

    assert not analyzer.to_dataframe().empty
    assert not analyzer.to_residue_dataframe().empty
    assert analyzer.hot_spot_residues(top_n=2)
    assert "fastPISA" in analyzer.summary()
