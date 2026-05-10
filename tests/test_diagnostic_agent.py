"""tests/test_diagnostic_agent.py - diagnostic_agent unit tests"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from agent.diagnostic_agent import FALLBACK_DIAGNOSIS, parse_diagnosis_output


def make_df(n=60):
    """Build a minimal test DataFrame."""
    import numpy as np
    ts = pd.date_range("2026-05-07 20:00", periods=n, freq="1min")
    return pd.DataFrame({
        "device_id": ["CNC_001"] * n,
        "device_name": ["CNC"] * n,
        "timestamp": ts,
        "temperature": 45.0 + np.random.randn(n) * 1.5,
        "vibration": 2.5 + np.random.randn(n) * 0.3,
        "current": 15.0 + np.random.randn(n) * 0.5,
        "rpm": 3000.0 + np.random.randn(n) * 15.0,
        "fault_type": ["bearing_wear"] * n,
        "rul": [0.05] * n,
    })


class TestFallbackDiagnosis:
    """FALLBACK_DIAGNOSIS integrity."""

    def test_all_types_have_required_keys(self):
        for k, v in FALLBACK_DIAGNOSIS.items():
            assert "fault_analysis" in v, f"{k} missing fault_analysis"
            assert "steps" in v, f"{k} missing steps"
            assert "parts_needed" in v, f"{k} missing parts_needed"

    def test_all_types_have_content(self):
        for k, v in FALLBACK_DIAGNOSIS.items():
            assert len(v["fault_analysis"]) > 10
            assert len(v["steps"]) >= 3
            assert len(v["parts_needed"]) >= 1


class TestParseDiagnosis:
    """parse_diagnosis_output tests."""

    def test_parses_sections(self):
        # Implementation looks for Chinese markers
        text = "## 故障根因分析\nRoot cause text\n## 建议操作步骤\n1. Step one\n## 所需备件清单\n- Part A"
        r = parse_diagnosis_output(text)
        assert "Root cause text" in r["fault_analysis"]
        assert len(r["steps"]) == 1
        assert len(r["parts_needed"]) == 1

    def test_handles_empty(self):
        r = parse_diagnosis_output("")
        assert r["fault_analysis"] == ""
        assert r["steps"] == []
        assert r["parts_needed"] == []


class TestDiagnoseIntegration:
    """diagnose() end-to-end via fallback path."""

    def test_returns_complete_structure(self):
        from agent.diagnostic_agent import diagnose
        df = make_df()
        report = diagnose("CNC", df.tail(60), {"fault_type": "bearing_wear", "outlier_sensors": ["vibration"]}, "CNC_001")
        for key in ("device_id", "timestamp", "fault_analysis", "steps", "parts_needed", "severity", "source"):
            assert key in report, f"Missing {key}"
        assert report["source"] == "fallback"

    def test_high_severity_low_rul(self):
        from agent.diagnostic_agent import diagnose
        df = make_df()
        df["rul"] = 0.05
        report = diagnose("CNC", df.tail(60), {"fault_type": "bearing_wear", "outlier_sensors": []}, "CNC_001")
        assert report["severity"] == "high"

    def test_low_severity_normal(self):
        from agent.diagnostic_agent import diagnose
        df = make_df()
        df["fault_type"] = "normal"
        df["rul"] = 1.0
        report = diagnose("CNC", df.tail(60), {"fault_type": "normal", "outlier_sensors": []}, "CNC_001")
        assert report["severity"] == "low"
