"""
End-to-end regression tests for the Monte-Carlo tracking study.

The study is a top-level script rather than an importable package, so it is
exercised as a subprocess and its console report is parsed. Assertions are
deliberately statistical rather than exact: the simulation is unseeded by
design, so what must hold is the *behaviour* of the estimator, not a
particular realisation of it.

Run with:  pytest -q
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kalman_missile_sim.py"
FIGURE = ROOT / "results" / "kalman_missile_intercept.png"

# Radar noise standard deviation declared by the study (metres, per axis).
SIGMA_R = 50.0
# Expected magnitude of a 2-D isotropic Gaussian error vector: sigma * sqrt(pi/2).
EXPECTED_RAW_ERROR = SIGMA_R * math.sqrt(math.pi / 2.0)  # ~= 62.7 m
# Lethal radius declared by the study (metres).
KILL_RADIUS = 300.0


def _grab(label: str, text: str) -> float:
    """Pull a numeric field out of the console report."""
    match = re.search(re.escape(label) + r"\s*:\s*([0-9.]+)", text)
    assert match is not None, "missing field " + label + " in report:\n" + text
    return float(match.group(1))


@pytest.fixture(scope="module")
def report() -> dict:
    """Run the study exactly once per test session and parse its output."""
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, "study exited non-zero:\n" + proc.stderr
    assert "SIMULATION COMPLETE" in proc.stdout

    out = proc.stdout
    return {
        "success": _grab("Success Rate", out),
        "reduction": _grab("Avg Error Reduc.", out),
        "raw": _grab("Avg Raw Error", out),
        "kf": _grab("Avg Kalman Error", out),
        "miss": _grab("Avg Miss Distance", out),
    }


def test_sensor_baseline_matches_theory(report):
    """The raw radar error must match sigma * sqrt(pi/2) for 2-D Gaussian noise.

    This validates the sensor model and the error metric before any claim is
    made about the filter itself.
    """
    assert report["raw"] == pytest.approx(EXPECTED_RAW_ERROR, rel=0.15)


def test_filter_beats_the_raw_sensor(report):
    """Filtering must reduce mean position error, not merely change it."""
    assert report["kf"] < report["raw"]


def test_error_reduction_is_substantial(report):
    """An NCV filter at this noise-to-manoeuvre ratio should cut error markedly."""
    assert report["reduction"] > 25.0


def test_reported_reduction_is_consistent_with_reported_errors(report):
    """Cross-check the headline percentage against the raw and filtered means.

    The reported figure averages per-mission reductions while this check works
    from ensemble means, so the two agree only up to a few percentage points.
    """
    implied = (1.0 - report["kf"] / report["raw"]) * 100.0
    assert implied == pytest.approx(report["reduction"], abs=8.0)


def test_intercept_success_rate_is_high(report):
    """Accurate state estimates should make the naive intercept solver reliable."""
    assert report["success"] >= 60.0


def test_mean_miss_distance_is_inside_the_lethal_radius(report):
    assert 0.0 < report["miss"] < KILL_RADIUS


def test_figure_is_regenerated(report):
    """The study must emit a non-trivial figure to results/."""
    assert FIGURE.is_file()
    assert FIGURE.stat().st_size > 20_000
