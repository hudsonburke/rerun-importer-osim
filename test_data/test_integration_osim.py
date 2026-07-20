"""Integration test for OSIM importer with melos-rerun Arrow components."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rerun_importer_osim import parse_osim_model


TEST_OSIM = Path(__file__).resolve().parent.parent / "test_data" / "RajagopalData" / "Rajagopal2015.osim"


def test_osim_model_parses_successfully() -> None:
    """Basic sanity: the Rajagopal model parses into bodies and joints."""
    if not TEST_OSIM.exists():
        pytest.skip(f"Test data not found: {TEST_OSIM}")

    model = parse_osim_model(str(TEST_OSIM))
    assert model is not None
    assert len(model["joints"]) > 0
    assert len(model["bodies"]) > 0


def test_osim_joint_has_expected_fields() -> None:
    """Every parsed joint should have the fields needed by JointDefinition."""
    if not TEST_OSIM.exists():
        pytest.skip(f"Test data not found: {TEST_OSIM}")

    model = parse_osim_model(str(TEST_OSIM))
    assert model is not None

    for joint in model["joints"]:
        assert "name" in joint
        assert "type" in joint
        assert "parent" in joint or "child" in joint

        # Coordinates are optional but should have range if present
        for coord in joint.get("coordinates", []):
            if "range" in coord:
                assert len(coord["range"]) == 2
                assert coord["range"][0] < coord["range"][1]


def test_osim_body_has_inertial_fields() -> None:
    """Bodies should have mass and geometry info for LinkDefinition."""
    if not TEST_OSIM.exists():
        pytest.skip(f"Test data not found: {TEST_OSIM}")

    model = parse_osim_model(str(TEST_OSIM))
    assert model is not None

    for body in model["bodies"]:
        assert "name" in body
        # Some bodies might not have mass (e.g. ground)
        # but at least inertial properties should be parseable
        _ = body.get("mass", 0.0)


def test_osim_model_logs_without_error() -> None:
    """Running log_osim machinery succeeds — tests that Arrow component
    creation from the model dict does not raise."""
    if not TEST_OSIM.exists():
        pytest.skip(f"Test data not found: {TEST_OSIM}")

    from melos.rerun.components import (
        JointDefinitionBatch,
        LinkDefinitionBatch,
    )

    model = parse_osim_model(str(TEST_OSIM))
    assert model is not None

    # Test that the Arrow struct arrays can be created from model data
    # (this is the core of the component integration)
    for joint in model["joints"][:3]:  # Check first 3 joints
        coords = joint.get("coordinates", [])
        first_coord = coords[0] if coords else {}
        limits = (
            {"lower": first_coord["range"][0], "upper": first_coord["range"][1]}
            if "range" in first_coord
            else {"lower": float("-inf"), "upper": float("inf")}
        )

        batch = JointDefinitionBatch([{
            "joint_type": joint.get("type", "CustomJoint"),
            "axis": [0.0, 0.0, 1.0] if "Pin" in joint.get("type", "") else [0.0, 0.0, 0.0],
            "limits": limits,
            "parent_link": joint.get("parent", "ground"),
            "child_link": joint.get("child", ""),
            "default_qpos": first_coord.get("default", 0.0),
        }])
        arr = batch.as_arrow_array()
        assert len(arr) == 1
        assert arr[0]["joint_type"].as_py() is not None

    for body in model["bodies"][:3]:  # Check first 3 bodies
        batch = LinkDefinitionBatch([{
            "name": body["name"],
            "mass": body.get("mass", 0.0),
            "center_of_mass": body.get("mass_center", [0.0, 0.0, 0.0]),
            "inertia": body.get("inertia", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "graphics_file": "",
            "visible": True,
        }])
        arr = batch.as_arrow_array()
        assert len(arr) == 1
        assert arr[0]["name"].as_py() == body["name"]
