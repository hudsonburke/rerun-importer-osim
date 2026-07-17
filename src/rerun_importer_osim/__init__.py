#!/usr/bin/env python3
"""
Rerun external importer for OpenSim model files and simulation results.

Handles:
  - .osim   OpenSim model XML (skeleton, joints, body properties)
  - .mot    Motion / storage files (IK results, scale output, GRF)
  - .sto    Storage files (ID results, muscle analysis, states, controls)
  - .trc    3D marker trajectory files

Entity path scheme:
  {prefix}/model/bodies/{body}          — static body properties
  {prefix}/model/joints/{joint}         — static joint transforms
  {prefix}/ik/{joint}                   — IK joint angles (time series)
  {prefix}/id/{joint}                   — ID joint moments (time series)
  {prefix}/muscles/{muscle}/{quantity}  — muscle analysis data
  {prefix}/grf/{component}              — ground reaction forces
  {prefix}/kinematics/{q|u|dudt}/{dof}  — CMC/RRA kinematics
  {prefix}/actuation/{force|power|speed} — actuation signals
  {prefix}/markers/{name}               — TRC marker trajectories
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rerun as rr


# ---------------------------------------------------------------------------
# .sto / .mot parser
# ---------------------------------------------------------------------------
# Both formats share the same structure:
#   line 1: description
#   version=1
#   nRows=N   nColumns=M
#   inDegrees=yes|no
#   endheader
#   col1\tcol2\t...
#   val1\tval2\t...

STORAGE_HEADER_RE = re.compile(r"^(nRows|nColumns|version|inDegrees|endheader)\s*=\s*(.*)", re.IGNORECASE)


def parse_storage(filepath: str) -> dict:
    """Parse a .mot or .sto file into a dict with ``columns``, ``data``, ``metadata``.

    Returns
    -------
    dict with keys:
      metadata  —  dict of header fields (version, nRows, nColumns, inDegrees)
      columns   —  list of column names
      data      —  2D numpy array (nRows × nColumns)
    """
    with open(filepath) as f:
        lines = f.readlines()

    metadata: dict[str, str] = {}
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("endheader"):
            header_end = i + 1
            break
        m = STORAGE_HEADER_RE.match(stripped)
        if m:
            metadata[m.group(1)] = m.group(2)
    else:
        # No endheader found — try the older .sto format where header is just
        # lines starting with special chars
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(("#", "%", "!", "\r")):
                header_end = i
                break

    # Read data from header_end onward
    data_lines = [l for l in lines[header_end:] if l.strip()]
    if not data_lines:
        return {"metadata": metadata, "columns": [], "data": np.empty((0, 0))}

    reader = csv.reader(data_lines, delimiter="\t", skipinitialspace=True)
    rows = [row for row in reader if row and any(c.strip() for c in row)]

    if not rows:
        return {"metadata": metadata, "columns": [], "data": np.empty((0, 0))}

    columns = [c.strip().replace("\r", "") for c in rows[0]]
    data_rows = rows[1:]

    nrows = int(metadata.get("nRows", 0))
    ncols = int(metadata.get("nColumns", 0))

    arr = np.zeros((len(data_rows), len(columns)), dtype=np.float64)
    for i, row in enumerate(data_rows):
        for j in range(min(len(row), len(columns))):
            try:
                arr[i, j] = float(row[j].replace("\r", ""))
            except (ValueError, IndexError):
                arr[i, j] = np.nan

    return {
        "metadata": metadata,
        "columns": columns,
        "data": arr,
    }


def time_column(data: dict) -> tuple[np.ndarray | None, int]:
    """Return (time_values, time_idx) or (None, -1) if missing."""
    cols = data["columns"]
    for label in ("time", "Time", "t", "T"):
        if label in cols:
            idx = cols.index(label)
            return data["data"][:, idx].copy(), idx
    return None, -1


# ---------------------------------------------------------------------------
# .osim model parser
# ---------------------------------------------------------------------------

def parse_osim_model(filepath: str) -> dict | None:
    """Extract bodies, joints, and their properties from an .osim model."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        return None

    ns = ""  # OpenSim XML doesn't generally use namespaces

    model_elem = root.find("Model") or root
    name = model_elem.get("name", Path(filepath).stem)

    bodies = []
    for body_elem in model_elem.iter("Body"):
        b = {"name": body_elem.get("name", "")}
        mass_elem = body_elem.find("mass")
        if mass_elem is not None and mass_elem.text:
            try:
                b["mass"] = float(mass_elem.text)
            except ValueError:
                pass
        inertia_elem = body_elem.find("inertia")
        if inertia_elem is not None and inertia_elem.text:
            parts = inertia_elem.text.strip().split()
            if len(parts) >= 6:
                b["inertia"] = [float(p) for p in parts[:6]]
        bodies.append(b)

    joints = []
    for joint_elem in model_elem.iter("Joint"):
        j = {
            "name": joint_elem.get("name", ""),
            "type": joint_elem.tag,
        }
        parent = joint_elem.find("parent_body")
        child = joint_elem.find("child_body")
        if parent is not None and parent.text:
            j["parent"] = parent.text.strip()
        if child is not None and child.text:
            j["child"] = child.text.strip()

        # Location/orientation in parent frame
        for loc_elem in joint_elem.iter("location_in_parent"):
            if loc_elem.text:
                parts = loc_elem.text.strip().split()
                if len(parts) >= 3:
                    j["location"] = [float(p) for p in parts[:3]]
        for orient_elem in joint_elem.iter("orientation_in_parent"):
            if orient_elem.text:
                parts = orient_elem.text.strip().split()
                if len(parts) >= 3:
                    j["orientation"] = [float(p) for p in parts[:3]]

        # Coordinate ranges (for joints with DoFs)
        coords = []
        for coord_elem in joint_elem.iter("Coordinate"):
            c = {"name": coord_elem.get("name", "")}
            range_elem = coord_elem.find("range")
            if range_elem is not None and range_elem.text:
                parts = range_elem.text.strip().split()
                if len(parts) >= 2:
                    c["range"] = [float(p) for p in parts[:2]]
            default_elem = coord_elem.find("default_value")
            if default_elem is not None and default_elem.text:
                try:
                    c["default"] = float(default_elem.text)
                except ValueError:
                    pass
            coords.append(c)
        if coords:
            j["coordinates"] = coords

        joints.append(j)

    return {
        "name": name,
        "bodies": bodies,
        "joints": joints,
    }


# ---------------------------------------------------------------------------
# .trc marker parser
# ---------------------------------------------------------------------------

def parse_trc(filepath: str) -> dict | None:
    """Parse a .trc 3D marker trajectory file (standard Motion Analysis format).

    Returns dict with:
      file_type, data_rate, camera_rate, num_frames, num_markers,
      marker_names, data, units
    """
    with open(filepath) as f:
        lines = f.readlines()

    if len(lines) < 6:
        return None

    # Line 1: PathFileType header
    header1 = lines[0].strip()

    # Line 2: column labels
    header2 = lines[1].strip().split("\t")

    # Line 3: numeric values
    header3 = lines[2].strip().split("\t")

    # Line 4: marker name row (Frame#  Time  Name1  ...  Name1  Name2  ...)
    name_row = lines[3].strip().split("\t")

    # Line 5: coordinate label row (   X1   Y1   Z1   X2   Y2   Z2 ...)
    coord_row = lines[4].strip().split("\t")

    # Parse header values
    result = {"file_type": header1}
    try:
        result["data_rate"] = float(header3[0])
        result["camera_rate"] = float(header3[1])
        result["num_frames"] = int(header3[2])
        result["num_markers"] = int(header3[3])
        result["units"] = header3[4].strip()
    except (ValueError, IndexError):
        pass

    # Extract marker names from the name row (skip Frame# and Time)
    marker_names = []
    for i in range(2, len(name_row)):
        name = name_row[i].strip()
        if name and (not coord_row[i - 2].strip().endswith("X") or i == 2 or name != marker_names[-1]):
            marker_names.append(name.replace(" ", "_"))

    result["marker_names"] = marker_names
    n_markers = len(marker_names)

    # Data rows start at line 6
    frames = []
    for line in lines[5:]:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            frame_num = int(parts[0])
            time_val = float(parts[1])
        except ValueError:
            continue

        positions = np.zeros((n_markers, 3), dtype=np.float64)
        for m_idx in range(n_markers):
            col_base = 2 + m_idx * 3
            if col_base + 2 < len(parts):
                try:
                    positions[m_idx, 0] = float(parts[col_base])
                    positions[m_idx, 1] = float(parts[col_base + 1])
                    positions[m_idx, 2] = float(parts[col_base + 2])
                except (ValueError, IndexError):
                    pass
        frames.append((time_val, positions))

    result["n_frames"] = len(frames)
    result["data"] = frames  # list of (time, (n_markers, 3))
    return result


# ---------------------------------------------------------------------------
# Rerun logging
# ---------------------------------------------------------------------------

def detect_file_type(filepath: str) -> str:
    """Return ``osim``, ``sto``, ``mot``, ``trc``, or ``unknown``."""
    ext = Path(filepath).suffix.lower()
    if ext == ".osim":
        return "osim"
    if ext == ".sto":
        return "sto"
    if ext == ".mot":
        return "mot"
    if ext == ".trc":
        return "trc"
    return "unknown"


def log_osim(filepath: str, prefix: str, recording: rr.RecordingStream) -> None:
    """Log an OpenSim model (.osim) to Rerun."""
    model = parse_osim_model(filepath)
    if model is None:
        return

    # Model info
    recording.log(
        f"{prefix}/model/info",
        rr.TextDocument(
            f"# {model['name']}\n\n"
            f"- **Bodies**: {len(model['bodies'])}\n"
            f"- **Joints**: {len(model['joints'])}\n",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    # Log body properties as static scalars
    for body in model["bodies"]:
        body_path = f"{prefix}/model/bodies/{body['name']}/mass"
        if "mass" in body:
            recording.log(body_path, rr.Scalars([body["mass"]]), static=True)

    # Log joint hierarchy as transforms
    for joint in model["joints"]:
        jpath = f"{prefix}/model/joints/{joint['name']}"
        recording.log(
            f"{jpath}/info",
            rr.TextDocument(
                f"**Type**: {joint['type']}\n"
                f"**Parent**: {joint.get('parent', 'ground')}\n"
                f"**Child**: {joint.get('child', '')}\n",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )

        # Log coordinates
        for coord in joint.get("coordinates", []):
            cpath = f"{jpath}/coordinate/{coord['name']}"
            if "range" in coord:
                recording.log(
                    f"{cpath}/range",
                    rr.Scalars([coord["range"][0], coord["range"][1]]),
                    static=True,
                )
            if "default" in coord:
                recording.log(
                    f"{cpath}/default",
                    rr.Scalars([coord["default"]]),
                    static=True,
                )


def log_storage(
    filepath: str,
    prefix: str,
    recording: rr.RecordingStream,
    kind: str = "",
) -> None:
    """Log a .mot or .sto file as batch time-series scalars."""
    data = parse_storage(filepath)
    if data["data"].size == 0 or not data["columns"]:
        return

    times, time_idx = time_column(data)
    if times is None:
        return

    n_frames = len(times)
    in_degrees = data["metadata"].get("inDegrees", "no").lower() == "yes"
    filename = Path(filepath).stem
    sub_prefix = kind if kind else filename

    # Build the time column once
    time_col = rr.TimeColumn("time", duration=times)

    for col_idx, col_name in enumerate(data["columns"]):
        if col_idx == time_idx:
            continue
        safe = sanitize(col_name)
        values = data["data"][:, col_idx]

        recording.send_columns(
            f"{prefix}/{sub_prefix}/{safe}",
            indexes=[time_col],
            columns=[rr.ComponentColumn(
                descriptor="scalar",
                component_batch=rr.components.ScalarBatch(values),
            )],
        )

    # Metadata about the file
    deg_str = " (degrees)" if in_degrees else ""
    recording.log(
        f"{prefix}/{sub_prefix}/info",
        rr.TextDocument(
            f"# {filename}\n\n"
            f"- **Columns**: {len(data['columns'])}\n"
            f"- **Rows**: {n_frames}\n"
            f"- **In degrees**: {in_degrees}{deg_str}\n",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )


def log_trc(filepath: str, prefix: str, recording: rr.RecordingStream) -> None:
    """Log a .trc marker file as per-frame Point3D."""
    trc = parse_trc(filepath)
    if trc is None or trc.get("n_frames", 0) == 0:
        return

    marker_names = trc["marker_names"]
    frames = trc["data"]
    n_markers = len(marker_names)

    # Log marker names
    recording.log(
        f"{prefix}/markers/info",
        rr.TextDocument(
            f"# Marker Data\n"
            f"- **Markers**: {n_markers}\n"
            f"- **Frames**: {trc.get('n_frames', 0)}\n"
            f"- **Units**: {trc.get('units', 'mm')}\n",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    for frame_idx, (time_val, positions) in enumerate(frames):
        recording.set_time("time", duration=time_val)
        recording.set_time("frame", sequence=frame_idx)

        recording.log(
            f"{prefix}/markers",
            rr.Points3D(
                positions,
                labels=marker_names,
                radii=np.full(n_markers, 0.01),
            ),
        )


def sanitize(name: str) -> str:
    """Make a string safe for use as a Rerun entity path component."""
    return name.replace("/", "_").replace(" ", "_").replace(".", "_").replace("(", "_").replace(")", "_")


# ---------------------------------------------------------------------------
# File dispatcher
# ---------------------------------------------------------------------------

FILE_KINDS: dict[str, str] = {
    "ik_output": "ik",
    "inverse_dynamics": "id",
    "grf": "grf",
    "scale_output": "scale",
    "cmc_states": "states",
    "cmc_Kinematics_q": "kinematics/q",
    "cmc_Kinematics_u": "kinematics/u",
    "cmc_Kinematics_dudt": "kinematics/dudt",
    "cmc_Actuation_force": "actuation/force",
    "cmc_Actuation_power": "actuation/power",
    "cmc_Actuation_speed": "actuation/speed",
    "cmc_controls": "controls",
    "cmc_pErr": "tracking_error",
    "cmc_MuscleAnalysis": "muscles",
    "rra_Kinematics_q": "kinematics/q",
    "rra_Kinematics_u": "kinematics/u",
    "rra_Kinematics_dudt": "kinematics/dudt",
    "rra_Actuation_force": "actuation/force",
    "rra_Actuation_power": "actuation/power",
    "rra_Actuation_speed": "actuation/speed",
    "rra_controls": "controls",
    "rra_pErr": "tracking_error",
    "rra_avgResiduals": "residuals",
    "rra_states": "states",
}


def classify_storage_file(filepath: str) -> str:
    """Return the entity sub-path prefix for a .mot/.sto file based on its name."""
    stem = Path(filepath).stem
    for pattern, kind in FILE_KINDS.items():
        if pattern in stem:
            return kind
    return stem


def log_file(filepath: str, prefix: str, recording: rr.RecordingStream) -> bool:
    """Log a file to Rerun based on its type. Returns True on success."""
    ftype = detect_file_type(filepath)
    if ftype == "osim":
        log_osim(filepath, prefix, recording)
        return True
    elif ftype == "trc":
        log_trc(filepath, prefix, recording)
        return True
    elif ftype in ("mot", "sto"):
        kind = classify_storage_file(filepath)
        log_storage(filepath, prefix, recording, kind=kind)
        return True
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""\
External Rerun importer for OpenSim files (.osim, .mot, .sto, .trc).

Any executable on $PATH whose name starts with ``rerun-importer-`` is
discovered by the Rerun Viewer as an external importer.

Usage:
    rerun-importer-osim path/to/model.osim
    rerun-importer-osim path/to/ik_output.mot
    rerun-importer-osim batch /data/dir -o /out/rrd
""",
    )

    # Single-file mode
    parser.add_argument("filepath", type=str, nargs="?", help="Path to the file to load")
    parser.add_argument("--application-id", type=str)
    parser.add_argument("--opened-application-id", type=str)
    parser.add_argument("--recording-id", type=str)
    parser.add_argument("--opened-recording-id", type=str)
    parser.add_argument("--entity-path-prefix", type=str)
    parser.add_argument("--static", action="store_true", default=False)
    parser.add_argument("--time", type=str, action="append")
    parser.add_argument("--sequence", type=str, action="append")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.filepath:
        parser.print_help()
        sys.exit(1)

    # Check supported extensions
    ext = Path(args.filepath).suffix.lower()
    if ext not in (".osim", ".mot", ".sto", ".trc") or not os.path.isfile(args.filepath):
        sys.exit(rr.EXTERNAL_IMPORTER_INCOMPATIBLE_EXIT_CODE)

    app_id = args.application_id or args.filepath
    rr.init(app_id, recording_id=args.recording_id)
    recording = rr.get_global_data_recording()
    if recording is None:
        recording = rr.RecordingStream(application_id=app_id, recording_id=args.recording_id)
    recording.stdout()

    _set_time_from_args(args, recording)

    prefix = args.entity_path_prefix or Path(args.filepath).stem
    log_file(args.filepath, prefix, recording)


def _set_time_from_args(args: argparse.Namespace, recording: rr.RecordingStream) -> None:
    if args.static:
        return
    if args.time:
        for time_str in args.time:
            parts = time_str.split("=")
            if len(parts) == 2:
                recording.set_time(parts[0], duration=float(parts[1]))
    if args.sequence:
        for seq_str in args.sequence:
            parts = seq_str.split("=")
            if len(parts) == 2:
                recording.set_time(parts[0], sequence=int(parts[1]))


if __name__ == "__main__":
    main()
