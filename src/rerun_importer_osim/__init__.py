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

    model_elem = root.find("Model") or root
    name = model_elem.get("name", Path(filepath).stem)

    bodies = []
    joints = []
    for body_elem in model_elem.iter("Body"):
        b = {"name": body_elem.get("name", "")}
        if b["name"] == "ground":
            continue
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

        # Geometry references
        geo_files = []
        for disp in body_elem.iter("DisplayGeometry"):
            gf = disp.find("geometry_file")
            if gf is not None and gf.text:
                scale = disp.find("scale_factors")
                s = scale.text.strip().split() if scale is not None and scale.text else ["1", "1", "1"]
                try:
                    svec = [float(x) for x in s[:3]]
                except ValueError:
                    svec = [1.0, 1.0, 1.0]
                geo_files.append({"file": gf.text.strip(), "scale": svec})
        if geo_files:
            b["geometry"] = geo_files
        bodies.append(b)

        # Extract joint from within this body's <Joint> child
        joint_wrapper = body_elem.find("Joint")
        if joint_wrapper is None:
            continue
        # The actual joint is one level deeper
        for joint_elem in joint_wrapper:
            tag = joint_elem.tag
            if not tag.endswith("Joint"):
                continue
            j = {
                "name": joint_elem.get("name", ""),
                "type": tag,
                "child": b["name"],
            }
            parent = joint_elem.find("parent_body")
            if parent is not None and parent.text:
                j["parent"] = parent.text.strip()
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
            # Coordinates
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
# .vtp mesh parser
# ---------------------------------------------------------------------------

def parse_vtp(filepath: str) -> dict | None:
    """Parse a .vtp (VTK PolyData XML) file into vertices and triangles.

    Returns dict with ``vertices`` (N, 3) and ``triangles`` (M, 3), or None.
    Handles ``Polys`` (triangles), ``Strips`` (triangle strips), and
    ``Verts`` (points-only).
    """
    try:
        tree = ET.parse(filepath)
    except ET.ParseError:
        return None
    root = tree.getroot()

    piece = root.find(".//Piece")
    if piece is None:
        return None

    # Vertices
    points_elem = piece.find("Points")
    if points_elem is None:
        return None
    data_arr = points_elem.find("DataArray")
    if data_arr is None or data_arr.text is None:
        return None
    verts = np.fromstring(data_arr.text.strip(), sep=" ", dtype=np.float64)
    if len(verts) == 0:
        return None
    verts = verts.reshape(-1, 3)
    n_verts = len(verts)

    # Try Polys first (most common for meshes)
    polys_elem = piece.find("Polys")
    tris = np.zeros((0, 3), dtype=np.int32)
    if polys_elem is not None:
        conn_arr = polys_elem.find("DataArray")
        if conn_arr is not None and conn_arr.text and conn_arr.text.strip():
            conn = np.fromstring(conn_arr.text.strip(), sep=" ", dtype=np.int32)
            # VTP polys stored as (n1, i1, i2, ..., n2, j1, j2, ...)
            tris_list = []
            pos = 0
            while pos < len(conn):
                n = int(conn[pos])
                if n == 3:
                    tris_list.append(conn[pos + 1 : pos + 4])
                pos += n + 1
            tris = np.array(tris_list, dtype=np.int32) if tris_list else tris

    # Try Strips if no triangles found (older VTP / strip geometry)
    if len(tris) == 0:
        strips_elem = piece.find("Strips")
        if strips_elem is not None:
            conn_arr = strips_elem.find("DataArray")
            if conn_arr is not None and conn_arr.text and conn_arr.text.strip():
                conn = np.fromstring(conn_arr.text.strip(), sep=" ", dtype=np.int32)
                # VTP strips: (n1, i1, i2, i3, ..., n2, j1, j2, j3, ...)
                # Each strip of length n produces (n-2) triangles
                strips_tris = []
                pos = 0
                while pos < len(conn):
                    n = int(conn[pos])
                    strip_verts = conn[pos + 1 : pos + n + 1]
                    for k in range(2, len(strip_verts)):
                        # Alternate CW vs CCW for each triangle in the strip
                        if k % 2 == 0:
                            strips_tris.append([int(strip_verts[k - 2]), int(strip_verts[k - 1]), int(strip_verts[k])])
                        else:
                            strips_tris.append([int(strip_verts[k - 1]), int(strip_verts[k - 2]), int(strip_verts[k])])
                    pos += n + 1
                tris = np.array(strips_tris, dtype=np.int32) if strips_tris else tris

    return {"vertices": verts, "triangles": tris}


def compute_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray | None:
    """Compute per-vertex normals from a triangle mesh."""
    if len(tris) == 0:
        return None
    v = verts[tris]  # (M, 3, 3)
    face_normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(face_normals, norms, out=np.zeros_like(face_normals), where=norms > 1e-12)

    vertex_normals = np.zeros_like(verts)
    for i in range(len(tris)):
        for j in range(3):
            vertex_normals[tris[i, j]] += face_normals[i]
    vn_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = np.divide(vertex_normals, vn_norms, out=np.zeros_like(vertex_normals), where=vn_norms > 1e-12)
    return vertex_normals.astype(np.float32)


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
# Kinematic tree and IK animation
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


def build_kinematic_tree(model: dict) -> list[dict]:
    """Build an ordered list of joints from root to leaves.

    Each entry::
        {name, type, parent, child, location, orientation, coords, axes}
    """
    joints = model["joints"]
    # Build child lookup
    children_of: dict[str, list] = {j["name"]: [] for j in joints}
    joint_map = {j["name"]: j for j in joints}
    roots = []
    for j in joints:
        p = j.get("parent", "ground")
        if p == "ground" or p not in joint_map:
            roots.append(j["name"])
        if p in joint_map:
            children_of[p].append(j["name"])
        elif p != "ground":
            roots.append(j["name"])

    # BFS from roots
    ordered = []
    queue = list(roots)
    visited = set()
    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        j = joint_map[name]
        ordered.append(j)
        for child in children_of.get(name, []):
            if child not in visited:
                queue.append(child)
    return ordered


def joint_coord_names(joint: dict) -> dict[str, dict]:
    """Return a dict mapping coordinate names to their definitions for a joint."""
    result = {}
    for c in joint.get("coordinates", []):
        cname = c["name"]
        result[cname] = c
    # For CustomJoint, get additional axis info
    return result


def compute_joint_transform_euler(orientation: list[float] | None) -> np.ndarray:
    """XYZ Euler angles → 3x3 rotation matrix (R = Rz * Ry * Rx)."""
    if not orientation or len(orientation) < 3:
        return np.eye(3)
    rx, ry, rz = np.radians(orientation)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    # R = Rz * Ry * Rx
    return np.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ])


def joint_local_transform(joint: dict, coords: dict[str, float], in_degrees: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute the local transform (translation, rotation_matrix) for a joint.

    Parameters
    ----------
    joint:
        Joint definition from the model parser.
    coords:
        Mapping of coordinate name → value from IK data.
    in_degrees:
        Whether coordinate values are in degrees (default True).

    Returns
    -------
    (translation, rotation_matrix) in the parent frame.
    """
    location = joint.get("location", [0.0, 0.0, 0.0])
    orientation = joint.get("orientation", [0.0, 0.0, 0.0])

    # Static transform from joint frame to parent frame
    static_R = compute_joint_transform_euler(orientation)
    static_t = np.array(location, dtype=np.float64)

    # Apply coordinate-dependent transform (in the joint frame)
    jtype = joint.get("type", "")
    joint_R = np.eye(3)
    joint_t = np.zeros(3)

    deg_factor = np.pi / 180.0 if in_degrees else 1.0

    coord_defs = {c["name"]: c for c in joint.get("coordinates", [])}

    if jtype == "PinJoint":
        # One coordinate, rotation about z-axis of joint frame
        for cname, cval in coords.items():
            a = cval * deg_factor
            ca, sa = np.cos(a), np.sin(a)
            joint_R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
            break  # PinJoint has only one coordinate

    elif jtype == "UniversalJoint":
        # Two coordinates: first about x, then about y (body-fixed)
        vals = list(coords.values())
        if len(vals) >= 2:
            a1, a2 = vals[0] * deg_factor, vals[1] * deg_factor
            ca1, sa1 = np.cos(a1), np.sin(a1)
            ca2, sa2 = np.cos(a2), np.sin(a2)
            # Rx(a1) then Ry(a2) (body-fixed = Ry * Rx in world frame)
            joint_R = np.array([
                [ca2, 0, sa2],
                [sa1 * sa2, ca1, -sa1 * ca2],
                [-ca1 * sa2, sa1, ca1 * ca2],
            ])
        elif len(vals) >= 1:
            a = vals[0] * deg_factor
            ca, sa = np.cos(a), np.sin(a)
            joint_R = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    elif jtype == "CustomJoint":
        # Apply SpatialTransform axes in order
        # We need the axes definitions from the model XML
        axes = joint.get("spatial_axes", [])
        for ax in axes:
            coord_name = ax.get("coord", "")
            axis_vec = np.array(ax.get("axis", [0, 0, 1]))
            ctype = ax.get("type", "rotation")  # rotation or translation
            cval = coords.get(coord_name, 0.0)
            if ctype == "rotation":
                a = cval * deg_factor
                ca, sa = np.cos(a), np.sin(a)
                uv = axis_vec / np.linalg.norm(axis_vec)
                # Rotation about arbitrary axis (Rodrigues)
                R = np.array([
                    [ca + uv[0]**2*(1-ca), uv[0]*uv[1]*(1-ca)-uv[2]*sa, uv[0]*uv[2]*(1-ca)+uv[1]*sa],
                    [uv[1]*uv[0]*(1-ca)+uv[2]*sa, ca+uv[1]**2*(1-ca), uv[1]*uv[2]*(1-ca)-uv[0]*sa],
                    [uv[2]*uv[0]*(1-ca)-uv[1]*sa, uv[2]*uv[1]*(1-ca)+uv[0]*sa, ca+uv[2]**2*(1-ca)],
                ])
                joint_R = R @ joint_R
            elif ctype == "translation":
                joint_t += axis_vec * cval

    # Compose: child frame = static_t * static_R * joint_t * joint_R
    # In the parent frame
    trans = static_t + static_R @ joint_t
    rot = static_R @ joint_R
    return trans, rot


def extract_spatial_axes(model_xml_root: ET.Element, joint_elem: ET.Element) -> list[dict]:
    """Extract SpatialTransform axis definitions for a CustomJoint."""
    axes = []
    transform = joint_elem.find("SpatialTransform")
    if transform is None:
        return axes
    for axis_elem in transform:
        coord_elem = axis_elem.find("coordinate")
        coord_name = coord_elem.text.strip() if coord_elem is not None and coord_elem.text else ""
        axis_xyz = axis_elem.find("axis")
        axis_vec = [0, 0, 1]
        if axis_xyz is not None and axis_xyz.text:
            try:
                axis_vec = [float(x) for x in axis_xyz.text.strip().split()[:3]]
            except ValueError:
                pass
        # Determine if rotation or translation
        # Look for child elements named 'rotation' or 'translation'
        is_rotation = False
        is_translation = False
        for child in axis_elem:
            if child.tag == "function":
                continue
            if child.tag in ("coordinate", "axis"):
                continue
            if "rotation" in child.tag.lower() or child.tag == "rotation":
                is_rotation = True
            if "translation" in child.tag.lower() or child.tag == "translation":
                is_translation = True
        # If neither tag found, infer from axis name
        if not is_rotation and not is_translation:
            # In OpenSim, axes without explicit type are rotations for the first 3
            # and translations for the next 3, but we can check by axis element presence
            # The OpenSim convention: first 3 axes are rotations, next 3 are translations
            # Actually the tag itself tells us: the element tag IS TransformAxis
            # We check parent_found or just default to rotation
            is_rotation = True  # default

        axes.append({"coord": coord_name, "axis": axis_vec, "type": "rotation" if is_rotation else "translation"})
    return axes


def log_osim_with_ik(
    model_path: str,
    prefix: str,
    recording: rr.RecordingStream,
    ik_path: str | None = None,
) -> None:
    """Log an OSIM model to Rerun, with optional IK-driven animation."""
    # Parse model
    tree = ET.parse(model_path)
    root_xml = tree.getroot()
    model_elem = root_xml.find("Model") or root_xml

    # Extract model with geometry
    model = parse_osim_model(model_path)
    if model is None:
        return

    # Pre-extract spatial axes for CustomJoints
    joint_defs = {}
    for joint_elem in model_elem.iter("Joint"):
        jname = joint_elem.get("name", "")
        if jname:
            joint_defs[jname] = joint_elem

    for j in model["joints"]:
        if j["type"] == "CustomJoint":
            xelem = joint_defs.get(j["name"])
            if xelem is not None:
                j["spatial_axes"] = extract_spatial_axes(root_xml, xelem)

    # Log model info and body geometry
    log_osim(model_path, prefix, recording, model=model)

    # Log IK animation if provided
    if ik_path:
        log_ik_animation(ik_path, model, prefix, recording)


def log_ik_animation(
    ik_path: str,
    model: dict,
    prefix: str,
    recording: rr.RecordingStream,
) -> None:
    """Log per-frame body transforms from IK data."""
    ik_data = parse_storage(ik_path)
    if ik_data["data"].size == 0:
        return

    times, time_idx = time_column(ik_data)
    if times is None:
        return

    n_frames = len(times)
    in_degrees = ik_data["metadata"].get("inDegrees", "yes").lower() != "no"

    # Build IK coordinate lookup: column_name → column_index
    ik_cols = {c: i for i, c in enumerate(ik_data["columns"]) if i != time_idx}

    # Build kinematic tree
    tree = build_kinematic_tree(model)

    # Build inverse lookup: child body → joint
    body_to_joint = {}
    for j in tree:
        child = j.get("child", "")
        if child:
            body_to_joint[child] = j

    # Find root bodies (no parent joint or connected to ground)
    all_children = set(j.get("child", "") for j in tree)
    root_bodies = [j["parent"] for j in tree if j.get("parent", "") == "ground"]
    # Also find bodies with no parent joint connecting them
    for j in tree:
        p = j.get("parent", "")
        if p != "ground" and p not in body_to_joint and j.get("child", ""):
            root_bodies.append(p)

    # Walk the tree for each frame
    for frame_idx in range(n_frames):
        recording.set_time("time", duration=times[frame_idx])
        recording.set_time("frame", sequence=frame_idx)

        # Compute world transforms for each body
        world_transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        body_processed: set[str] = set()

        def get_world_transform(body_name: str) -> tuple[np.ndarray, np.ndarray]:
            """Compute world transform for a body recursively."""
            if body_name in world_transforms:
                return world_transforms[body_name]

            if body_name == "ground":
                t, R = np.zeros(3), np.eye(3)
                world_transforms["ground"] = (t, R)
                return t, R

            if body_name in body_processed:
                return world_transforms.get(body_name, (np.zeros(3), np.eye(3)))

            body_processed.add(body_name)

            # Find the joint whose child is this body
            joint = body_to_joint.get(body_name)
            if joint is None:
                # Check if this is a root body connected to ground
                t, R = np.zeros(3), np.eye(3)
                world_transforms[body_name] = (t, R)
                return t, R

            # Get IK coordinate values for this joint
            coords: dict[str, float] = {}
            for c in joint.get("coordinates", []):
                cname = c["name"]
                if cname in ik_cols:
                    coords[cname] = ik_data["data"][frame_idx, ik_cols[cname]]

            # Local transform from joint
            local_t, local_R = joint_local_transform(joint, coords, in_degrees)

            # Parent world transform
            parent_name = joint.get("parent", "ground")
            parent_t, parent_R = get_world_transform(parent_name)

            # Compose: world = parent_R * local_t + parent_t, parent_R * local_R
            world_t = parent_t + parent_R @ local_t
            world_R = parent_R @ local_R

            world_transforms[body_name] = (world_t, world_R)
            return world_t, world_R

        # Compute transforms for all bodies in the tree
        for j in tree:
            child = j.get("child", "")
            if child:
                get_world_transform(child)

        # Log transforms for each body
        for body_name, (world_t, world_R) in world_transforms.items():
            if body_name == "ground":
                continue

            body_path = f"{prefix}/model/bodies/{body_name}"

            # Convert rotation matrix to quaternion
            # (w, x, y, z) format for rerun
            trace = np.trace(world_R)
            if trace > 0:
                S = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * S
                qx = (world_R[2, 1] - world_R[1, 2]) / S
                qy = (world_R[0, 2] - world_R[2, 0]) / S
                qz = (world_R[1, 0] - world_R[0, 1]) / S
            elif world_R[0, 0] > world_R[1, 1] and world_R[0, 0] > world_R[2, 2]:
                S = np.sqrt(1.0 + world_R[0, 0] - world_R[1, 1] - world_R[2, 2]) * 2
                qw = (world_R[2, 1] - world_R[1, 2]) / S
                qx = 0.25 * S
                qy = (world_R[0, 1] + world_R[1, 0]) / S
                qz = (world_R[0, 2] + world_R[2, 0]) / S
            elif world_R[1, 1] > world_R[2, 2]:
                S = np.sqrt(1.0 + world_R[1, 1] - world_R[0, 0] - world_R[2, 2]) * 2
                qw = (world_R[0, 2] - world_R[2, 0]) / S
                qx = (world_R[0, 1] + world_R[1, 0]) / S
                qy = 0.25 * S
                qz = (world_R[1, 2] + world_R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + world_R[2, 2] - world_R[0, 0] - world_R[1, 1]) * 2
                qw = (world_R[1, 0] - world_R[0, 1]) / S
                qx = (world_R[0, 2] + world_R[2, 0]) / S
                qy = (world_R[1, 2] + world_R[2, 1]) / S
                qz = 0.25 * S

            recording.log(
                body_path,
                rr.Transform3D(
                    translation=world_t,
                    quaternion=(qw, qx, qy, qz),
                ),
            )

    log_info_text = f"IK animation: {n_frames} frames, {len(world_transforms)} bodies"
    recording.log(f"{prefix}/model/animation_info", rr.TextLog(log_info_text), static=True)


def log_osim(
    filepath: str,
    prefix: str,
    recording: rr.RecordingStream,
    model: dict | None = None,
) -> None:
    """Log an OpenSim model (.osim) to Rerun."""
    if model is None:
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
        body_path = f"{prefix}/model/bodies/{body['name']}"
        if "mass" in body:
            recording.log(f"{body_path}/mass", rr.Scalars([body["mass"]]), static=True)

        # Log mesh geometry
        geo_dir = Path(filepath).parent / "Geometry"
        for geo_info in body.get("geometry", []):
            geo_path = geo_dir / geo_info["file"]
            if not geo_path.exists():
                continue
            mesh = parse_vtp(str(geo_path))
            if mesh is None or len(mesh["vertices"]) == 0:
                continue
            scale = geo_info.get("scale", [1.0, 1.0, 1.0])
            verts = mesh["vertices"] * scale

            if len(mesh["triangles"]) > 0:
                normals = compute_normals(verts, mesh["triangles"])
                recording.log(
                    f"{body_path}/mesh",
                    rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=mesh["triangles"],
                        vertex_normals=normals,
                    ),
                    static=True,
                )
            else:
                # Verts-only: log as point cloud
                recording.log(
                    f"{body_path}/mesh",
                    rr.Points3D(
                        verts,
                        radii=0.005,
                    ),
                    static=True,
                )

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
    parser.add_argument("--animate", type=str, default=None,
        help="Path to IK .mot file for model animation")

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

    if ext == ".osim" and args.animate:
        log_osim_with_ik(args.filepath, prefix, recording, ik_path=args.animate)
    else:
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
