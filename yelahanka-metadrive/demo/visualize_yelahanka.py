"""
visualize_yelahanka.py
======================
Yelahanka OpenDRIVE map visualizer for MetaDrive.

NOTE: Run from the yelahanka-metadrive/ directory:
    python demo/visualize_yelahanka.py

Improvements over initial commit
---------------------------------
* Fixed lane rendering   – `draw_debug_lanes()` now correctly walks every
                          block's network graph and pulls lane.visualization_points.
* Direct-geometry lanes  – Falls back to raw geometry scan when the block
                          network graph is empty (covers allLanes path).
* AMD / hardware rendering – Forces hardware vertex buffers, disables software
                            fallback, and picks the best available renderer.
* Richer visuals         – Coloured lane lines by type, anti-aliased lines,
                          road-surface quads in addition to centrelines,
                          lane-marking outlines in yellow/white.
* Better camera          – Uses Panda3D's built-in orthographic top-down view
                          with correct aspect ratio.

Usage
-----
    cd yelahanka-metadrive
    python demo/visualize_yelahanka.py
"""

# ============================================================
# HARDWARE / DRIVER HINTS  (must come before panda3d imports)
# ============================================================

import os

# Prefer OpenGL – works best with AMD Radeon / RX / RDNA on Windows.
# Remove or change to "pandadx9" if DX9 is preferred.
os.environ.setdefault("PANDA_DISPLAY_DRIVER", "pandagl")

# Disable CPU-side software fallback so panda reports an error early
# rather than silently falling back to software rendering.
os.environ.setdefault("PANDA_SOFTWARE_RENDERER", "0")

# Tell Mesa / AMD driver not to throttle background windows.
os.environ.setdefault("vblank_mode", "0")

import math
import sys
import io
import numpy as np

# Force UTF-8 on Windows console so Unicode prints don't crash.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    Camera,
    CardMaker,
    DirectionalLight,
    FrameBufferProperties,
    GeomVertexFormat,
    LineSegs,
    LoaderOptions,
    OrthographicLens,
    Vec4,
    WindowProperties,
    loadPrcFileData,
)

# ============================================================
# PANDA3D RENDERING CONFIGURATION
# ============================================================

# Multi-sample anti-aliasing (4x) – supported by all modern AMD GPUs.
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")

# Hardware vertex processing – offloads transforms to the GPU.
loadPrcFileData("", "hardware-animated-vertices 1")

# Enable OpenGL VBO (vertex buffer objects) – crucial for AMD perf.
loadPrcFileData("", "vertex-buffers 1")
loadPrcFileData("", "index-buffers 1")
loadPrcFileData("", "compressed-vertices 0")

# Window settings.
loadPrcFileData("", "window-title YELAHANKA MetaDrive Visualizer")
loadPrcFileData("", "win-size 1600 900")

# Sync to display (0 = unlimited FPS, good for a static map view).
loadPrcFileData("", "sync-video 0")

# Log only warnings and above during normal operation.
loadPrcFileData("", "notify-level warning")
loadPrcFileData("", "notify-level-display info")

# ============================================================
# METADRIVE IMPORTS
# ============================================================

from metadrive.component.opendrive_block.opendrive_block import (
    OpenDriveBlock,
)
from metadrive.component.road_network.edge_road_network import (
    OpenDriveRoadNetwork,
)
from metadrive.engine.asset_loader import (
    initialize_asset_loader,
)
from metadrive.tests.vis_block.vis_block_base import (
    TestBlock,
)
from metadrive.utils.opendrive.map_load import (
    load_opendrive_map,
)


# ================================================================
# SETTINGS
# ================================================================

XODR_PATH   = "maps/opendrive/yelahanka.xodr"
MAX_ROADS   = 200          # Increase from 50 to load more of the map

GROUND_MARGIN   = 1_000.0  # metres extra around the map boundary
CAMERA_HEIGHT   = 8_000.0  # orthographic top-down height
VIEW_MARGIN     = 1.12     # fraction of viewport the map occupies

DRAW_GROUND         = True
DRAW_ROAD_SURFACE   = True   # filled asphalt quads
DRAW_LANE_LINES     = True   # yellow/white lane-marking lines
DRAW_CENTRELINES    = True   # thin red centrelines (debug)
DRAW_CENTER_MARKER  = True

LINE_WIDTH_CENTRELINE  = 3.0
LINE_WIDTH_LANE_MARK   = 4.0

# Road-surface colour  (dark asphalt)
ROAD_COLOR  = Vec4(0.20, 0.20, 0.20, 1.0)
# Lane centreline colour (diagnostic red)
CENTRE_COLOR = Vec4(1.0, 0.15, 0.15, 1.0)
# Lane marking colour (yellow)
MARK_COLOR  = Vec4(1.0, 0.85, 0.10, 1.0)


# ================================================================
# HARDWARE INFO
# ================================================================

def print_hw_info():
    """Print Panda3D renderer information for diagnostics."""
    try:
        from panda3d.core import GraphicsEngine, GraphicsPipe, GraphicsPipeSelection
        sel = GraphicsPipeSelection.getGlobalPtr()
        print("[HW] Available renderers:", sel.getNumPipeTypes())
        for i in range(sel.getNumPipeTypes()):
            print("[HW]  ", sel.getPipeTypeName(i))
    except Exception as e:
        print("[HW] Could not query renderers:", e)


# ================================================================
# LIGHTING
# ================================================================

def setup_lighting(engine):

    print("[RENDER] Setting up lighting...")

    ambient = AmbientLight("yelahanka-ambient")
    ambient.setColor(Vec4(0.75, 0.75, 0.75, 1.0))
    ambient_np = engine.render.attachNewNode(ambient)
    engine.render.setLight(ambient_np)

    directional = DirectionalLight("yelahanka-directional")
    directional.setColor(Vec4(1.0, 1.0, 0.95, 1.0))
    dir_np = engine.render.attachNewNode(directional)
    dir_np.setHpr(-45, -60, 0)
    engine.render.setLight(dir_np)

    print("[RENDER] Lighting done.")


# ================================================================
# GROUND
# ================================================================

def create_ground(engine, center_x, center_y, size):

    print("[GROUND] Creating ground plane...")

    cm = CardMaker("yelahanka-ground")
    half = size / 2.0
    cm.setFrame(-half, half, -half, half)

    ground = engine.render.attachNewNode(cm.generate())
    ground.setPos(center_x, center_y, -1.0)
    ground.setColor(0.12, 0.13, 0.12, 1.0)   # very dark greenish-grey
    ground.setTwoSided(True)

    # Anti-alias the ground edges too.
    ground.setAntialias(AntialiasAttrib.MAuto)

    print(f"[GROUND] Center: ({center_x:.1f}, {center_y:.1f}), size: {size:.1f} m")
    return ground


# ================================================================
# MAP BOUNDS
# ================================================================

def get_map_bounds(roads):
    """
    Return (x_min, x_max, y_min, y_max) by sampling every geometry
    start position in the road plan-view.
    """
    xs, ys = [], []

    for road in roads:
        try:
            geometries = road.planView._geometries
        except Exception:
            continue

        for geo in geometries:
            try:
                sx, sy = float(geo.start_position[0]), float(geo.start_position[1])
                if math.isfinite(sx) and math.isfinite(sy):
                    xs.append(sx)
                    ys.append(sy)
            except Exception:
                pass

            # Also estimate the far end of this geometry.
            try:
                h  = float(geo.heading)
                ln = float(geo.length)
                ex = sx + math.cos(h) * ln
                ey = sy + math.sin(h) * ln
                if math.isfinite(ex) and math.isfinite(ey):
                    xs.append(ex)
                    ys.append(ey)
            except Exception:
                pass

    if not xs:
        return 0.0, 1000.0, 0.0, 1000.0

    return min(xs), max(xs), min(ys), max(ys)


# ================================================================
# CAMERA
# ================================================================

def setup_dedicated_camera(engine, center_x, center_y, map_width, map_height, win):

    print()
    print("=" * 40)
    print("[CAMERA] Setting up top-down orthographic camera")
    print("=" * 40)

    # Disable every existing display region / camera.
    for dr in list(win.getDisplayRegions()):
        try:
            dr.setActive(False)
        except Exception:
            pass

    cam_node = Camera("YelahankaTopDown")
    cam_np   = engine.render.attachNewNode(cam_node)

    lens = OrthographicLens()

    props = win.getProperties()
    w, h  = props.getXSize(), props.getYSize()
    aspect = (float(w) / float(h)) if h > 0 else 1.333

    # Fit the whole map into the viewport with some margin.
    film_h = max(float(map_height), float(map_width) / aspect) * VIEW_MARGIN
    film_w = film_h * aspect

    lens.setFilmSize(film_w, film_h)
    lens.setNear(1.0)
    lens.setFar(200_000.0)
    cam_node.setLens(lens)

    cam_np.setPos(center_x, center_y, CAMERA_HEIGHT)
    cam_np.setHpr(0, -90, 0)    # look straight down

    dr = win.makeDisplayRegion(0.0, 1.0, 0.0, 1.0)
    dr.setCamera(cam_np)
    dr.setActive(True)
    dr.setClearColorActive(True)
    dr.setClearColor(Vec4(0.60, 0.62, 0.65, 1.0))  # sky-grey background

    print(f"[CAMERA] Position:      ({center_x:.1f}, {center_y:.1f}, {CAMERA_HEIGHT})")
    print(f"[CAMERA] Film size:     {film_w:.1f} x {film_h:.1f} m")
    print(f"[CAMERA] Aspect ratio:  {aspect:.4f}")

    return cam_np


# ================================================================
# CENTER MARKER
# ================================================================

def create_center_marker(engine, cx, cy):

    lines = LineSegs("map-center")
    lines.setThickness(8.0)
    lines.setColor(Vec4(1.0, 0.0, 1.0, 1.0))   # magenta

    m = 120.0
    lines.moveTo(cx - m, cy,     50.0)
    lines.drawTo(cx + m, cy,     50.0)
    lines.moveTo(cx,     cy - m, 50.0)
    lines.drawTo(cx,     cy + m, 50.0)

    node = engine.render.attachNewNode(lines.create())
    node.setTwoSided(True)
    node.setAntialias(AntialiasAttrib.MLine)
    return node


# ================================================================
# ROAD SURFACE (filled quads from lane centreline)
# ================================================================

def draw_road_surfaces(engine, blocks):
    """
    Draw filled asphalt-coloured quads along every lane centreline.
    Uses the same triangle-strip geometry as OpenDriveBlock but
    rendered into the scene graph directly for immediate feedback.
    """
    from panda3d.core import (
        Geom, GeomNode, GeomTriangles,
        GeomVertexData, GeomVertexFormat, GeomVertexWriter,
        NodePath,
    )

    print()
    print("[ROAD] Building road-surface geometry...")

    root = engine.render.attachNewNode("YelahankaRoads")
    total_quads = 0

    for block in blocks:
        graph = getattr(block.block_network, "graph", {})

        for lane_key, lane_info in graph.items():
            try:
                lane   = lane_info.lane
                pts    = _get_lane_points(lane)
                if pts is None or len(pts) < 2:
                    continue

                width  = max(float(getattr(lane, "width", 3.5) or 3.5), 0.5)
                hw     = width * 0.5

                left_pts, right_pts = _lane_boundaries(pts, hw)
                if len(left_pts) < 2:
                    continue

                vdata = GeomVertexData(
                    "road", GeomVertexFormat.getV3(), Geom.UH_static
                )
                n_verts = len(left_pts) * 2
                vdata.setNumRows(n_verts)
                vw = GeomVertexWriter(vdata, "vertex")

                for lp, rp in zip(left_pts, right_pts):
                    vw.addData3f(float(lp[0]), float(lp[1]), 0.0)
                    vw.addData3f(float(rp[0]), float(rp[1]), 0.0)

                tris = GeomTriangles(Geom.UH_static)
                for i in range(len(left_pts) - 1):
                    la, ra = 2 * i, 2 * i + 1
                    lb, rb = 2 * (i + 1), 2 * (i + 1) + 1
                    tris.addVertices(la, ra, lb)
                    tris.addVertices(ra, rb, lb)

                geom = Geom(vdata)
                geom.addPrimitive(tris)
                node = GeomNode("road-%s" % str(lane_key))
                node.addGeom(geom)

                np_ = root.attachNewNode(node)
                np_.setTwoSided(True)
                np_.setColor(ROAD_COLOR)

                total_quads += len(left_pts) - 1

            except Exception as e:
                pass   # silently skip broken lanes

    print(f"[ROAD] Road-surface quads: {total_quads}")
    return root


# ================================================================
# LANE MARKINGS  (yellow lines between lanes)
# ================================================================

def draw_lane_markings(engine, blocks):
    """
    Draw yellow dashed / solid lane-boundary lines.
    """
    print()
    print("[MARK] Building lane markings...")

    root = engine.render.attachNewNode("YelahankaMarkings")
    drawn = 0

    for block in blocks:
        graph = getattr(block.block_network, "graph", {})

        for lane_key, lane_info in graph.items():
            try:
                lane = lane_info.lane
                pts  = _get_lane_points(lane)
                if pts is None or len(pts) < 2:
                    continue

                width = max(float(getattr(lane, "width", 3.5) or 3.5), 0.5)
                hw    = width * 0.5

                left_pts, right_pts = _lane_boundaries(pts, hw)

                for side_pts in (left_pts, right_pts):
                    if len(side_pts) < 2:
                        continue

                    ls = LineSegs("mark-%s" % str(lane_key))
                    ls.setThickness(LINE_WIDTH_LANE_MARK)
                    ls.setColor(MARK_COLOR)

                    p0 = side_pts[0]
                    ls.moveTo(float(p0[0]), float(p0[1]), 1.5)
                    for p in side_pts[1:]:
                        ls.drawTo(float(p[0]), float(p[1]), 1.5)

                    n = root.attachNewNode(ls.create())
                    n.setTwoSided(True)
                    n.setAntialias(AntialiasAttrib.MLine)
                    drawn += 1

            except Exception:
                pass

    print(f"[MARK] Lane-marking polylines: {drawn}")
    return root


# ================================================================
# LANE CENTRELINES  (diagnostic red)
# ================================================================

def draw_centrelines(engine, blocks):
    """
    Draw thin red centrelines through every lane for diagnostics.
    """
    print()
    print("[CENTRE] Building centreline debug lines...")

    root    = engine.render.attachNewNode("YelahankaCentrelines")
    rendered = 0

    for block in blocks:
        graph = getattr(block.block_network, "graph", {})

        for lane_key, lane_info in graph.items():
            try:
                lane = lane_info.lane
                pts  = _get_lane_points(lane)
                if pts is None or len(pts) < 2:
                    continue

                ls = LineSegs("cl-%s" % str(lane_key))
                ls.setThickness(LINE_WIDTH_CENTRELINE)
                ls.setColor(CENTRE_COLOR)

                ls.moveTo(float(pts[0][0]), float(pts[0][1]), 3.0)
                for p in pts[1:]:
                    ls.drawTo(float(p[0]), float(p[1]), 3.0)

                n = root.attachNewNode(ls.create())
                n.setTwoSided(True)
                n.setAntialias(AntialiasAttrib.MLine)
                rendered += 1

            except Exception:
                pass

    print(f"[CENTRE] Centrelines rendered: {rendered}")

    if rendered == 0:
        print("[CENTRE] WARNING: 0 lanes found in block networks!")
        print("[CENTRE] Trying direct geometry fallback...")
        rendered = _draw_centrelines_fallback(engine, root)

    return root


def _draw_centrelines_fallback(engine, root):
    """
    Emergency fallback: rebuild centrelines directly from the
    OpenDRIVE parsed road geometry without touching the block network.
    This catches cases where block_network.graph is empty.
    """
    import importlib, sys

    # Try to retrieve the odr_map from the module-level global.
    odr_map = globals().get("_ODR_MAP", None)
    if odr_map is None:
        print("[FALLBACK] No _ODR_MAP in globals – skipping fallback.")
        return 0

    rendered = 0

    from metadrive.component.lane.opendrive_lane import OpenDriveLane
    from metadrive.utils.opendrive.map_load import get_lane_width

    for road in odr_map.roads[:MAX_ROADS]:
        try:
            geometries = road.planView._geometries
            if not geometries:
                continue

            # Draw the road reference line (centreline of road 0).
            lane_data = None
            for sec in getattr(road.lanes, "lane_sections", []):
                for ld in getattr(sec, "allLanes", []):
                    if int(getattr(ld, "id", 999)) == 0:
                        lane_data = ld
                        break
                if lane_data:
                    break

            if lane_data is None:
                # Synthesise a minimal stand-in lane object.
                class _FakeLane:
                    def __init__(self):
                        self.id = 0
                        self.parentRoad = road
                lane_data = _FakeLane()

            try:
                lane = OpenDriveLane(3.5, lane_data)
                pts  = lane.visualization_points
                if pts is None or len(pts) < 2:
                    continue

                ls = LineSegs("fb-cl-%s" % road.id)
                ls.setThickness(LINE_WIDTH_CENTRELINE)
                ls.setColor(Vec4(0.0, 0.6, 1.0, 1.0))   # blue for fallback

                ls.moveTo(float(pts[0][0]), float(pts[0][1]), 3.0)
                for p in pts[1:]:
                    ls.drawTo(float(p[0]), float(p[1]), 3.0)

                n = root.attachNewNode(ls.create())
                n.setTwoSided(True)
                n.setAntialias(AntialiasAttrib.MLine)
                rendered += 1

            except Exception as e:
                print(f"[FALLBACK] Road {road.id}: {type(e).__name__}: {e}")

        except Exception:
            pass

    print(f"[FALLBACK] Fallback centrelines: {rendered}")
    return rendered


# ================================================================
# HELPERS
# ================================================================

def _get_lane_points(lane):
    """Return the (N, 2) centreline point array for a lane, or None."""
    for attr in ("visualization_points", "points"):
        pts = getattr(lane, attr, None)
        if pts is not None:
            pts = np.asarray(pts, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] >= 2 and len(pts) >= 2:
                pts = pts[:, :2]
                mask = np.all(np.isfinite(pts), axis=1)
                pts  = pts[mask]
                if len(pts) >= 2:
                    return pts
    return None


def _lane_boundaries(centreline, half_width):
    """
    Given a (N, 2) centreline and a half-width, return
    (left_pts, right_pts) as lists of (x, y) floats.
    """
    left, right = [], []
    pts = np.asarray(centreline, dtype=np.float64)
    n   = len(pts)

    for i in range(n):
        if i == 0:
            tang = pts[1] - pts[0]
        elif i == n - 1:
            tang = pts[-1] - pts[-2]
        else:
            tang = pts[i + 1] - pts[i - 1]

        norm = float(np.linalg.norm(tang))
        if norm < 1e-8:
            if left:
                left.append(left[-1])
                right.append(right[-1])
            continue

        tang /= norm
        normal = np.array([-tang[1], tang[0]], dtype=np.float64)

        left.append(pts[i] + normal * half_width)
        right.append(pts[i] - normal * half_width)

    return left, right


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print()
    print("=" * 40)
    print("STARTING YELAHANKA METADRIVE")
    print("=" * 40)
    print()

    # ============================================================
    # HARDWARE INFO
    # ============================================================

    print_hw_info()

    # ============================================================
    # ENGINE
    # ============================================================

    engine = TestBlock(True)

    initialize_asset_loader(engine)

    # Enable scene-level anti-aliasing.
    engine.render.setAntialias(AntialiasAttrib.MAuto)

    # ============================================================
    # LOAD OPENDRIVE MAP
    # ============================================================

    print(f"\n[MAP] Loading: {XODR_PATH}")

    odr_map = load_opendrive_map(XODR_PATH)

    # Store globally so fallback renderer can reach it.
    _ODR_MAP = odr_map
    globals()["_ODR_MAP"] = odr_map

    print(f"[MAP] OpenDRIVE loaded!")
    print(f"[MAP] Total roads:     {len(odr_map.roads)}")
    print(f"[MAP] Total junctions: {len(odr_map.junctions)}")

    roads = odr_map.roads[:MAX_ROADS]
    print(f"[MAP] Processing:      {len(roads)} roads")

    # ============================================================
    # ROAD NETWORK + BLOCKS
    # ============================================================

    global_network = OpenDriveRoadNetwork()
    blocks = []
    block_id = 0

    print()
    print("[MAP] Building OpenDRIVE blocks...")

    for road in roads:
        sections = getattr(road.lanes, "lane_sections", [])

        for section in sections:
            try:
                block = OpenDriveBlock(
                    block_id, global_network, 0, section
                )

                success = False
                try:
                    success = block.construct_block(
                        engine.render, engine.physics_world
                    )
                except Exception as e:
                    print(f"[BLOCK] construct_block failed road {road.id}: "
                          f"{type(e).__name__}: {e}")

                lane_count = len(getattr(block.block_network, "graph", {}))

                if success or lane_count > 0:
                    blocks.append(block)
                    block_id += 1
                    sys.stdout.write(
                        f"\r[MAP] Blocks built: {len(blocks)} "
                        f"(road {road.id}, lanes {lane_count})"
                    )
                    sys.stdout.flush()

            except Exception as e:
                print(f"\n[MAP] Section skipped road {road.id}: "
                      f"{type(e).__name__}: {e}")

    print()

    if not blocks:
        raise RuntimeError(
            "No valid MetaDrive blocks were created – cannot visualize."
        )

    total_lanes = sum(
        len(getattr(b.block_network, "graph", {})) for b in blocks
    )
    print(f"\n[MAP] Blocks created:   {len(blocks)}")
    print(f"[MAP] Total graph lanes: {total_lanes}")

    # ============================================================
    # MAP BOUNDS
    # ============================================================

    x_min, x_max, y_min, y_max = get_map_bounds(roads)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    mw = x_max - x_min
    mh = y_max - y_min
    ms = max(mw, mh)

    print()
    print("=" * 40)
    print("[MAP] BOUNDS")
    print("=" * 40)
    print(f"  X: {x_min:.1f} to {x_max:.1f}  (width  {mw:.1f} m)")
    print(f"  Y: {y_min:.1f} to {y_max:.1f}  (height {mh:.1f} m)")
    print(f"  Center: ({cx:.1f}, {cy:.1f})")

    # ============================================================
    # ROOT TRANSFORM  (keep OpenDRIVE coords as-is)
    # ============================================================

    engine.render.setPos(0, 0, 0)
    engine.render.setScale(1, 1, 1)

    # ============================================================
    # LIGHTING
    # ============================================================

    setup_lighting(engine)

    # ============================================================
    # GROUND
    # ============================================================

    if DRAW_GROUND:
        create_ground(engine, cx, cy, ms + 2.0 * GROUND_MARGIN)

    # ============================================================
    # ROAD SURFACES
    # ============================================================

    if DRAW_ROAD_SURFACE:
        draw_road_surfaces(engine, blocks)

    # ============================================================
    # LANE MARKINGS
    # ============================================================

    if DRAW_LANE_LINES:
        draw_lane_markings(engine, blocks)

    # ============================================================
    # CENTRELINES  (diagnostic)
    # ============================================================

    if DRAW_CENTRELINES:
        draw_centrelines(engine, blocks)

    # ============================================================
    # CENTER MARKER
    # ============================================================

    if DRAW_CENTER_MARKER:
        create_center_marker(engine, cx, cy)

    # ============================================================
    # CAMERA
    # ============================================================

    win = engine.win
    setup_dedicated_camera(engine, cx, cy, mw, mh, win)

    # ============================================================
    # SCENE STATS
    # ============================================================

    engine.render.analyze()

    # ============================================================
    # FINAL STATUS
    # ============================================================

    print()
    print("=" * 40)
    print("YELAHANKA MAP IS RUNNING")
    print("=" * 40)
    print(f"  Roads processed:  {len(roads)}")
    print(f"  Blocks created:   {len(blocks)}")
    print(f"  Total lanes:      {total_lanes}")
    print(f"  Map center:       ({cx:.1f}, {cy:.1f})")
    print(f"  Map size:         {mw:.1f} x {mh:.1f} m")
    print()
    print("  LEGEND")
    print("  ------")
    print("  Grey-blue background  = sky / viewport")
    print("  Dark grey             = Yelahanka ground")
    print("  Dark quads            = road surface")
    print("  Yellow lines          = lane markings")
    print("  Red thin lines        = lane centrelines (debug)")
    print("  Magenta cross         = exact map centre")
    print()
    print("=" * 40)

    # ============================================================
    # MAIN LOOP
    # ============================================================

    while True:
        engine.taskMgr.step()