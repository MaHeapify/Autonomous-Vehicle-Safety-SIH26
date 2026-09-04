import math
import numpy as np

from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    CardMaker,
    Vec4,
    LineSegs,
    Camera,
    OrthographicLens,
)

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

XODR_PATH = "maps/opendrive/yelahanka.xodr"

MAX_ROADS = 50

# Extra space around map.
GROUND_MARGIN = 1000.0

# Height is not important with orthographic projection, but keep
# the camera comfortably above the map.
CAMERA_HEIGHT = 5000.0

# Fraction of viewport occupied by map.
VIEW_MARGIN = 1.15

# Debug rendering.
DRAW_GROUND = True
DRAW_DEBUG_LANES = True
DRAW_CENTER_MARKER = True

# Debug lane appearance.
DEBUG_LINE_WIDTH = 5.0


# ================================================================
# LIGHTING
# ================================================================

def setup_lighting(engine):

    print("[RENDER] Creating ambient light...")

    ambient = AmbientLight(
        "yelahanka-ambient"
    )

    ambient.setColor(
        Vec4(
            0.8,
            0.8,
            0.8,
            1.0
        )
    )

    ambient_np = engine.render.attachNewNode(
        ambient
    )

    engine.render.setLight(
        ambient_np
    )

    print("[RENDER] Creating directional light...")

    directional = DirectionalLight(
        "yelahanka-directional"
    )

    directional.setColor(
        Vec4(
            1.0,
            1.0,
            1.0,
            1.0
        )
    )

    directional_np = engine.render.attachNewNode(
        directional
    )

    directional_np.setHpr(
        -45,
        -60,
        0
    )

    engine.render.setLight(
        directional_np
    )


# ================================================================
# GROUND
# ================================================================

def create_ground(
    engine,
    center_x,
    center_y,
    size
):

    print("[GROUND] Creating ground...")

    cm = CardMaker(
        "yelahanka-ground"
    )

    half = size / 2.0

    cm.setFrame(
        -half,
        half,
        -half,
        half
    )

    ground = engine.render.attachNewNode(
        cm.generate()
    )

    ground.setPos(
        center_x,
        center_y,
        -10.0
    )

    # Dark grey.
    ground.setColor(
        0.15,
        0.15,
        0.15,
        1.0
    )

    ground.setTwoSided(
        True
    )

    print(
        "[GROUND] Position:",
        ground.getPos()
    )

    print(
        "[GROUND] Size:",
        size
    )

    return ground


# ================================================================
# MAP BOUNDS
# ================================================================

def get_map_bounds(roads):

    points = []

    for road in roads:

        try:

            geometries = (
                road.planView._geometries
            )

        except Exception:

            continue

        for geo in geometries:

            try:

                start = np.asarray(
                    geo.start_position,
                    dtype=np.float64
                )

            except Exception:

                continue

            if (
                start.ndim != 1
                or len(start) < 2
                or not np.all(
                    np.isfinite(start[:2])
                )
            ):

                continue

            start = start[:2]

            points.append(
                start
            )

            try:

                length = float(
                    geo.length
                )

            except Exception:

                length = 0.0

            try:

                heading = float(
                    geo.heading
                )

            except Exception:

                heading = 0.0

            if length > 0:

                end = (
                    start
                    + np.array(
                        [
                            np.cos(heading) * length,
                            np.sin(heading) * length
                        ],
                        dtype=np.float64
                    )
                )

                if np.all(
                    np.isfinite(end)
                ):

                    points.append(
                        end
                    )

    if len(points) < 2:

        raise RuntimeError(
            "Unable to determine OpenDRIVE map bounds."
        )

    points = np.asarray(
        points,
        dtype=np.float64
    )

    x_min = float(
        np.min(points[:, 0])
    )

    x_max = float(
        np.max(points[:, 0])
    )

    y_min = float(
        np.min(points[:, 1])
    )

    y_max = float(
        np.max(points[:, 1])
    )

    return (
        x_min,
        x_max,
        y_min,
        y_max
    )


# ================================================================
# DEDICATED CAMERA
# ================================================================

def setup_dedicated_camera(
    engine,
    center_x,
    center_y,
    map_width,
    map_height
):

    print()
    print("========================================")
    print("[CAMERA] Creating dedicated Panda3D camera")
    print("========================================")

    # ------------------------------------------------------------
    # Get the window.
    # ------------------------------------------------------------

    win = engine.win

    if win is None:

        raise RuntimeError(
            "MetaDrive window is not available."
        )

    # ------------------------------------------------------------
    # Disable ALL existing display regions.
    #
    # This is important.
    #
    # We do not want TestBlock's default camera/display region
    # competing with our camera.
    # ------------------------------------------------------------

    existing_regions = (
        win.getDisplayRegions()
    )

    print(
        "[CAMERA] Existing display regions:",
        len(existing_regions)
    )

    for region in existing_regions:

        try:

            region.setActive(
                False
            )

        except Exception as e:

            print(
                "[CAMERA] Could not disable region:",
                type(e).__name__,
                str(e)
            )

    # ------------------------------------------------------------
    # Create our own camera.
    # ------------------------------------------------------------

    camera_node = Camera(
        "YelahankaTopDownCamera"
    )

    camera_np = engine.render.attachNewNode(
        camera_node
    )

    # ------------------------------------------------------------
    # Orthographic projection.
    #
    # This is intentionally used instead of perspective.
    #
    # Therefore:
    #
    #   - no perspective distortion
    #   - no FOV problems
    #   - map scale is predictable
    #   - entire map can be fitted exactly
    # ------------------------------------------------------------

    lens = OrthographicLens()

    aspect = 1.0

    try:

        properties = (
            win.getProperties()
        )

        width = properties.getXSize()
        height = properties.getYSize()

        if height > 0:

            aspect = (
                float(width)
                / float(height)
            )

    except Exception:

        aspect = 1.0

    map_width = max(
        float(map_width),
        1.0
    )

    map_height = max(
        float(map_height),
        1.0
    )

    # Fit both dimensions into the viewport.
    film_height = (
        max(
            map_height,
            map_width / aspect
        )
        * VIEW_MARGIN
    )

    film_width = (
        film_height
        * aspect
    )

    lens.setFilmSize(
        film_width,
        film_height
    )

    lens.setNear(
        0.1
    )

    lens.setFar(
        100000.0
    )

    camera_node.setLens(
        lens
    )

    # ------------------------------------------------------------
    # Camera position.
    # ------------------------------------------------------------

    camera_np.setPos(
        center_x,
        center_y,
        CAMERA_HEIGHT
    )

    # ------------------------------------------------------------
    # Explicitly look at the exact map center.
    # ------------------------------------------------------------

    camera_np.lookAt(
        center_x,
        center_y,
        0.0
    )

    # ------------------------------------------------------------
    # Create our OWN display region.
    # ------------------------------------------------------------

    display_region = (
        win.makeDisplayRegion(
            0.0,
            1.0,
            0.0,
            1.0
        )
    )

    display_region.setCamera(
        camera_np
    )

    display_region.setActive(
        True
    )

    # ------------------------------------------------------------
    # Clear to a visible background.
    # ------------------------------------------------------------

    display_region.setClearColorActive(
        True
    )

    display_region.setClearColor(
        Vec4(
            0.75,
            0.75,
            0.75,
            1.0
        )
    )

    # ------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------

    print(
        "[CAMERA] Camera position:",
        camera_np.getPos()
    )

    print(
        "[CAMERA] Camera HPR:",
        camera_np.getHpr()
    )

    print(
        "[CAMERA] Looking at:",
        center_x,
        center_y,
        0.0
    )

    print(
        "[CAMERA] Orthographic film:",
        film_width,
        "x",
        film_height
    )

    print(
        "[CAMERA] Aspect ratio:",
        aspect
    )

    print(
        "[CAMERA] Near:",
        lens.getNear()
    )

    print(
        "[CAMERA] Far:",
        lens.getFar()
    )

    print(
        "[CAMERA] Dedicated display region active."
    )

    return camera_np


# ================================================================
# CENTER MARKER
# ================================================================

def create_center_marker(
    engine,
    center_x,
    center_y
):

    print(
        "[DEBUG] Creating center marker..."
    )

    # Use a cross rather than a square so it is clearly visible.
    lines = LineSegs(
        "map-center"
    )

    lines.setThickness(
        8.0
    )

    # MAGENTA.
    lines.setColor(
        1.0,
        0.0,
        1.0,
        1.0
    )

    marker_size = 100.0

    lines.moveTo(
        center_x - marker_size,
        center_y,
        50.0
    )

    lines.drawTo(
        center_x + marker_size,
        center_y,
        50.0
    )

    lines.moveTo(
        center_x,
        center_y - marker_size,
        50.0
    )

    lines.drawTo(
        center_x,
        center_y + marker_size,
        50.0
    )

    node = engine.render.attachNewNode(
        lines.create()
    )

    node.setTwoSided(
        True
    )

    return node


# ================================================================
# DEBUG LANE RENDERING
# ================================================================

def draw_debug_lanes(
    engine,
    blocks
):

    print()
    print(
        "[DEBUG] Rendering lane centerlines..."
    )

    root = engine.render.attachNewNode(
        "YelahankaDebugLanes"
    )

    rendered = 0
    total_points = 0

    for block in blocks:

        graph = getattr(
            block.block_network,
            "graph",
            {}
        )

        for lane_key, lane_info in graph.items():

            try:

                lane = getattr(
                    lane_info,
                    "lane",
                    None
                )

                if lane is None:

                    continue

                points = getattr(
                    lane,
                    "visualization_points",
                    None
                )

                if points is None:

                    points = getattr(
                        lane,
                        "points",
                        None
                    )

                if points is None:

                    continue

                if len(points) < 2:

                    continue

                lines = LineSegs(
                    "OpenDRIVE-lane"
                )

                lines.setThickness(
                    DEBUG_LINE_WIDTH
                )

                # RED.
                lines.setColor(
                    1.0,
                    0.0,
                    0.0,
                    1.0
                )

                first = points[0]

                lines.moveTo(
                    float(first[0]),
                    float(first[1]),
                    30.0
                )

                for point in points[1:]:

                    lines.drawTo(
                        float(point[0]),
                        float(point[1]),
                        30.0
                    )

                geom = lines.create()

                node = root.attachNewNode(
                    geom
                )

                node.setTwoSided(
                    True
                )

                rendered += 1

                total_points += len(
                    points
                )

            except Exception as e:

                print(
                    "[DEBUG] Lane rendering failed:",
                    lane_key,
                    type(e).__name__,
                    str(e)
                )

    print(
        "[DEBUG] Directly rendered lanes:",
        rendered
    )

    print(
        "[DEBUG] Total lane points:",
        total_points
    )

    return root


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("STARTING YELAHANKA METADRIVE")
    print("========================================")
    print()

    # ============================================================
    # ENGINE
    # ============================================================

    engine = TestBlock(
        True
    )

    initialize_asset_loader(
        engine
    )

    # ============================================================
    # LOAD OPEN DRIVE
    # ============================================================

    print(
        "[MAP] Loading:",
        XODR_PATH
    )

    odr_map = load_opendrive_map(
        XODR_PATH
    )

    print(
        "[MAP] OpenDRIVE loaded!"
    )

    print(
        "[MAP] Total roads:",
        len(odr_map.roads)
    )

    print(
        "[MAP] Total junctions:",
        len(odr_map.junctions)
    )

    roads = odr_map.roads[
        :MAX_ROADS
    ]

    print(
        "[MAP] Processing:",
        len(roads),
        "roads"
    )

    # ============================================================
    # ROAD NETWORK
    # ============================================================

    global_network = OpenDriveRoadNetwork()

    blocks = []

    block_id = 0

    # ============================================================
    # BUILD BLOCKS
    # ============================================================

    print()
    print(
        "[MAP] Building OpenDRIVE blocks..."
    )

    for road in roads:

        sections = getattr(
            road.lanes,
            "lane_sections",
            []
        )

        for section in sections:

            try:

                block = OpenDriveBlock(
                    block_id,
                    global_network,
                    0,
                    section
                )

                success = False

                try:

                    success = (
                        block.construct_block(
                            engine.render,
                            engine.physics_world
                        )
                    )

                except Exception as e:

                    print(
                        "[MAP] Construction failed:",
                        road.id,
                        type(e).__name__,
                        str(e)
                    )

                lane_count = 0

                try:

                    graph = (
                        block.block_network.graph
                    )

                    lane_count = len(
                        graph
                    )

                except Exception:

                    lane_count = 0

                print(
                    "[WORLD] road:",
                    road.id,
                    "lanes:",
                    lane_count
                )

                if success or lane_count > 0:

                    blocks.append(
                        block
                    )

                    block_id += 1

                else:

                    print(
                        "[MAP] Skipping road:",
                        road.id
                    )

            except Exception as e:

                print(
                    "[MAP] Section skipped:",
                    road.id,
                    type(e).__name__,
                    str(e)
                )

    # ============================================================
    # VALIDATION
    # ============================================================

    if not blocks:

        raise RuntimeError(
            "No valid MetaDrive blocks were created."
        )

    print()
    print(
        "[MAP] Finished!"
    )

    print(
        "[MAP] Blocks created:",
        len(blocks)
    )

    # ============================================================
    # MAP BOUNDS
    # ============================================================

    (
        x_min,
        x_max,
        y_min,
        y_max
    ) = get_map_bounds(
        roads
    )

    center_x = (
        x_min + x_max
    ) / 2.0

    center_y = (
        y_min + y_max
    ) / 2.0

    map_width = (
        x_max - x_min
    )

    map_height = (
        y_max - y_min
    )

    map_size = max(
        map_width,
        map_height
    )

    print()
    print("========================================")
    print("[MAP] BOUNDS")
    print("========================================")

    print(
        "X:",
        x_min,
        "to",
        x_max
    )

    print(
        "Y:",
        y_min,
        "to",
        y_max
    )

    print(
        "Width:",
        map_width
    )

    print(
        "Height:",
        map_height
    )

    print(
        "Center:",
        center_x,
        center_y
    )

    # ============================================================
    # ROOT TRANSFORM
    # ============================================================

    # Keep OpenDRIVE coordinates exactly as they are.
    engine.render.setPos(
        0,
        0,
        0
    )

    engine.render.setScale(
        1,
        1,
        1
    )

    # ============================================================
    # LIGHTING
    # ============================================================

    setup_lighting(
        engine
    )

    # ============================================================
    # GROUND
    # ============================================================

    if DRAW_GROUND:

        ground_size = (
            map_size
            + 2.0 * GROUND_MARGIN
        )

        create_ground(
            engine,
            center_x,
            center_y,
            ground_size
        )

    # ============================================================
    # CENTER MARKER
    # ============================================================

    if DRAW_CENTER_MARKER:

        create_center_marker(
            engine,
            center_x,
            center_y
        )

    # ============================================================
    # DEBUG LANES
    # ============================================================

    if DRAW_DEBUG_LANES:

        draw_debug_lanes(
            engine,
            blocks
        )

    # ============================================================
    # DEDICATED CAMERA
    # ============================================================

    setup_dedicated_camera(
        engine,
        center_x,
        center_y,
        map_width,
        map_height
    )

    # ============================================================
    # FINAL STATUS
    # ============================================================

    print()
    print("========================================")
    print("YELAHANKA MAP IS RUNNING")
    print("========================================")

    print(
        "Roads processed:",
        len(roads)
    )

    print(
        "Blocks created:",
        len(blocks)
    )

    print(
        "Map center:",
        center_x,
        center_y
    )

    print(
        "Map size:",
        map_width,
        "x",
        map_height
    )

    print()
    print("EXPECTED:")
    print("  GREY BACKGROUND = camera viewport")
    print("  DARK GREY       = Yelahanka ground")
    print("  RED             = OpenDRIVE lanes")
    print("  MAGENTA CROSS   = exact map center")
    print()
    print("========================================")

    # ============================================================
    # MAIN LOOP
    # ============================================================

    while True:

        engine.taskMgr.step()