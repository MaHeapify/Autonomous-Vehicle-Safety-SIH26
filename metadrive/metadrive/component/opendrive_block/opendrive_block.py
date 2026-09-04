import numpy as np

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.opendrive_lane import OpenDriveLane
from metadrive.component.road_network.edge_road_network import (
    OpenDriveRoadNetwork,
)
from metadrive.utils.opendrive.map_load import get_lane_width


class OpenDriveBlock(BaseBlock):
    """
    OpenDRIVE lane block.

    Responsibilities:
      1. Build OpenDriveLane objects and put them into the MetaDrive
         road network.
      2. Keep MetaDrive's lane-localization Bullet geometry.
      3. Independently create visible road-surface meshes.

    The important fix is that visual road geometry is NOT delegated to
    BaseBlock._construct_lane(). That method is for lane-localization
    physics, not for rendering the road surface.
    """

    ROAD_Z = 0.05
    ROAD_HALF_WIDTH_FALLBACK = 1.75
    MIN_SEGMENT_LENGTH = 1e-4

    def __init__(
        self,
        block_index,
        global_network,
        random_seed,
        section_data,
    ):
        self.section_data = section_data
        self._visual_nodes = []

        super(OpenDriveBlock, self).__init__(
            block_index,
            global_network,
            random_seed,
        )

    # ==============================================================
    # TOPOLOGY
    # ==============================================================

    def _sample_topology(self):
        created = 0

        lanes = getattr(
            self.section_data,
            "allLanes",
            [],
        )

        print("[LANE] OpenDRIVE lanes:", len(lanes))

        for lane_data in lanes:
            try:
                lane_id = getattr(
                    lane_data,
                    "id",
                    None,
                )

                if lane_id is None:
                    lane_id = getattr(
                        lane_data,
                        "index",
                        None,
                    )

                if lane_id is None:
                    getter = getattr(
                        lane_data,
                        "get_lane_id",
                        None,
                    )

                    if callable(getter):
                        try:
                            lane_id = getter()
                        except Exception:
                            lane_id = None

                if lane_id is None:
                    print(
                        "[LANE] Skipped: lane has no valid ID"
                    )
                    continue

                lane_id = int(lane_id)

                width = get_lane_width(
                    lane_data
                )

                if width is None:
                    width = 3.5

                width = float(width)

                if not np.isfinite(width) or width <= 0:
                    width = 3.5

                lane = OpenDriveLane(
                    width,
                    lane_data,
                )

                lane.index = lane_id

                if hasattr(lane, "_index"):
                    lane._index = lane_id

                if getattr(lane, "index", None) is None:
                    raise RuntimeError(
                        "Lane index is still None after creation"
                    )

                print(
                    "[LANE] Adding:",
                    lane_id,
                    "points:",
                    len(lane.visualization_points),
                    "width:",
                    width,
                )

                self.block_network.add_lane(
                    lane
                )

                created += 1

            except Exception as e:
                print(
                    "[LANE] Skipped:",
                    type(e).__name__,
                    str(e),
                )

        print(
            "[LANE] Created:",
            created
        )

        return created > 0

    # ==============================================================
    # WORLD
    # ==============================================================

    def create_in_world(self):
        """
        Create both:

          - actual visible lane surfaces
          - Bullet lane-localization geometry

        The visible surface is generated directly from each lane's
        centerline. This avoids depending on MetaDrive's internal
        visualization assumptions for PointLane.
        """

        graph = self.block_network.graph

        print(
            "[WORLD] lanes:",
            len(graph)
        )

        visible = 0
        physics = 0

        for lane_key, lane_info in graph.items():

            try:
                lane = lane_info.lane

                points = getattr(
                    lane,
                    "visualization_points",
                    None,
                )

                if points is None:
                    points = getattr(
                        lane,
                        "points",
                        None,
                    )

                if points is None:
                    print(
                        "[WORLD] No geometry:",
                        lane_key
                    )
                    continue

                points = np.asarray(
                    points,
                    dtype=np.float64,
                )

                if (
                    points.ndim != 2
                    or points.shape[1] < 2
                ):
                    print(
                        "[WORLD] Invalid geometry:",
                        lane_key
                    )
                    continue

                points = points[:, :2]

                # --------------------------------------------------
                # Remove invalid points.
                # --------------------------------------------------

                finite = np.all(
                    np.isfinite(points),
                    axis=1,
                )

                points = points[finite]

                if len(points) < 2:
                    print(
                        "[WORLD] Too few points:",
                        lane_key
                    )
                    continue

                # --------------------------------------------------
                # Remove consecutive duplicate points.
                # --------------------------------------------------

                cleaned = [
                    points[0]
                ]

                for p in points[1:]:

                    if np.linalg.norm(
                        p - cleaned[-1]
                    ) > self.MIN_SEGMENT_LENGTH:

                        cleaned.append(
                            p
                        )

                points = np.asarray(
                    cleaned,
                    dtype=np.float64,
                )

                if len(points) < 2:
                    continue

                width = float(
                    getattr(
                        lane,
                        "width",
                        3.5,
                    )
                    or 3.5
                )

                if (
                    not np.isfinite(width)
                    or width <= 0
                ):
                    width = 3.5

                # --------------------------------------------------
                # VISIBLE ROAD SURFACE
                # --------------------------------------------------

                road_np = self._create_lane_surface(
                    lane_key,
                    points,
                    width,
                )

                if road_np is not None:

                    road_np.reparentTo(
                        self.lane_node_path
                    )

                    self._visual_nodes.append(
                        road_np
                    )

                    visible += 1

                # --------------------------------------------------
                # PHYSICS / LANE LOCALIZATION
                # --------------------------------------------------

                try:

                    self._construct_lane(
                        lane,
                        lane_index=lane_key,
                    )

                    physics += 1

                except Exception as e:

                    print(
                        "[PHYSICS] Lane localization failed:",
                        lane_key,
                        type(e).__name__,
                        str(e),
                    )

            except Exception as e:

                print(
                    "[WORLD] Failed:",
                    lane_key,
                    type(e).__name__,
                    str(e),
                )

        print(
            "[WORLD] Visible lane surfaces:",
            visible,
        )

        print(
            "[WORLD] Physics lanes:",
            physics,
        )

    # ==============================================================
    # ROAD SURFACE MESH
    # ==============================================================

    def _create_lane_surface(
        self,
        lane_key,
        centerline,
        width,
    ):
        """
        Build a real triangle mesh around the lane centerline.

        For every centerline point, create a left and right boundary.

        Consecutive boundary pairs create quads:

            left A ---------------- left B
              |                       |
              |       ROAD            |
              |                       |
            right A --------------- right B

        Each quad is split into two triangles.

        This produces filled road geometry rather than line
        primitives.
        """

        if len(centerline) < 2:
            return None

        half_width = max(
            float(width) * 0.5,
            self.ROAD_HALF_WIDTH_FALLBACK,
        )

        left = []
        right = []

        for i in range(
            len(centerline)
        ):

            p = centerline[i]

            if i == 0:

                tangent = (
                    centerline[1]
                    - centerline[0]
                )

            elif i == len(centerline) - 1:

                tangent = (
                    centerline[-1]
                    - centerline[-2]
                )

            else:

                tangent = (
                    centerline[i + 1]
                    - centerline[i - 1]
                )

            norm = float(
                np.linalg.norm(
                    tangent
                )
            )

            if norm < self.MIN_SEGMENT_LENGTH:

                if i > 0:

                    tangent = (
                        centerline[i]
                        - centerline[i - 1]
                    )

                    norm = float(
                        np.linalg.norm(
                            tangent
                        )
                    )

                if norm < self.MIN_SEGMENT_LENGTH:
                    continue

            tangent = tangent / norm

            # ------------------------------------------------------
            # Left-hand normal.
            # ------------------------------------------------------

            normal = np.array(
                [
                    -tangent[1],
                    tangent[0],
                ],
                dtype=np.float64,
            )

            left.append(
                p + normal * half_width
            )

            right.append(
                p - normal * half_width
            )

        if len(left) < 2:
            return None

        # ----------------------------------------------------------
        # Vertex layout:
        #
        #   0 = left  point 0
        #   1 = right point 0
        #   2 = left  point 1
        #   3 = right point 1
        #   ...
        # ----------------------------------------------------------

        vertices = []

        for i in range(
            len(left)
        ):

            vertices.append(
                (
                    float(left[i][0]),
                    float(left[i][1]),
                    float(self.ROAD_Z),
                )
            )

            vertices.append(
                (
                    float(right[i][0]),
                    float(right[i][1]),
                    float(self.ROAD_Z),
                )
            )

        # ----------------------------------------------------------
        # Panda3D vertex data.
        # ----------------------------------------------------------

        vertex_data = GeomVertexData(
            "opendrive-road",
            GeomVertexFormat.getV3(),
            Geom.UH_static,
        )

        vertex_data.setNumRows(
            len(vertices)
        )

        vertex_writer = GeomVertexWriter(
            vertex_data,
            "vertex",
        )

        for x, y, z in vertices:

            vertex_writer.addData3f(
                x,
                y,
                z,
            )

        # ----------------------------------------------------------
        # Triangle indices.
        # ----------------------------------------------------------

        triangles = GeomTriangles(
            Geom.UH_static
        )

        for i in range(
            len(vertices) // 2 - 1
        ):

            left_a = 2 * i
            right_a = 2 * i + 1

            left_b = 2 * (i + 1)
            right_b = 2 * (i + 1) + 1

            triangles.addVertices(
                left_a,
                right_a,
                left_b,
            )

            triangles.addVertices(
                right_a,
                right_b,
                left_b,
            )

        # ----------------------------------------------------------
        # Create geometry node.
        # ----------------------------------------------------------

        geom = Geom(
            vertex_data
        )

        geom.addPrimitive(
            triangles
        )

        node = GeomNode(
            "OpenDRIVE-Road-%s"
            % str(lane_key)
        )

        node.addGeom(
            geom
        )

        road_np = NodePath(
            node
        )

        # ----------------------------------------------------------
        # Render from both sides.
        # ----------------------------------------------------------

        road_np.setTwoSided(
            True
        )

        # ----------------------------------------------------------
        # Dark road surface.
        # ----------------------------------------------------------

        road_np.setColor(
            0.22,
            0.22,
            0.22,
            1.0,
        )

        return road_np

    # ==============================================================
    # NETWORK TYPE
    # ==============================================================

    @property
    def block_network_type(self):
        return OpenDriveRoadNetwork

    # ==============================================================
    # CLEANUP
    # ==============================================================

    def destroy(self):

        self._visual_nodes.clear()

        self.section_data = None

        super(
            OpenDriveBlock,
            self
        ).destroy()