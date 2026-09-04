import math
import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.utils.opendrive.elements.geometry import (
    Line,
    Arc,
    ParamPoly3,
)


class OpenDriveLane(PointLane):

    ARC_SEGMENT_LENGTH = 5.0
    PARAMPOLY3_SEGMENT_LENGTH = 5.0

    def __init__(
        self,
        width,
        lane_data
    ):

        self.lane_data = lane_data

        self.width = max(
            float(width),
            0.1
        )

        # ========================================================
        # LANE ID
        # ========================================================

        lane_id = getattr(
            lane_data,
            "id",
            None
        )

        if lane_id is None:

            lane_id = getattr(
                lane_data,
                "index",
                None
            )

        if lane_id is None:

            getter = getattr(
                lane_data,
                "get_lane_id",
                None
            )

            if callable(getter):

                try:
                    lane_id = getter()

                except Exception:
                    lane_id = None

        if lane_id is None:

            raise ValueError(
                "OpenDRIVE lane has no id/index"
            )

        lane_id = int(
            lane_id
        )

        self.index = lane_id

        # ========================================================
        # METADATA
        # ========================================================

        lane_section = getattr(
            lane_data,
            "lane_section",
            None
        )

        self.single_side = getattr(
            lane_section,
            "singleSide",
            False
        )

        self._section_index = getattr(
            lane_section,
            "idx",
            0
        )

        road_mark = getattr(
            lane_data,
            "roadMark",
            {}
        )

        if road_mark is None:
            road_mark = {}

        self.roadMark_color = road_mark.get(
            "color",
            None
        )

        self.roadMark_type = road_mark.get(
            "type",
            None
        )

        self.roadMark_material = road_mark.get(
            "material",
            None
        )

        # ========================================================
        # ROAD
        # ========================================================

        parent_road = getattr(
            lane_data,
            "parentRoad",
            None
        )

        if parent_road is None:

            raise ValueError(
                "OpenDRIVE lane has no parentRoad"
            )

        plan_view = getattr(
            parent_road,
            "planView",
            None
        )

        if plan_view is None:

            raise ValueError(
                "OpenDRIVE road has no planView"
            )

        geometries = getattr(
            plan_view,
            "_geometries",
            None
        )

        if not geometries:

            raise ValueError(
                "OpenDRIVE road has no geometries"
            )

        # ========================================================
        # REFERENCE LINE
        # ========================================================

        points = self._build_lane_centerline(
            geometries
        )

        if len(points) < 2:

            raise ValueError(
                "Could not generate lane centerline"
            )

        points = np.asarray(
            points,
            dtype=np.float64
        )

        # ========================================================
        # FINITE
        # ========================================================

        finite = np.all(
            np.isfinite(points),
            axis=1
        )

        points = points[
            finite
        ]

        if len(points) < 2:

            raise ValueError(
                "Lane contains fewer than two finite points"
            )

        # ========================================================
        # REMOVE DUPLICATES
        # ========================================================

        cleaned = [
            points[0]
        ]

        for point in points[1:]:

            if np.linalg.norm(
                point - cleaned[-1]
            ) > 1e-5:

                cleaned.append(
                    point
                )

        points = np.asarray(
            cleaned,
            dtype=np.float64
        )

        # ========================================================
        # OFFSET
        # ========================================================

        points = self._offset_lane(
            points
        )

        if not np.all(
            np.isfinite(points)
        ):

            raise ValueError(
                "Offset lane contains invalid coordinates"
            )

        # ========================================================
        # POINT LANE
        # ========================================================

        super(
            OpenDriveLane,
            self
        ).__init__(
            points,
            self.width
        )

        # ========================================================
        # OWN STORAGE
        # ========================================================

        self.points = np.asarray(
            points,
            dtype=np.float32
        ).copy()

        self.visualization_points = (
            self.points.copy()
        )

        self.start = (
            self.points[0].copy()
        )

        self.end = (
            self.points[-1].copy()
        )

        self.index = lane_id

        if hasattr(
            self,
            "_index"
        ):

            self._index = lane_id

    # ============================================================
    # REFERENCE CENTERLINE
    # ============================================================

    def _build_lane_centerline(
        self,
        geometries
    ):

        points = []

        for geo in geometries:

            # ====================================================
            # LINE
            # ====================================================

            if isinstance(
                geo,
                Line
            ):

                length = float(
                    geo.length
                )

                if length <= 1e-6:
                    continue

                start = np.asarray(
                    geo.start_position,
                    dtype=np.float64
                )

                heading = float(
                    geo.heading
                )

                count = max(
                    2,
                    int(
                        math.ceil(
                            length
                            / self.PARAMPOLY3_SEGMENT_LENGTH
                        )
                    ) + 1
                )

                for i in range(
                    count
                ):

                    s = (
                        length
                        * i
                        / (count - 1)
                    )

                    point = (
                        start
                        + np.array(
                            [
                                np.cos(heading) * s,
                                np.sin(heading) * s
                            ],
                            dtype=np.float64
                        )
                    )

                    self._append_point(
                        points,
                        point
                    )

            # ====================================================
            # ARC
            # ====================================================

            elif isinstance(
                geo,
                Arc
            ):

                length = float(
                    geo.length
                )

                if length <= 1e-6:
                    continue

                for point in self._arc_points(
                    geo.start_position,
                    float(geo.heading),
                    float(geo.curvature),
                    length
                ):

                    self._append_point(
                        points,
                        point
                    )

            # ====================================================
            # PARAMPOLY3
            # ====================================================

            elif isinstance(
                geo,
                ParamPoly3
            ):

                length = float(
                    geo.length
                )

                if length <= 1e-6:
                    continue

                count = max(
                    2,
                    int(
                        math.ceil(
                            length
                            / self.PARAMPOLY3_SEGMENT_LENGTH
                        )
                    ) + 1
                )

                for i in range(
                    count
                ):

                    s = (
                        length
                        * i
                        / (count - 1)
                    )

                    try:

                        position, _ = (
                            geo.calc_position(
                                float(s)
                            )
                        )

                    except Exception:

                        continue

                    position = np.asarray(
                        position,
                        dtype=np.float64
                    )

                    if np.all(
                        np.isfinite(position)
                    ):

                        self._append_point(
                            points,
                            position
                        )

        return points

    # ============================================================
    # OFFSET
    # ============================================================

    def _offset_lane(
        self,
        points
    ):

        lane_id = int(
            self.index
        )

        if lane_id == 0:

            return points

        offset = (
            abs(lane_id) - 0.5
        ) * self.width

        result = []

        for i in range(
            len(points)
        ):

            current = points[i]

            if i == 0:

                tangent = (
                    points[1]
                    - points[0]
                )

            elif i == len(points) - 1:

                tangent = (
                    points[-1]
                    - points[-2]
                )

            else:

                tangent = (
                    points[i + 1]
                    - points[i - 1]
                )

            norm = np.linalg.norm(
                tangent
            )

            if norm < 1e-8:

                result.append(
                    current.copy()
                )

                continue

            tangent = tangent / norm

            normal = np.array(
                [
                    -tangent[1],
                    tangent[0]
                ],
                dtype=np.float64
            )

            signed_offset = (
                offset
                if lane_id > 0
                else -offset
            )

            result.append(
                current
                + normal * signed_offset
            )

        return np.asarray(
            result,
            dtype=np.float64
        )

    # ============================================================
    # ARC
    # ============================================================

    def _arc_points(
        self,
        start_position,
        heading,
        curvature,
        length
    ):

        start = np.asarray(
            start_position,
            dtype=np.float64
        )

        heading = float(
            heading
        )

        curvature = float(
            curvature
        )

        length = float(
            length
        )

        if length <= 1e-6:
            return []

        count = max(
            2,
            int(
                math.ceil(
                    length
                    / self.ARC_SEGMENT_LENGTH
                )
            ) + 1
        )

        points = []

        if abs(
            curvature
        ) < 1e-10:

            for i in range(
                count
            ):

                s = (
                    length
                    * i
                    / (count - 1)
                )

                points.append(
                    start
                    + np.array(
                        [
                            np.cos(heading) * s,
                            np.sin(heading) * s
                        ],
                        dtype=np.float64
                    )
                )

            return points

        for i in range(
            count
        ):

            s = (
                length
                * i
                / (count - 1)
            )

            angle = (
                heading
                + curvature * s
            )

            x = (
                start[0]
                + (
                    np.sin(angle)
                    - np.sin(heading)
                )
                / curvature
            )

            y = (
                start[1]
                - (
                    np.cos(angle)
                    - np.cos(heading)
                )
                / curvature
            )

            point = np.array(
                [
                    x,
                    y
                ],
                dtype=np.float64
            )

            if np.all(
                np.isfinite(point)
            ):

                points.append(
                    point
                )

        return points

    # ============================================================
    # APPEND
    # ============================================================

    @staticmethod
    def _append_point(
        points,
        point
    ):

        point = np.asarray(
            point,
            dtype=np.float64
        )

        if point.shape[0] < 2:
            return

        point = point[:2]

        if not np.all(
            np.isfinite(point)
        ):
            return

        if not points:

            points.append(
                point
            )

            return

        if np.linalg.norm(
            point - points[-1]
        ) > 1e-5:

            points.append(
                point
            )

    # ============================================================
    # ROAD MARKING
    # ============================================================

    def is_lane_line(self):

        return self.roadMark_type in [
            "solid",
            "broken",
            "broken broken",
            "solid solid",
            "solid broken",
            "broken solid",
        ]

    # ============================================================
    # CLEANUP
    # ============================================================

    def destroy(self):

        self.width = None
        self.points = None
        self.visualization_points = None
        self.start = None
        self.end = None
        self.lane_data = None

        super(
            OpenDriveLane,
            self
        ).destroy()