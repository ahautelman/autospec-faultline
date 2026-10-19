"""Room-level graph analysis for a full-edge-door room grid.

Doors on an n-rooms layout open a whole room edge, so continuous reachability
between rooms is exactly room-graph connectivity. This module therefore works
on room indices only: no continuous geometry is needed.

A door map handed to :class:`RoomGrid` must contain only full-edge intervals.
A partially open gap (the 9-rooms narrow-door scenario uses one) is rejected
rather than guessed at, because room adjacency would no longer be a yes/no
question. The generic :class:`~faultline.envs.rooms.layout.RoomsLayout` still
accepts partial gaps for the continuous environment and the 9-rooms catalogue;
only this analysis path is strict.

The module has no dependency on the gymnasium environment, so it can move to a
fault-injection analysis package unchanged.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isclose

from faultline.envs.rooms.layout import (
    DEFAULT_ROOM_SIZE,
    DoorInterval,
    WallKey,
    full_edge_interval,
)
from faultline.envs.rooms.regions import RegionBlock

RoomKey = tuple[int, int]

_WALL_KEY_LENGTH = 3  # (axis, col, row)


@dataclass(frozen=True)
class RoomPath:
    """A shortest room path through open doors.

    Args:
        rooms: The room sequence, including both endpoints.

    How:
        ``length`` is the number of door crossings, so a one-room path has
        length zero. That is the unit the horizon check multiplies by
        ``room_size / speed``.
    """

    rooms: tuple[RoomKey, ...]

    @property
    def length(self) -> int:
        """Return the number of door crossings on the path.

        How:
            Counts the door crossings between consecutive rooms.

        Returns:
            The path length in door crossings.
        """
        return max(0, len(self.rooms) - 1)


class RoomGrid:
    """Connectivity analysis over an ``n x n`` grid of rooms and full-edge doors.

    Args:
        rooms_per_side: Number of rooms along each axis; must be at least 1.
        doors: Wall key to full-edge open interval mapping.
        room_size: Width and height of one room.

    How:
        Validates that every door covers exactly its room edge, then answers
        connectivity, shortest-path, cut-room, sealed-room and viability
        queries. Neighbour order and component order are deterministic, so
        results are reproducible.

    Raises:
        ValueError: If a grid size is invalid, a wall key is outside the grid,
            or a door interval is not the wall's full room edge.
    """

    def __init__(
        self,
        rooms_per_side: int,
        doors: Mapping[WallKey, DoorInterval],
        room_size: float = DEFAULT_ROOM_SIZE,
    ) -> None:
        self.rooms_per_side = int(rooms_per_side)
        self.room_size = float(room_size)
        if self.rooms_per_side < 1:
            raise ValueError(f"rooms_per_side must be positive, got {rooms_per_side}")
        if self.room_size <= 0:
            raise ValueError(f"room_size must be positive, got {room_size}")
        self._doors = {
            key: self._validated_interval(key, interval) for key, interval in doors.items()
        }
        self._rooms = tuple(
            (col, row)
            for row in range(self.rooms_per_side)
            for col in range(self.rooms_per_side)
        )

    @property
    def doors(self) -> dict[WallKey, DoorInterval]:
        """Return a copy of the validated full-edge door map.

        How:
            Copies the stored map so callers cannot change the grid.

        Returns:
            The validated full-edge door map.
        """
        return dict(self._doors)

    @property
    def rooms(self) -> tuple[RoomKey, ...]:
        """Return every room in row-major order.

        How:
            Returns the tuple built once at construction.

        Returns:
            Every room key, row-major.
        """
        return self._rooms

    def _validated_interval(self, key: WallKey, interval: DoorInterval) -> DoorInterval:
        """Validate one wall key and its full-edge interval.

        Args:
            key: The wall key to validate.
            interval: The claimed open interval.

        How:
            Checks the axis tag, the wall's position inside the grid, and that
            both interval endpoints match the wall's whole room edge. Floating
            point tolerance is tiny; generated intervals are exact products of
            the room size.

        Returns:
            The interval as a pair of floats.

        Raises:
            ValueError: If the key or interval is invalid.
        """
        if not isinstance(key, tuple) or len(key) != _WALL_KEY_LENGTH:
            raise ValueError(f"Invalid wall key {key!r}")
        axis, col, row = key
        n = self.rooms_per_side
        if axis == "v":
            if not (0 <= col < n - 1 and 0 <= row < n):
                raise ValueError(f"Vertical wall {key!r} is outside a {n}x{n} grid")
        elif axis == "h":
            if not (0 <= col < n and 0 <= row < n - 1):
                raise ValueError(f"Horizontal wall {key!r} is outside a {n}x{n} grid")
        else:
            raise ValueError(f"Wall axis must be 'v' or 'h', got {axis!r}")
        expected = full_edge_interval(key, self.room_size)
        low, high = float(interval[0]), float(interval[1])
        if not (
            isclose(low, expected[0], abs_tol=1e-9)
            and isclose(high, expected[1], abs_tol=1e-9)
        ):
            raise ValueError(
                f"RoomGrid needs full-edge doors: {key!r} has {interval!r}, "
                f"expected {expected!r}"
            )
        return (low, high)

    def _wall_between(self, first: RoomKey, second: RoomKey) -> WallKey | None:
        """Return the wall key joining two orthogonally adjacent rooms.

        Args:
            first: One room.
            second: Another room.

        How:
            A vertical wall joins rooms that share a row and differ by one
            column; a horizontal wall joins rooms that share a column and differ
            by one row. Anything else is not adjacent.

        Returns:
            The shared wall key, or ``None`` when the rooms are not adjacent.
        """
        (c0, r0), (c1, r1) = first, second
        if r0 == r1 and abs(c0 - c1) == 1:
            return ("v", min(c0, c1), r0)
        if c0 == c1 and abs(r0 - r1) == 1:
            return ("h", c0, min(r0, r1))
        return None

    def _inside(self, room: RoomKey) -> bool:
        """Return whether a room key lies inside the grid.

        Args:
            room: The room key to test.

        How:
            Compares both indices with the grid bounds.

        Returns:
            ``True`` when the room is inside the grid.
        """
        col, row = room
        return 0 <= col < self.rooms_per_side and 0 <= row < self.rooms_per_side

    def _check_rooms(self, rooms: Iterable[RoomKey], label: str) -> frozenset[RoomKey]:
        """Validate a room collection and return it as a set.

        Args:
            rooms: Room keys to validate.
            label: Name used in the error message.

        How:
            Rejects any room outside the grid. An empty collection is allowed;
            callers that need a non-empty one check it themselves.

        Returns:
            The rooms as a frozenset.

        Raises:
            ValueError: If any room is outside the grid.
        """
        checked = frozenset(rooms)
        for room in checked:
            if not self._inside(room):
                raise ValueError(f"{label} room {room!r} is outside the grid")
        return checked

    def neighbours(self, room: RoomKey) -> tuple[RoomKey, ...]:
        """Return the adjacent rooms reachable through an open door.

        Args:
            room: The room to look around.

        How:
            Tests the up to four orthogonal neighbours and keeps those whose
            shared wall key is in the full-edge door map. The result is sorted,
            so traversal order is deterministic.

        Returns:
            The connected neighbouring rooms.

        Raises:
            ValueError: If ``room`` is outside the grid.
        """
        checked = self._check_rooms((room,), "Neighbour")
        (col, row), = checked
        candidates = ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1))
        return tuple(
            sorted(
                neighbour
                for neighbour in candidates
                if self._inside(neighbour)
                and self._wall_between((col, row), neighbour) in self._doors
            )
        )

    def components(self) -> tuple[frozenset[RoomKey], ...]:
        """Return the connected components of the room graph.

        How:
            Floods from the smallest unvisited room until every room is covered.
            Components come out in ascending order of their smallest room, and
            each is a frozenset, so the result is deterministic.

        Returns:
            Every connected component.
        """
        remaining = set(self._rooms)
        components: list[frozenset[RoomKey]] = []
        while remaining:
            component = frozenset(self._flood({min(remaining)}))
            components.append(component)
            remaining -= component
        return tuple(components)

    def _flood(
        self, sources: Iterable[RoomKey], within: frozenset[RoomKey] | set[RoomKey] | None = None
    ) -> set[RoomKey]:
        """Return every room reachable from ``sources`` through open doors.

        Args:
            sources: Rooms the flood starts from; always part of the result.
            within: If given, the flood only enters rooms in this set.

        How:
            Breadth-first search with sorted neighbours. It is the one traversal
            behind components, reachability, viability and region splits.

        Returns:
            The reached rooms.
        """
        reached = set(sources)
        queue: deque[RoomKey] = deque(sorted(reached))
        while queue:
            room = queue.popleft()
            for neighbour in self.neighbours(room):
                if neighbour not in reached and (within is None or neighbour in within):
                    reached.add(neighbour)
                    queue.append(neighbour)
        return reached

    def component_of(self, room: RoomKey) -> frozenset[RoomKey]:
        """Return the connected component containing ``room``.

        Args:
            room: The room to locate.

        How:
            Scans :meth:`components` for the component holding the room.

        Returns:
            The component.

        Raises:
            ValueError: If ``room`` is outside the grid.
        """
        checked = self._check_rooms((room,), "Component")
        (target,) = checked
        for component in self.components():
            if target in component:
                return component
        raise AssertionError(f"Room {target!r} is missing from every component")

    def shortest_path(
        self, source_rooms: Iterable[RoomKey], target_rooms: Iterable[RoomKey]
    ) -> RoomPath | None:
        """Return a shortest path between any source and any target room.

        Args:
            source_rooms: Non-empty collection of start rooms.
            target_rooms: Non-empty collection of goal rooms.

        How:
            Multi-source breadth-first search through open doors. Sources and
            neighbours are visited in sorted order, so ties resolve
            deterministically.

        Returns:
            The path, or ``None`` when no source can reach any target.

        Raises:
            ValueError: If either collection is empty or contains an outside room.
        """
        sources = self._check_rooms(source_rooms, "Source")
        targets = self._check_rooms(target_rooms, "Target")
        if not sources or not targets:
            raise ValueError("shortest_path needs at least one source and one target room")
        overlap = sources & targets
        if overlap:
            return RoomPath((min(overlap),))
        parents: dict[RoomKey, RoomKey] = {}
        visited = set(sources)
        queue: deque[RoomKey] = deque(sorted(sources))
        while queue:
            room = queue.popleft()
            if room in targets:
                return RoomPath(self._reconstruct(room, parents))
            for neighbour in self.neighbours(room):
                if neighbour not in visited:
                    visited.add(neighbour)
                    parents[neighbour] = room
                    queue.append(neighbour)
        return None

    def _reconstruct(
        self, room: RoomKey, parents: Mapping[RoomKey, RoomKey]
    ) -> tuple[RoomKey, ...]:
        """Reconstruct a path from breadth-first parent links.

        Args:
            room: The room the path ends at.
            parents: Breadth-first parent links.

        How:
            Follows the parent links back to a source and reverses them.

        Returns:
            The rooms from a source to ``room``.
        """
        path = [room]
        while path[-1] in parents:
            path.append(parents[path[-1]])
        return tuple(reversed(path))

    def _reachable(
        self,
        sources: Iterable[RoomKey],
        targets: Iterable[RoomKey],
        blocked: frozenset[RoomKey] = frozenset(),
    ) -> bool:
        """Return whether any source reaches any target while avoiding ``blocked``.

        Args:
            sources: Rooms the search starts from.
            targets: Rooms to reach.
            blocked: Rooms the route may not enter.

        How:
            Floods from the unblocked sources without entering a blocked room.
            Used by the brute-force cut-room scan and kept private so the public
            API exposes only graph facts.

        Returns:
            ``True`` when an unblocked source reaches an unblocked target.
        """
        open_sources = set(sources) - blocked
        open_targets = set(targets) - blocked
        allowed = set(self._rooms) - blocked
        return bool(self._flood(open_sources, within=allowed) & open_targets)

    def cut_rooms(
        self, source_rooms: Iterable[RoomKey], target_rooms: Iterable[RoomKey]
    ) -> frozenset[RoomKey]:
        """Return the rooms whose removal disconnects sources from targets.

        Args:
            source_rooms: Rooms the route starts in.
            target_rooms: Rooms the route must reach.

        How:
            If no route exists already, there is nothing to cut and the result is
            empty. Otherwise every room other than the endpoints is removed in
            turn and reachability is re-tested. The endpoints are never reported
            as cut rooms. An exhaustive scan is cheap at n=10.

        Returns:
            The cut rooms, as a frozenset.

        Raises:
            ValueError: If either collection is empty or contains an outside room.
        """
        sources = self._check_rooms(source_rooms, "Source")
        targets = self._check_rooms(target_rooms, "Target")
        if not sources or not targets:
            raise ValueError("cut_rooms needs at least one source and one target room")
        if not self._reachable(sources, targets):
            return frozenset()
        candidates = [room for room in self._rooms if room not in sources | targets]
        return frozenset(
            room
            for room in candidates
            if not self._reachable(sources, targets, frozenset((room,)))
        )

    def sealed_rooms(self, reference_rooms: Iterable[RoomKey]) -> frozenset[RoomKey]:
        """Return the rooms outside the reference component.

        Args:
            reference_rooms: Non-empty collection of rooms that define the main
                component.

        How:
            Uses the component of the first room in sorted order. This makes the
            result deterministic when the reference rooms happen to span
            separate components, which a verification report flags separately.

        Returns:
            Every room not connected to the reference.

        Raises:
            ValueError: If the collection is empty or contains an outside room.
        """
        references = self._check_rooms(reference_rooms, "Reference")
        if not references:
            raise ValueError("sealed_rooms needs at least one reference room")
        main = self.component_of(min(references))
        return frozenset(set(self._rooms) - set(main))

    def region_rooms(self, block: RegionBlock) -> tuple[RoomKey, ...]:
        """Return a block's four rooms.

        Args:
            block: The 2x2 region block.

        How:
            Delegates to :meth:`RegionBlock.rooms` after checking the block is
            inside the grid.

        Returns:
            The block's rooms.

        Raises:
            ValueError: If any block room is outside the grid.
        """
        rooms = block.rooms()
        self._check_rooms(rooms, "Region")
        return rooms

    def region_internal_components(self, block: RegionBlock) -> tuple[frozenset[RoomKey], ...]:
        """Return the connected components inside one region block.

        Args:
            block: The 2x2 region block.

        How:
            Runs connectivity only over the block's own four rooms, so a region
            split by an internal solid wall reports more than one component even
            when both halves connect to the rest of the grid. This is the
            diagnostic behind a predicted PastRefine symptom.

        Returns:
            The block's internal components, ordered by their smallest room.

        Raises:
            ValueError: If any block room is outside the grid.
        """
        rooms = frozenset(self.region_rooms(block))
        remaining = set(rooms)
        components: list[frozenset[RoomKey]] = []
        while remaining:
            component = frozenset(self._flood({min(remaining)}, within=rooms))
            components.append(component)
            remaining -= component
        return tuple(components)

    def viability(
        self, target_rooms: Iterable[RoomKey], safe_rooms: Iterable[RoomKey]
    ) -> frozenset[RoomKey]:
        """Return the rooms that can reach a target while staying inside ``safe_rooms``.

        Args:
            target_rooms: Rooms the agent wants to reach.
            safe_rooms: Rooms the whole route must stay inside.

        How:
            Backward breadth-first search from the target rooms over the
            undirected door graph, expanding only through rooms in
            ``safe_rooms``. Targets are always included even if the caller left
            them out of ``safe_rooms``. This is the spec's viability set: the
            rooms from which a safe route to a target still exists.

        Returns:
            The viable rooms, as a frozenset.

        Raises:
            ValueError: If a target is outside the grid.
        """
        targets = self._check_rooms(target_rooms, "Target")
        safe = self._check_rooms(safe_rooms, "Safe")
        return frozenset(self._flood(targets, within=safe | targets))
