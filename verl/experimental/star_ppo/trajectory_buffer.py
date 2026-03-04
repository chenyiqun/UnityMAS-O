import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from verl import DataProto


@dataclass
class TrajectoryEntry:
    traj_id: str
    model_id: str
    query_id: str
    agent_id: str
    fat_data: DataProto
    reward: Optional[torch.Tensor] = None
    done: bool = False
    created_at: float = field(default_factory=time.time)


class TrajectoryBuffer:
    """In-worker trajectory buffer for fat-data residency."""

    def __init__(self, max_items: int, ttl_seconds: int):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.entries: OrderedDict[str, TrajectoryEntry] = OrderedDict()
        self.ready_queue: deque[str] = deque()

    def _evict_expired(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        expired = [k for k, v in self.entries.items() if now - v.created_at > self.ttl_seconds]
        for key in expired:
            self.entries.pop(key, None)

    def _evict_overflow(self) -> None:
        while len(self.entries) > self.max_items:
            oldest_key = next(iter(self.entries.keys()))
            self.entries.pop(oldest_key, None)

    def put(self, entry: TrajectoryEntry) -> None:
        self._evict_expired()
        self.entries[entry.traj_id] = entry
        self._evict_overflow()

    def commit_reward(self, traj_id: str, reward: torch.Tensor, done: bool) -> bool:
        self._evict_expired()
        entry = self.entries.get(traj_id)
        if entry is None:
            return False

        entry.reward = reward.detach().cpu()
        entry.done = bool(done)
        if entry.done:
            self.ready_queue.append(traj_id)
        return True

    def pop_ready(self, max_items: int | None = None) -> list[TrajectoryEntry]:
        self._evict_expired()
        out: list[TrajectoryEntry] = []
        limit = max_items if max_items is not None and max_items > 0 else len(self.ready_queue)
        for _ in range(min(limit, len(self.ready_queue))):
            traj_id = self.ready_queue.popleft()
            entry = self.entries.pop(traj_id, None)
            if entry is not None:
                out.append(entry)
        return out

    def stats(self) -> dict[str, int]:
        self._evict_expired()
        return {
            "buffer/total": len(self.entries),
            "buffer/ready": len(self.ready_queue),
        }
