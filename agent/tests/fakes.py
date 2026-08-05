"""The fakes the agent turn seam stands on. Gemini and FoodData Central are
faked by `NotWired`, which raises the moment a node reaches for them."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from nutrigraph_agent.models import Profile

DEMO_PROFILE = Profile(
    user_id="demo-user-1",
    name="Lou",
    sex="M",
    age=24,
    height_cm=172,
    weight_kg=78,
    target_weight_kg=72,
    activity_level="light",
    diet_pattern="omnivore",
    allergies=["peanut"],
)


@dataclass
class StoredMessage:
    user_id: str
    turn_id: UUID
    role: str
    raw_text: str


@dataclass
class FakeDatabase:
    profiles: dict[str, Profile] = field(
        default_factory=lambda: {DEMO_PROFILE.user_id: DEMO_PROFILE}
    )
    messages: list[StoredMessage] = field(default_factory=list)
    fail_on_load: bool = False

    async def load_profile(self, user_id: str) -> Profile | None:
        if self.fail_on_load:
            raise ConnectionError("database is gone")
        return self.profiles.get(user_id)

    async def store_message(
        self, *, user_id: str, turn_id: UUID, role: str, raw_text: str
    ) -> None:
        self.messages.append(StoredMessage(user_id, turn_id, role, raw_text))
