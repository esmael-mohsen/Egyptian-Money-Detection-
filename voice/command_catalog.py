from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voice.commands import CommandId, normalize_text


@dataclass(frozen=True)
class CommandSpec:
    command_id: CommandId
    aliases_en: tuple[str, ...]
    aliases_ar: tuple[str, ...]
    misrecognitions: tuple[str, ...]
    requires_amount: bool = False
    confirmation_required: bool = False

    @property
    def all_aliases(self) -> tuple[str, ...]:
        return self.aliases_en + self.aliases_ar + self.misrecognitions


class CommandCatalog:
    def __init__(self, specs: tuple[CommandSpec, ...]) -> None:
        self.specs = specs

    @classmethod
    def load(cls, path: Path) -> "CommandCatalog":
        data = json.loads(path.read_text(encoding="utf-8"))
        specs: list[CommandSpec] = []
        for command_name, values in data.get("commands", {}).items():
            specs.append(
                CommandSpec(
                    command_id=CommandId(command_name),
                    aliases_en=tuple(values.get("aliases_en", ())),
                    aliases_ar=tuple(values.get("aliases_ar", ())),
                    misrecognitions=tuple(values.get("misrecognitions", ())),
                    requires_amount=bool(values.get("requires_amount", False)),
                    confirmation_required=bool(values.get("confirmation_required", False)),
                )
            )
        return cls(tuple(specs))

    @classmethod
    def default(cls) -> "CommandCatalog":
        return cls.load(Path("voice/command_catalog.yaml"))

    def spec_for(self, command_id: CommandId) -> CommandSpec | None:
        for spec in self.specs:
            if spec.command_id == command_id:
                return spec
        return None

    def normalized_aliases(self) -> list[tuple[str, CommandSpec]]:
        aliases: list[tuple[str, CommandSpec]] = []
        for spec in self.specs:
            for alias in spec.all_aliases:
                aliases.append((normalize_text(alias), spec))
        return sorted(aliases, key=lambda item: len(item[0]), reverse=True)
