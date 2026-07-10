"""Per-molecule manifest: the record + the idempotency ledger.

`manifest.json` is both the human-readable provenance for a molecule and the
state that lets each pipeline step be skipped on re-run once it has completed.
Every step appends a `StepRecord` (status, outputs, timing); `is_done` lets the
pipeline resume where it left off.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class StepRecord:
    name: str
    status: str  # "ok" | "failed" | "skipped"
    seconds: float = 0.0
    outputs: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


@dataclass
class Manifest:
    mol_id: str
    smiles: str
    path: Path
    mol_class: str = ""
    class_evidence: list[str] = field(default_factory=list)
    needs_review: bool = False
    ph: float | None = None
    protonation_state: str = ""
    net_charge: int | None = None
    charge_method: str = ""
    forcefield: str = ""
    water_model_planned: str = ""
    random_seed: int | None = None
    tool_versions: dict = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)
    handoff_files: list[str] = field(default_factory=list)

    # --- step ledger --------------------------------------------------------
    def record(self, step: StepRecord) -> None:
        self.steps = [s for s in self.steps if s.name != step.name] + [step]
        self.save()

    def is_done(self, name: str) -> bool:
        return any(s.name == name and s.status == "ok" for s in self.steps)

    # --- persistence --------------------------------------------------------
    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["path"] = str(self.path)
        (self.path / "manifest.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load_or_new(cls, mol_id: str, smiles: str, path: Path) -> "Manifest":
        f = path / "manifest.json"
        if f.exists():
            data = json.loads(f.read_text())
            steps = [StepRecord(**s) for s in data.pop("steps", [])]
            data.pop("path", None)
            m = cls(path=path, steps=steps, **data)
            return m
        return cls(mol_id=mol_id, smiles=smiles, path=path)
