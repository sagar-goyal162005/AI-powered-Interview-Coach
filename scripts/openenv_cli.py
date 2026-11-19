import sys
from pathlib import Path

import yaml


REQUIRED_TOP_LEVEL = [
    "name",
    "version",
    "description",
    "metadata",
    "observation",
    "action",
    "reward",
    "interface",
    "tasks",
]


def _fail(message: str) -> int:
    print(f"openenv validate: FAIL - {message}")
    return 1


def _validate_yaml(path: Path) -> int:
    if not path.exists():
        return _fail(f"file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        return _fail(f"invalid yaml: {exc}")

    if not isinstance(data, dict):
        return _fail("top-level YAML must be a mapping")

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in data]
    if missing:
        return _fail(f"missing top-level keys: {', '.join(missing)}")

    interface = data.get("interface", {})
    if not isinstance(interface, dict):
        return _fail("interface section must be a mapping")

    for method_key in ["reset()", "step(action)", "state()"]:
        if method_key not in interface:
            return _fail(f"interface missing method: {method_key}")

    tasks = data.get("tasks", {})
    if not isinstance(tasks, dict):
        return _fail("tasks section must be a mapping")

    for level in ["easy", "medium", "hard"]:
        if level not in tasks:
            return _fail(f"tasks missing level: {level}")

    print("openenv validate: PASS")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: openenv validate <path-to-openenv.yaml>")
        return 2

    command = argv[1].strip().lower()
    if command != "validate":
        print(f"Unsupported command: {command}")
        print("Supported commands: validate")
        return 2

    if len(argv) < 3:
        print("Usage: openenv validate <path-to-openenv.yaml>")
        return 2

    target = Path(argv[2])
    return _validate_yaml(target)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
