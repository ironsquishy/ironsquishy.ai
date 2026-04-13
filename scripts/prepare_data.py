from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from utils.get_system_prompt import get_system_prompt

def validate_messages(messages: list[dict]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    allowed_roles = {"system", "user", "assistant"}
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"message {i} must be an object")
        if msg.get("role") not in allowed_roles:
            raise ValueError(f"message {i} has invalid role")
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            raise ValueError(f"message {i} must have non-empty content")


def add_system_prompt(messages: list[dict]) -> list[dict]:
    system_prompt = get_system_prompt()
    output = []
    for message in messages:
        new_message = message
        if new_message["role"] == "system":
            new_message["content"] = system_prompt
        
        output.append(message)

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                messages = row["messages"]
                validate_messages(messages)
                messages = add_system_prompt(messages)
                outfile.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                kept += 1
            except Exception as exc:
                skipped += 1
                print(f"Skipping line {line_num}: {exc}")

    print(f"Done. kept={kept} skipped={skipped} output={output_path}")


if __name__ == "__main__":
    main()