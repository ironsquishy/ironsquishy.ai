from pathlib import Path

SYSTEM_PROMPT = ""

base_dir = Path(__file__).resolve().parent.parent

system_prompt_file_path = base_dir / "prompts" / "system.prompt.md"

try:
    print("Attempting to opening system prompt")
    with open(system_prompt_file_path, "r", encoding="utf-8") as file:
        content = file.read()
        SYSTEM_PROMPT= content.strip()

except FileNotFoundError:

    print("Creating a new file instead...")
    with open(system_prompt_file_path, 'w') as file:

        content = """
        You are the infrastructure assistant:

        Rules:
            - prefer Tailscale for private traffic
            - recommend secure reverse proxy defaults
            - do not expose the model API publicly
            - be concise, practical, and explicit about tradeoffs
            - return shell commands in fenced bash blocks when relevant
        """

        SYSTEM_PROMPT = content.strip()

        file.write(content)


def build_prompt(user_request: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_request}\n"
        f"<|assistant|>\n"
    )
