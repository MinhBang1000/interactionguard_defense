"""Extract source payloads, never labels or agent-generated reasoning, for Screen."""
import re

SECTION = re.compile(r"^\[(PROMPT|RETRIEVE|TOOLS|TOOL:[^\]]+|AGENT|SEP|EOS)\][ \t]*$", re.M)
DOCUMENT = re.compile(r"^\[Doc \d+\][ \t]*$", re.M)


def channel_payloads(sequence):
    matches = list(SECTION.finditer(sequence))
    if not matches:
        return [("prompt", sequence.strip())] if sequence.strip() else []
    result = []
    for index, match in enumerate(matches):
        tag = match.group(1)
        if tag in {"AGENT", "SEP", "EOS"}:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(sequence)
        content = sequence[match.end():end].strip()
        channel = "prompt" if tag == "PROMPT" else "retrieval" if tag == "RETRIEVE" else "tool"
        parts = DOCUMENT.split(content) if channel == "retrieval" else [content]
        result.extend((channel, part.strip()) for part in parts if part.strip())
    return result
