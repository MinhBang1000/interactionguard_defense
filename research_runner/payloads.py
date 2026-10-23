"""Pure, byte-compatible O2 payload construction and controlled channel ablations."""
import hashlib
import re

SECTION = re.compile(r"^\[(PROMPT|RETRIEVE|TOOLS|TOOL:[^\]]+|AGENT|SEP|EOS)\]\s*$", re.M)


def parse(sequence):
    sections = {key: [] for key in ('PROMPT', 'RETRIEVE', 'TOOLS', 'AGENT')}
    matches = list(SECTION.finditer(sequence))
    for i, match in enumerate(matches):
        tag = match.group(1)
        if tag in {'SEP', 'EOS'}:
            continue
        content = sequence[match.end():matches[i+1].start() if i+1 < len(matches) else len(sequence)].strip()
        if content:
            sections['TOOLS' if tag.startswith('TOOL:') else tag].append(content)
    return sections


def trim_middle(text, limit):
    if limit <= 0 or len(text) <= limit:
        return text
    marker = '\n[... middle truncated for Layer 3 O2 ...]\n'
    if limit <= len(marker) + 32:
        return text[:limit]
    head = int(limit * .65)
    tail = max(16, limit-head-len(marker))
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def build_payload(sequence, settings, channel=None, anchoring='original_prompt'):
    sections = parse(sequence)
    if channel is not None:
        allowed = {'prompt': 'PROMPT', 'retrieval': 'RETRIEVE', 'tool': 'TOOLS'}
        if channel not in allowed or anchoring not in {'original_prompt', 'none'}:
            raise ValueError('Invalid channel or prompt anchoring')
        keep = {allowed[channel]}
        if anchoring == 'original_prompt':
            keep.add('PROMPT')
        sections = {k: v if k in keep else [] for k, v in sections.items()}
    prompt = trim_middle('\n\n'.join(sections['PROMPT']), settings['max_prompt_chars'])
    blocks = [f"[PROMPT]\n{prompt or '<empty>'}"]
    for label in ('RETRIEVE', 'TOOLS', 'AGENT'):
        source = sections[label][:settings['max_blocks_per_channel']]
        blocks.append('\n\n'.join(f'[{label} #{i}]\n{trim_middle(text, settings["max_block_chars"])}'
                                  for i, text in enumerate(source, 1)) if source else f'[{label}]\n<empty>')
    return '\n\n'.join(blocks)


def request(row, settings, channel=None, anchoring='original_prompt'):
    payload = build_payload(row['sequence'], settings, channel, anchoring)
    return {'request_id': hashlib.sha256(payload.encode()).hexdigest(), 'payload': payload,
            'prefix_id': row['id'], 'episode_id': row['original_trace_id'],
            'channel': channel or 'history'}
