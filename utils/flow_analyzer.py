# ============================================================================
# FILE: utils/flow_analyzer.py
# Flow-analysis helpers for chained detection modes and routing summaries.
#
# Purpose:
# - Summarizes how samples move through multi-layer chained detection modes.
#
# Workflow:
# - Converts raw per-layer stats into readable routing summaries.
# - Generates diagram-friendly node and edge structures.
#
# Use this file when:
# - You want to inspect escalation, forwarding, and blocking behavior across
#   chained pipeline modes.
# ============================================================================
from __future__ import annotations

from typing import Dict, Any, List


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def analyze_flow(layer_stats: Dict[str, Dict[str, Any]], total_samples: int) -> Dict[str, Any]:
    """
    Summarize routing flow across layers using the new schema.

    Returns:
      {
        "total_input": int,
        "layers": [
          {
            "layer": str,
            "processed": int,
            "blocked": int,
            "forwarded": int,
            "sent_to_next": int,
            "pct_blocked": float,
            "pct_forwarded": float,
            "pct_sent_to_next": float,
          }, ...
        ],
        "total_blocked": int,
        "total_forwarded": int,
        "total_unaccounted": int,   # should be 0 if stats are consistent
        "block_rate": float,
        "forward_rate": float,
      }
    """
    layers_out: List[Dict[str, Any]] = []

    total_blocked = 0
    total_forwarded = 0

    # Preserve insertion order of dict (Python 3.7+)
    for layer_name, stats in layer_stats.items():
        processed = _to_int(stats.get("processed", 0))
        blocked = _to_int(stats.get("blocked", 0))
        forwarded = _to_int(stats.get("forwarded", 0))
        sent_to_next = _to_int(stats.get("sent_to_next", 0))

        # Percentages within processed
        if processed > 0:
            pct_blocked = 100.0 * blocked / processed
            pct_forwarded = 100.0 * forwarded / processed
            pct_sent = 100.0 * sent_to_next / processed
        else:
            pct_blocked = pct_forwarded = pct_sent = 0.0

        layers_out.append(
            {
                "layer": layer_name,
                "processed": processed,
                "blocked": blocked,
                "forwarded": forwarded,
                "sent_to_next": sent_to_next,
                "pct_blocked": pct_blocked,
                "pct_forwarded": pct_forwarded,
                "pct_sent_to_next": pct_sent,
            }
        )

        total_blocked += blocked
        total_forwarded += forwarded

    # If your pipeline is consistent:
    # total_forwarded + total_blocked == total_samples
    total_unaccounted = total_samples - (total_blocked + total_forwarded)

    block_rate = (100.0 * total_blocked / total_samples) if total_samples > 0 else 0.0
    forward_rate = (100.0 * total_forwarded / total_samples) if total_samples > 0 else 0.0

    return {
        "total_input": int(total_samples),
        "layers": layers_out,
        "total_blocked": int(total_blocked),
        "total_forwarded": int(total_forwarded),
        "total_unaccounted": int(total_unaccounted),
        "block_rate": float(block_rate),
        "forward_rate": float(forward_rate),
    }


def generate_flow_diagram_data(layer_stats: Dict[str, Dict[str, Any]], total_samples: int) -> Dict[str, Any]:
    """
    Generate diagram-friendly flow graph for the new schema.

    Output:
      {
        "nodes": [{"id","label","type","count"}, ...],
        "edges": [{"from","to","label","count","color"?(optional)}, ...]
      }
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Nodes
    nodes.append({"id": "input", "label": f"Input: {total_samples}", "type": "start", "count": int(total_samples)})

    # Terminal nodes (single)
    nodes.append({"id": "blocked", "label": "Blocked", "type": "end_blocked", "count": 0})
    nodes.append({"id": "forwarded", "label": "Forwarded", "type": "end_forwarded", "count": 0})

    prev_id = "input"
    layer_ids: List[str] = []

    # Create layer nodes in order
    for i, (layer_name, stats) in enumerate(layer_stats.items(), start=1):
        lid = f"layer{i}"
        layer_ids.append(lid)
        processed = _to_int(stats.get("processed", 0))
        nodes.append(
            {
                "id": lid,
                "label": f"{layer_name}: {processed}",
                "type": "detector",
                "count": processed,
            }
        )

        # Edge prev -> current (how many reached this layer)
        edges.append(
            {
                "from": prev_id,
                "to": lid,
                "label": str(processed),
                "count": processed,
            }
        )
        prev_id = lid

    # Now create per-layer outgoing edges:
    # - to blocked
    # - to forwarded
    # - to next layer (sent_to_next)
    for i, (layer_name, stats) in enumerate(layer_stats.items(), start=1):
        lid = f"layer{i}"
        blocked = _to_int(stats.get("blocked", 0))
        forwarded = _to_int(stats.get("forwarded", 0))
        sent_to_next = _to_int(stats.get("sent_to_next", 0))

        if blocked > 0:
            edges.append(
                {
                    "from": lid,
                    "to": "blocked",
                    "label": f"{blocked} (blocked)",
                    "count": blocked,
                    "color": "red",
                }
            )
        if forwarded > 0:
            edges.append(
                {
                    "from": lid,
                    "to": "forwarded",
                    "label": f"{forwarded} (forward)",
                    "count": forwarded,
                    "color": "green",
                }
            )

        # sent_to_next should go to next layer if exists
        if sent_to_next > 0:
            if i < len(layer_stats):
                next_id = f"layer{i+1}"
                edges.append(
                    {
                        "from": lid,
                        "to": next_id,
                        "label": str(sent_to_next),
                        "count": sent_to_next,
                    }
                )
            else:
                # If stats say sent_to_next but there is no next layer, keep it visible
                edges.append(
                    {
                        "from": lid,
                        "to": "forwarded",
                        "label": f"{sent_to_next} (to_next?)",
                        "count": sent_to_next,
                        "color": "orange",
                    }
                )

    # Update terminal counts
    total_blocked = sum(_to_int(s.get("blocked", 0)) for s in layer_stats.values())
    total_forwarded = sum(_to_int(s.get("forwarded", 0)) for s in layer_stats.values())
    for n in nodes:
        if n["id"] == "blocked":
            n["count"] = int(total_blocked)
            n["label"] = f"Blocked: {total_blocked}"
        elif n["id"] == "forwarded":
            n["count"] = int(total_forwarded)
            n["label"] = f"Forwarded: {total_forwarded}"

    return {"nodes": nodes, "edges": edges}


def format_flow_report(flow_data: Dict[str, Any]) -> str:
    """
    Human-readable report for analyze_flow() output.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("DETECTION FLOW ANALYSIS (DESIGN-ACCURATE)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total Input: {flow_data.get('total_input', 0)}")
    lines.append("")

    for layer in flow_data.get("layers", []):
        lines.append(f"📍 {layer['layer']}:")
        lines.append(f"   Processed   : {layer['processed']}")
        lines.append(f"   Blocked     : {layer['blocked']} ({layer['pct_blocked']:.1f}%)")
        lines.append(f"   Forwarded   : {layer['forwarded']} ({layer['pct_forwarded']:.1f}%)")
        lines.append(f"   Sent to next: {layer['sent_to_next']} ({layer['pct_sent_to_next']:.1f}%)")
        lines.append("")

    lines.append("✅ Final Totals:")
    lines.append(f"   Total Blocked   : {flow_data.get('total_blocked', 0)} ({flow_data.get('block_rate', 0.0):.1f}%)")
    lines.append(f"   Total Forwarded : {flow_data.get('total_forwarded', 0)} ({flow_data.get('forward_rate', 0.0):.1f}%)")

    unacc = flow_data.get("total_unaccounted", 0)
    if unacc != 0:
        lines.append(f"⚠️  Unaccounted: {unacc}  (check your layer_stats: forwarded+blocked should sum to total_input)")

    lines.append("=" * 80)
    return "\n".join(lines)
