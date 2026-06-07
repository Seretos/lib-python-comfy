"""Emit the ComfyUI UI workflow format from a :class:`GraphBuilder`."""
from __future__ import annotations

from .builder import GraphBuilder, _Link


def to_ui(graph: GraphBuilder) -> dict:
    """Convert *graph* to the ComfyUI UI workflow format.

    The returned dict has the shape::

        {
            "nodes": [...],
            "links": [...],
            "groups": [],
            "config": {},
            "extra": {},
            "version": "0.4",
        }

    Node positions are auto-laid out horizontally: node with ``id=N`` gets
    position ``[(N - 1) * 200, 0]``.  All slot types are emitted as the
    sentinel string ``"UNKNOWN"`` because type information is not tracked.

    Args:
        graph: A :class:`GraphBuilder` instance (may be empty).

    Returns:
        A plain dict suitable for JSON serialisation and loading into the
        ComfyUI web UI as a saved workflow.
    """
    nodes_out: list[dict] = []
    links_out: list[list] = []

    # Index links by destination and source for slot-order derivation.
    # dst_links[node_id] = list of _Link in insertion order (dedup by dst_input).
    dst_links: dict[int, list[_Link]] = {}
    for lnk in graph._link_list:
        bucket = dst_links.setdefault(lnk.dst_node_id, [])
        # Dedup by dst_input: first-seen wins.
        existing_names = {l.dst_input for l in bucket}
        if lnk.dst_input not in existing_names:
            bucket.append(lnk)

    # src_links[node_id] = list of _Link from that node.
    src_links: dict[int, list[_Link]] = {}
    for lnk in graph._link_list:
        src_links.setdefault(lnk.src_node_id, []).append(lnk)

    for index, node in enumerate(graph._node_list):
        # --- Input slots ---
        input_slots: list[dict] = []
        for lnk in dst_links.get(node.id, []):
            input_slots.append({"name": lnk.dst_input, "type": "UNKNOWN", "link": lnk.id})

        # --- Output slots ---
        # Group outgoing links by src_slot; preserve order of first appearance.
        slot_order: list[int] = []
        slot_links: dict[int, list[int]] = {}
        for lnk in src_links.get(node.id, []):
            if lnk.src_slot not in slot_links:
                slot_order.append(lnk.src_slot)
                slot_links[lnk.src_slot] = []
            slot_links[lnk.src_slot].append(lnk.id)

        output_slots: list[dict] = [
            {"name": f"OUTPUT_{slot}", "type": "UNKNOWN", "links": slot_links[slot]}
            for slot in slot_order
        ]

        nodes_out.append(
            {
                "id": node.id,
                "type": node.class_type,
                "pos": [(node.id - 1) * 200, 0],
                "size": {"0": 140, "1": 80},
                "flags": {},
                "order": index,
                "mode": 0,
                "inputs": input_slots,
                "outputs": output_slots,
                "properties": {},
            }
        )

    # --- Links list ---
    # Build a lookup: for each dst node, what is the 0-based slot index for
    # each input name?
    dst_slot_index: dict[int, dict[str, int]] = {}
    for node_id, lnk_list in dst_links.items():
        dst_slot_index[node_id] = {lnk.dst_input: i for i, lnk in enumerate(lnk_list)}

    for lnk in graph._link_list:
        slot_idx = dst_slot_index.get(lnk.dst_node_id, {}).get(lnk.dst_input, 0)
        links_out.append(
            [lnk.id, lnk.src_node_id, lnk.src_slot, lnk.dst_node_id, slot_idx, "UNKNOWN"]
        )

    return {
        "nodes": nodes_out,
        "links": links_out,
        "groups": [],
        "config": {},
        "extra": {},
        "version": "0.4",
    }
