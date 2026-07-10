"""Emit the ComfyUI UI workflow format from a :class:`GraphBuilder`."""
from __future__ import annotations

from .builder import GraphBuilder, _Link
from .node_meta import input_type, output_slot, widget_order, widget_value


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
            "last_node_id": ...,
            "last_link_id": ...,
        }

    Node positions are auto-laid out horizontally: node with ``id=N`` gets
    position ``[(N - 1) * 200, 0]``.  Slot types and ``widgets_values`` are
    resolved from the static registry in :mod:`node_meta` for known scaffold
    node types (see ``scaffolds.py``); node types not present in that
    registry fall back to the sentinel string ``"UNKNOWN"`` for slot types,
    ``OUTPUT_{n}`` for output slot names, and omit ``widgets_values``
    entirely.

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
            slot_type = input_type(node.class_type, lnk.dst_input)
            input_slots.append({"name": lnk.dst_input, "type": slot_type, "link": lnk.id})

        # --- Output slots ---
        # Group outgoing links by src_slot; preserve order of first appearance.
        slot_order: list[int] = []
        slot_links: dict[int, list[int]] = {}
        for lnk in src_links.get(node.id, []):
            if lnk.src_slot not in slot_links:
                slot_order.append(lnk.src_slot)
                slot_links[lnk.src_slot] = []
            slot_links[lnk.src_slot].append(lnk.id)

        output_slots: list[dict] = []
        for slot in slot_order:
            slot_name, slot_type = output_slot(node.class_type, slot)
            output_slots.append({"name": slot_name, "type": slot_type, "links": slot_links[slot]})

        node_out: dict = {
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

        # --- widgets_values ---
        widgets = widget_order(node.class_type)
        if widgets:
            node_out["widgets_values"] = [
                widget_value(node.raw_inputs, name) for name in widgets
            ]

        nodes_out.append(node_out)

    # --- Links list ---
    # Build a lookup: for each dst node, what is the 0-based slot index for
    # each input name?
    dst_slot_index: dict[int, dict[str, int]] = {}
    for node_id, lnk_list in dst_links.items():
        dst_slot_index[node_id] = {lnk.dst_input: i for i, lnk in enumerate(lnk_list)}

    # Look up each destination node's class_type so link entries can carry
    # the real slot type instead of "UNKNOWN".
    class_type_by_id: dict[int, str] = {node.id: node.class_type for node in graph._node_list}

    for lnk in graph._link_list:
        slot_idx = dst_slot_index.get(lnk.dst_node_id, {}).get(lnk.dst_input, 0)
        dst_class_type = class_type_by_id.get(lnk.dst_node_id, "")
        link_type = input_type(dst_class_type, lnk.dst_input)
        links_out.append(
            [lnk.id, lnk.src_node_id, lnk.src_slot, lnk.dst_node_id, slot_idx, link_type]
        )

    last_node_id = max((n.id for n in graph._node_list), default=0)
    last_link_id = max((l.id for l in graph._link_list), default=0)

    return {
        "nodes": nodes_out,
        "links": links_out,
        "groups": [],
        "config": {},
        "extra": {},
        "version": "0.4",
        "last_node_id": last_node_id,
        "last_link_id": last_link_id,
    }
