"""Emit the ComfyUI API prompt format from a :class:`GraphBuilder`."""
from __future__ import annotations

from .builder import GraphBuilder


def to_api(graph: GraphBuilder) -> dict:
    """Convert *graph* to the ComfyUI API prompt format.

    The returned dict has the shape::

        {
            "1": {"class_type": "...", "inputs": {...}},
            "2": {"class_type": "...", "inputs": {...}},
            ...
        }

    Literal inputs supplied at :meth:`~GraphBuilder.add_node` time are
    preserved verbatim.  Each edge added via :meth:`~GraphBuilder.link`
    overlays the destination node's ``inputs`` dict with a two-element list
    ``[src_node_id_str, src_output_index]`` keyed by the destination input
    name.  If a link and a literal share the same input name, the link wins.

    Args:
        graph: A :class:`GraphBuilder` instance (may be empty).

    Returns:
        A plain dict suitable for JSON serialisation and submission to the
        ComfyUI ``/prompt`` endpoint.
    """
    result: dict = {}

    # Seed every node with its literal inputs.
    for node in graph._node_list:
        result[str(node.id)] = {
            "class_type": node.class_type,
            "inputs": dict(node.raw_inputs),
        }

    # Overlay links — links win over literals when the key is the same.
    for lnk in graph._link_list:
        result[str(lnk.dst_node_id)]["inputs"][lnk.dst_input] = [
            str(lnk.src_node_id),
            lnk.src_slot,
        ]

    return result
