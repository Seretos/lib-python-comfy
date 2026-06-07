"""Graph builder core — media-agnostic in-memory ComfyUI graph representation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Node:
    """Internal record for a single node added to the graph."""

    id: int
    class_type: str
    raw_inputs: dict


@dataclass
class _Link:
    """Internal record for a directed edge between two nodes."""

    id: int
    src_node_id: int
    src_slot: int
    dst_node_id: int
    dst_input: str


@dataclass(frozen=True)
class NodeRef:
    """Immutable reference to a node returned by :meth:`GraphBuilder.add_node`.

    Consumers use this to wire edges via :meth:`GraphBuilder.link`.  The
    ``id`` is the stable numeric identity that appears in both the API-format
    and the UI-format outputs.
    """

    id: int
    class_type: str


class GraphBuilder:
    """Generic, media-agnostic in-memory ComfyUI graph builder.

    Usage::

        g = GraphBuilder()
        a = g.add_node("CheckpointLoaderSimple", {"ckpt_name": "v1-5.safetensors"})
        b = g.add_node("KSampler", {"steps": 20, "cfg": 7.0})
        g.link(a, 0, b, "model")
        api_dict = to_api(g)
        ui_dict  = to_ui(g)
    """

    def __init__(self) -> None:
        self._next_node_id: int = 1
        self._next_link_id: int = 1
        self._nodes: list[_Node] = []
        self._links: list[_Link] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, class_type: str, inputs: dict | None = None) -> NodeRef:
        """Register a new node and return a :class:`NodeRef`.

        Args:
            class_type: The ComfyUI node class name (e.g. ``"KSampler"``).
            inputs: Optional mapping of literal (non-link) input values.
                    Links added later via :meth:`link` are overlaid on top.

        Returns:
            A frozen :class:`NodeRef` whose ``id`` is unique within this graph.
        """
        raw_inputs: dict = dict(inputs) if inputs is not None else {}
        node = _Node(id=self._next_node_id, class_type=class_type, raw_inputs=raw_inputs)
        self._nodes.append(node)
        self._next_node_id += 1
        return NodeRef(id=node.id, class_type=class_type)

    def link(
        self,
        src: NodeRef,
        src_output_index: int,
        dst: NodeRef,
        dst_input: str,
    ) -> None:
        """Add a directed edge from *src* output slot to *dst* named input.

        Args:
            src: Source node reference (returned by :meth:`add_node`).
            src_output_index: 0-based output slot index on the source node.
            dst: Destination node reference.
            dst_input: Name of the input slot on the destination node.

        Raises:
            ValueError: If a link already targets the same ``(dst_node_id,
                dst_input)`` pair.  ComfyUI inputs are single-valued; allowing
                two links to the same slot would cause ``to_api`` and ``to_ui``
                to disagree on which one wins.
        """
        for existing in self._links:
            if existing.dst_node_id == dst.id and existing.dst_input == dst_input:
                raise ValueError(
                    f"A link already targets input '{dst_input}' on node {dst.id} "
                    f"(existing link id={existing.id}).  Each destination input may "
                    "only be connected once."
                )
        lnk = _Link(
            id=self._next_link_id,
            src_node_id=src.id,
            src_slot=src_output_index,
            dst_node_id=dst.id,
            dst_input=dst_input,
        )
        self._links.append(lnk)
        self._next_link_id += 1

    # ------------------------------------------------------------------
    # Internal accessors (used by format modules)
    # ------------------------------------------------------------------

    @property
    def _node_list(self) -> list[_Node]:
        return self._nodes

    @property
    def _link_list(self) -> list[_Link]:
        return self._links
