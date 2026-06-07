"""Tests for the graph-builder core (ticket #8).

All imports go through the public ``lib_python_comfy`` namespace to verify
that re-exports and ``__all__`` are wired correctly.
"""
from __future__ import annotations

import pytest

from lib_python_comfy import GraphBuilder, NodeRef, to_api, to_ui


# ---------------------------------------------------------------------------
# NodeRef / add_node
# ---------------------------------------------------------------------------


def test_add_node_returns_node_ref():
    """add_node returns a NodeRef; first id=1, second id=2."""
    g = GraphBuilder()
    a = g.add_node("LoadImage")
    b = g.add_node("SaveImage")
    assert isinstance(a, NodeRef)
    assert isinstance(b, NodeRef)
    assert a.id == 1
    assert b.id == 2
    assert a.class_type == "LoadImage"
    assert b.class_type == "SaveImage"


def test_node_ref_not_mutated_by_link():
    """NodeRef.id and class_type are unchanged after link()."""
    g = GraphBuilder()
    a = g.add_node("A")
    b = g.add_node("B")
    a_id_before = a.id
    a_type_before = a.class_type
    g.link(a, 0, b, "input")
    assert a.id == a_id_before
    assert a.class_type == a_type_before


# ---------------------------------------------------------------------------
# API format
# ---------------------------------------------------------------------------


def test_api_format_structure():
    """Two-node graph produces exactly two keys; each has class_type and inputs."""
    g = GraphBuilder()
    g.add_node("A")
    g.add_node("B")
    result = to_api(g)
    assert len(result) == 2
    for val in result.values():
        assert "class_type" in val
        assert "inputs" in val


def test_api_format_link_encoding():
    """link(a, 0, b, 'image') encodes as [str(a.id), 0] in b's inputs."""
    g = GraphBuilder()
    a = g.add_node("Loader")
    b = g.add_node("Sampler")
    g.link(a, 0, b, "image")
    result = to_api(g)
    assert result[str(b.id)]["inputs"]["image"] == [str(a.id), 0]


def test_api_format_literal_inputs():
    """Literal inputs supplied at add_node time are preserved verbatim."""
    g = GraphBuilder()
    g.add_node("KSampler", {"steps": 20, "cfg": 7.0})
    result = to_api(g)
    node_entry = result["1"]["inputs"]
    assert node_entry["steps"] == 20
    assert node_entry["cfg"] == 7.0


def test_literal_inputs_not_overwritten_by_unrelated_link():
    """A link into a different input key does not clobber an unrelated literal."""
    g = GraphBuilder()
    a = g.add_node("Source")
    b = g.add_node("KSampler", {"steps": 20})
    g.link(a, 0, b, "model")          # link goes into "model", not "steps"
    result = to_api(g)
    assert result[str(b.id)]["inputs"]["steps"] == 20
    assert result[str(b.id)]["inputs"]["model"] == [str(a.id), 0]


# ---------------------------------------------------------------------------
# UI format — top-level structure
# ---------------------------------------------------------------------------


def test_ui_format_top_level_keys():
    """Result has the required top-level keys."""
    g = GraphBuilder()
    result = to_ui(g)
    for key in ("nodes", "links", "groups", "config", "extra", "version"):
        assert key in result


def test_ui_format_node_count_matches():
    """len(result['nodes']) equals the number of add_node calls."""
    g = GraphBuilder()
    g.add_node("A")
    g.add_node("B")
    g.add_node("C")
    result = to_ui(g)
    assert len(result["nodes"]) == 3


def test_ui_format_link_count_matches():
    """len(result['links']) equals the number of link() calls."""
    g = GraphBuilder()
    a = g.add_node("A")
    b = g.add_node("B")
    c = g.add_node("C")
    g.link(a, 0, b, "x")
    g.link(b, 0, c, "y")
    result = to_ui(g)
    assert len(result["links"]) == 2


def test_ui_format_node_fields():
    """Each node dict contains the required fields."""
    g = GraphBuilder()
    g.add_node("A")
    result = to_ui(g)
    node = result["nodes"][0]
    for field in ("id", "type", "pos", "size", "inputs", "outputs"):
        assert field in node


def test_ui_format_link_is_six_element_array():
    """Each link entry is a list of length 6; indices 1 and 3 match src/dst IDs."""
    g = GraphBuilder()
    a = g.add_node("A")
    b = g.add_node("B")
    g.link(a, 0, b, "in")
    result = to_ui(g)
    lnk = result["links"][0]
    assert isinstance(lnk, list)
    assert len(lnk) == 6
    assert lnk[1] == a.id   # src_node_id
    assert lnk[3] == b.id   # dst_node_id


def test_ui_format_output_slots_derived_from_links():
    """Node with one outgoing link on slot 0 has outputs length 1 named 'OUTPUT_0'.
    Node with no outgoing links has outputs == [].
    """
    g = GraphBuilder()
    a = g.add_node("Source")
    b = g.add_node("Sink")
    g.link(a, 0, b, "inp")
    result = to_ui(g)
    node_a = next(n for n in result["nodes"] if n["id"] == a.id)
    node_b = next(n for n in result["nodes"] if n["id"] == b.id)
    assert len(node_a["outputs"]) == 1
    assert node_a["outputs"][0]["name"] == "OUTPUT_0"
    assert node_b["outputs"] == []


def test_ui_format_input_slots_ordered_by_insertion():
    """Input slot names on node c respect link-insertion order."""
    g = GraphBuilder()
    a = g.add_node("A")
    b = g.add_node("B")
    c = g.add_node("C")
    g.link(a, 0, c, "alpha")
    g.link(b, 0, c, "beta")
    result = to_ui(g)
    node_c = next(n for n in result["nodes"] if n["id"] == c.id)
    names = [slot["name"] for slot in node_c["inputs"]]
    assert names == ["alpha", "beta"]


def test_auto_layout_positions_are_deterministic():
    """Node id=1 gets pos [0, 0]; node id=2 gets pos [200, 0]."""
    g = GraphBuilder()
    g.add_node("A")
    g.add_node("B")
    result = to_ui(g)
    nodes_by_id = {n["id"]: n for n in result["nodes"]}
    assert nodes_by_id[1]["pos"] == [0, 0]
    assert nodes_by_id[2]["pos"] == [200, 0]


# ---------------------------------------------------------------------------
# ID parity
# ---------------------------------------------------------------------------


def test_id_parity_between_formats():
    """Node IDs in to_api (string keys cast to int) and to_ui ('id' fields) are equal sets."""
    g = GraphBuilder()
    a = g.add_node("A")
    b = g.add_node("B")
    c = g.add_node("C")
    g.link(a, 0, b, "x")
    g.link(b, 0, c, "y")
    api_ids = {int(k) for k in to_api(g).keys()}
    ui_ids = {n["id"] for n in to_ui(g)["nodes"]}
    assert api_ids == ui_ids


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_graph():
    """Empty GraphBuilder produces valid empty structures without raising."""
    g = GraphBuilder()
    api = to_api(g)
    ui = to_ui(g)
    assert api == {}
    assert ui["nodes"] == []
    assert ui["links"] == []
    assert ui["groups"] == []


def test_single_node_no_links():
    """One node, no links: to_api has one key; to_ui node has inputs=[] and outputs=[]."""
    g = GraphBuilder()
    g.add_node("Solo")
    api = to_api(g)
    ui = to_ui(g)
    assert len(api) == 1
    assert len(ui["nodes"]) == 1
    node = ui["nodes"][0]
    assert node["inputs"] == []
    assert node["outputs"] == []


def test_chain_a_b_c():
    """Three-node linear chain A→B→C: both emitters produce 3 nodes, 2 links; ID parity holds;
    B is both a destination (has inputs) and a source (has outputs).
    """
    g = GraphBuilder()
    a = g.add_node("A")
    b = g.add_node("B")
    c = g.add_node("C")
    g.link(a, 0, b, "in")
    g.link(b, 0, c, "in")

    api = to_api(g)
    ui = to_ui(g)

    # Node count
    assert len(api) == 3
    assert len(ui["nodes"]) == 3

    # Link count
    assert len(ui["links"]) == 2

    # ID parity
    api_ids = {int(k) for k in api.keys()}
    ui_ids = {n["id"] for n in ui["nodes"]}
    assert api_ids == ui_ids

    # B is both destination and source
    node_b = next(n for n in ui["nodes"] if n["id"] == b.id)
    assert len(node_b["inputs"]) == 1    # receives from A
    assert len(node_b["outputs"]) == 1   # feeds into C


# ---------------------------------------------------------------------------
# Regression / behavioural contract tests added for review findings
# ---------------------------------------------------------------------------


def test_link_raises_on_duplicate_dst_input():
    """link() raises ValueError when a second link targets the same (dst_node, dst_input).

    Regression: previously both to_api (last-wins) and to_ui (first-wins) would
    silently accept the duplicate and disagree on the resulting graph.  The guard
    in GraphBuilder.link() prevents that divergence at the source.
    """
    g = GraphBuilder()
    src1 = g.add_node("SrcA")
    src2 = g.add_node("SrcB")
    dst = g.add_node("Dst")
    g.link(src1, 0, dst, "model")
    with pytest.raises(ValueError):
        g.link(src2, 0, dst, "model")


def test_link_wins_over_literal_in_api_format():
    """A link to an input name overrides a literal value supplied at add_node time.

    The 'link wins over literal' rule must be reflected in to_api output: the
    input value must be the two-element list [str(src_id), slot], not the
    original string literal.
    """
    g = GraphBuilder()
    src = g.add_node("SrcNode")
    dst = g.add_node("B", {"model": "literal_value"})
    g.link(src, 0, dst, "model")
    result = to_api(g)
    assert result[str(dst.id)]["inputs"]["model"] == [str(src.id), 0]
