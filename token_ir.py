"""
token_ir.py - Compiler Intermediate Representation (T-IR) & AST Token Graph
Docket: IDF-GGL-2026-TOKENOMICS-COMPILER
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Replaces heuristic string/regex matching with formal compiler intermediate representation
operating directly on AST nodes, token dependency tensors, and vocabulary DAG graphs.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Tuple
from pydantic import BaseModel, Field


class IRNodeType(Enum):
    """Formal intermediate representation node classifications."""
    ROOT_TASK = auto()           # Top-level operational objective
    HARD_INVARIANT = auto()      # Negative constraint, security boundary, type check
    SCHEMA_CONTRACT = auto()     # OpenAPI / JSON parameter contract
    STATIC_CONTEXT = auto()      # Immutable reference tables, DDL schemas, policies
    DYNAMIC_QUERY = auto()       # Per-invocation variable bindings
    CONVERSATIONAL_PADDING = auto() # Non-binding pleasantries, filler
    ATTENTION_LANDMARK = auto()   # Synthetically injected uppercase attention sink


class InvariantType(Enum):
    """Taxonomy of hard domain invariants."""
    PROHIBITION = "PROHIBITION"        # "Never disclose", "Do not drop"
    MANDATORY_FIELD = "MANDATORY_FIELD" # "Required parameter", "Not null"
    TYPE_RESTRICTION = "TYPE_RESTRICTION" # "Must be integer", "Enum bound"
    TRANSACTION_BOUNDARY = "TRANSACTION_BOUNDARY" # "Isolation serializable", "Rollback on failure"
    OUTPUT_CONTRACT = "OUTPUT_CONTRACT" # "JSON schema format contract"


class TokenIRNode(BaseModel):
    """Atomic node in the Token Intermediate Representation (T-IR) graph."""
    node_id: str
    node_type: IRNodeType
    raw_content: str
    canonical_content: str = ""
    invariant_type: Optional[InvariantType] = None
    variable_bindings: Set[str] = Field(default_factory=set)
    estimated_raw_tokens: int = 0
    estimated_canonical_tokens: int = 0
    is_prunable: bool = False
    children: List[str] = Field(default_factory=list)


class TokenASTGraph:
    """
    Directed Acyclic Graph (DAG) representing prompt token dependencies.
    Executes formal compiler optimization passes (-O1, -O2, -O3).
    """

    def __init__(self):
        self.nodes: Dict[str, TokenIRNode] = {}
        self.root_nodes: List[str] = []

    def add_node(self, node: TokenIRNode, parent_id: Optional[str] = None):
        self.nodes[node.node_id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node.node_id)
        else:
            self.root_nodes.append(node.node_id)

    def optimize_pass_dead_code_elimination(self) -> int:
        """
        Compiler Pass 1: Prunes AST nodes tagged as CONVERSATIONAL_PADDING
        or nodes that establish zero variable bindings or constraints.
        Returns count of pruned nodes.
        """
        pruned_count = 0
        for node_id, node in list(self.nodes.items()):
            if node.node_type == IRNodeType.CONVERSATIONAL_PADDING or node.is_prunable:
                del self.nodes[node_id]
                pruned_count += 1
                # Remove from root nodes or parent children
                if node_id in self.root_nodes:
                    self.root_nodes.remove(node_id)
                for other_node in self.nodes.values():
                    if node_id in other_node.children:
                        other_node.children.remove(node_id)
        return pruned_count

    def optimize_pass_invariant_deduplication(self) -> int:
        """
        Compiler Pass 2: Identifies duplicate invariant nodes sharing
        identical invariant types and variable bindings, folding them
        into a single canonical invariant representation.
        """
        seen_invariants: Dict[Tuple[Optional[InvariantType], frozenset], str] = {}
        folded_count = 0

        for node_id, node in list(self.nodes.items()):
            if node.node_type == IRNodeType.HARD_INVARIANT:
                sig = (node.invariant_type, frozenset(node.variable_bindings))
                if sig in seen_invariants:
                    # Fold duplicate into existing invariant
                    del self.nodes[node_id]
                    if node_id in self.root_nodes:
                        self.root_nodes.remove(node_id)
                    folded_count += 1
                else:
                    seen_invariants[sig] = node_id
        return folded_count
