#!/usr/bin/env python3
"""
OpenCog AtomSpace Integration for Deep Tree Echo

Implements ECAN-aware atomspace with attention allocation mechanisms
for cognitive processing in the Deep Tree Echo architecture.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class AtomType(Enum):
    """OpenCog-compatible atom types."""
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    SCHEMA_NODE = "SchemaNode"
    EVALUATION_LINK = "EvaluationLink"
    INHERITANCE_LINK = "InheritanceLink"
    SIMILARITY_LINK = "SimilarityLink"
    MEMBER_LINK = "MemberLink"
    EXECUTION_LINK = "ExecutionLink"
    LIST_LINK = "ListLink"


@dataclass
class AttentionValue:
    """
    ECAN Attention Values for atoms.
    
    STI (Short-Term Importance): Immediate attention allocation
    LTI (Long-Term Importance): Historical significance
    VLTI (Very Long-Term Importance): Persistent structural importance
    """
    sti: float = 0.0  # Short-Term Importance
    lti: float = 0.0  # Long-Term Importance
    vlti: float = 0.0  # Very Long-Term Importance
    
    def decay(self, sti_decay: float = 0.1, lti_decay: float = 0.01):
        """Apply ECAN decay to attention values."""
        self.sti = max(0.0, self.sti - sti_decay)
        self.lti = max(0.0, self.lti - lti_decay)
    
    def total_importance(self) -> float:
        """Calculate total attention importance."""
        return self.sti + 0.5 * self.lti + 0.3 * self.vlti


@dataclass
class TruthValue:
    """OpenCog truth value representation."""
    strength: float = 0.5  # Probability/confidence [0, 1]
    confidence: float = 0.0  # Amount of evidence [0, 1]
    
    def update(self, new_strength: float, new_confidence: float):
        """Update truth value with new evidence."""
        # Simple Bayesian-like update
        total_conf = self.confidence + new_confidence
        if total_conf > 0:
            self.strength = (
                self.strength * self.confidence + new_strength * new_confidence
            ) / total_conf
            self.confidence = min(1.0, total_conf)


@dataclass
class Atom:
    """Base atom class representing nodes and links in the atomspace."""
    id: str
    atom_type: AtomType
    name: Optional[str] = None
    outgoing: List[str] = field(default_factory=list)  # IDs of connected atoms
    attention: AttentionValue = field(default_factory=AttentionValue)
    truth_value: TruthValue = field(default_factory=TruthValue)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def is_node(self) -> bool:
        """Check if atom is a node type."""
        return "Node" in self.atom_type.value
    
    def is_link(self) -> bool:
        """Check if atom is a link type."""
        return "Link" in self.atom_type.value
    
    def touch(self):
        """Mark atom as accessed, updating timestamp."""
        self.last_accessed = datetime.now()


class ECANAttentionAllocator:
    """
    Economic Attention Network (ECAN) allocator.
    
    Implements attention spreading and resource allocation based on
    cognitive economics principles from OpenCog.
    """
    
    def __init__(
        self,
        sti_funds: float = 100000.0,
        lti_funds: float = 100000.0,
        sti_decay_rate: float = 0.1,
        lti_decay_rate: float = 0.01,
        spreading_factor: float = 0.5,
        attention_threshold: float = 10.0
    ):
        self.sti_funds = sti_funds
        self.lti_funds = lti_funds
        self.sti_decay_rate = sti_decay_rate
        self.lti_decay_rate = lti_decay_rate
        self.spreading_factor = spreading_factor
        self.attention_threshold = attention_threshold
        
        self.total_sti_allocated = 0.0
        self.total_lti_allocated = 0.0
    
    def allocate_attention(self, atom: Atom, sti_amount: float, lti_amount: float = 0.0):
        """Allocate attention resources to an atom."""
        # Check available funds
        sti_available = self.sti_funds - self.total_sti_allocated
        lti_available = self.lti_funds - self.total_lti_allocated
        
        # Allocate what's available
        actual_sti = min(sti_amount, sti_available)
        actual_lti = min(lti_amount, lti_available)
        
        atom.attention.sti += actual_sti
        atom.attention.lti += actual_lti
        
        self.total_sti_allocated += actual_sti
        self.total_lti_allocated += actual_lti
        
        logger.debug(
            f"Allocated STI={actual_sti:.2f}, LTI={actual_lti:.2f} to atom {atom.id}"
        )
    
    def spread_attention(
        self,
        source_atom: Atom,
        target_atoms: List[Atom],
        atomspace: 'OpenCogAtomSpace'
    ):
        """
        Spread attention from source to connected target atoms.
        Implements hebbian-like attention spreading.
        """
        if source_atom.attention.sti < self.attention_threshold:
            return
        
        # Calculate spreading amount
        spread_amount = source_atom.attention.sti * self.spreading_factor
        per_target = spread_amount / max(1, len(target_atoms))
        
        for target in target_atoms:
            # Spread is modulated by link strength if available
            link = atomspace.get_link_between(source_atom.id, target.id)
            modulation = link.truth_value.strength if link else 1.0
            
            actual_spread = per_target * modulation
            self.allocate_attention(target, actual_spread, 0.0)
    
    def decay_all(self, atoms: List[Atom]):
        """Apply decay to all atoms in the collection."""
        for atom in atoms:
            old_sti = atom.attention.sti
            old_lti = atom.attention.lti
            
            atom.attention.decay(self.sti_decay_rate, self.lti_decay_rate)
            
            # Return decayed attention to funds
            self.total_sti_allocated -= (old_sti - atom.attention.sti)
            self.total_lti_allocated -= (old_lti - atom.attention.lti)
    
    def get_attentional_focus(self, atoms: List[Atom], top_k: int = 10) -> List[Atom]:
        """Get the top-k atoms in attentional focus."""
        sorted_atoms = sorted(
            atoms,
            key=lambda a: a.attention.total_importance(),
            reverse=True
        )
        return sorted_atoms[:top_k]


class OpenCogAtomSpace:
    """
    OpenCog-compatible AtomSpace implementation with ECAN awareness.
    
    Provides atom storage, retrieval, pattern matching, and attention allocation
    for Deep Tree Echo cognitive architecture.
    """
    
    def __init__(
        self,
        enable_ecan: bool = True,
        sti_funds: float = 100000.0,
        lti_funds: float = 100000.0
    ):
        self.atoms: Dict[str, Atom] = {}
        self.name_index: Dict[str, Set[str]] = {}  # name -> atom_ids
        self.type_index: Dict[AtomType, Set[str]] = {}  # type -> atom_ids
        self.incoming_index: Dict[str, Set[str]] = {}  # atom_id -> link_ids
        
        self.enable_ecan = enable_ecan
        self.ecan = ECANAttentionAllocator(sti_funds, lti_funds) if enable_ecan else None
        
        logger.info("OpenCogAtomSpace initialized with ECAN" if enable_ecan else "OpenCogAtomSpace initialized")
    
    def add_node(
        self,
        atom_type: AtomType,
        name: str,
        truth_value: Optional[TruthValue] = None,
        initial_sti: float = 0.0
    ) -> Atom:
        """Add a node atom to the atomspace."""
        if not name:
            raise ValueError("Node must have a name")
        
        # Check if node already exists
        existing_id = self._find_node(atom_type, name)
        if existing_id:
            atom = self.atoms[existing_id]
            if truth_value:
                atom.truth_value.update(truth_value.strength, truth_value.confidence)
            return atom
        
        # Create new node
        atom_id = str(uuid.uuid4())
        atom = Atom(
            id=atom_id,
            atom_type=atom_type,
            name=name,
            truth_value=truth_value or TruthValue()
        )
        
        self.atoms[atom_id] = atom
        
        # Update indexes
        self.name_index.setdefault(name, set()).add(atom_id)
        self.type_index.setdefault(atom_type, set()).add(atom_id)
        
        # Allocate initial attention
        if self.enable_ecan and initial_sti > 0:
            self.ecan.allocate_attention(atom, initial_sti, 0.0)
        
        return atom
    
    def add_link(
        self,
        atom_type: AtomType,
        outgoing: List[str],
        truth_value: Optional[TruthValue] = None,
        initial_sti: float = 0.0
    ) -> Atom:
        """Add a link atom to the atomspace."""
        if not outgoing:
            raise ValueError("Link must have outgoing atoms")
        
        # Verify all outgoing atoms exist
        for atom_id in outgoing:
            if atom_id not in self.atoms:
                raise ValueError(f"Outgoing atom {atom_id} does not exist")
        
        # Check if link already exists
        existing_id = self._find_link(atom_type, outgoing)
        if existing_id:
            atom = self.atoms[existing_id]
            if truth_value:
                atom.truth_value.update(truth_value.strength, truth_value.confidence)
            return atom
        
        # Create new link
        atom_id = str(uuid.uuid4())
        atom = Atom(
            id=atom_id,
            atom_type=atom_type,
            outgoing=outgoing,
            truth_value=truth_value or TruthValue()
        )
        
        self.atoms[atom_id] = atom
        
        # Update indexes
        self.type_index.setdefault(atom_type, set()).add(atom_id)
        for out_id in outgoing:
            self.incoming_index.setdefault(out_id, set()).add(atom_id)
        
        # Allocate initial attention
        if self.enable_ecan and initial_sti > 0:
            self.ecan.allocate_attention(atom, initial_sti, 0.0)
        
        return atom
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Retrieve atom by ID."""
        atom = self.atoms.get(atom_id)
        if atom:
            atom.touch()
        return atom
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """Get all atoms with given name."""
        atom_ids = self.name_index.get(name, set())
        return [self.atoms[aid] for aid in atom_ids]
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of given type."""
        atom_ids = self.type_index.get(atom_type, set())
        return [self.atoms[aid] for aid in atom_ids]
    
    def get_incoming(self, atom_id: str) -> List[Atom]:
        """Get all links pointing to this atom."""
        link_ids = self.incoming_index.get(atom_id, set())
        return [self.atoms[lid] for lid in link_ids]
    
    def get_link_between(self, atom_id1: str, atom_id2: str) -> Optional[Atom]:
        """Find link connecting two atoms."""
        incoming = self.get_incoming(atom_id1)
        for link in incoming:
            if atom_id2 in link.outgoing:
                return link
        return None
    
    def pattern_match(
        self,
        pattern: Dict[str, Any],
        max_results: int = 100
    ) -> List[Dict[str, Atom]]:
        """
        Simple pattern matching for atomspace queries.
        
        Pattern format:
        {
            'atom_type': AtomType.CONCEPT_NODE,
            'name': 'deep-tree-echo',  # optional
            'outgoing': [...]  # optional for links
        }
        """
        results = []
        
        # Get candidate atoms by type
        atom_type = pattern.get('atom_type')
        if atom_type:
            candidates = self.get_atoms_by_type(atom_type)
        else:
            candidates = list(self.atoms.values())
        
        # Filter by name if specified
        name = pattern.get('name')
        if name:
            candidates = [a for a in candidates if a.name == name]
        
        # Filter by outgoing if specified (for links)
        outgoing = pattern.get('outgoing')
        if outgoing:
            candidates = [
                a for a in candidates
                if a.is_link() and set(a.outgoing) == set(outgoing)
            ]
        
        for atom in candidates[:max_results]:
            results.append({'atom': atom})
        
        return results
    
    def spread_activation(self, source_atom_id: str, depth: int = 2):
        """Spread activation/attention from source atom through the graph."""
        if not self.enable_ecan:
            logger.warning("ECAN not enabled, cannot spread activation")
            return
        
        visited = set()
        queue = [(source_atom_id, 0)]
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_id in visited or current_depth >= depth:
                continue
            
            visited.add(current_id)
            current_atom = self.get_atom(current_id)
            if not current_atom:
                continue
            
            # Get connected atoms
            connected = []
            
            # Add outgoing targets (for links)
            if current_atom.is_link():
                for out_id in current_atom.outgoing:
                    connected.append(self.get_atom(out_id))
            
            # Add incoming links
            for link in self.get_incoming(current_id):
                connected.append(link)
                # Also add other nodes in the link
                for out_id in link.outgoing:
                    if out_id != current_id:
                        connected.append(self.get_atom(out_id))
            
            # Filter None values
            connected = [a for a in connected if a is not None]
            
            # Spread attention
            self.ecan.spread_attention(current_atom, connected, self)
            
            # Add to queue for next level
            for atom in connected:
                if atom.id not in visited:
                    queue.append((atom.id, current_depth + 1))
    
    def update_ecan(self):
        """Run ECAN update cycle: decay and compute attentional focus."""
        if not self.enable_ecan:
            return
        
        all_atoms = list(self.atoms.values())
        self.ecan.decay_all(all_atoms)
    
    def get_attentional_focus(self, top_k: int = 10) -> List[Atom]:
        """Get atoms currently in attentional focus."""
        if not self.enable_ecan:
            return []
        
        all_atoms = list(self.atoms.values())
        return self.ecan.get_attentional_focus(all_atoms, top_k)
    
    def _find_node(self, atom_type: AtomType, name: str) -> Optional[str]:
        """Find existing node by type and name."""
        candidates = self.name_index.get(name, set())
        for atom_id in candidates:
            atom = self.atoms[atom_id]
            if atom.atom_type == atom_type:
                return atom_id
        return None
    
    def _find_link(self, atom_type: AtomType, outgoing: List[str]) -> Optional[str]:
        """Find existing link by type and outgoing."""
        candidates = self.type_index.get(atom_type, set())
        outgoing_set = set(outgoing)
        for atom_id in candidates:
            atom = self.atoms[atom_id]
            if set(atom.outgoing) == outgoing_set:
                return atom_id
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get atomspace statistics."""
        stats = {
            'total_atoms': len(self.atoms),
            'nodes': sum(1 for a in self.atoms.values() if a.is_node()),
            'links': sum(1 for a in self.atoms.values() if a.is_link()),
            'atoms_by_type': {
                atom_type.value: len(ids)
                for atom_type, ids in self.type_index.items()
            }
        }
        
        if self.enable_ecan:
            stats['ecan'] = {
                'sti_allocated': self.ecan.total_sti_allocated,
                'lti_allocated': self.ecan.total_lti_allocated,
                'sti_available': self.ecan.sti_funds - self.ecan.total_sti_allocated,
                'lti_available': self.ecan.lti_funds - self.ecan.total_lti_allocated,
                'attentional_focus_size': len(self.get_attentional_focus())
            }
        
        return stats


# Example usage and initialization
def initialize_deep_tree_echo_atomspace() -> OpenCogAtomSpace:
    """Initialize atomspace with Deep Tree Echo concepts."""
    atomspace = OpenCogAtomSpace(enable_ecan=True)
    
    # Add foundational concepts
    dte_node = atomspace.add_node(
        AtomType.CONCEPT_NODE,
        "deep-tree-echo",
        TruthValue(1.0, 0.9),
        initial_sti=100.0
    )
    
    cognitive_arch = atomspace.add_node(
        AtomType.CONCEPT_NODE,
        "cognitive-architecture",
        TruthValue(1.0, 0.85),
        initial_sti=80.0
    )
    
    # Create inheritance relationship
    atomspace.add_link(
        AtomType.INHERITANCE_LINK,
        [dte_node.id, cognitive_arch.id],
        TruthValue(0.95, 0.9),
        initial_sti=50.0
    )
    
    # Add memory types
    for memory_type in ["declarative", "procedural", "episodic", "intentional"]:
        mem_node = atomspace.add_node(
            AtomType.CONCEPT_NODE,
            f"memory-{memory_type}",
            TruthValue(1.0, 0.8),
            initial_sti=60.0
        )
        atomspace.add_link(
            AtomType.MEMBER_LINK,
            [mem_node.id, dte_node.id],
            TruthValue(1.0, 0.85)
        )
    
    logger.info("Deep Tree Echo atomspace initialized with foundational concepts")
    return atomspace
