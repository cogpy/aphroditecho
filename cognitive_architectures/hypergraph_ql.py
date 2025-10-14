#!/usr/bin/env python3
"""
HypergraphQL Interface for OpenCog AtomSpace

Provides GraphQL-like query language for atomspace pattern matching and traversal,
enabling flexible querying of cognitive structures in Deep Tree Echo.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .opencog_atomspace import OpenCogAtomSpace, Atom, AtomType, TruthValue, AttentionValue

logger = logging.getLogger(__name__)


class QueryOperator(Enum):
    """Query operators for HypergraphQL."""
    EQ = "eq"  # Equal
    GT = "gt"  # Greater than
    LT = "lt"  # Less than
    GTE = "gte"  # Greater than or equal
    LTE = "lte"  # Less than or equal
    CONTAINS = "contains"  # String contains
    IN = "in"  # Value in list
    AND = "and"  # Logical AND
    OR = "or"  # Logical OR
    NOT = "not"  # Logical NOT


@dataclass
class QueryResult:
    """Result of a HypergraphQL query."""
    atoms: List[Atom]
    count: int
    query_time_ms: float
    has_more: bool = False
    cursor: Optional[str] = None


class HypergraphQLParser:
    """
    Parser for HypergraphQL query language.
    
    Query syntax examples:
    {
        "select": ["id", "name", "truth_value", "attention"],
        "where": {
            "atom_type": "ConceptNode",
            "name": {"contains": "echo"},
            "truth_value.strength": {"gte": 0.7},
            "attention.sti": {"gt": 50}
        },
        "traverse": {
            "direction": "outgoing",
            "depth": 2,
            "filter": {"atom_type": "InheritanceLink"}
        },
        "order_by": [{"field": "attention.sti", "direction": "desc"}],
        "limit": 10
    }
    """
    
    def __init__(self, atomspace: OpenCogAtomSpace):
        self.atomspace = atomspace
    
    def parse_and_execute(self, query: Dict[str, Any]) -> QueryResult:
        """Parse and execute a HypergraphQL query."""
        import time
        start_time = time.time()
        
        # Extract query components
        select_fields = query.get('select', ['*'])
        where_clause = query.get('where', {})
        traverse_spec = query.get('traverse')
        order_by = query.get('order_by', [])
        limit = query.get('limit', 100)
        offset = query.get('offset', 0)
        
        # Execute query
        atoms = self._execute_where(where_clause)
        
        # Apply traversal if specified
        if traverse_spec:
            atoms = self._execute_traversal(atoms, traverse_spec)
        
        # Apply ordering
        if order_by:
            atoms = self._apply_ordering(atoms, order_by)
        
        # Apply pagination
        total_count = len(atoms)
        atoms = atoms[offset:offset + limit]
        has_more = (offset + limit) < total_count
        
        query_time = (time.time() - start_time) * 1000
        
        return QueryResult(
            atoms=atoms,
            count=len(atoms),
            query_time_ms=query_time,
            has_more=has_more
        )
    
    def _execute_where(self, where_clause: Dict[str, Any]) -> List[Atom]:
        """Execute WHERE clause filtering."""
        if not where_clause:
            return list(self.atomspace.atoms.values())
        
        # Start with all atoms or filtered by type
        if 'atom_type' in where_clause:
            atom_type_str = where_clause['atom_type']
            try:
                # Try direct conversion first
                atom_type = AtomType[atom_type_str.upper().replace('-', '_').replace('NODE', '_NODE').replace('LINK', '_LINK')]
                candidates = self.atomspace.get_atoms_by_type(atom_type)
            except KeyError:
                # Try matching by value
                try:
                    matching_types = [at for at in AtomType if at.value == atom_type_str or at.name == atom_type_str.upper().replace('-', '_')]
                    if matching_types:
                        atom_type = matching_types[0]
                        candidates = self.atomspace.get_atoms_by_type(atom_type)
                    else:
                        logger.warning(f"Unknown atom type: {atom_type_str}")
                        return []
                except Exception as e:
                    logger.warning(f"Error parsing atom type {atom_type_str}: {e}")
                    return []
        else:
            candidates = list(self.atomspace.atoms.values())
        
        # Apply filters
        filtered = []
        for atom in candidates:
            if self._matches_filters(atom, where_clause):
                filtered.append(atom)
        
        return filtered
    
    def _matches_filters(self, atom: Atom, filters: Dict[str, Any]) -> bool:
        """Check if atom matches all filters."""
        for field, condition in filters.items():
            if field == 'atom_type':
                continue  # Already handled
            
            # Get field value from atom
            value = self._get_nested_value(atom, field)
            
            # Check condition
            if isinstance(condition, dict):
                # Condition with operator
                if not self._evaluate_condition(value, condition):
                    return False
            else:
                # Direct equality
                if value != condition:
                    return False
        
        return True
    
    def _get_nested_value(self, atom: Atom, field_path: str) -> Any:
        """Get nested field value from atom using dot notation."""
        parts = field_path.split('.')
        current = atom
        
        for part in parts:
            if isinstance(current, Atom):
                if part == 'truth_value':
                    current = current.truth_value
                elif part == 'attention':
                    current = current.attention
                elif hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            elif isinstance(current, (TruthValue, AttentionValue)):
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        
        return current
    
    def _evaluate_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate a condition against a value."""
        for op, target in condition.items():
            try:
                operator = QueryOperator(op)
            except ValueError:
                logger.warning(f"Unknown operator: {op}")
                continue
            
            if operator == QueryOperator.EQ:
                if value != target:
                    return False
            elif operator == QueryOperator.GT:
                if not (value > target):
                    return False
            elif operator == QueryOperator.LT:
                if not (value < target):
                    return False
            elif operator == QueryOperator.GTE:
                if not (value >= target):
                    return False
            elif operator == QueryOperator.LTE:
                if not (value <= target):
                    return False
            elif operator == QueryOperator.CONTAINS:
                if isinstance(value, str) and target not in value:
                    return False
            elif operator == QueryOperator.IN:
                if value not in target:
                    return False
        
        return True
    
    def _execute_traversal(
        self,
        start_atoms: List[Atom],
        traverse_spec: Dict[str, Any]
    ) -> List[Atom]:
        """Execute graph traversal from starting atoms."""
        direction = traverse_spec.get('direction', 'outgoing')
        depth = traverse_spec.get('depth', 1)
        filter_spec = traverse_spec.get('filter', {})
        
        # Use dict keyed by atom ID to track unique atoms
        result_atoms_dict = {atom.id: atom for atom in start_atoms}
        
        for start_atom in start_atoms:
            traversed = self._traverse_from_atom(
                start_atom,
                direction,
                depth,
                filter_spec
            )
            for atom in traversed:
                result_atoms_dict[atom.id] = atom
        
        return list(result_atoms_dict.values())
    
    def _traverse_from_atom(
        self,
        atom: Atom,
        direction: str,
        depth: int,
        filter_spec: Dict[str, Any],
        visited: Optional[set] = None
    ) -> List[Atom]:
        """Traverse from a single atom."""
        if visited is None:
            visited = set()
        
        if atom.id in visited or depth <= 0:
            return []
        
        visited.add(atom.id)
        result = []
        
        # Get connected atoms based on direction
        if direction == 'outgoing':
            # Follow outgoing links
            if atom.is_link():
                for out_id in atom.outgoing:
                    out_atom = self.atomspace.get_atom(out_id)
                    if out_atom and self._matches_filters(out_atom, filter_spec):
                        result.append(out_atom)
                        if depth > 1:
                            result.extend(
                                self._traverse_from_atom(
                                    out_atom, direction, depth - 1, filter_spec, visited
                                )
                            )
        
        elif direction == 'incoming':
            # Follow incoming links
            incoming = self.atomspace.get_incoming(atom.id)
            for link in incoming:
                if self._matches_filters(link, filter_spec):
                    result.append(link)
                    if depth > 1:
                        result.extend(
                            self._traverse_from_atom(
                                link, direction, depth - 1, filter_spec, visited
                            )
                        )
        
        elif direction == 'both':
            # Follow both directions
            result.extend(
                self._traverse_from_atom(atom, 'outgoing', depth, filter_spec, visited)
            )
            result.extend(
                self._traverse_from_atom(atom, 'incoming', depth, filter_spec, visited)
            )
        
        return result
    
    def _apply_ordering(
        self,
        atoms: List[Atom],
        order_specs: List[Dict[str, Any]]
    ) -> List[Atom]:
        """Apply ordering to atom list."""
        for order_spec in reversed(order_specs):
            field = order_spec.get('field')
            direction = order_spec.get('direction', 'asc')
            
            atoms = sorted(
                atoms,
                key=lambda a: self._get_nested_value(a, field) or 0,
                reverse=(direction == 'desc')
            )
        
        return atoms


class HypergraphQLEngine:
    """
    Main HypergraphQL query engine.
    
    Provides high-level interface for querying atomspace with GraphQL-like syntax.
    """
    
    def __init__(self, atomspace: OpenCogAtomSpace):
        self.atomspace = atomspace
        self.parser = HypergraphQLParser(atomspace)
    
    def query(self, query: Union[str, Dict[str, Any]]) -> QueryResult:
        """
        Execute a HypergraphQL query.
        
        Args:
            query: Query string (JSON) or dictionary
        
        Returns:
            QueryResult with matching atoms
        """
        if isinstance(query, str):
            import json
            query = json.loads(query)
        
        return self.parser.parse_and_execute(query)
    
    def find_concept(
        self,
        name: str,
        min_truth: float = 0.0,
        min_attention: float = 0.0
    ) -> List[Atom]:
        """Convenience method to find concept nodes by name."""
        query = {
            'where': {
                'atom_type': 'ConceptNode',
                'name': name if not '*' in name else {'contains': name.replace('*', '')}
            },
            'limit': 100
        }
        
        if min_truth > 0:
            query['where']['truth_value.strength'] = {'gte': min_truth}
        
        if min_attention > 0:
            query['where']['attention.sti'] = {'gte': min_attention}
        
        result = self.query(query)
        return result.atoms
    
    def find_related(
        self,
        atom_id: str,
        link_type: Optional[str] = None,
        depth: int = 1,
        direction: str = 'both'
    ) -> List[Atom]:
        """Find atoms related to given atom."""
        atom = self.atomspace.get_atom(atom_id)
        if not atom:
            return []
        
        query = {
            'where': {'id': atom_id},
            'traverse': {
                'direction': direction,
                'depth': depth
            }
        }
        
        if link_type:
            query['traverse']['filter'] = {'atom_type': link_type}
        
        result = self.query(query)
        return result.atoms
    
    def find_by_attention(
        self,
        min_sti: float = 0.0,
        min_lti: float = 0.0,
        limit: int = 10
    ) -> List[Atom]:
        """Find atoms by attention values."""
        query = {
            'where': {},
            'order_by': [{'field': 'attention.sti', 'direction': 'desc'}],
            'limit': limit
        }
        
        if min_sti > 0:
            query['where']['attention.sti'] = {'gte': min_sti}
        
        if min_lti > 0:
            query['where']['attention.lti'] = {'gte': min_lti}
        
        result = self.query(query)
        return result.atoms
    
    def pattern_match(
        self,
        pattern: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Atom]]:
        """
        Advanced pattern matching with variable binding.
        
        Example pattern:
        {
            'nodes': {
                '$x': {'atom_type': 'ConceptNode', 'name': 'deep-tree-echo'},
                '$y': {'atom_type': 'ConceptNode'}
            },
            'links': [
                {'atom_type': 'InheritanceLink', 'outgoing': ['$x', '$y']}
            ]
        }
        """
        # This is a simplified pattern matcher
        # Full implementation would handle variable substitution and unification
        
        results = []
        node_patterns = pattern.get('nodes', {})
        link_patterns = pattern.get('links', [])
        
        # Match nodes first
        node_bindings = {}
        for var_name, node_pattern in node_patterns.items():
            query_result = self.query({'where': node_pattern, 'limit': 100})
            if query_result.atoms:
                node_bindings[var_name] = query_result.atoms
            else:
                return []  # No matches for required variable
        
        # Try to find compatible link combinations
        # This is simplified - full implementation would use constraint satisfaction
        for link_pattern in link_patterns:
            link_query = {'where': {'atom_type': link_pattern.get('atom_type')}}
            link_result = self.query(link_query)
            
            # Check if any links match the pattern with node bindings
            for link in link_result.atoms:
                binding = {}
                is_match = True
                
                # Simplified matching - just checks if outgoing matches variables
                for i, out_id in enumerate(link.outgoing):
                    var_name = link_pattern['outgoing'][i] if i < len(link_pattern['outgoing']) else None
                    if var_name and var_name.startswith('$'):
                        if var_name in node_bindings:
                            # Check if this matches one of the bound nodes
                            if not any(n.id == out_id for n in node_bindings[var_name]):
                                is_match = False
                                break
                            binding[var_name] = self.atomspace.get_atom(out_id)
                
                if is_match and binding:
                    results.append(binding)
        
        return results
    
    def explain_query(self, query: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Explain query execution plan without executing."""
        if isinstance(query, str):
            import json
            query = json.loads(query)
        
        explanation = {
            'query': query,
            'execution_plan': [],
            'estimated_cost': 0
        }
        
        # Analyze WHERE clause
        where_clause = query.get('where', {})
        if 'atom_type' in where_clause:
            atom_type = where_clause['atom_type']
            type_count = len(self.atomspace.get_atoms_by_type(
                AtomType[atom_type.upper().replace('-', '_')]
            ))
            explanation['execution_plan'].append({
                'step': 'Filter by type',
                'type': atom_type,
                'estimated_results': type_count
            })
            explanation['estimated_cost'] += type_count
        else:
            explanation['execution_plan'].append({
                'step': 'Full scan',
                'estimated_results': len(self.atomspace.atoms)
            })
            explanation['estimated_cost'] += len(self.atomspace.atoms)
        
        # Analyze traversal
        if 'traverse' in query:
            traverse_spec = query['traverse']
            depth = traverse_spec.get('depth', 1)
            explanation['execution_plan'].append({
                'step': 'Graph traversal',
                'direction': traverse_spec.get('direction', 'outgoing'),
                'depth': depth,
                'estimated_cost_multiplier': depth * 2
            })
            explanation['estimated_cost'] *= (depth * 2)
        
        # Analyze ordering
        if 'order_by' in query:
            explanation['execution_plan'].append({
                'step': 'Sort results',
                'fields': query['order_by']
            })
        
        return explanation
