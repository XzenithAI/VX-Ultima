"""
VX-ARCHEOS Extended Capabilities
=================================

Additional reasoning capabilities:
- Constraint satisfaction with AC-3 and backtracking
- Semantic knowledge graph with embeddings
- Abstract pattern recognition
- Temporal reasoning
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib

# ============================================================================
# CONSTRAINT SATISFACTION PROBLEM SOLVER
# ============================================================================

@dataclass
class Variable:
    """CSP Variable"""
    name: str
    domain: Set[Any]

    def __hash__(self):
        return hash(self.name)

@dataclass
class Constraint:
    """Constraint between variables"""
    variables: Tuple[str, ...]
    satisfied: Callable[[Dict[str, Any]], bool]
    description: str = ""

class ConstraintSatisfactionSolver:
    """
    AC-3 arc consistency + backtracking search for CSPs
    """

    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        self.solution_count = 0

    def add_variable(self, name: str, domain: Set[Any]):
        """Add a variable to the CSP"""
        self.variables[name] = Variable(name, domain.copy())

    def add_constraint(self, variables: Tuple[str, ...],
                      condition: Callable[[Dict[str, Any]], bool],
                      description: str = ""):
        """Add a constraint"""
        self.constraints.append(Constraint(variables, condition, description))

    def solve(self, find_all: bool = False) -> List[Dict[str, Any]]:
        """
        Solve the CSP using AC-3 + backtracking.

        Args:
            find_all: If True, find all solutions. Otherwise, find first solution.

        Returns:
            List of solutions (variable assignments)
        """
        # Apply AC-3 first to reduce domains
        if not self.ac3():
            return []  # No solution possible

        # Use backtracking to find solution(s)
        solutions = []
        self._backtrack({}, solutions, find_all)
        return solutions

    def ac3(self) -> bool:
        """
        AC-3 arc consistency algorithm.
        Returns True if domains remain non-empty, False if inconsistency detected.
        """
        # Create queue of arcs
        queue = deque()

        for constraint in self.constraints:
            for var in constraint.variables:
                for other_var in constraint.variables:
                    if var != other_var:
                        queue.append((var, other_var, constraint))

        while queue:
            xi, xj, constraint = queue.popleft()

            if self._revise(xi, xj, constraint):
                if len(self.variables[xi].domain) == 0:
                    return False  # Inconsistency detected

                # Add arcs for neighbors of xi
                for other_constraint in self.constraints:
                    if xi in other_constraint.variables:
                        for xk in other_constraint.variables:
                            if xk != xi and xk != xj:
                                queue.append((xk, xi, other_constraint))

        return True

    def _revise(self, xi: str, xj: str, constraint: Constraint) -> bool:
        """
        Make xi arc-consistent with respect to xj.
        Returns True if domain of xi was revised.
        """
        revised = False
        domain_i = list(self.variables[xi].domain)

        for vi in domain_i:
            # Check if there exists a value in domain of xj that satisfies constraint
            satisfiable = False

            for vj in self.variables[xj].domain:
                assignment = {xi: vi, xj: vj}
                if constraint.satisfied(assignment):
                    satisfiable = True
                    break

            if not satisfiable:
                self.variables[xi].domain.remove(vi)
                revised = True

        return revised

    def _backtrack(self, assignment: Dict[str, Any],
                   solutions: List[Dict[str, Any]],
                   find_all: bool):
        """Backtracking search"""
        if len(assignment) == len(self.variables):
            # Complete assignment found
            solutions.append(assignment.copy())
            self.solution_count += 1
            return not find_all  # Stop if we only want one solution

        # Select unassigned variable (using MRV heuristic)
        var = self._select_unassigned_variable(assignment)

        # Try each value in domain (ordered by LCV heuristic)
        for value in self._order_domain_values(var, assignment):
            assignment[var] = value

            if self._consistent(assignment):
                result = self._backtrack(assignment, solutions, find_all)
                if result:  # Found solution and should stop
                    return True

            del assignment[var]

        return False

    def _select_unassigned_variable(self, assignment: Dict[str, Any]) -> str:
        """Select next variable using Minimum Remaining Values heuristic"""
        unassigned = [v for v in self.variables if v not in assignment]

        # MRV: choose variable with smallest domain
        return min(unassigned,
                  key=lambda v: len(self.variables[v].domain))

    def _order_domain_values(self, var: str, assignment: Dict[str, Any]) -> List[Any]:
        """Order domain values using Least Constraining Value heuristic"""
        # Simplified: just return domain values in order
        # Full LCV would count how many values are ruled out in neighbors
        return list(self.variables[var].domain)

    def _consistent(self, assignment: Dict[str, Any]) -> bool:
        """Check if current assignment satisfies all applicable constraints"""
        for constraint in self.constraints:
            # Check if all variables in constraint are assigned
            if all(v in assignment for v in constraint.variables):
                if not constraint.satisfied(assignment):
                    return False

        return True

# ============================================================================
# SEMANTIC KNOWLEDGE GRAPH
# ============================================================================

class KnowledgeNode:
    """Node in knowledge graph"""
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        self.id = node_id
        self.type = node_type
        self.properties = properties or {}
        self.embedding: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, KnowledgeNode) and self.id == other.id

@dataclass
class KnowledgeEdge:
    """Edge in knowledge graph"""
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

class SemanticKnowledgeGraph:
    """
    Knowledge graph with semantic embeddings for similarity computation
    """

    def __init__(self, embedding_dim: int = 64):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.embedding_dim = embedding_dim

        # Adjacency structure
        self.outgoing: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        self.incoming: Dict[str, List[KnowledgeEdge]] = defaultdict(list)

    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        """Add node to graph"""
        if node_id not in self.nodes:
            node = KnowledgeNode(node_id, node_type, properties)
            node.embedding = self._compute_embedding(node)
            self.nodes[node_id] = node

    def add_edge(self, source: str, target: str, relation: str,
                properties: Dict[str, Any] = None, weight: float = 1.0):
        """Add edge to graph"""
        edge = KnowledgeEdge(source, target, relation, properties or {}, weight)
        self.edges.append(edge)
        self.outgoing[source].append(edge)
        self.incoming[target].append(edge)

    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[str]:
        """Get neighboring nodes, optionally filtered by relation"""
        neighbors = []
        for edge in self.outgoing[node_id]:
            if relation is None or edge.relation == relation:
                neighbors.append(edge.target)
        return neighbors

    def shortest_path(self, source: str, target: str,
                     max_length: int = 5) -> Optional[List[str]]:
        """Find shortest path between nodes using BFS"""
        if source not in self.nodes or target not in self.nodes:
            return None

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if len(path) > max_length:
                continue

            if current == target:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def find_similar_nodes(self, node_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar nodes based on embedding similarity"""
        if node_id not in self.nodes or self.nodes[node_id].embedding is None:
            return []

        query_emb = self.nodes[node_id].embedding
        similarities = []

        for other_id, other_node in self.nodes.items():
            if other_id == node_id or other_node.embedding is None:
                continue

            # Cosine similarity
            similarity = np.dot(query_emb, other_node.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(other_node.embedding) + 1e-8
            )
            similarities.append((other_id, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def subgraph_around(self, node_id: str, radius: int = 2) -> 'SemanticKnowledgeGraph':
        """Extract subgraph within radius of node"""
        if node_id not in self.nodes:
            return SemanticKnowledgeGraph(self.embedding_dim)

        # BFS to find nodes within radius
        nodes_to_include = set()
        queue = deque([(node_id, 0)])
        visited = {node_id}

        while queue:
            current, dist = queue.popleft()
            nodes_to_include.add(current)

            if dist < radius:
                for neighbor in self.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Create subgraph
        subgraph = SemanticKnowledgeGraph(self.embedding_dim)

        for nid in nodes_to_include:
            node = self.nodes[nid]
            subgraph.add_node(nid, node.type, node.properties)

        for edge in self.edges:
            if edge.source in nodes_to_include and edge.target in nodes_to_include:
                subgraph.add_edge(edge.source, edge.target, edge.relation,
                                edge.properties, edge.weight)

        return subgraph

    def _compute_embedding(self, node: KnowledgeNode) -> np.ndarray:
        """Compute node embedding (simplified random projection)"""
        # In production, use actual embedding model
        # This is a deterministic hash-based embedding

        # Combine node ID and type
        content = f"{node.id}:{node.type}"
        hash_val = int(hashlib.sha256(content.encode()).hexdigest(), 16)

        # Use hash as seed for reproducible random embedding
        rng = np.random.RandomState(hash_val % (2**32))
        embedding = rng.randn(self.embedding_dim)

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def query_path_patterns(self, pattern: List[str]) -> List[List[str]]:
        """
        Find paths matching a relation pattern.
        Example: pattern = ['causes', 'leads_to'] finds X -causes-> Y -leads_to-> Z
        """
        if not pattern:
            return []

        results = []

        # Start from all nodes
        for start_node in self.nodes:
            paths = self._match_pattern_from(start_node, pattern, [])
            results.extend(paths)

        return results

    def _match_pattern_from(self, current: str, pattern: List[str],
                           path: List[str]) -> List[List[str]]:
        """Recursively match pattern from current node"""
        if not pattern:
            return [path + [current]]

        next_relation = pattern[0]
        remaining_pattern = pattern[1:]
        results = []

        for edge in self.outgoing[current]:
            if edge.relation == next_relation:
                sub_results = self._match_pattern_from(
                    edge.target, remaining_pattern, path + [current]
                )
                results.extend(sub_results)

        return results

# ============================================================================
# ABSTRACT PATTERN RECOGNITION
# ============================================================================

class AbstractPattern:
    """Represents an abstract pattern extracted from examples"""
    def __init__(self, pattern_type: str):
        self.pattern_type = pattern_type
        self.variables: Set[str] = set()
        self.constraints: List[Callable] = []
        self.examples: List[Any] = []
        self.confidence = 0.0

@dataclass
class Transformation:
    """A transformation between representations"""
    name: str
    source_form: str
    target_form: str
    transform_fn: Callable[[Any], Any]

class PatternRecognizer:
    """Recognizes abstract patterns across different domains"""

    def __init__(self):
        self.patterns: List[AbstractPattern] = []
        self.transformations: List[Transformation] = []

    def learn_pattern(self, examples: List[Any], pattern_type: str) -> AbstractPattern:
        """
        Learn abstract pattern from examples.
        """
        pattern = AbstractPattern(pattern_type)
        pattern.examples = examples

        if pattern_type == 'sequence':
            pattern = self._learn_sequence_pattern(examples)
        elif pattern_type == 'structure':
            pattern = self._learn_structure_pattern(examples)
        elif pattern_type == 'functional':
            pattern = self._learn_functional_pattern(examples)

        self.patterns.append(pattern)
        return pattern

    def _learn_sequence_pattern(self, examples: List[Any]) -> AbstractPattern:
        """Learn sequential pattern (e.g., Fibonacci, arithmetic progression)"""
        pattern = AbstractPattern('sequence')

        if not examples or len(examples) < 3:
            return pattern

        # Check for arithmetic progression
        if all(isinstance(x, (int, float)) for x in examples):
            diffs = [examples[i+1] - examples[i] for i in range(len(examples)-1)]

            if len(set(diffs)) == 1:  # Constant difference
                pattern.variables = {'start', 'step'}
                pattern.confidence = 0.9
                pattern.constraints.append(
                    lambda n, start=examples[0], step=diffs[0]: start + n * step
                )

        # Check for geometric progression
        if all(isinstance(x, (int, float)) and x != 0 for x in examples):
            ratios = [examples[i+1] / examples[i] for i in range(len(examples)-1)]

            if all(abs(r - ratios[0]) < 0.01 for r in ratios):  # Constant ratio
                pattern.variables = {'start', 'ratio'}
                pattern.confidence = 0.85
                ratio = ratios[0]
                pattern.constraints.append(
                    lambda n, start=examples[0], ratio=ratio: start * (ratio ** n)
                )

        return pattern

    def _learn_structure_pattern(self, examples: List[Any]) -> AbstractPattern:
        """Learn structural pattern"""
        pattern = AbstractPattern('structure')
        # Simplified - would use graph isomorphism in full version
        pattern.confidence = 0.5
        return pattern

    def _learn_functional_pattern(self, examples: List[Tuple[Any, Any]]) -> AbstractPattern:
        """Learn functional relationship pattern"""
        pattern = AbstractPattern('functional')

        if not examples:
            return pattern

        # Simple linear regression
        if all(isinstance(x, (int, float)) and isinstance(y, (int, float))
               for x, y in examples):

            X = np.array([x for x, y in examples])
            Y = np.array([y for x, y in examples])

            # Fit y = mx + b
            if len(X) > 1:
                m = np.cov(X, Y)[0, 1] / (np.var(X) + 1e-8)
                b = np.mean(Y) - m * np.mean(X)

                pattern.variables = {'slope', 'intercept'}
                pattern.confidence = 0.7
                pattern.constraints.append(lambda x: m * x + b)

        return pattern

    def apply_pattern(self, pattern: AbstractPattern, input_data: Any) -> Any:
        """Apply learned pattern to new data"""
        if not pattern.constraints:
            return None

        # Use first constraint as the pattern function
        try:
            return pattern.constraints[0](input_data)
        except:
            return None

    def transfer_pattern(self, pattern: AbstractPattern,
                        source_domain: str, target_domain: str) -> Optional[AbstractPattern]:
        """Transfer pattern from one domain to another"""
        # Find transformation between domains
        transform = None
        for t in self.transformations:
            if t.source_form == source_domain and t.target_form == target_domain:
                transform = t
                break

        if not transform:
            return None

        # Apply transformation to pattern
        transferred = AbstractPattern(pattern.pattern_type)
        transferred.variables = pattern.variables.copy()
        transferred.confidence = pattern.confidence * 0.8  # Reduce confidence for transfer

        # Transform examples
        try:
            transferred.examples = [transform.transform_fn(ex) for ex in pattern.examples]
        except:
            return None

        return transferred

# ============================================================================
# TESTING
# ============================================================================

def test_csp_solver():
    """Test constraint satisfaction solver"""
    print("=== Testing CSP Solver ===\n")

    solver = ConstraintSatisfactionSolver()

    # Classic N-Queens problem (4-Queens)
    n = 4
    for i in range(n):
        solver.add_variable(f"Q{i}", set(range(n)))

    # Add constraints: no two queens in same row, column, or diagonal
    def no_attack(assignment, i, j):
        if f"Q{i}" not in assignment or f"Q{j}" not in assignment:
            return True
        qi = assignment[f"Q{i}"]
        qj = assignment[f"Q{j}"]
        # Different columns (implicit), different rows, different diagonals
        return qi != qj and abs(qi - qj) != abs(i - j)

    for i in range(n):
        for j in range(i+1, n):
            solver.add_constraint(
                (f"Q{i}", f"Q{j}"),
                lambda a, i=i, j=j: no_attack(a, i, j),
                f"Q{i} and Q{j} don't attack"
            )

    solutions = solver.solve(find_all=True)
    print(f"4-Queens problem: Found {len(solutions)} solutions")
    if solutions:
        print(f"Example solution: {solutions[0]}")

    return len(solutions) > 0

def test_knowledge_graph():
    """Test semantic knowledge graph"""
    print("\n=== Testing Knowledge Graph ===\n")

    kg = SemanticKnowledgeGraph(embedding_dim=32)

    # Build a simple ontology
    kg.add_node("mammal", "class", {"level": "high"})
    kg.add_node("dog", "class", {"level": "medium"})
    kg.add_node("cat", "class", {"level": "medium"})
    kg.add_node("fido", "instance", {"name": "Fido"})
    kg.add_node("whiskers", "instance", {"name": "Whiskers"})

    kg.add_edge("dog", "mammal", "is_a")
    kg.add_edge("cat", "mammal", "is_a")
    kg.add_edge("fido", "dog", "instance_of")
    kg.add_edge("whiskers", "cat", "instance_of")

    # Test path finding
    path = kg.shortest_path("fido", "mammal")
    print(f"Path from Fido to Mammal: {path}")

    # Test similarity
    similar = kg.find_similar_nodes("dog", top_k=2)
    print(f"Nodes similar to 'dog': {similar}")

    return path is not None

def test_pattern_recognition():
    """Test pattern recognition"""
    print("\n=== Testing Pattern Recognition ===\n")

    recognizer = PatternRecognizer()

    # Learn arithmetic sequence
    arith_seq = [2, 5, 8, 11, 14]
    pattern = recognizer.learn_pattern(arith_seq, 'sequence')

    if pattern.constraints:
        predicted = pattern.constraints[0](5)  # Predict 6th term
        print(f"Arithmetic sequence: {arith_seq}")
        print(f"Predicted next term (index 5): {predicted}")
        print(f"Pattern confidence: {pattern.confidence}")

    # Learn functional relationship
    func_examples = [(1, 3), (2, 5), (3, 7), (4, 9)]  # y = 2x + 1
    func_pattern = recognizer.learn_pattern(func_examples, 'functional')

    if func_pattern.constraints:
        predicted = func_pattern.constraints[0](5)
        print(f"\nFunctional pattern from {func_examples}")
        print(f"Predicted f(5): {predicted}")

    return pattern.confidence > 0.5

def run_extended_tests():
    """Run all extended tests"""
    print("=" * 60)
    print("VX-ARCHEOS Extended Capabilities - Test Suite")
    print("=" * 60 + "\n")

    results = {
        'CSP Solver': test_csp_solver(),
        'Knowledge Graph': test_knowledge_graph(),
        'Pattern Recognition': test_pattern_recognition()
    }

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test:.<40} {status}")

    return all(results.values())

if __name__ == "__main__":
    run_extended_tests()
