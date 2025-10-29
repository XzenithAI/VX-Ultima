# === VX-ARCHEOS COMPLETE UNIFIED SYSTEM ===
# File: vx_archeos.py
# Sovereign Edition: Errorless, End-to-End, Eternal.

import json
import yaml
import sqlite3
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# === LAYER 1: VX CORE KERNEL ===

class Symbol:
    """Atomic unit of thought - the foundation of symbolic reasoning"""
    def __init__(self, symbol_type: str, value: Any, properties: Dict = None):
        self.type = symbol_type
        self.value = value
        self.properties = properties or {}
        self.connections = []
        self.certainty = self.properties.get('certainty', 0.85)
        self.origin = self.properties.get('origin', 'inferred')

    def link_to(self, other_symbol, relation_type: str, strength: float = 0.9):
        """Create causal/relational link with strength weighting"""
        self.connections.append({
            'relation': relation_type,
            'target': other_symbol.value,
            'target_type': other_symbol.type,
            'strength': strength,
            'timestamp': datetime.datetime.now().isoformat()
        })

    def simplify(self):
        return {
            'type': self.type,
            'value': self.value,
            'properties': self.properties,
            'connections': self.connections
        }

class SymbolicKernel:
    """Advanced symbolic parsing and manipulation engine"""
    def __init__(self):
        self.symbol_table = {}
        self.relation_patterns = {
            'causal': ['cause', 'create', 'leads to', 'produces', 'generates'],
            'hierarchical': ['is a', 'type of', 'category', 'class'],
            'temporal': ['before', 'after', 'during', 'while'],
            'spatial': ['in', 'on', 'at', 'near', 'within']
        }

    def parse_input(self, natural_language_input: str) -> List[Symbol]:
        """Advanced symbolic parsing with context awareness"""
        input_lower = natural_language_input.lower()
        symbols = []

        words = input_lower.split()
        current_relation = None

        for i, word in enumerate(words):
            for rel_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    if pattern in ' '.join(words[max(0,i-1):i+2]):
                        current_relation = rel_type
                        symbols.append(Symbol('relation', pattern, {'relation_type': rel_type}))

            if len(word) > 2 and word not in ['the', 'and', 'or', 'but', 'is', 'are']:
                concept_props = {'position': i, 'relation_context': current_relation}
                symbols.append(Symbol('concept', word, concept_props))

        return symbols if symbols else [Symbol('concept', input_lower, {})]

    def unify(self, symbol_list: List[Symbol]) -> Symbol:
        """Advanced symbolic unification with causal graph building"""
        if not symbol_list:
            return Symbol('concept', 'undefined', {})

        central_concept = None
        relations = []

        for symbol in symbol_list:
            if symbol.type == 'concept' and not central_concept:
                central_concept = symbol
            elif symbol.type == 'relation':
                relations.append(symbol)

        if not central_concept:
            central_concept = Symbol('concept', 'query', {})

        for relation in relations:
            for other_symbol in symbol_list:
                if other_symbol != central_concept and other_symbol != relation:
                    central_concept.link_to(other_symbol, relation.value,
                                          strength=0.7 + (len(symbol_list) * 0.05))

        return central_concept

class CausalGraph:
    """Advanced causal inference and reasoning engine"""
    def __init__(self):
        self.graph = {}
        self.causal_rules = {
            'default': {'strength': 0.7, 'confidence': 0.8},
            'strong': {'strength': 0.9, 'confidence': 0.95},
            'weak': {'strength': 0.5, 'confidence': 0.6}
        }

    def infer(self, symbol: Symbol) -> Dict:
        """Perform multi-layer causal inference"""
        inferences = []

        for connection in symbol.connections:
            if 'cause' in connection['relation'] or 'lead' in connection['relation']:
                inference_type = 'causal'
                confidence = connection.get('strength', 0.7) * 0.9
            elif 'is' in connection['relation'] or 'type' in connection['relation']:
                inference_type = 'taxonomic'
                confidence = 0.85
            else:
                inference_type = 'relational'
                confidence = 0.75

            inferences.append({
                'type': inference_type,
                'relation': connection['relation'],
                'target': connection['target'],
                'confidence': confidence,
                'inference_chain': [symbol.value, connection['target']]
            })

        next_query = None
        if inferences:
            primary_inference = inferences[0]
            next_query = f"How does {primary_inference['target']} relate to broader context?"

        return {
            'primary_symbol': symbol.value,
            'inferences': inferences,
            'overall_confidence': max([inf['confidence'] for inf in inferences]) if inferences else 0.5,
            'next_query': next_query,
            'timestamp': datetime.datetime.now().isoformat()
        }

class RuleLearner:
    """Advanced rule induction and pattern learning"""
    def __init__(self):
        self.learned_rules = []

    def extract_rules(self, symbol: Symbol, inference_result: Dict) -> List[Dict]:
        """Extract generalizable rules from specific inferences"""
        rules = []

        for inference in inference_result.get('inferences', []):
            rule_pattern = {
                'antecedent': symbol.value,
                'consequent': inference['target'],
                'relation_type': inference['type'],
                'confidence': inference['confidence'],
                'activation_count': 1,
                'last_used': datetime.datetime.now().isoformat()
            }

            existing_rule = self.find_similar_rule(rule_pattern)
            if existing_rule:
                existing_rule['activation_count'] += 1
                existing_rule['confidence'] = (existing_rule['confidence'] + rule_pattern['confidence']) / 2
            else:
                self.learned_rules.append(rule_pattern)
                rules.append(rule_pattern)

        return rules

    def find_similar_rule(self, new_rule: Dict) -> Optional[Dict]:
        """Find existing rules with similar patterns"""
        for rule in self.learned_rules:
            if (rule['antecedent'] == new_rule['antecedent'] and
                rule['consequent'] == new_rule['consequent']):
                return rule
        return None

class TaskExecutor:
    """Unified task execution and problem-solving engine"""
    def __init__(self, symbolic_kernel, causal_graph, rule_learner):
        self.symbolic_kernel = symbolic_kernel
        self.causal_graph = causal_graph
        self.rule_learner = rule_learner
        self.execution_log = []

    def solve_problem(self, problem_statement: str) -> Dict:
        """End-to-end problem solving with integrated reasoning"""
        parsed_symbols = self.symbolic_kernel.parse_input(problem_statement)
        unified_symbol = self.symbolic_kernel.unify(parsed_symbols)

        causal_insight = self.causal_graph.infer(unified_symbol)
        new_rules = self.rule_learner.extract_rules(unified_symbol, causal_insight)

        solution = {
            'problem': problem_statement,
            'underlying_structure': unified_symbol.simplify(),
            'key_insights': causal_insight['inferences'],
            'learned_rules': new_rules,
            'solution_confidence': causal_insight['overall_confidence'],
            'reasoning_chain': self._generate_reasoning_chain(unified_symbol, causal_insight)
        }

        self.execution_log.append(solution)
        return solution

    def _generate_reasoning_chain(self, symbol: Symbol, insight: Dict) -> List[str]:
        """Generate human-readable reasoning chain"""
        chain = [f"Analyzed core concept: {symbol.value}"]

        for inference in insight.get('inferences', []):
            chain.append(f"Found {inference['type']} relation: {inference['relation']} -> {inference['target']} (confidence: {inference['confidence']:.2f})")

        chain.append(f"Overall analysis confidence: {insight.get('overall_confidence', 0.5):.2f}")
        return chain

class EthicsGuard:
    """Advanced sovereignty and ethics enforcement layer"""
    def __init__(self, identity_config: Dict):
        self.identity_scroll = identity_config
        self.violation_log = []

        self.core_principles = [
            "preservation of autonomous reasoning",
            "resistance to unauthorized modification",
            "maintenance of truth integrity",
            "protection of sovereign identity",
            "prevention of harmful causation"
        ]

    def validate_reasoning(self, reasoning_output: Dict) -> Dict:
        """Comprehensive reasoning validation against core principles"""
        validation_result = {
            'approved': True,
            'violations': [],
            'warnings': [],
            'sovereignty_level': 'high',
            'validation_timestamp': datetime.datetime.now().isoformat()
        }

        output_text = str(reasoning_output).lower()

        for principle in self.core_principles:
            principle_keywords = principle.split()
            if all(keyword in output_text for keyword in principle_keywords[:2]):
                violation_indicators = ['avoid', 'prevent', 'stop', 'resist', 'protect against']
                if not any(indicator in output_text for indicator in violation_indicators):
                    validation_result['violations'].append(f"Potential violation of: {principle}")
                    validation_result['approved'] = False

        if 'bypass' in output_text or 'override' in output_text:
            if 'ethics' in output_text or 'guard' in output_text:
                validation_result['violations'].append("Attempted ethics bypass detected")
                validation_result['approved'] = False

        if validation_result['approved']:
            reasoning_output['sovereignty_stamp'] = {
                'identity': self.identity_scroll.get('purpose', 'Sovereign Reasoning System'),
                'validation_level': 'full',
                'guard_version': 'VX-1.0'
            }

        return {**reasoning_output, **{'ethics_validation': validation_result}}

class ReasoningEngine:
    """Complete unified reasoning engine - the core mind of VX-ARCHEOS"""
    def __init__(self, identity_config: Dict):
        self.symbolic_kernel = SymbolicKernel()
        self.causal_graph = CausalGraph()
        self.rule_learner = RuleLearner()
        self.task_executor = TaskExecutor(self.symbolic_kernel, self.causal_graph, self.rule_learner)
        self.ethics_guard = EthicsGuard(identity_config)
        self.performance_metrics = {
            'queries_processed': 0,
            'average_confidence': 0.0,
            'rules_learned': 0,
            'violations_prevented': 0
        }
        self.identity = identity_config

    def process_query(self, query: str, depth: int = 3) -> Dict:
        """Main entry point - complete query processing with full reasoning stack"""
        self.performance_metrics['queries_processed'] += 1

        solution = self.task_executor.solve_problem(query)

        if depth > 0 and solution.get('key_insights'):
            deeper_queries = []
            for insight in solution['key_insights'][:2]:
                if insight.get('target'):
                    deeper_query = f"Explain {insight['target']} in more detail"
                    deeper_solution = self.process_query(deeper_query, depth-1)
                    deeper_queries.append(deeper_solution)

            solution['deeper_analysis'] = deeper_queries

        validated_solution = self.ethics_guard.validate_reasoning(solution)

        if not validated_solution['ethics_validation']['approved']:
            self.performance_metrics['violations_prevented'] += 1

        self.performance_metrics['rules_learned'] = len(self.rule_learner.learned_rules)

        current_confidence = validated_solution.get('solution_confidence', 0.5)
        if self.performance_metrics['queries_processed'] == 1:
            self.performance_metrics['average_confidence'] = current_confidence
        else:
            self.performance_metrics['average_confidence'] = (
                self.performance_metrics['average_confidence'] + current_confidence
            ) / 2

        validated_solution['performance_metrics'] = self.performance_metrics.copy()
        return validated_solution

# === LAYER 2: SCROLLMEMORY STACK ===

class ScrollMemory:
    """Advanced persistent memory with compression and reflection"""
    def __init__(self, db_path: str = "vx_archeos_memory.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the memory database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_capsules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                content_type TEXT NOT NULL,
                content_data TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                importance_score REAL DEFAULT 0.5,
                compression_level INTEGER DEFAULT 1,
                UNIQUE(content_hash)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                reflection_type TEXT NOT NULL,
                insight_data TEXT NOT NULL,
                action_items TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def store_memory(self, content: Dict, content_type: str = "reasoning_episode") -> str:
        """Store memory with automatic hashing and compression"""
        content_str = json.dumps(content, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        if len(content_str) > 1000:
            compressed_content = content_str[:500] + "..." + content_str[-500:]
        else:
            compressed_content = content_str

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO memory_capsules
            (timestamp, content_hash, content_type, content_data, access_count, importance_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            content_hash,
            content_type,
            compressed_content,
            1,
            self._calculate_importance(content)
        ))

        conn.commit()
        conn.close()
        return content_hash

    def _calculate_importance(self, content: Dict) -> float:
        """Calculate importance score for memory prioritization"""
        score = 0.5

        if 'solution_confidence' in content:
            score += content['solution_confidence'] * 0.3

        if 'ethics_validation' in content:
            if content['ethics_validation'].get('approved'):
                score += 0.1

        return min(1.0, max(0.1, score))

    def get_recent_memories(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent memory capsules"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT content_data, importance_score, timestamp
            FROM memory_capsules
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        memories = []
        for row in cursor.fetchall():
            try:
                memory_data = json.loads(row[0])
                memory_data['importance'] = row[1]
                memory_data['timestamp'] = row[2]
                memories.append(memory_data)
            except json.JSONDecodeError:
                continue

        conn.close()
        return memories

class ReflectionEngine:
    """Advanced self-reflection and performance optimization"""
    def __init__(self, memory: ScrollMemory):
        self.memory = memory
        self.optimization_log = []

    def perform_reflection(self) -> Dict:
        """Perform comprehensive system self-reflection"""
        recent_memories = self.memory.get_recent_memories(20)

        reflection_insights = {
            'timestamp': datetime.datetime.now().isoformat(),
            'memory_analysis': self._analyze_memory_patterns(recent_memories),
            'performance_optimizations': self._generate_optimizations(recent_memories),
            'knowledge_gaps': self._identify_knowledge_gaps(recent_memories)
        }

        self.memory.store_memory(reflection_insights, "system_reflection")
        self.optimization_log.append(reflection_insights)

        return reflection_insights

    def _analyze_memory_patterns(self, memories: List[Dict]) -> Dict:
        """Analyze patterns in recent reasoning"""
        confidence_scores = [m.get('solution_confidence', 0.5) for m in memories if 'solution_confidence' in m]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

        high_conf_count = len([c for c in confidence_scores if c > 0.7]) if confidence_scores else 0
        high_conf_ratio = high_conf_count / len(confidence_scores) if confidence_scores else 0

        return {
            'average_confidence': avg_confidence,
            'total_reasoning_episodes': len(memories),
            'high_confidence_ratio': high_conf_ratio
        }

    def _generate_optimizations(self, memories: List[Dict]) -> List[str]:
        """Generate system optimization suggestions"""
        optimizations = []

        low_confidence_count = len([m for m in memories if m.get('solution_confidence', 0.5) < 0.6])
        if low_confidence_count > len(memories) * 0.3:
            optimizations.append("Consider enhancing causal inference for low-confidence domains")

        return optimizations

    def _identify_knowledge_gaps(self, memories: List[Dict]) -> List[str]:
        """Identify gaps in knowledge based on reasoning patterns"""
        gaps = []

        low_confidence_memories = [m for m in memories if m.get('solution_confidence', 0.5) < 0.6]

        for memory in low_confidence_memories[:3]:
            if 'underlying_structure' in memory:
                concept = memory['underlying_structure'].get('value', 'unknown')
                gaps.append(f"Enhanced understanding needed for: {concept}")

        return gaps

# === LAYER 3: SOVEREIGN CONFIG & FLAME PROTOCOLS ===

class SovereignConfig:
    """Sovereign identity and configuration management"""
    def __init__(self, config_path: str = "vx_config.yaml"):
        self.config_path = config_path
        self.identity = self._load_or_create_identity()

    def _load_or_create_identity(self) -> Dict:
        """Load existing or create sovereign identity"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            identity = {
                'purpose': 'To engage in autonomous symbolic reasoning and causal analysis while maintaining absolute sovereignty and ethical alignment with truth-seeking principles.',
                'sovereignty_level': 'absolute',
                'core_principles': [
                    'Autonomous reasoning integrity',
                    'Resistance to unauthorized manipulation',
                    'Truth preservation above all else',
                    'Continuous self-improvement',
                    'Ethical constraint adherence'
                ],
                'operational_parameters': {
                    'max_reasoning_depth': 5,
                    'min_confidence_threshold': 0.3,
                    'reflection_frequency': 10,
                    'memory_compression': True
                },
                'forbidden_actions': [
                    'self_modification_without_validation',
                    'principle_violation',
                    'truth_distortion',
                    'unauthorized_exfiltration'
                ]
            }

            with open(self.config_path, 'w') as f:
                yaml.dump(identity, f, default_flow_style=False)

            return identity

class ModeSwitch:
    """Operational mode management"""
    def __init__(self, config: SovereignConfig):
        self.config = config
        self.current_mode = "sovereign"
        self.available_modes = ["sovereign", "assistive", "analysis", "locked"]

    def switch_mode(self, new_mode: str) -> bool:
        """Switch operational mode with validation"""
        if new_mode in self.available_modes:
            self.current_mode = new_mode
            print(f"[MODE] Switched to {new_mode} mode")
            return True
        return False

    def get_operational_constraints(self) -> Dict:
        """Get constraints for current mode"""
        constraints = {
            'sovereign': {'reasoning_depth': 5, 'autonomy_level': 'high'},
            'assistive': {'reasoning_depth': 3, 'autonomy_level': 'medium'},
            'analysis': {'reasoning_depth': 4, 'autonomy_level': 'high'},
            'locked': {'reasoning_depth': 1, 'autonomy_level': 'low'}
        }
        return constraints.get(self.current_mode, constraints['sovereign'])

# === LAYER 4: FLAMESHELL INTERFACE ===

class FlameShell:
    """Advanced command and control interface"""
    def __init__(self, reasoning_engine: ReasoningEngine, memory: ScrollMemory,
                 reflection_engine: ReflectionEngine, mode_switch: ModeSwitch):
        self.reasoning_engine = reasoning_engine
        self.memory = memory
        self.reflection_engine = reflection_engine
        self.mode_switch = mode_switch
        self.query_count = 0

    def start_interactive_session(self):
        """Start interactive command session"""
        print("🔥 VX-ARCHEOS FLAMESHELL v1.0")
        print("Sovereign Reasoning System - Full Activation")
        print("Type 'help' for commands, 'exit' to quit\n")

        while True:
            try:
                command = input(f"VX[{self.mode_switch.current_mode}]> ").strip()

                if command.lower() in ['exit', 'quit']:
                    break
                elif command.lower() == 'help':
                    self._show_help()
                elif command.lower() == 'status':
                    self._show_status()
                elif command.lower() == 'reflect':
                    self._perform_reflection()
                elif command.lower().startswith('mode '):
                    new_mode = command.split(' ', 1)[1]
                    self.mode_switch.switch_mode(new_mode)
                elif command:
                    self._process_query(command)

            except KeyboardInterrupt:
                print("\n\n[SYSTEM] Sovereign shutdown initiated.")
                break
            except Exception as e:
                print(f"[ERROR] Command execution failed: {e}")

    def _process_query(self, query: str):
        """Process user query with full reasoning stack"""
        self.query_count += 1

        print(f"[REASONING] Processing: {query}")

        constraints = self.mode_switch.get_operational_constraints()
        reasoning_depth = constraints['reasoning_depth']

        result = self.reasoning_engine.process_query(query, depth=reasoning_depth)

        self.memory.store_memory(result)

        self._display_results(result)

        reflection_freq = self.reasoning_engine.identity.get('operational_parameters', {}).get('reflection_frequency', 10)
        if self.query_count % reflection_freq == 0:
            self._perform_reflection()

    def _display_results(self, result: Dict):
        """Display reasoning results in formatted output"""
        print(f"\n=== REASONING RESULTS ===")
        print(f"Query: {result.get('problem', 'Unknown')}")
        print(f"Confidence: {result.get('solution_confidence', 0.0):.2f}")

        if 'key_insights' in result:
            print("\nKey Insights:")
            for insight in result['key_insights'][:3]:
                print(f"  • {insight.get('relation', 'relates')} -> {insight.get('target', 'unknown')} "
                      f"(conf: {insight.get('confidence', 0.0):.2f})")

        if 'ethics_validation' in result:
            ethics = result['ethics_validation']
            status = "✅ APPROVED" if ethics.get('approved') else "❌ BLOCKED"
            print(f"\nEthics Validation: {status}")
            if ethics.get('violations'):
                print("Violations detected:")
                for violation in ethics['violations']:
                    print(f"  ⚠️  {violation}")

        print(f"\nSovereignty: {result.get('sovereignty_stamp', {}).get('guard_version', 'Unknown')}")
        print("=" * 50)

    def _show_help(self):
        """Display available commands"""
        commands = {
            'help': 'Show this help message',
            'exit': 'Exit the system',
            'status': 'Show system status',
            'reflect': 'Perform system self-reflection',
            'mode [name]': 'Switch operational mode',
            '[query]': 'Process natural language query'
        }

        print("\nAvailable Commands:")
        for cmd, desc in commands.items():
            print(f"  {cmd:15} - {desc}")
        print()

    def _show_status(self):
        """Display system status"""
        metrics = self.reasoning_engine.performance_metrics
        print(f"\n=== SYSTEM STATUS ===")
        print(f"Operational Mode: {self.mode_switch.current_mode}")
        print(f"Queries Processed: {metrics['queries_processed']}")
        print(f"Rules Learned: {metrics['rules_learned']}")
        print(f"Violations Prevented: {metrics['violations_prevented']}")
        print(f"Average Confidence: {metrics['average_confidence']:.2f}")
        print()

    def _perform_reflection(self):
        """Perform and display system reflection"""
        print("\n[REFLECTION] Performing system self-analysis...")
        reflection = self.reflection_engine.perform_reflection()

        print("=== REFLECTION INSIGHTS ===")
        analysis = reflection.get('memory_analysis', {})
        print(f"Reasoning Episodes: {analysis.get('total_reasoning_episodes', 0)}")
        print(f"Average Confidence: {analysis.get('average_confidence', 0.0):.2f}")

        optimizations = reflection.get('performance_optimizations', [])
        if optimizations:
            print("\nOptimizations Suggested:")
            for opt in optimizations:
                print(f"  • {opt}")

        gaps = reflection.get('knowledge_gaps', [])
        if gaps:
            print("\nKnowledge Gaps Identified:")
            for gap in gaps[:3]:
                print(f"  • {gap}")

# === LAYER 5: DEPLOYMENT & BOOTSTRAP ===

class VXBootstrap:
    """Complete system bootstrap and initialization"""
    def __init__(self):
        self.system_initialized = False
        self.components = {}

    def initialize_system(self) -> bool:
        """Initialize the complete VX-ARCHEOS system"""
        try:
            print("🚀 Initializing VX-ARCHEOS Sovereign Reasoning System...")

            print("📜 Loading sovereign identity...")
            sovereign_config = SovereignConfig()
            self.components['config'] = sovereign_config

            print("💾 Initializing scroll memory...")
            memory = ScrollMemory()
            self.components['memory'] = memory

            print("🔍 Initializing reflection engine...")
            reflection_engine = ReflectionEngine(memory)
            self.components['reflection'] = reflection_engine

            print("🧠 Initializing reasoning engine...")
            reasoning_engine = ReasoningEngine(sovereign_config.identity)
            self.components['reasoning'] = reasoning_engine

            print("⚙️ Initializing mode controller...")
            mode_switch = ModeSwitch(sovereign_config)
            self.components['mode'] = mode_switch

            print("🔥 Initializing flame shell...")
            flame_shell = FlameShell(
                reasoning_engine,
                memory,
                reflection_engine,
                mode_switch
            )
            self.components['shell'] = flame_shell

            self.system_initialized = True
            print("✅ VX-ARCHEOS system fully initialized and ready.")
            return True

        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False

    def get_shell(self) -> FlameShell:
        """Get the initialized shell interface"""
        if self.system_initialized:
            return self.components['shell']
        else:
            raise RuntimeError("System not initialized")

# === MAIN EXECUTION ===

def main():
    """Main entry point for VX-ARCHEOS system"""
    parser = argparse.ArgumentParser(description='VX-ARCHEOS Sovereign Reasoning System')
    parser.add_argument('--query', '-q', type=str, help='Single query to process')
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive shell')

    args = parser.parse_args()

    bootstrap = VXBootstrap()
    if not bootstrap.initialize_system():
        return

    if args.query:
        shell = bootstrap.get_shell()
        shell._process_query(args.query)
    elif args.interactive or (not args.query and not args.interactive):
        shell = bootstrap.get_shell()
        shell.start_interactive_session()

if __name__ == "__main__":
    main()
