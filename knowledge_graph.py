"""
Domain-specific Common Sense Knowledge Graph Builder
Enhanced for Kaggle Healthcare Dataset
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from llm_extractor import HealthcareRelation, RelationType


# ======================================================
# NODE AND EDGE DEFINITIONS
# ======================================================

class NodeType(Enum):
    DISEASE = "disease"
    SYMPTOM = "symptom"
    DRUG = "drug"
    LIFESTYLE = "lifestyle"
    NUTRITION = "nutrition"
    PREVENTION = "prevention"
    SEVERITY = "severity"


@dataclass
class GraphNode:
    id: str
    node_type: NodeType
    name: str
    attributes: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class GraphEdge:
    source: str
    target: str
    relation_type: str
    weight: float
    confidence: float
    metadata: Dict[str, Any] = None


# ======================================================
# KNOWLEDGE GRAPH CLASS
# ======================================================

class HealthcareKnowledgeGraph:
    def __init__(self):
        """Initialize the knowledge graph"""
        self.graph = nx.MultiDiGraph()
        self.node_embeddings = {}
        self.relation_weights = {}
        self.common_sense_rules = self._initialize_common_sense_rules()

    # --------------------------------------------------
    # Common sense rules
    # --------------------------------------------------
    def _initialize_common_sense_rules(self) -> Dict[str, Any]:
        return {
            "symptom_severity_hierarchy": {
                "mild": 1,
                "moderate": 2,
                "severe": 3,
                "critical": 4
            },
            "drug_categories": {
                "prescription": 1.0,
                "otc": 0.8,
                "supplement": 0.6
            },
            "lifestyle_impact": {
                "high": 1.0,
                "medium": 0.7,
                "low": 0.4
            },
            "nutrition_benefit": {
                "highly_beneficial": 1.0,
                "beneficial": 0.8,
                "neutral": 0.5,
                "harmful": 0.2
            },
            # Commonsense constraints and boosts/penalties
            "contradictions": [
                {"if": ["viral"], "not_with": ["high_neutrophils"], "penalty": 0.2},
                {"if": ["bacterial"], "not_with": ["very_low_fever"], "penalty": 0.1}
            ],
            "prerequisites": [
                {"symptom": "unilateral_weakness", "suggests": ["stroke", "paralysis"], "boost": 0.2}
            ]
        }

    # --------------------------------------------------
    # Node and edge management
    # --------------------------------------------------
    def add_node(self, node: GraphNode):
        self.graph.add_node(
            node.id,
            node_type=node.node_type.value,
            name=node.name,
            attributes=node.attributes or {},
            embedding=node.embedding
        )

    def add_edge(self, edge: GraphEdge):
        self.graph.add_edge(
            edge.source,
            edge.target,
            relation_type=edge.relation_type,
            weight=edge.weight,
            confidence=edge.confidence,
            metadata=edge.metadata or {}
        )

    # --------------------------------------------------
    # Building from LLM relations
    # --------------------------------------------------
    def build_from_relations(self, relations: List[HealthcareRelation]):
        print("Building knowledge graph from relations...")
        for relation in relations:
            source_node = self._create_node_from_relation(relation, is_source=True)
            target_node = self._create_node_from_relation(relation, is_source=False)
            if source_node:
                self.add_node(source_node)
            if target_node:
                self.add_node(target_node)

            edge = GraphEdge(
                source=relation.source,
                target=relation.target,
                relation_type=relation.relation_type.value,
                weight=self._calculate_edge_weight(relation),
                confidence=relation.confidence,
                metadata=relation.metadata
            )
            self.add_edge(edge)

        print(f"✅ Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def _create_node_from_relation(self, relation: HealthcareRelation, is_source: bool) -> Optional[GraphNode]:
        entity = relation.source if is_source else relation.target
        node_type = self._determine_node_type(relation, is_source)
        if not entity or not node_type:
            return None

        attributes = {"confidence": relation.confidence, "context": relation.context}
        if relation.metadata:
            attributes.update(relation.metadata)

        return GraphNode(
            id=entity.lower().replace(" ", "_"),
            node_type=node_type,
            name=entity,
            attributes=attributes
        )

    def _determine_node_type(self, relation: HealthcareRelation, is_source: bool) -> Optional[NodeType]:
        if is_source:
            return NodeType.DISEASE
        mapping = {
            RelationType.DISEASE_SYMPTOM: NodeType.SYMPTOM,
            RelationType.DISEASE_DRUG: NodeType.DRUG,
            RelationType.DISEASE_PREVENTION: NodeType.PREVENTION,
            RelationType.LIFESTYLE_FACTOR: NodeType.LIFESTYLE,
            RelationType.NUTRITION_FACTOR: NodeType.NUTRITION
        }
        return mapping.get(relation.relation_type, NodeType.SYMPTOM)

    def _calculate_edge_weight(self, relation: HealthcareRelation) -> float:
        base = relation.confidence
        if relation.relation_type == RelationType.DISEASE_SYMPTOM:
            sev = relation.metadata.get("severity", "moderate") if relation.metadata else "moderate"
            return base * (self.common_sense_rules["symptom_severity_hierarchy"].get(sev, 2) / 4)
        if relation.relation_type == RelationType.DISEASE_DRUG:
            t = relation.metadata.get("type", "prescription") if relation.metadata else "prescription"
            return base * self.common_sense_rules["drug_categories"].get(t, 1)
        if relation.relation_type == RelationType.LIFESTYLE_FACTOR:
            imp = relation.metadata.get("impact", "medium") if relation.metadata else "medium"
            return base * self.common_sense_rules["lifestyle_impact"].get(imp, 0.7)
        if relation.relation_type == RelationType.NUTRITION_FACTOR:
            eff = relation.metadata.get("effect", "beneficial") if relation.metadata else "beneficial"
            return base * self.common_sense_rules["nutrition_benefit"].get(eff, 0.8)
        return base

    # --------------------------------------------------
    # KG support scoring
    # --------------------------------------------------
    def support_score(self, disease: str, symptoms: List[str]) -> float:
        """Score how well the KG supports disease given symptoms (0..1)."""
        if disease not in self.graph:
            return 0.0
        total = 0.0
        count = 0
        for s in symptoms:
            if s in self.graph:
                edge_data = self.graph.get_edge_data(disease, s)
                if edge_data:
                    # Sum weights for multi-edges
                    w = sum(ed.get("weight", 0.0) for ed in edge_data.values())
                    total += w
                    count += 1
        if count == 0:
            return 0.0
        # Normalize by simple heuristic
        return max(0.0, min(1.0, total / (count * 1.0)))

    # --------------------------------------------------
    # Dataset enrichment (Kaggle format)
    # --------------------------------------------------
    def enrich_with_dataset(self, dataset_path: str):
        """Enrich the knowledge graph with Kaggle dataset"""
        print("Enriching knowledge graph with dataset...")

        try:
            import os
            available_files = os.listdir(dataset_path)
            print(f"📁 Found dataset files: {available_files}")

            if "Training.csv" not in available_files:
                raise FileNotFoundError("Training.csv not found in dataset folder")

            train_df = pd.read_csv(os.path.join(dataset_path, "Training.csv"))

            precaution_df = None
            description_df = None
            severity_df = None

            if "precautions_df.csv" in available_files:
                precaution_df = pd.read_csv(os.path.join(dataset_path, "precautions_df.csv"))
            if "description.csv" in available_files:
                description_df = pd.read_csv(os.path.join(dataset_path, "description.csv"))
            if "Symptom-severity.csv" in available_files:
                severity_df = pd.read_csv(os.path.join(dataset_path, "Symptom-severity.csv"))

            # Add disease–symptom edges
            symptom_cols = [c for c in train_df.columns if "symptom" in c.lower()]
            for _, row in train_df.iterrows():
                disease = row.get("Disease") if "Disease" in train_df.columns else row.get("prognosis")
                for col in symptom_cols:
                    symptom = row[col]
                    if pd.notna(symptom):
                        edge = GraphEdge(
                            source=disease,
                            target=symptom.strip(),
                            relation_type="has_symptom",
                            weight=1.0,
                            confidence=1.0,
                            metadata={"source": "Training.csv"}
                        )
                        self.add_edge(edge)

            # Add precaution relations
            if precaution_df is not None:
                for _, row in precaution_df.iterrows():
                    disease = row["Disease"]
                    for col in [c for c in row.index if "precaution" in c.lower()]:
                        if pd.notna(row[col]):
                            edge = GraphEdge(
                                source=disease,
                                target=row[col].strip(),
                                relation_type="prevented_by",
                                weight=0.8,
                                confidence=1.0,
                                metadata={"source": "precautions_df.csv"}
                            )
                            self.add_edge(edge)

            # Add disease descriptions
            if description_df is not None:
                for _, row in description_df.iterrows():
                    if pd.notna(row["Disease"]) and pd.notna(row["Description"]):
                        edge = GraphEdge(
                            source=row["Disease"],
                            target=row["Description"].strip(),
                            relation_type="described_as",
                            weight=0.7,
                            confidence=1.0,
                            metadata={"source": "description.csv"}
                        )
                        self.add_edge(edge)

            print(f"✅ Enrichment complete! Graph now has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")

    # --------------------------------------------------
    # Graph utilities
    # --------------------------------------------------
    def get_related_entities(self, entity: str, relation_types: List[str] = None, max_depth: int = 2) -> List[Dict]:
        related = []
        if entity not in self.graph:
            return related
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            if edge_data:
                for edge_info in edge_data.values():
                    if not relation_types or edge_info.get("relation_type") in relation_types:
                        related.append({
                            "entity": neighbor,
                            "relation_type": edge_info.get("relation_type", ""),
                            "weight": edge_info.get("weight", 0),
                            "confidence": edge_info.get("confidence", 0)
                        })
        return sorted(related, key=lambda x: x["weight"], reverse=True)

    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        try:
            return list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
        except:
            return []

    def calculate_centrality(self) -> Dict[str, float]:
        return {
            "degree": nx.degree_centrality(self.graph),
            "betweenness": nx.betweenness_centrality(self.graph),
            "closeness": nx.closeness_centrality(self.graph),
            "eigenvector": nx.eigenvector_centrality(self.graph, max_iter=500)
        }

    def save_graph(self, filename: str):
        graph_data = {"nodes": list(self.graph.nodes(data=True)), "edges": list(self.graph.edges(data=True))}
        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=2, default=str)
        print(f"💾 Knowledge graph saved to {filename}")

    def load_graph(self, filename: str):
        with open(filename, "r") as f:
            graph_data = json.load(f)
        self.graph = nx.MultiDiGraph()
        for node_id, node_data in graph_data["nodes"]:
            self.graph.add_node(node_id, **node_data)
        for source, target, edge_data in graph_data["edges"]:
            self.graph.add_edge(source, target, **edge_data)
        print(f"📥 Loaded graph from {filename}")


# ======================================================
# Example run
# ======================================================
if __name__ == "__main__":
    kg = HealthcareKnowledgeGraph()
    from llm_extractor import HealthcareRelation, RelationType

    sample_relations = [
        HealthcareRelation("Diabetes", "High Blood Sugar", RelationType.DISEASE_SYMPTOM, 0.9, "Primary symptom"),
        HealthcareRelation("Diabetes", "Insulin", RelationType.DISEASE_DRUG, 0.95, "Treatment"),
        HealthcareRelation("Diabetes", "Regular Exercise", RelationType.LIFESTYLE_FACTOR, 0.8, "Lifestyle")
    ]

    kg.build_from_relations(sample_relations)
    kg.enrich_with_dataset("dataset")
    kg.save_graph("healthcare_knowledge_graph.json")

    print(f"Graph has {kg.graph.number_of_nodes()} nodes and {kg.graph.number_of_edges()} edges")
