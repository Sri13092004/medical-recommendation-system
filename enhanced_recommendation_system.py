"""
Enhanced Medical Recommendation System with Proposed Framework
This module implements the complete framework integration:
1. LLM-based healthcare relation extraction
2. Domain-specific common sense knowledge graph
3. GAT-based graph fusion
4. Preventive knowledge enrichment
"""

import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn

from llm_extractor import LLMHealthcareExtractor, HealthcareRelation, RelationType
from knowledge_graph import HealthcareKnowledgeGraph, GraphNode, GraphEdge, NodeType
from gat_fusion import KnowledgeGraphFusion

@dataclass
class EnhancedRecommendation:
    """Enhanced recommendation with framework components"""
    disease: str
    confidence: float
    symptoms: List[str]
    medications: List[Dict[str, Any]]
    lifestyle_recommendations: List[Dict[str, Any]]
    nutrition_recommendations: List[Dict[str, Any]]
    prevention_strategies: List[Dict[str, Any]]
    knowledge_graph_paths: List[List[str]]
    attention_weights: Optional[Dict[str, float]] = None

class EnhancedMedicalRecommendationSystem:
    """
    Enhanced medical recommendation system with the proposed framework
    """
    
    def __init__(self, 
                 model_path: str = "model/RandomForest.pkl",
                 dataset_path: str = "kaggle_dataset",
                 llm_api_key: str = None):
        """
        Initialize the enhanced system
        
        Args:
            model_path: Path to the trained Random Forest model
            dataset_path: Path to the dataset directory
            llm_api_key: OpenAI API key for LLM extraction
        """
        # Load existing model and data
        self.rf_model = pickle.load(open(model_path, 'rb'))
        self.load_datasets(dataset_path)
        
        # Initialize framework components
        self.llm_extractor = LLMHealthcareExtractor(api_key=llm_api_key)
        self.knowledge_graph = HealthcareKnowledgeGraph()
        self.fusion_system = KnowledgeGraphFusion()
        
        # Enhanced knowledge storage
        self.enhanced_knowledge = {}
        self.preventive_knowledge = {}
        
        # Initialize the enhanced system
        self._initialize_enhanced_system()
    
    def load_datasets(self, dataset_path: str):
        """Load existing datasets"""
        self.sym_des = pd.read_csv(f"{dataset_path}/symptoms_df.csv")
        self.precautions = pd.read_csv(f"{dataset_path}/precautions_df.csv")
        self.workout = pd.read_csv(f"{dataset_path}/workout_df.csv")
        self.description = pd.read_csv(f"{dataset_path}/description.csv")
        self.medications = pd.read_csv(f"{dataset_path}/medications.csv")
        self.diets = pd.read_csv(f"{dataset_path}/diets.csv")
        
        # Create symptoms and diseases mappings
        self.symptoms_list = self._create_symptoms_mapping()
        self.diseases_list = self._create_diseases_mapping()
        self.symptoms_list_processed = {
            symptom.replace('_', ' ').lower(): value 
            for symptom, value in self.symptoms_list.items()
        }
    
    def _create_symptoms_mapping(self) -> Dict[str, int]:
        """Create symptoms mapping from the dataset"""
        symptoms = {}
        for i, col in enumerate(self.sym_des.columns):
            if col != 'Disease':
                symptoms[col] = i
        return symptoms
    
    def _create_diseases_mapping(self) -> Dict[int, str]:
        """Create diseases mapping from the dataset"""
        diseases = {}
        unique_diseases = self.sym_des['Disease'].unique()
        for i, disease in enumerate(unique_diseases):
            diseases[i] = disease
        return diseases
    
    def _initialize_enhanced_system(self):
        """Initialize the enhanced system with framework components"""
        print("Initializing enhanced medical recommendation system...")
        
        # Extract relations using LLM
        self._extract_llm_relations()
        
        # Build knowledge graph
        self._build_knowledge_graph()
        
        # Enrich with preventive knowledge
        self._enrich_preventive_knowledge()
        
        print("Enhanced system initialization completed!")
    
    def _extract_llm_relations(self):
        """Extract healthcare relations using LLM"""
        print("Extracting healthcare relations using LLM...")
        
        # Get unique diseases from dataset
        diseases = self.sym_des['Disease'].unique().tolist()
        
        # Extract relations for each disease
        self.llm_relations = self.llm_extractor.extract_from_dataset(diseases[:10])  # Limit for demo
        
        print(f"Extracted {len(self.llm_relations)} relations from LLM")
    
    def _build_knowledge_graph(self):
        """Build the domain-specific knowledge graph"""
        print("Building knowledge graph...")
        
        # Build from LLM relations
        self.knowledge_graph.build_from_relations(self.llm_relations)
        
        # Enrich with existing dataset
        self.knowledge_graph.enrich_with_dataset("kaggle_dataset")
        
        print(f"Knowledge graph built with {self.knowledge_graph.graph.number_of_nodes()} nodes and {self.knowledge_graph.graph.number_of_edges()} edges")
    
    def _enrich_preventive_knowledge(self):
        """Enrich the system with preventive knowledge"""
        print("Enriching with preventive knowledge...")
        
        # Extract preventive knowledge for each disease
        diseases = self.sym_des['Disease'].unique().tolist()
        
        for disease in diseases[:5]:  # Limit for demo
            try:
                # Extract lifestyle and nutrition factors
                preventive_relations = self.llm_extractor.extract_relations(
                    disease, 
                    [RelationType.LIFESTYLE_FACTOR, RelationType.NUTRITION_FACTOR]
                )
                
                self.preventive_knowledge[disease] = preventive_relations
                
            except Exception as e:
                print(f"Error extracting preventive knowledge for {disease}: {e}")
                continue
        
        print(f"Enriched preventive knowledge for {len(self.preventive_knowledge)} diseases")
    
    def predict_disease_enhanced(self, symptoms: List[str]) -> EnhancedRecommendation:
        """
        Enhanced disease prediction with framework components
        
        Args:
            symptoms: List of patient symptoms
            
        Returns:
            Enhanced recommendation with framework insights
        """
        # Basic disease prediction
        predicted_disease = self._predict_disease_basic(symptoms)
        
        if predicted_disease == "Unknown Disease":
            return self._create_unknown_disease_recommendation(symptoms)
        
        # Get enhanced information using knowledge graph
        enhanced_info = self._get_enhanced_disease_info(predicted_disease, symptoms)
        
        # Create enhanced recommendation
        recommendation = EnhancedRecommendation(
            disease=predicted_disease,
            confidence=enhanced_info['confidence'],
            symptoms=enhanced_info['symptoms'],
            medications=enhanced_info['medications'],
            lifestyle_recommendations=enhanced_info['lifestyle'],
            nutrition_recommendations=enhanced_info['nutrition'],
            prevention_strategies=enhanced_info['prevention'],
            knowledge_graph_paths=enhanced_info['paths'],
            attention_weights=enhanced_info.get('attention_weights')
        )
        
        return recommendation
    
    def _predict_disease_basic(self, symptoms: List[str]) -> str:
        """Basic disease prediction using Random Forest"""
        try:
            # Create input vector
            expected_features = self.rf_model.n_features_in_
            i_vector = np.zeros(expected_features)
            
            for symptom in symptoms:
                if symptom in self.symptoms_list_processed:
                    i_vector[self.symptoms_list_processed[symptom]] = 1
            
            # Convert to DataFrame to avoid sklearn warning
            import pandas as pd
            # Use training feature names if available
            feature_names = [c for c in self.sym_des.columns if c != 'Disease']
            if len(feature_names) == expected_features:
                df_input = pd.DataFrame([i_vector], columns=feature_names)
            else:
                df_input = pd.DataFrame([i_vector])
            
            # Predict disease
            predicted_label = self.rf_model.predict(df_input)[0]
            return self.diseases_list.get(predicted_label, "Unknown Disease")
            
        except Exception as e:
            print(f"Error in basic prediction: {e}")
            return "Unknown Disease"
    
    def _get_enhanced_disease_info(self, disease: str, symptoms: List[str]) -> Dict[str, Any]:
        """Get enhanced disease information using knowledge graph"""
        enhanced_info = {
            'confidence': 0.8,
            'symptoms': [],
            'medications': [],
            'lifestyle': [],
            'nutrition': [],
            'prevention': [],
            'paths': []
        }
        
        try:
            # Get related entities from knowledge graph
            related_entities = self.knowledge_graph.get_related_entities(disease)
            
            # Categorize related entities
            for entity_info in related_entities:
                entity = entity_info['entity']
                relation_type = entity_info['relation_type']
                confidence = entity_info['confidence']
                
                if relation_type == 'has_symptom' or relation_type == 'disease_symptom':
                    enhanced_info['symptoms'].append({
                        'name': entity,
                        'confidence': confidence,
                        'source': 'knowledge_graph'
                    })
                
                elif relation_type == 'treated_by' or relation_type == 'disease_drug':
                    enhanced_info['medications'].append({
                        'name': entity,
                        'confidence': confidence,
                        'source': 'knowledge_graph'
                    })
                
                elif relation_type == 'lifestyle_factor':
                    enhanced_info['lifestyle'].append({
                        'factor': entity,
                        'confidence': confidence,
                        'source': 'knowledge_graph'
                    })
                
                elif relation_type == 'nutrition_factor':
                    enhanced_info['nutrition'].append({
                        'nutrient': entity,
                        'confidence': confidence,
                        'source': 'knowledge_graph'
                    })
                
                elif relation_type == 'prevented_by':
                    enhanced_info['prevention'].append({
                        'strategy': entity,
                        'confidence': confidence,
                        'source': 'knowledge_graph'
                    })
            
            # Find paths between symptoms and disease
            for symptom in symptoms:
                paths = self.knowledge_graph.find_paths(symptom, disease, max_length=3)
                enhanced_info['paths'].extend(paths)
            
            # Add traditional dataset information
            self._add_traditional_info(disease, enhanced_info)
            
            # Add preventive knowledge
            if disease in self.preventive_knowledge:
                preventive_relations = self.preventive_knowledge[disease]
                for relation in preventive_relations:
                    if relation.relation_type == RelationType.LIFESTYLE_FACTOR:
                        enhanced_info['lifestyle'].append({
                            'factor': relation.target,
                            'confidence': relation.confidence,
                            'source': 'llm_extraction'
                        })
                    elif relation.relation_type == RelationType.NUTRITION_FACTOR:
                        enhanced_info['nutrition'].append({
                            'nutrient': relation.target,
                            'confidence': relation.confidence,
                            'source': 'llm_extraction'
                        })
            
        except Exception as e:
            print(f"Error getting enhanced info: {e}")
        
        return enhanced_info
    
    def _add_traditional_info(self, disease: str, enhanced_info: Dict[str, Any]):
        """Add traditional dataset information"""
        try:
            # Add traditional medications
            disease_medications = self.medications.loc[
                self.medications['Disease'] == disease, 'Medication'
            ].values
            
            if len(disease_medications) > 0:
                medications = disease_medications[0].split(",")
                for med in medications:
                    enhanced_info['medications'].append({
                        'name': med.strip(),
                        'confidence': 1.0,
                        'source': 'dataset'
                    })
            
            # Add traditional diet recommendations
            disease_diet = self.diets.loc[
                self.diets['Disease'] == disease, 'Diet'
            ].values
            
            if len(disease_diet) > 0:
                diets = disease_diet[0].split(",")
                for diet in diets:
                    enhanced_info['nutrition'].append({
                        'nutrient': diet.strip(),
                        'confidence': 1.0,
                        'source': 'dataset'
                    })
            
            # Add traditional workout recommendations
            disease_workout = self.workout.loc[
                self.workout['disease'] == disease, 'workout'
            ].values
            
            if len(disease_workout) > 0:
                workouts = disease_workout[0].split(",")
                for workout in workouts:
                    enhanced_info['lifestyle'].append({
                        'factor': workout.strip(),
                        'confidence': 1.0,
                        'source': 'dataset'
                    })
            
        except Exception as e:
            print(f"Error adding traditional info: {e}")
    
    def _create_unknown_disease_recommendation(self, symptoms: List[str]) -> EnhancedRecommendation:
        """Create recommendation for unknown disease"""
        return EnhancedRecommendation(
            disease="Unknown Disease",
            confidence=0.0,
            symptoms=symptoms,
            medications=[],
            lifestyle_recommendations=[],
            nutrition_recommendations=[],
            prevention_strategies=[],
            knowledge_graph_paths=[]
        )
    
    def get_knowledge_graph_insights(self, entity: str) -> Dict[str, Any]:
        """Get insights from the knowledge graph for a specific entity"""
        insights = {
            'entity': entity,
            'related_entities': [],
            'centrality_measures': {},
            'graph_statistics': {}
        }
        
        try:
            # Get related entities
            related = self.knowledge_graph.get_related_entities(entity)
            insights['related_entities'] = related
            
            # Calculate centrality measures
            centrality = self.knowledge_graph.calculate_centrality()
            if entity in centrality['degree']:
                insights['centrality_measures'] = {
                    'degree': centrality['degree'][entity],
                    'betweenness': centrality['betweenness'][entity],
                    'closeness': centrality['closeness'][entity],
                    'eigenvector': centrality['eigenvector'][entity]
                }
            
            # Graph statistics
            insights['graph_statistics'] = {
                'total_nodes': self.knowledge_graph.graph.number_of_nodes(),
                'total_edges': self.knowledge_graph.graph.number_of_edges(),
                'entity_connections': len(related)
            }
            
        except Exception as e:
            print(f"Error getting graph insights: {e}")
        
        return insights
    
    def save_enhanced_system(self, filepath: str):
        """Save the enhanced system state"""
        system_state = {
            'knowledge_graph': {
                'nodes': list(self.knowledge_graph.graph.nodes(data=True)),
                'edges': list(self.knowledge_graph.graph.edges(data=True))
            },
            'preventive_knowledge': self.preventive_knowledge,
            'enhanced_knowledge': self.enhanced_knowledge
        }
        
        with open(filepath, 'w') as f:
            json.dump(system_state, f, indent=2, default=str)
        
        print(f"Enhanced system saved to {filepath}")
    
    def load_enhanced_system(self, filepath: str):
        """Load the enhanced system state"""
        with open(filepath, 'r') as f:
            system_state = json.load(f)
        
        # Rebuild knowledge graph
        self.knowledge_graph = HealthcareKnowledgeGraph()
        for node_id, node_data in system_state['knowledge_graph']['nodes']:
            self.knowledge_graph.add_node(GraphNode(
                id=node_id,
                node_type=NodeType(node_data.get('node_type', 'disease')),
                name=node_data.get('name', node_id),
                attributes=node_data
            ))
        
        for source, target, edge_data in system_state['knowledge_graph']['edges']:
            self.knowledge_graph.add_edge(GraphEdge(
                source=source,
                target=target,
                relation_type=edge_data.get('relation_type', ''),
                weight=edge_data.get('weight', 1.0),
                confidence=edge_data.get('confidence', 1.0),
                metadata=edge_data.get('metadata', {})
            ))
        
        self.preventive_knowledge = system_state['preventive_knowledge']
        self.enhanced_knowledge = system_state['enhanced_knowledge']
        
        print(f"Enhanced system loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Initialize enhanced system
    enhanced_system = EnhancedMedicalRecommendationSystem()
    
    # Test with sample symptoms
    test_symptoms = ["high fever", "headache", "fatigue"]
    
    # Get enhanced recommendation
    recommendation = enhanced_system.predict_disease_enhanced(test_symptoms)
    
    print(f"Predicted Disease: {recommendation.disease}")
    print(f"Confidence: {recommendation.confidence}")
    print(f"Medications: {[med['name'] for med in recommendation.medications]}")
    print(f"Lifestyle: {[life['factor'] for life in recommendation.lifestyle_recommendations]}")
    print(f"Nutrition: {[nut['nutrient'] for nut in recommendation.nutrition_recommendations]}")
    
    # Get knowledge graph insights
    insights = enhanced_system.get_knowledge_graph_insights(recommendation.disease)
    print(f"Graph insights: {insights}")
