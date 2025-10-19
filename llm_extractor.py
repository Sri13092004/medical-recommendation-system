"""
Healthcare LLM Extractor using OpenAI API
-----------------------------------------
This module extracts healthcare knowledge (relations between disease, symptom,
drug, lifestyle, prevention, etc.) using OpenAI's chat completions API.
"""

import os
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# ---------------------------
#  ENUM DEFINITIONS
# ---------------------------

class RelationType(Enum):
    DISEASE_SYMPTOM = "disease_symptom"
    DISEASE_DRUG = "disease_drug"
    DISEASE_PREVENTION = "disease_prevention"
    LIFESTYLE_FACTOR = "lifestyle_factor"
    NUTRITION_FACTOR = "nutrition_factor"


# ---------------------------
#  RELATION DATACLASS
# ---------------------------

@dataclass
class HealthcareRelation:
    source: str                    # e.g. "Diabetes"
    target: str                    # e.g. "Insulin"
    relation_type: RelationType    # e.g. RelationType.DISEASE_DRUG
    confidence: float = 1.0
    context: str = ""
    metadata: Dict[str, Any] = None


# ---------------------------
#  EXTRACTOR CLASS
# ---------------------------

class LLMHealthcareExtractor:
    """
    Extracts domain-specific healthcare relations using OpenAI API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        # Prefer explicit api_key, fallback to env var
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("❌ OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI()
        self.model = model

    # ---------------------------
    #  Low-level LLM helpers
    # ---------------------------
    def _call_llm(self, prompt: str, system: str = "You are a medical knowledge extractor. Return only the answer requested.") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def query(self, prompt: str) -> str:
        """General-purpose query helper used by other modules."""
        return self._call_llm(prompt)

    def safe_json_parse(self, text: str) -> Any:
        """Best-effort JSON parsing that tolerates fenced code blocks and extra text."""
        if not text:
            return {}
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1] if "```" in cleaned else cleaned
            cleaned = cleaned.replace("json\n", "").replace("json\r\n", "").strip()
        # Try to locate the first JSON object/array
        start_obj = cleaned.find("{")
        start_arr = cleaned.find("[")
        start = min([i for i in [start_obj, start_arr] if i != -1], default=-1)
        if start > 0:
            cleaned = cleaned[start:]
        # Try object
        try:
            return json.loads(cleaned)
        except Exception:
            # Try to truncate to last matching brace/bracket
            end_obj = cleaned.rfind("}")
            end_arr = cleaned.rfind("]")
            end = max(end_obj, end_arr)
            if end != -1:
                try:
                    return json.loads(cleaned[: end + 1])
                except Exception:
                    return {}
            return {}

    def _safe_confidence(self, value, default=0.8):
        """Safely convert confidence value to float, handling string values like 'high', 'moderate'"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value_lower = value.lower().strip()
            if value_lower in ['high', 'severe', 'critical']:
                return 0.9
            elif value_lower in ['moderate', 'medium']:
                return 0.7
            elif value_lower in ['low', 'mild']:
                return 0.5
            else:
                try:
                    return float(value)
                except ValueError:
                    return default
        return default

    # ---------------------------
    #  MAIN EXTRACT FUNCTION
    # ---------------------------

    def extract_relations_batch(self, diseases: List[str]) -> List[HealthcareRelation]:
        """
        Extract structured relations for multiple diseases in one API call.
        """
        print("\n🧠 Extracting relations for multiple diseases in one batch...")

        prompt = f"""
        You are a medical knowledge extractor. 
        For each of the following diseases, provide structured JSON with:
        - symptoms (list of dicts with 'name' and 'confidence')
        - drugs (list of dicts with 'name' and 'confidence')
        - prevention (list of dicts with 'strategy' and 'confidence')
        - lifestyle (list of dicts with 'factor' and 'confidence')
        - nutrition (list of dicts with 'item', 'effect', and 'confidence')

        Diseases:
        {', '.join(diseases)}

        Format response as *valid JSON array*, e.g.:

        [
          {{
            "disease": "Diabetes",
            "symptoms": [{{"name": "frequent urination", "confidence": 0.9}}],
            "drugs": [{{"name": "insulin", "confidence": 0.95}}],
            "prevention": [{{"strategy": "regular exercise", "confidence": 0.8}}],
            "lifestyle": [{{"factor": "low sugar diet", "confidence": 0.85}}],
            "nutrition": [{{"item": "whole grains", "effect": "beneficial", "confidence": 0.9}}]
          }}
        ]
        """

        try:
            raw_text = self._call_llm(prompt)
            data = self.safe_json_parse(raw_text)
            if not isinstance(data, list):
                return []

            relations = []
            for entry in data:
                disease = entry.get("disease")
                if not disease:
                    continue

                # SYMPTOMS
                for s in entry.get("symptoms", []):
                    relations.append(HealthcareRelation(
                        source=disease,
                        target=s["name"],
                        relation_type=RelationType.DISEASE_SYMPTOM,
                        confidence=self._safe_confidence(s.get("confidence", 0.8)),
                        context="Extracted from OpenAI batch",
                        metadata={}
                    ))

                # DRUGS
                for d in entry.get("drugs", []):
                    relations.append(HealthcareRelation(
                        source=disease,
                        target=d["name"],
                        relation_type=RelationType.DISEASE_DRUG,
                        confidence=self._safe_confidence(d.get("confidence", 0.8)),
                        context="Extracted from OpenAI batch",
                        metadata={}
                    ))

                # PREVENTION
                for p in entry.get("prevention", []):
                    relations.append(HealthcareRelation(
                        source=disease,
                        target=p["strategy"],
                        relation_type=RelationType.DISEASE_PREVENTION,
                        confidence=self._safe_confidence(p.get("confidence", 0.8)),
                        context="Extracted from OpenAI batch",
                        metadata={}
                    ))

                # LIFESTYLE
                for lf in entry.get("lifestyle", []):
                    relations.append(HealthcareRelation(
                        source=disease,
                        target=lf["factor"],
                        relation_type=RelationType.LIFESTYLE_FACTOR,
                        confidence=self._safe_confidence(lf.get("confidence", 0.8)),
                        context="Extracted from OpenAI batch",
                        metadata={}
                    ))

                # NUTRITION
                for nf in entry.get("nutrition", []):
                    relations.append(HealthcareRelation(
                        source=disease,
                        target=nf["item"],
                        relation_type=RelationType.NUTRITION_FACTOR,
                        confidence=self._safe_confidence(nf.get("confidence", 0.8)),
                        context="Extracted from OpenAI batch",
                        metadata={"effect": nf.get("effect", "beneficial")}
                    ))

            print(f"✅ Extracted {len(relations)} total relations (batch mode).")
            return relations

        except Exception as e:
            print(f"❌ Error extracting relations: {e}")
            return []

    # ---------------------------
    #  WRAPPER FOR DATASET
    # ---------------------------

    def extract_from_dataset(self, disease_list: List[str]) -> List[HealthcareRelation]:
        """
        Extract relations for diseases found in dataset.
        """
        print("\n📘 Extracting healthcare relations using OpenAI for dataset...")
        all_relations = []
        batch_size = 5

        for i in range(0, len(disease_list), batch_size):
            batch = disease_list[i:i+batch_size]
            batch_relations = self.extract_relations_batch(batch)
            all_relations.extend(batch_relations)

        print(f"✅ Total relations extracted from dataset: {len(all_relations)}")
        return all_relations

    # ---------------------------
    #  Extract specific relation types for a disease
    # ---------------------------
    def extract_relations(self, disease: str, relation_types: List[RelationType]) -> List[HealthcareRelation]:
        requested = []
        if RelationType.LIFESTYLE_FACTOR in relation_types:
            requested.append("lifestyle (list of {factor, confidence})")
        if RelationType.NUTRITION_FACTOR in relation_types:
            requested.append("nutrition (list of {item, effect, confidence})")
        if RelationType.DISEASE_SYMPTOM in relation_types:
            requested.append("symptoms (list of {name, confidence})")
        if RelationType.DISEASE_DRUG in relation_types:
            requested.append("drugs (list of {name, confidence})")
        if RelationType.DISEASE_PREVENTION in relation_types:
            requested.append("prevention (list of {strategy, confidence})")

        sections = "\n- ".join(requested) if requested else "symptoms, drugs, prevention, lifestyle, nutrition"
        prompt = f"""
        You are a medical knowledge extractor.
        For the disease: {disease}
        Provide structured JSON with the following sections:
        - {sections}

        Return a single JSON object with keys present only for available sections.
        """

        try:
            raw_text = self._call_llm(prompt)
            data = self.safe_json_parse(raw_text) or {}
            relations: List[HealthcareRelation] = []

            for s in data.get("symptoms", []) if isinstance(data, dict) else []:
                relations.append(HealthcareRelation(
                    source=disease,
                    target=s.get("name", ""),
                    relation_type=RelationType.DISEASE_SYMPTOM,
                    confidence=self._safe_confidence(s.get("confidence", 0.8)),
                    context="Extracted from OpenAI",
                    metadata={}
                ))

            for d in data.get("drugs", []) if isinstance(data, dict) else []:
                relations.append(HealthcareRelation(
                    source=disease,
                    target=d.get("name", ""),
                    relation_type=RelationType.DISEASE_DRUG,
                    confidence=self._safe_confidence(d.get("confidence", 0.8)),
                    context="Extracted from OpenAI",
                    metadata={}
                ))

            for p in data.get("prevention", []) if isinstance(data, dict) else []:
                relations.append(HealthcareRelation(
                    source=disease,
                    target=p.get("strategy", ""),
                    relation_type=RelationType.DISEASE_PREVENTION,
                    confidence=self._safe_confidence(p.get("confidence", 0.8)),
                    context="Extracted from OpenAI",
                    metadata={}
                ))

            for lf in data.get("lifestyle", []) if isinstance(data, dict) else []:
                relations.append(HealthcareRelation(
                    source=disease,
                    target=lf.get("factor", ""),
                    relation_type=RelationType.LIFESTYLE_FACTOR,
                    confidence=self._safe_confidence(lf.get("confidence", 0.8)),
                    context="Extracted from OpenAI",
                    metadata={}
                ))

            for nf in data.get("nutrition", []) if isinstance(data, dict) else []:
                relations.append(HealthcareRelation(
                    source=disease,
                    target=nf.get("item", ""),
                    relation_type=RelationType.NUTRITION_FACTOR,
                    confidence=self._safe_confidence(nf.get("confidence", 0.8)),
                    context="Extracted from OpenAI",
                    metadata={"effect": nf.get("effect", "beneficial")}
                ))

            return [r for r in relations if r.target]
        except Exception as e:
            print(f"❌ Error extracting relations for {disease}: {e}")
            return []

    # ---------------------------
    #  Symptom normalization via LLM
    # ---------------------------
    def normalize_symptoms(self, free_text_symptoms: List[str]) -> List[str]:
        """
        Map lay terms to canonical dataset-like tokens (snake_case) using LLM.
        Returns a list like ["high_fever", "headache"].
        """
        if not free_text_symptoms:
            return []
        prompt = f"""
        Normalize these symptoms into short, canonical tokens (snake_case, lowercase),
        suitable for clinical datasets. Use synonyms where appropriate. Return JSON list of strings only.
        Symptoms: {free_text_symptoms}
        Example output: ["high_fever", "headache", "abdominal_pain"]
        """
        try:
            raw = self._call_llm(prompt, system="You convert symptoms to canonical tokens only.")
            data = self.safe_json_parse(raw)
            if isinstance(data, list):
                return [str(x).strip() for x in data if isinstance(x, str)]
        except Exception:
            pass
        # Fallback to simple normalization
        return [s.lower().strip().replace(" ", "_") for s in free_text_symptoms]

    # ---------------------------
    #  Post-prediction verification
    # ---------------------------
    def verify_prediction(self, symptoms: List[str], disease: str) -> Dict[str, Any]:
        """Ask LLM to verify plausibility. Returns {"plausible": bool, "confidence": float, "reason": str}."""
        prompt = f"""
        Given symptoms: {symptoms}
        Candidate disease: {disease}
        Is this diagnosis medically plausible? Reply as JSON object with keys:
        {{"plausible": true|false, "confidence": 0.0-1.0, "reason": "brief"}}
        If uncertain, set plausible=false and confidence<=0.5.
        """
        try:
            raw = self._call_llm(prompt, system="You critically verify medical plausibility and are conservative.")
            data = self.safe_json_parse(raw)
            if isinstance(data, dict):
                return {
                    "plausible": bool(data.get("plausible", False)),
                    "confidence": float(data.get("confidence", 0.5)) if isinstance(data.get("confidence", 0.5), (int, float, str)) else 0.5,
                    "reason": str(data.get("reason", ""))
                }
        except Exception:
            pass
        return {"plausible": False, "confidence": 0.0, "reason": "verification_failed"}
