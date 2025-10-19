import os
import pandas as pd
import networkx as nx
from flask import Flask, render_template, request
from difflib import get_close_matches
from fuzzywuzzy import process
from llm_extractor import LLMHealthcareExtractor  # Uses OpenAI API

# ------------------------------
# Initialize Flask
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Global variables
# ------------------------------
G = nx.Graph()
df_combined = None
symptoms_list_processed = {}
extractor = None
disease_col = "Disease"

# ------------------------------
# System Initialization
# ------------------------------
def initialize_enhanced_system():
    global extractor, G, df_combined, symptoms_list_processed

    print("Initializing enhanced medical recommendation system...")

    dataset_path = os.path.join(os.getcwd(), "kaggle_dataset")
    print(f"📂 Loading dataset from: {dataset_path}")

    all_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    dfs = []

    for file in all_files:
        df = pd.read_csv(os.path.join(dataset_path, file))
        if "Disease" in df.columns:
            df.rename(columns={"Disease": "prognosis"}, inplace=True)
        if "prognosis" not in df.columns:
            print(f"⚠️ Skipping {file} (no Disease/prognosis column)")
            continue
        dfs.append(df)

    df_combined = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"✅ Loaded {len(df_combined)} rows from dataset.")

    # ✅ Initialize extractor
    extractor = LLMHealthcareExtractor()

    # Build a basic symptom list
    if "Symptom_1" in df_combined.columns:
        all_symptoms = []
        for col in df_combined.columns:
            if col.lower().startswith("symptom"):
                all_symptoms.extend(df_combined[col].dropna().unique().tolist())
        symptoms_list_processed = {s.lower().strip(): s for s in set(all_symptoms)}

    # ✅ Build a simple Knowledge Graph
    print("🧠 Building Knowledge Graph...")
    for _, row in df_combined.iterrows():
        disease = row["prognosis"]
        for col in df_combined.columns:
            if col.lower().startswith("symptom") and isinstance(row[col], str):
                G.add_edge(disease.strip(), row[col].strip())

    print(f"✅ Knowledge Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges")
    print("✅ System ready with LLM + KG hybrid reasoning.")
    return extractor, G, df_combined, symptoms_list_processed


# ------------------------------
# Hybrid Reasoning Function
# ------------------------------
def hybrid_reasoning(symptoms_list):
    print(f"🧠 Starting Hybrid Reasoning with symptoms:", symptoms_list)

    # Normalize symptoms
    normalized_symptoms = [s.strip().lower() for s in symptoms_list if s.strip()]
    
    # Apply commonsense mappings FIRST (before LLM normalization)
    commonsense_mappings = {
        "cold": "runny_nose",  # Map "cold" to runny nose, not cold hands
        "common_cold": "runny_nose",
        "flu": "high_fever",
        "influenza": "high_fever",
        "fever": "high_fever",
        "headache": "headache",
        "sore_throat": "throat_irritation",
        "nausea": "nausea",
        "vomiting": "vomiting",
        "diarrhea": "diarrhoea",
        "diarrhoea": "diarrhoea"
    }
    
    # Apply commonsense mappings
    for i, symptom in enumerate(normalized_symptoms):
        if symptom in commonsense_mappings:
            normalized_symptoms[i] = commonsense_mappings[symptom]
    
    print(f"🔧 After commonsense mapping:", normalized_symptoms)

    # 0️⃣ Normalize with LLM to canonical tokens (only if not already mapped)
    try:
        llm_normalized = extractor.normalize_symptoms(normalized_symptoms)
        # Only use LLM normalization for symptoms not already commonsense-mapped
        for i, symptom in enumerate(normalized_symptoms):
            if symptom not in commonsense_mappings.values():
                if i < len(llm_normalized):
                    normalized_symptoms[i] = llm_normalized[i]
    except Exception:
        pass

    # 1️⃣ Step 1: Use Knowledge Graph to find related diseases
    candidate_diseases = set()
    candidate_scores = {}
    
    # Heuristic augmentation: map GERD-related lay terms to dataset tokens
    tokens_set = set(normalized_symptoms)
    gerd_triggers = {"regurgitation", "sour_taste", "acid_reflux", "heartburn", "burning_chest_pain"}
    if tokens_set & gerd_triggers:
        if "acidity" not in tokens_set:
            normalized_symptoms.append("acidity")
    for symptom in normalized_symptoms:
        # Clean and fuzzy match to nodes
        possible_matches = [n for n in G.nodes if n.lower().strip() == symptom]
        if not possible_matches:
            match, score = process.extractOne(symptom, list(G.nodes))
            if score >= 80:
                possible_matches.append(match)

        for match in possible_matches:
            try:
                connected_nodes = list(G.neighbors(match))
                disease_set = [n for n in connected_nodes if n in df_combined["prognosis"].unique()]
                candidate_diseases.update(disease_set)
                for dis in disease_set:
                    has_edge = 1 if (G.has_edge(dis, match) or G.has_edge(match, dis)) else 0
                    candidate_scores[dis] = candidate_scores.get(dis, 0) + has_edge
            except Exception as e:
                print(f"⚠️ Skipping unknown symptom '{symptom}' ({e})")

    # Ensure common illnesses are considered when appropriate patterns present
    common_illness_patterns = {
        "common_cold": ["runny_nose", "cough", "congestion"],
        "flu": ["high_fever", "cough", "fatigue"],
        "gastroenteritis": ["nausea", "vomiting", "diarrhoea"]
    }
    
    for illness, pattern in common_illness_patterns.items():
        if any(s in normalized_symptoms for s in pattern):
            candidate_diseases.add(illness)
            candidate_scores[illness] = candidate_scores.get(illness, 0) + 1.0

    # Ensure GERD is considered when acidity + chest_pain pattern present
    if ("acidity" in normalized_symptoms) and any(s in normalized_symptoms for s in ["chest_pain", "burning_chest_pain"]):
        candidate_diseases.add("GERD")
        candidate_scores["GERD"] = candidate_scores.get("GERD", 0) + 1.5

    # Normalize candidate_scores to [0..1] based on coverage fraction
    total_syms = max(1, len([s for s in normalized_symptoms if s in G.nodes]))
    for dis in list(candidate_scores.keys()):
        candidate_scores[dis] = round(min(1.0, candidate_scores[dis] / total_syms), 3)

    top_candidates = sorted(list(candidate_diseases), key=lambda d: candidate_scores.get(d, 0), reverse=True)[:5]

    print(f"🔍 KG Candidate diseases: {top_candidates} ...")

    # 2️⃣ Step 2: Use LLM to reason final prediction with stronger commonsense instruction and few-shot
    examples = [
        {
            "symptoms": ["high_fever", "dry_cough", "loss_of_smell"],
            "candidates": ["Common Cold", "COVID-19"],
            "answer": {"Disease": "COVID-19", "Description": "The presence of high fever, dry cough, and loss of smell (anosmia) strongly suggests COVID-19 infection. While fever and cough are common in many respiratory illnesses, the sudden loss of smell is a distinctive symptom of COVID-19 that occurs due to the virus affecting olfactory nerve cells. This combination of symptoms, particularly the anosmia, makes COVID-19 the most likely diagnosis over other viral respiratory infections like the common cold.", "Precautions": "Isolate, mask", "Medications": "Symptomatic", "Workout": "Rest", "Diet": "Hydration"}
        },
        {
            "symptoms": ["burning_chest_pain", "regurgitation", "sour_taste"],
            "candidates": ["Gastritis", "GERD"],
            "answer": {"Disease": "GERD", "Description": "The combination of burning chest pain, regurgitation, and sour taste strongly indicates Gastroesophageal Reflux Disease (GERD). These symptoms occur when stomach acid flows back into the esophagus due to a weakened lower esophageal sphincter, causing the characteristic burning sensation and acidic taste. The regurgitation of stomach contents, particularly the sour taste, is a hallmark sign of acid reflux that distinguishes GERD from other gastrointestinal conditions.", "Precautions": "Elevate head", "Medications": "PPIs", "Workout": "Light", "Diet": "Avoid trigger foods"}
        }
    ]

    prompt = f"""
    You are an advanced medical reasoning AI that integrates a knowledge graph and a classifier.
    Prioritize medical plausibility over dataset frequency. If uncertain, abstain.
    Provide a brief rationale.

    Few-shot examples:\n{examples}

    Given the symptoms (consider ALL of them): {normalized_symptoms}
    Candidate diseases with KG coverage scores (0..1): {candidate_scores}
    Focus on these top candidates: {top_candidates}
    
    IMPORTANT: Write a comprehensive, detailed description (2-3 sentences) explaining:
    1. Why this disease fits the symptom pattern
    2. How the symptoms relate to the disease mechanism
    3. What makes this diagnosis most likely given the evidence
    
    Return strict JSON:
    {{
      "Disease": "|Unknown| or disease name",
      "Description": "Comprehensive 2-3 sentence explanation of why this disease matches the symptoms, including medical reasoning and symptom-disease relationships",
      "Precautions": "",
      "Medications": "",
      "Workout": "",
      "Diet": "",
      "Rationale": "why"
    }}
    """

    try:
        llm_response = extractor.query(prompt)
        result = extractor.safe_json_parse(llm_response)

        # 3️⃣ Verification pass
        predicted = result.get("Disease", "Unknown") if isinstance(result, dict) else "Unknown"
        verification = extractor.verify_prediction(normalized_symptoms, predicted)

        # 4️⃣ Fusion of scores (lightweight)
        # KG support: proportion of symptoms directly connected to predicted disease in current G
        def kg_support_score(disease: str, symptoms_norm: list[str]) -> float:
            if not disease or disease not in G.nodes:
                return 0.0
            if not symptoms_norm:
                return 0.0
            hits = 0
            total = 0
            for s in symptoms_norm:
                if s in G.nodes:
                    total += 1
                    if G.has_edge(disease, s) or G.has_edge(s, disease):
                        hits += 1
            return hits / total if total else 0.0

        kg_score = kg_support_score(predicted, normalized_symptoms)
        llm_score = verification.get("confidence", 0.6)
        rf_score = 0.0  # avoid heavy RF init inside request path
        final_score = 0.45 * llm_score + 0.35 * kg_score + 0.20 * rf_score

        if not verification.get("plausible", False) and kg_score == 0.0:
            # fallback to best KG candidate if available
            fallback = list(candidate_diseases)[0] if candidate_diseases else "Unknown"
            if isinstance(result, dict):
                result["Disease"] = fallback
                result["Rationale"] = "Adjusted by sanity checks"

        if isinstance(result, dict):
            result["Confidence"] = round(final_score, 2)
        print("🤖 LLM reasoning successful.")
        return result

    except Exception as e:
        print("⚠️ LLM reasoning failed:", e)
        if candidate_diseases:
            fallback_disease = list(candidate_diseases)[0]
            return {
                "Disease": fallback_disease,
                "Description": "Detected from dataset using KG fallback.",
                "Precautions": "Rest and stay hydrated. Seek medical advice if symptoms persist.",
                "Medications": "Use only doctor-prescribed medicines.",
                "Workout": "Mild exercise recommended.",
                "Diet": "Eat light, nutritious meals."
            }
        else:
            return {
                "Disease": "Unknown",
                "Description": "No disease found in KG or LLM.",
                "Precautions": "Consult a doctor immediately.",
                "Medications": "Not available.",
                "Workout": "Not available.",
                "Diet": "Not available."
            }


# ------------------------------
# Flask Routes
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    symptoms_input = request.form.get("symptoms", "").strip()
    if not symptoms_input:
        return render_template("index.html", message="⚠️ Please enter symptoms.")

    user_symptoms = [s.strip().lower() for s in symptoms_input.split(",") if s.strip()]
    corrected_symptoms = []

    # Try dataset fuzzy match first
    for symptom in user_symptoms:
        if symptoms_list_processed:
            match, score = process.extractOne(symptom, list(symptoms_list_processed.keys()))
            if score >= 70:
                corrected_symptoms.append(symptoms_list_processed[match])

    # If nothing matched, fall back to LLM normalization and proceed
    if not corrected_symptoms:
        try:
            normalized = extractor.normalize_symptoms(user_symptoms)
            corrected_symptoms = normalized if normalized else user_symptoms
        except Exception:
            corrected_symptoms = user_symptoms

    # 🔄 Always use hybrid reasoning (KG + LLM)
    result = hybrid_reasoning(corrected_symptoms)
    
    # Calculate KG support score for display
    kg_score = 0.0
    verification = {"confidence": 0.0}
    
    try:
        predicted_disease = result.get("Disease", "Unknown")
        if predicted_disease and predicted_disease in G.nodes:
            hits = 0
            total = 0
            for s in corrected_symptoms:
                if s in G.nodes:
                    total += 1
                    if G.has_edge(predicted_disease, s) or G.has_edge(s, predicted_disease):
                        hits += 1
            kg_score = hits / total if total else 0.0
    except Exception:
        pass
    
    try:
        verification = extractor.verify_prediction(corrected_symptoms, result.get("Disease", "Unknown"))
    except Exception:
        pass

    return render_template(
        "index.html",
        prediction=True,
        disease=result.get("Disease", "Unknown"),
        description=result.get("Description", "No description available."),
        precautions=result.get("Precautions", "Not available."),
        medications=result.get("Medications", "Not available."),
        workout=result.get("Workout", "Not available."),
        diet=result.get("Diet", "Not available."),
        symptoms=corrected_symptoms,
        user_symptoms=user_symptoms,
        confidence=round(result.get("Confidence", 0.0) * 100, 1) if result.get("Confidence") else None,
        rationale=result.get("Rationale", ""),
        kg_support=round(kg_score * 100, 1),
        llm_confidence=round(verification.get("confidence", 0.0) * 100, 1),
    )


# ------------------------------
# Run Flask App
# ------------------------------
if __name__ == "__main__":
    extractor, G, df_combined, symptoms_list_processed = initialize_enhanced_system()
    app.run(debug=True)
