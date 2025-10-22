import argparse
import os
import sys

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from enhanced_recommendation_system import EnhancedMedicalRecommendationSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="model/fused_kg.json")
    parser.add_argument("--dataset", default="kaggle_dataset")
    args = parser.parse_args()

    system = EnhancedMedicalRecommendationSystem(dataset_path=args.dataset)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    system.knowledge_graph.save_graph(args.out)
    print("Saved fused KG to", args.out)


if __name__ == "__main__":
    main()


