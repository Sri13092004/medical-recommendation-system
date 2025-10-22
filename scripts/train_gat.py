import argparse
import os
import sys

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from knowledge_graph import HealthcareKnowledgeGraph
from gat_fusion import save_gat_weights, train_gat_stub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="model/gat_fusion.pt")
    args = parser.parse_args()

    kg = HealthcareKnowledgeGraph()  # replace with real fused KG if available
    state_dict = train_gat_stub(kg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ok = save_gat_weights(state_dict, args.out)
    print("Saved GAT weights to", args.out if ok else "FAILED")


if __name__ == "__main__":
    main()


