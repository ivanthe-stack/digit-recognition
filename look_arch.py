import sys
import pickle

def print_network_layers(pkl_file):
    # Load the saved parameters
    with open(pkl_file, "rb") as f:
        network = pickle.load(f)

    # Print layer sizes
    for i, layer in enumerate(network):
        weights = layer["weights"]
        print(f"Layer {i}: {len(weights)} x {len(weights[0])}")
    print("----")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_layers.py <params.pkl>")
        sys.exit(1)

    print_network_layers(sys.argv[1])