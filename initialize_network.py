import math
import random
from typing import Dict, Any, List

from config_store import load_config
from param_io import save_params
import network as net


PARAM_CACHE: Dict[str, Any] = {}


def initialize_network_layer(layer1: int, layer2: int) -> Dict[str, List[float]]:
    """Initialize a layer using Xavier/Glorot uniform suited for ReLU."""
    a = math.sqrt(6 / (layer1 + layer2))
    weights: List[List[float]] = []
    biases: List[float] = []
    for _ in range(layer2):
        biases.append(0.0)

    for i in range(layer1):
        weights.append([])
        for _ in range(layer2):
            weights[i].append((random.random() * 2 - 1) * a)
    return {"weights": weights, "biases": biases}


def refresh_params() -> Dict[str, Any]:
    global PARAM_CACHE
    cfg = load_config()
    PARAM_CACHE = cfg
    return dict(cfg)


def build_network() -> List[Dict[str, List[float]]]:
    cfg = PARAM_CACHE or refresh_params()
    sizes = list(cfg["network_size"])
    network_layers: List[Dict[str, List[float]]] = []
    for i in range(1, len(sizes)):
        network_layers.append(
            initialize_network_layer(int(sizes[i - 1]), int(sizes[i]))
        )
    return network_layers


def save_new_params() -> List[Dict[str, List[float]]]:
    net.refresh_config()
    network_layers = build_network()
    save_params(network_layers)
    return network_layers


if __name__ == "__main__":
    network_layers = build_network()
    for idx, layer in enumerate(network_layers):
        print("Layer", idx, "", len(layer["weights"]), "x", len(layer["weights"][0]))
    print("----")
    save_params(network_layers)
