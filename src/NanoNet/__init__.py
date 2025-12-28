# Top-level API for NanoNet
from .network import Network, load_from_file
from .optimizer import ADAM, SGD, SGD_Momentum, RMSProp # Ensure it's Prop, not Promp

from .costFunction import (
    QuadraticCost, 
    CategorialCrossEntropy, 
    BinaryCrossEntropy, 
    MeanAbsoluteCost,
)

from .activationFunction import (
    Sigmoid, 
    SoftMax,
    ReLu
)

# Adding data utilities here makes the library much easier to use
from .data.data_loader import DataLoader
from .data.dataset import Dataset