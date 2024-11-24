import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.grid import *
from src.simulation.particles import *

def test_compute_weight():
    p = Particle((1, 1, 1), velocity=(0, -1, 0), mass=1)
    node = GridNode((1, 3, 1))
    weight = node.compute_weight(p)
    print(weight)

test_compute_weight()