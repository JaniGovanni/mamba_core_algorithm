# Mamba Core Algorithm Implementation

This Jupyter notebook contains an implementation and detailed explanation of the core algorithms used in the Mamba architecture, a state-space model for efficient sequence modeling.

## Contents

1. Mamba 1 Selective Scan Algorithm
   - Implementation of the selective scan algorithm
   - Step-by-step breakdown of the algorithm with mathematical explanations

2. Mamba 2 SSD (State Space Decomposition) Algorithm
   - Implementation of the SSD algorithm
   - Detailed explanations of each component:
     - Computation of L (lookahead-mask) and Y_diag (diagonal blocks output)
     - Computation of states
     - Computation of new_states
     - Computation of Y_off (off-diagonal blocks output)

## Key Features

- TensorFlow and PyTorch implementations
- Extensive use of Einstein summation for efficient tensor operations
- Detailed mathematical breakdowns and visualizations of the algorithms
- Step-by-step explanations of the computations involved

This notebook serves as both an implementation guide and a deep dive into the mathematics behind the Mamba architecture, making it valuable for researchers and practitioners working with state-space models in deep learning.