# Rust-ETH-Gas-ML 

Rust regression model training and inference for predicting Gas Price of next Ethereum chain block using ML with tch-rs.

## Prerequisites

- Rust (latest stable)
- LibTorch will be auto-downloaded by tch-rs

## Quick Start

### Windows (PowerShell/CMD)
```powershell
# Clone or create the project
mkdir tch-hft-predictor
cd tch-hft-predictor

# Copy all the provided files into the project

# Build and run
cargo run --release
```

### macOS/Linux
```bash
# Clone or create the project
mkdir tch-hft-predictor
cd tch-hft-predictor

# Copy all the provided files into the project

# Build and run
cargo run --release
```

## Available Commands

If you have `just` installed:
```bash
just          # Show available commands
just run      # Train model and run inference demo
just clean    # Clean build artifacts and saved model
just fmt      # Format code
```

Without `just`, use cargo directly:
```bash
cargo build --release    # Build optimized binary
cargo run --release      # Run the demo
cargo clean             # Clean build artifacts
```

## Project Structure

```
tch-hft-predictor/
├── Cargo.toml          # Project dependencies
├── Justfile            # Command shortcuts
├── README.md           # This file
├── src/
│   ├── main.rs         # Entry point & demo
│   ├── model.rs        # Neural network definition
│   ├── data.rs         # Synthetic data generation
│   └── train.rs        # Training loop
└── gas_model.pt        # Saved model (created after training)
```

## Model Architecture

- **Input**: 7 features
  - Current base fee (gwei)
  - Pending transaction count
  - Average gas used (last 5 blocks)
  - Block utilization %
  - Hour of day (UTC)
  - High-priority transaction count
  - Weekend flag

- **Network**: 2-layer feedforward
  - Input layer: 7 neurons
  - Hidden layer: 64 neurons (ReLU activation)
  - Output layer: 1 neuron (gas price prediction)

- **Training**: 
  - 100 epochs
  - Adam optimizer (lr=0.001)
  - MSE loss
  - Batch size: 32

## Sample Output

```
Ethereum Gas Price Predictor
================================

Generating synthetic gas price data...
Generated 10000 training samples with 7 features

Training neural network...
Epoch  10/100: Train Loss: 234.5678, Val Loss: 245.1234
Epoch  20/100: Train Loss: 125.4321, Val Loss: 132.5678
...
Model saved to gas_model.pt

Inference Demo
================

Example 1:
  Base Fee: 150 gwei
  Pending Transactions: 500
  Avg Gas Used (last 5 blocks): 85.0%
  Block Utilization: 90.0%
  Hour (UTC): 14
  High Priority TXs: 200
  Weekend: No
  -> Predicted Gas Price: 187.45 gwei
```

## Learning Points

This tutorial demonstrates:
- Setting up tch-rs (PyTorch bindings for Rust)
- Creating a simple feedforward neural network
- Generating synthetic training data
- Training loop with validation
- Saving/loading models
- Running inference on new data

## Notes

- The model trains on synthetic data that mimics gas price patterns
- Real gas price prediction would require historical blockchain data
- Training takes ~30 seconds on CPU
- The saved model (`gas_model.pt`) can be reused for inference

## Troubleshooting

If you encounter LibTorch download issues:
1. tch-rs will auto-download LibTorch on first build
2. For manual installation, see: https://pytorch.org/get-started/locally/
3. Set `LIBTORCH` environment variable to point to your installation

## Resources

- [tch-rs documentation](https://docs.rs/tch/)
- [PyTorch documentation](https://pytorch.org/docs/)
- [Ethereum gas mechanics](https://ethereum.org/en/developers/docs/gas/)