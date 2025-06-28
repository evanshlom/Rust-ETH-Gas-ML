// Import neural network module from tch
// nn provides layers and model building blocks
use tch::{nn, nn::Module, Tensor};

// Define constants for network architecture
// These control the size and capacity of our model
// Number of input features (7 gas market indicators)
const INPUT_SIZE: i64 = 7;
// Number of neurons in hidden layer
// 64 provides good capacity without overfitting
const HIDDEN_SIZE: i64 = 64;
// Number of outputs (1 for regression)
// Single value: predicted gas price
const OUTPUT_SIZE: i64 = 1;

// Define our neural network structure
// Implements a 2-layer feedforward network
pub struct GasPriceNet {
    // First fully connected layer: input -> hidden
    // Transforms 7 features to 64 hidden units
    fc1: nn::Linear,
    // Second fully connected layer: hidden -> output  
    // Transforms 64 hidden units to 1 prediction
    fc2: nn::Linear,
}

// Implementation block for model methods
impl GasPriceNet {
    // Constructor to create new model instance
    // Takes a path from variable store for parameter registration
    pub fn new(vs: &nn::Path) -> Self {
        // Initialize first layer with Xavier/He initialization
        // / "fc1" creates a subpath for these parameters
        let fc1 = nn::linear(vs / "fc1", INPUT_SIZE, HIDDEN_SIZE, Default::default());
        // Initialize output layer
        // / "fc2" keeps parameters organized
        let fc2 = nn::linear(vs / "fc2", HIDDEN_SIZE, OUTPUT_SIZE, Default::default());
        
        // Return the constructed model
        // Both layers are now registered in the variable store
        Self { fc1, fc2 }
    }
}

// Implement the Module trait for forward pass
// This defines how data flows through the network
impl Module for GasPriceNet {
    // Forward propagation function
    // Takes input tensor and returns predictions
    fn forward(&self, xs: &Tensor) -> Tensor {
        // Apply first linear transformation
        // Converts input features to hidden representation
        xs.apply(&self.fc1)
            // Apply ReLU activation function
            // Introduces non-linearity for learning complex patterns
            .relu()
            // Apply second linear transformation
            // Produces final gas price prediction
            .apply(&self.fc2)
    }
}