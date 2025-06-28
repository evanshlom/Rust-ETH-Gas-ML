// Import model structure from our model module
use crate::model::GasPriceNet;
// Import colored output for training progress
use ansi_term::Colour::{Green, Red, Yellow};
// Import PyTorch components for training
use tch::{nn, nn::OptimizerConfig, Device, Tensor};

// Main training function
// Takes data, device, and model path, returns trained variable store
pub fn train_model(
    train_features: &Tensor,
    train_labels: &Tensor,
    val_features: &Tensor,
    val_labels: &Tensor,
    device: Device,
    model_path: &str,
) -> nn::VarStore {
    // Create variable store to hold model parameters
    // This manages all trainable weights and biases
    let mut vs = nn::VarStore::new(device);
    
    // Initialize the model with the variable store
    // Creates layers and registers parameters
    let model = GasPriceNet::new(&vs.root());
    
    // Create Adam optimizer with learning rate 0.001
    // Adam adapts learning rate per parameter
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    
    // Training hyperparameters
    // Number of complete passes through the dataset
    let n_epochs = 100;
    // Number of samples per gradient update
    // Smaller batch = noisier but more frequent updates
    let batch_size = 32;
    
    // Calculate number of batches per epoch
    // Integer division for complete batches only
    let n_batches = train_features.size()[0] / batch_size;
    
    // Move data to computation device
    // Ensures CPU/GPU consistency
    let train_features = train_features.to_device(device);
    let train_labels = train_labels.to_device(device);
    let val_features = val_features.to_device(device);
    let val_labels = val_labels.to_device(device);
    
    // Main training loop
    // Iterate through all epochs
    for epoch in 1..=n_epochs {
        // Accumulate loss for epoch statistics
        let mut epoch_loss = 0.0;
        
        // Mini-batch training loop
        // Process data in small chunks for efficiency
        for batch_idx in 0..n_batches {
            // Calculate batch start and end indices
            // Ensures we don't exceed data bounds
            let start = batch_idx * batch_size;
            let end = ((batch_idx + 1) * batch_size).min(train_features.size()[0]);
            
            // Extract batch of features
            // narrow creates a view without copying
            let batch_features = train_features.narrow(0, start, end - start);
            let batch_labels = train_labels.narrow(0, start, end - start);
            
            // Forward pass: compute predictions
            // Model processes batch of inputs
            let predictions = model.forward(&batch_features);
            
            // Compute mean squared error loss
            // Measures prediction accuracy
            let loss = predictions.mse_loss(&batch_labels, tch::Reduction::Mean);
            
            // Backward pass: compute gradients
            // Updates all parameters based on loss
            opt.backward_step(&loss);
            
            // Accumulate batch loss
            // Convert tensor to f64 for statistics
            epoch_loss += f64::from(&loss);
        }
        
        // Compute average epoch loss
        // Normalizes by number of batches
        epoch_loss /= n_batches as f64;
        
        // Validation pass every 10 epochs
        // Monitors model performance on unseen data
        if epoch % 10 == 0 {
            // Disable gradient computation for validation
            // Saves memory and computation
            let val_predictions = tch::no_grad(|| model.forward(&val_features));
            // Compute validation loss
            let val_loss = val_predictions.mse_loss(&val_labels, tch::Reduction::Mean);
            let val_loss_value = f64::from(&val_loss);
            
            // Print training progress with colors
            // Green for good progress, yellow for warnings
            println!(
                "Epoch {:3}/{}: {} {:.4}, {} {:.4}",
                epoch,
                n_epochs,
                Yellow.paint("Train Loss:"),
                epoch_loss,
                Green.paint("Val Loss:"),
                val_loss_value
            );
            
            // Early stopping check
            // Prevents overfitting when validation stops improving
            if val_loss_value > epoch_loss * 1.5 {
                println!("{}", Red.paint("Warning: Possible overfitting detected"));
            }
        }
    }
    
    // Save trained model to disk
    // Allows loading for inference later
    vs.save(model_path).unwrap();
    println!("{}", Green.bold().paint(format!("Model saved to {}", model_path)));
    
    // Return the variable store
    // Contains all trained parameters
    vs
}