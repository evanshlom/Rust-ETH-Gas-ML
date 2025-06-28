// Import required modules from our project
// These modules contain model definition, data generation, and training logic
mod data;
mod model;
mod train;

// Import necessary items from tch crate
// Device represents CPU/GPU, Tensor is the main data structure
use tch::{Device, Tensor};
// Import colored output for better terminal display
use ansi_term::Colour::{Blue, Green, Red, Yellow};

// Main entry point of our program
// This function orchestrates training and inference demo
fn main() {
    // Print welcome message with colored output
    // Makes it clear the program has started
    println!("{}", Blue.bold().paint("\nEthereum Gas Price Predictor"));
    println!("{}", Blue.paint("================================\n"));

    // Set the computation device (CPU in this case)
    // Could be Device::Cuda(0) for GPU if available
    let device = Device::Cpu;
    
    // Define model save path
    // This is where we'll save/load our trained model
    let model_path = "gas_model.pt";

    // Generate synthetic training data
    // Returns features and labels for training
    println!("{}", Yellow.paint("Generating synthetic gas price data..."));
    // Create 10,000 training samples with our 7 features
    // More data generally leads to better model performance
    let (train_features, train_labels) = data::generate_gas_data(10000);
    // Create 1,000 validation samples for monitoring training
    // Helps detect overfitting during training
    let (val_features, val_labels) = data::generate_gas_data(1000);
    
    // Print data shape information
    // Useful for debugging and understanding data flow
            println!("{}", Green.paint(format!(
        "Generated {} training samples with {} features",
        train_features.size()[0],
        train_features.size()[1]
    )));

    // Train the model
    // This is where the neural network learns patterns
    println!("\n{}", Yellow.paint("Training neural network..."));
    // Call our training function with all necessary parameters
    // Returns the trained variable store containing model weights
    let vs = train::train_model(
        &train_features,
        &train_labels,
        &val_features,
        &val_labels,
        device,
        model_path,
    );

    // Inference demo section
    // Shows how to use the trained model for predictions
    println!("\n{}", Blue.bold().paint("Inference Demo"));
    println!("{}", Blue.paint("================\n"));

    // Create some example inputs for prediction
    // These represent realistic gas market conditions
    let examples = vec![
        // Example 1: High congestion scenario
        // base_fee, pending_tx, avg_gas, util%, hour, high_priority, weekend
        vec![150.0, 500.0, 0.85, 0.90, 14.0, 200.0, 0.0], // Weekday afternoon high load
        // Example 2: Low congestion scenario
        vec![30.0, 100.0, 0.45, 0.40, 3.0, 20.0, 1.0],    // Weekend early morning
        // Example 3: Medium congestion scenario
        vec![80.0, 300.0, 0.65, 0.70, 9.0, 100.0, 0.0],   // Weekday morning
    ];

    // Load the trained model for inference
    // Create a new model instance with the saved weights
    let model = model::GasPriceNet::new(&vs.root());
    
    // Process each example and make predictions
    // Demonstrates real-world usage of the model
    for (i, example) in examples.iter().enumerate() {
        // Convert example vector to tensor
        // Reshape to [1, 7] for single sample batch
        let input = Tensor::of_slice(example)
            .to_device(device)
            .unsqueeze(0);
        
        // Run inference (forward pass)
        // no_grad prevents gradient computation for efficiency
        let prediction = tch::no_grad(|| model.forward(&input));
        
        // Extract prediction value as f64
        // Convert from tensor to regular Rust type
        let predicted_price = f64::from(prediction);
        
        // Print formatted results
        // Shows both input conditions and predicted price
        println!("{}", Yellow.paint(format!("Example {}:", i + 1)));
        println!("  Base Fee: {} gwei", example[0]);
        println!("  Pending Transactions: {}", example[1]);
        println!("  Avg Gas Used (last 5 blocks): {:.1}%", example[2] * 100.0);
        println!("  Block Utilization: {:.1}%", example[3] * 100.0);
        println!("  Hour (UTC): {}", example[4]);
        println!("  High Priority TXs: {}", example[5]);
        println!("  Weekend: {}", if example[6] == 1.0 { "Yes" } else { "No" });
        // Highlight the prediction in green
        println!("{}", Green.bold().paint(format!(
            "  -> Predicted Gas Price: {:.2} gwei\n", 
            predicted_price
        )));
    }

    // Print completion message
    // Indicates successful execution
    println!("{}", Blue.bold().paint("Demo complete!"));
}