// Import random number generation utilities
// Used to create realistic synthetic data
use rand::Rng;
// Import tensor type from tch
// Our main data structure for neural network inputs
use tch::Tensor;

// Function to generate synthetic gas price data
// Returns tuple of (features, labels) tensors
pub fn generate_gas_data(n_samples: usize) -> (Tensor, Tensor) {
    // Create thread-local random number generator
    // More efficient than creating new one each time
    let mut rng = rand::thread_rng();
    
    // Pre-allocate vectors for features and labels
    // More efficient than growing dynamically
    let mut features = Vec::with_capacity(n_samples * 7);
    let mut labels = Vec::with_capacity(n_samples);
    
    // Generate each sample
    // Loop creates realistic gas market scenarios
    for _ in 0..n_samples {
        // Feature 1: Current base fee (20-200 gwei)
        // Base fee fluctuates with network demand
        let base_fee = rng.gen_range(20.0..200.0);
        
        // Feature 2: Pending transaction count (50-1000)
        // More pending txs = higher congestion
        let pending_tx = rng.gen_range(50.0..1000.0);
        
        // Feature 3: Average gas used last 5 blocks (0.3-0.95)
        // Indicates recent network utilization
        let avg_gas_used = rng.gen_range(0.3..0.95);
        
        // Feature 4: Current block utilization (0.2-1.0)
        // How full the current block is
        let block_util = rng.gen_range(0.2..1.0);
        
        // Feature 5: Hour of day (0-23)
        // Gas prices vary by time of day
        let hour = rng.gen_range(0.0..24.0);
        
        // Feature 6: High priority transaction count
        // Transactions paying >2x base fee
        let high_priority = rng.gen_range(0.0..300.0);
        
        // Feature 7: Weekend flag (0 or 1)
        // Lower activity on weekends typically
        let weekend = if rng.gen_bool(2.0/7.0) { 1.0 } else { 0.0 };
        
        // Calculate realistic gas price based on features
        // This formula mimics real gas price dynamics
        let gas_price = 
            // Base component from base fee
            base_fee * 1.1
            // Congestion factor from pending transactions
            + (pending_tx / 1000.0) * 50.0
            // Utilization pressure
            + avg_gas_used * 40.0
            // Current block pressure
            + block_util * 30.0
            // Time of day factor (higher during business hours)
            + if hour >= 9.0 && hour <= 17.0 { 15.0 } else { -5.0 }
            // High priority transaction pressure
            + (high_priority / 300.0) * 25.0
            // Weekend discount
            + if weekend == 1.0 { -10.0 } else { 5.0 }
            // Add some random noise for realism
            + rng.gen_range(-5.0..5.0);
        
        // Clamp gas price to reasonable range
        // Prevents unrealistic negative or extreme values
        let gas_price = gas_price.max(15.0).min(300.0);
        
        // Add features to vector in order
        // Order must match model input expectations
        features.push(base_fee);
        features.push(pending_tx);
        features.push(avg_gas_used);
        features.push(block_util);
        features.push(hour);
        features.push(high_priority);
        features.push(weekend);
        
        // Add corresponding label
        // This is what the model will learn to predict
        labels.push(gas_price);
    }
    
    // Convert vectors to tensors
    // Reshape features to [n_samples, 7] matrix
    let features_tensor = Tensor::of_slice(&features)
        .reshape(&[n_samples as i64, 7]);
    // Labels remain as [n_samples] vector
    let labels_tensor = Tensor::of_slice(&labels);
    
    // Return both tensors
    // Ready for training or evaluation
    (features_tensor, labels_tensor)
}