# Justfile - Command runner for the project
# Similar to Makefile but simpler syntax
# Run commands with: just <command-name>

# Default recipe that runs when you type 'just' with no arguments
# Shows available commands to the user
default:
    # List all available commands in this Justfile
    # The @ symbol prevents printing the command itself
    @just --list

# Build the project in release mode with optimizations
# This creates a faster binary for inference
build:
    # Compile with --release flag for optimized performance
    # tch operations are much faster in release mode
    cargo build --release

# Run the main program which trains and demos the model
# Uses release mode for faster training
run:
    # Execute the release binary directly
    # Faster than 'cargo run' since it skips compilation check
    cargo run --release

# Quick development run without optimizations
# Use this for debugging and development
dev:
    # Run in debug mode with better error messages
    # Slower but provides more debugging information
    cargo run

# Clean build artifacts and saved models
# Useful for starting fresh
clean:
    # Remove all compiled files and dependencies
    cargo clean
    # Remove saved model file if it exists
    # The - prefix ignores errors if file doesn't exist
    -rm gas_model.pt

# Run clippy linter to catch common mistakes
# Helps maintain code quality
lint:
    # Run clippy with all targets and stricter warnings
    # --all-targets checks tests and examples too
    cargo clippy --all-targets -- -W clippy::all

# Format code according to Rust standards
# Ensures consistent code style
fmt:
    # Format all Rust files in the project
    # Makes code easier to read and review
    cargo fmt

# Run tests if we add any
# Good practice for production code
test:
    # Run all unit and integration tests
    # Currently no tests but structure is here
    cargo test

# Check if project compiles without building
# Faster than full build for syntax checking
check:
    # Verify code compiles but don't generate binary
    # Useful for quick syntax validation
    cargo check

# Watch for file changes and auto-rebuild
# Requires cargo-watch: cargo install cargo-watch
watch:
    # Automatically rebuild on file changes
    # Great for development workflow
    cargo watch -x check -x run

# Show project tree structure
# Useful for documentation and debugging
tree:
    # Display project file structure
    # Uses tree command if available, otherwise find
    @echo "Project structure:"
    @tree -I target || find . -name target -prune -o -type f -print | grep -v target | sort