#!/bin/bash
# Development script for Fractal-Mind

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Fractal-Mind Development Tool${NC}"
echo ""

# Default features
FEATURES="pdf"

# Parse command line arguments
case "${1:-run}" in
    "run")
        echo -e "${YELLOW}Starting server with features: $FEATURES${NC}"
        RUST_LOG=info cargo run --features "$FEATURES"
        ;;
    "build")
        echo -e "${YELLOW}Building with features: $FEATURES${NC}"
        cargo build --features "$FEATURES"
        ;;
    "build-release")
        echo -e "${YELLOW}Building release with features: $FEATURES${NC}"
        cargo build --release --features "$FEATURES"
        ;;
    "test")
        echo -e "${YELLOW}Running tests with features: $FEATURES${NC}"
        cargo test --features "$FEATURES"
        ;;
    "clean")
        echo -e "${YELLOW}Cleaning build artifacts${NC}"
        cargo clean
        ;;
    "help")
        echo "Usage: ./dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  run            Start the development server (default)"
        echo "  build          Build the project"
        echo "  build-release  Build release version"
        echo "  test           Run tests"
        echo "  clean          Clean build artifacts"
        echo "  help           Show this help message"
        echo ""
        echo "Features enabled: $FEATURES"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './dev.sh help' for usage information"
        exit 1
        ;;
esac
