#!/bin/bash
# Development script for FractalMind

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}FractalMind Development Tool${NC}"
echo ""

# Default features
FEATURES="pdf"

# Check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker daemon is not running${NC}"
        exit 1
    fi
}

# Function to start Docker containers
start_docker() {
    echo -e "${YELLOW}Starting Docker containers...${NC}"
    
    # Check if containers are already running
    if docker-compose ps --services --filter "status=running" | grep -q . 2>/dev/null; then
        echo -e "${GREEN}✓ Docker containers already running${NC}"
    else
        # Start searxng by default (profile enabled)
        docker-compose up -d surrealdb searxng
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Docker containers started successfully${NC}"
            echo ""
            echo -e "${GREEN}✅ Services:${NC}"
            echo "  - surrealdb: http://localhost:8000"
            echo "  - searxng:   http://localhost:8080 (web search)"
            echo ""
            echo -e "${YELLOW}Note:${NC} SearXNG is enabled by default for REM phase web search."
            echo "         Set WEB_SEARCH_ENABLED=false to disable (REM phase will be off)."
        else
            echo -e "${RED}✗ Failed to start Docker containers${NC}"
            return 1
        fi
    fi
}

# Function to start backend
start_backend() {
    echo -e "${YELLOW}Starting backend server...${NC}"
    RUST_LOG=debug cargo run --features "$FEATURES" &
    BACKEND_PID=$!
    echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
}

# Function to start frontend
start_frontend() {
    if [ -d "ui" ]; then
        echo -e "${YELLOW}Starting frontend...${NC}"
        cd ui
        if [ ! -d "node_modules" ]; then
            npm install
        fi
        npm run dev &
        FRONTEND_PID=$!
        cd ..
        echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Frontend directory not found, skipping${NC}"
    fi
}

# Parse command line arguments
case "${1:-run}" in
    "run")
        echo -e "${YELLOW}Starting full development environment...${NC}"
        check_docker
        start_docker || exit 1
        sleep 3  # Wait for searxng to start
        start_backend
        start_frontend
        
        echo ""
        echo -e "${GREEN}═══════════════════════════════════════${NC}"
        echo -e "${GREEN}Development environment is running!${NC}"
        echo -e "${GREEN}═══════════════════════════════════════${NC}"
        echo ""
        echo "Services:"
        echo "  Database:  http://localhost:8000"
        echo "  SearXNG:   http://localhost:8080 (web search)"
        echo "  Backend:   http://localhost:3000"
        echo "  Frontend:  http://localhost:5173 (or configured port)"
        echo ""
        echo "REM Phase:"
        echo "  ✓ SearXNG is enabled - REM web search active"
        echo ""
        echo "Press Ctrl+C to stop all services"
        wait
        ;;
    "docker")
        check_docker
        start_docker
        ;;
    "backend")
        echo -e "${YELLOW}Building and running backend only...${NC}"
        cargo build --features "$FEATURES"
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
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        cargo clean
        docker-compose down -v
        echo -e "${GREEN}✓ Clean completed${NC}"
        ;;
    "stop")
        echo -e "${YELLOW}Stopping Docker containers...${NC}"
        docker-compose down
        echo -e "${GREEN}✓ Docker containers stopped${NC}"
        ;;
    "help")
        echo "Usage: ./dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  run            Start full dev environment (docker + backend + frontend) [default]"
        echo "  docker         Start only Docker containers (SearXNG enabled by default)"
        echo "  backend        Start only backend server"
        echo "  build          Build the project"
        echo "  build-release  Build release version"
        echo "  test           Run tests"
        echo "  stop           Stop Docker containers"
        echo "  clean          Clean build artifacts and containers"
        echo "  help           Show this help message"
        echo ""
        echo "Features enabled: $FEATURES"
        echo ""
        echo "SearXNG:"
        echo "  - Enabled by default via docker-compose profile"
        echo "  - Provides web search for REM phase"
        echo "  - Disable by setting WEB_SEARCH_ENABLED=false"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './dev.sh help' for usage information"
        exit 1
        ;;
esac
