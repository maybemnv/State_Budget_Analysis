#!/bin/bash

# Simple script to manage the project environment

function show_help() {
    echo "Usage: ./manage.sh [command]"
    echo ""
    echo "Commands:"
    echo "  dev      Start the project in development mode (with hot-reloading)"
    echo "  prod     Start the project in production mode"
    echo "  stop     Stop all containers"
    echo "  build    Build/Rebuild Docker images"
    echo "  test     Run backend and frontend tests"
    echo "  clean    Remove Docker volumes and containers"
    echo "  help     Show this help message"
}

case "$1" in
    dev)
        echo "Starting in development mode..."
        docker compose up --build
        ;;
    prod)
        echo "Starting in production mode..."
        docker compose -f docker-compose.prod.yaml up --build -d
        ;;
    stop)
        echo "Stopping containers..."
        docker compose down
        ;;
    build)
        echo "Building images..."
        docker compose build
        ;;
    test)
        echo "Running backend tests..."
        docker compose run --rm backend pytest backend/tests
        echo "Running frontend tests..."
        cd frontend && npm run test
        ;;
    clean)
        echo "Cleaning up..."
        docker compose down -v
        ;;
    *)
        show_help
        ;;
esac
