#!/bin/bash
set -e

# Print GPU information
echo "=== GPU Information ==="
nvidia-smi

# Check Python environment
echo "=== Python Environment ==="
python --version
pip --version

# Check CUDA availability
echo "=== CUDA Check ==="
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# Setup logging directory
mkdir -p /app/logs

# Initialize database if needed
echo "=== Database Setup ==="
if [ ! -f "/app/.db_initialized" ]; then
    echo "Initializing database..."
    python -c "
import time
import psycopg2
import os

# Wait for postgres to be ready
max_retries = 30
for i in range(max_retries):
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            user=os.getenv('POSTGRES_USER', 'consciousness'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('POSTGRES_DB', 'consciousness_dev')
        )
        conn.close()
        print('Database connection successful!')
        break
    except psycopg2.OperationalError:
        print(f'Waiting for database... ({i+1}/{max_retries})')
        time.sleep(2)
else:
    print('Failed to connect to database after 30 attempts')
    exit(1)
"
    
    # Run migrations if alembic is configured
    if [ -f "/app/alembic.ini" ]; then
        echo "Running database migrations..."
        alembic upgrade head
    fi
    
    touch /app/.db_initialized
    echo "Database initialization complete"
fi

# Start the application
echo "=== Starting Application ==="
exec "$@" 