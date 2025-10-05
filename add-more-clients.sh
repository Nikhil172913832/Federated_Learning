#!/bin/bash

# Script to add more client containers to the federated learning setup
# This follows Step 7 from the official Flower documentation

set -e

echo "🔧 Adding More Client Containers to Federated Learning Setup"
echo "============================================================"

# Check if we're in the right directory
if [ ! -d "complete" ]; then
    echo "❌ Error: complete directory not found. Run setup-docker-compose.sh first"
    exit 1
fi

cd complete

echo "📋 Current setup:"
echo "  - 1 SuperLink (coordination)"
echo "  - 1 SuperExec-ServerApp (server)"
echo "  - 2 SuperNodes (client nodes)"
echo "  - 2 SuperExec-ClientApps (client containers)"
echo ""

echo "🔧 Adding SuperNode-3 and SuperExec-ClientApp-3..."
echo ""

# Create a backup of the original compose.yml
cp compose.yml compose.yml.backup
echo "✅ Created backup: compose.yml.backup"

# Uncomment the SuperNode-3 section
echo "🔧 Enabling SuperNode-3..."
sed -i 's/^  # supernode-3:/  supernode-3:/' compose.yml
sed -i 's/^  #   image:/    image:/' compose.yml
sed -i 's/^  #   command:/    command:/' compose.yml
sed -i 's/^  #     - --insecure/      - --insecure/' compose.yml
sed -i 's/^  #     - --superlink/      - --superlink/' compose.yml
sed -i 's/^  #     - superlink:9092/      - superlink:9092/' compose.yml
sed -i 's/^  #     - --clientappio-api-address/      - --clientappio-api-address/' compose.yml
sed -i 's/^  #     - 0.0.0.0:9096/      - 0.0.0.0:9096/' compose.yml
sed -i 's/^  #     - --isolation/      - --isolation/' compose.yml
sed -i 's/^  #     - process/      - process/' compose.yml
sed -i 's/^  #     - --node-config/      - --node-config/' compose.yml
sed -i 's/^  #     - "partition-id=1 num-partitions=2"/      - "partition-id=2 num-partitions=3"/' compose.yml
sed -i 's/^  #   depends_on:/    depends_on:/' compose.yml
sed -i 's/^  #     - superlink/      - superlink/' compose.yml

# Uncomment the SuperExec-ClientApp-3 section
echo "🔧 Enabling SuperExec-ClientApp-3..."
sed -i 's/^  # superexec-clientapp-3:/  superexec-clientapp-3:/' compose.yml
sed -i 's/^  #   build:/    build:/' compose.yml
sed -i 's/^  #     context:/      context:/' compose.yml
sed -i 's/^  #     dockerfile_inline:/      dockerfile_inline:/' compose.yml
sed -i 's/^  #       FROM flwr\/superexec:/        FROM flwr\/superexec:/' compose.yml
sed -i 's/^  #       USER root/        USER root/' compose.yml
sed -i 's/^  #       RUN apt-get update/        RUN apt-get update/' compose.yml
sed -i 's/^  #           && apt-get -y --no-install-recommends install/            && apt-get -y --no-install-recommends install/' compose.yml
sed -i 's/^  #           build-essential/            build-essential/' compose.yml
sed -i 's/^  #           && rm -rf \/var\/lib\/apt\/lists\*//' compose.yml
sed -i 's/^  #       USER app/        USER app/' compose.yml
sed -i 's/^  #       WORKDIR \/app/        WORKDIR \/app/' compose.yml
sed -i 's/^  #       COPY --chown=app:app pyproject.toml ./        COPY --chown=app:app complete\/fl\/ \/app\//' compose.yml
sed -i 's/^  #       RUN sed -i/        RUN sed -i/' compose.yml
sed -i 's/^  #         && python -m pip install -U --no-cache-dir ./          && python -m pip install -e . --no-cache-dir/' compose.yml
sed -i 's/^  #       ENTRYPOINT \["flwr-superexec"\]/        ENTRYPOINT ["flower-superexec"]/' compose.yml
sed -i 's/^  #   command:/    command:/' compose.yml
sed -i 's/^  #     - --insecure/      - --insecure/' compose.yml
sed -i 's/^  #     - --plugin-type/      - --plugin-type/' compose.yml
sed -i 's/^  #     - clientapp/      - clientapp/' compose.yml
sed -i 's/^  #     - --appio-api-address/      - --appio-api-address/' compose.yml
sed -i 's/^  #     - supernode-3:9096/      - supernode-3:9096/' compose.yml
sed -i 's/^  #   deploy:/    deploy:/' compose.yml
sed -i 's/^  #     resources:/      resources:/' compose.yml
sed -i 's/^  #       limits:/        limits:/' compose.yml
sed -i 's/^  #         cpus:/          cpus:/' compose.yml
sed -i 's/^  #   stop_signal:/    stop_signal:/' compose.yml
sed -i 's/^  #   depends_on:/    depends_on:/' compose.yml
sed -i 's/^  #     - supernode-3/      - supernode-3/' compose.yml

echo "✅ Successfully enabled SuperNode-3 and SuperExec-ClientApp-3"
echo ""

echo "📋 Updated setup:"
echo "  - 1 SuperLink (coordination)"
echo "  - 1 SuperExec-ServerApp (server)"
echo "  - 3 SuperNodes (client nodes) - ports 9094, 9095, 9096"
echo "  - 3 SuperExec-ClientApps (client containers)"
echo ""

echo "🔄 Restarting services with new client containers..."
if [ -z "$PROJECT_DIR" ]; then
    export PROJECT_DIR="quickstart-compose"
fi

docker compose down
docker compose up --build -d

echo ""
echo "⏳ Waiting for new services to be ready..."
sleep 30

echo ""
echo "📊 Checking all services..."
docker compose ps

echo ""
echo "🎯 Ready to run federated learning with 3 client containers!"
echo ""
echo "To run federated learning:"
echo "  cd .."
echo "  flwr run $PROJECT_DIR local-deployment --stream"
echo ""
echo "To view logs from the new client:"
echo "  docker compose logs superexec-clientapp-3"
echo ""
echo "To revert changes:"
echo "  cp compose.yml.backup compose.yml"
echo "  docker compose down && docker compose up --build -d"
