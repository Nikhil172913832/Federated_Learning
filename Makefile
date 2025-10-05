PYTHON=python
PIP=pip
PROJECT_DIR=quickstart-compose

.PHONY: install test run-local lint build simulate docker-setup docker-up docker-down docker-test docker-verify docker-state docker-tls docker-certs docker-clients docker-add-clients platform platform-stop platform-logs showcase

install:
	cd complete/fl && $(PIP) install -e .

test:
	cd complete/fl && pytest -q

simulate:
	cd complete/fl && flwr run .

run-local: simulate

lint:
	cd complete/fl && python -m flake8 . && python -m black --check .

build:
	cd complete/fl && python -m build

# Docker Compose commands
docker-setup:
	@echo "Setting up Docker Compose environment..."
	git clone --depth=1 --branch v1.22.0 https://github.com/adap/flower.git _tmp \
	&& mv _tmp/framework/docker/complete . \
	&& rm -rf _tmp
	@echo "Creating Flower project..."
	flwr new $(PROJECT_DIR) --framework PyTorch --username flower
	@echo "Setting PROJECT_DIR environment variable..."
	@echo "export PROJECT_DIR=$(PROJECT_DIR)" >> .env

docker-up:
	@echo "Starting Docker Compose services..."
	export PROJECT_DIR=$(PROJECT_DIR) && docker compose up --build -d

docker-down:
	@echo "Stopping Docker Compose services..."
	docker compose down

docker-test:
	@echo "Testing Docker Compose setup..."
	export PROJECT_DIR=$(PROJECT_DIR) && docker compose up --build -d
	@echo "Waiting for services to be ready..."
	sleep 30
	@echo "Running Flower project..."
	flwr run $(PROJECT_DIR) local-deployment --stream
	@echo "Stopping services..."
	docker compose down

docker-logs:
	docker compose logs -f

docker-verify:
	@echo "Verifying Docker Compose setup..."
	./test-docker-compose.sh

docker-state:
	@echo "Starting services with state persistence..."
	cd complete && export PROJECT_DIR=$(PROJECT_DIR) && docker compose -f compose.yml -f with-state.yml up --build -d

docker-tls:
	@echo "Starting services with TLS encryption..."
	cd complete && export PROJECT_DIR=$(PROJECT_DIR) && docker compose -f compose.yml -f with-tls.yml up --build -d

docker-certs:
	@echo "Generating TLS certificates..."
	cd complete && docker compose -f certs.yml run --rm --build gen-certs

docker-combined:
	@echo "Starting services with state persistence and TLS..."
	cd complete && export PROJECT_DIR=$(PROJECT_DIR) && docker compose -f compose.yml -f with-tls.yml -f with-state.yml up --build -d

docker-clients:
	@echo "Running federated learning with Docker container clients..."
	./run-docker-clients.sh

docker-add-clients:
	@echo "Adding more client containers to the setup..."
	./add-more-clients.sh

# Platform commands
platform:
	@echo "Launching complete federated learning platform with UI..."
	./launch-platform.sh

platform-stop:
	@echo "Stopping federated learning platform..."
	cd complete && docker compose -f compose-with-ui.yml down

platform-logs:
	@echo "Viewing platform logs..."
	cd complete && docker compose -f compose-with-ui.yml logs -f

showcase:
	@echo "Starting platform showcase..."
	./showcase-platform.sh


