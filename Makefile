SHELL = /bin/bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

help:
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean:
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

setup:
	pip install pre-commit==3.8.0
	pre-commit install

format:
	pre-commit run -a

build-package:
	poetry build

publish-test:
	poetry publish -r testpypi

publish:
	poetry publish

version-bump:
	@if [ "$(BUMP)" = "" ]; then \
		echo "Please specify BUMP as patch, minor, or major"; \
		exit 1; \
	fi
	poetry version $(BUMP)
	@echo "Version bumped to $$(poetry version -s)"
	@echo "Don't forget to commit the changes and create a new tag!"

release: clean build-package publish

debug:
	docker-compose build --build-arg INSTALL_DEV=true
	docker-compose up --no-build -d
	docker-compose exec dev bash
	docker image prune -f || true

shutdown:
	docker-compose down --remove-orphans

run-ci:
	docker-compose -f docker-compose.ci.yaml build
	docker-compose -f docker-compose.ci.yaml up --no-build -d
	docker-compose -f docker-compose.ci.yaml exec -T ci bash scripts/run_ci.sh

logs:
	@if [ -z "$(CONTAINER)" ]; then \
		echo "Showing logs for all containers..."; \
		docker-compose logs -f; \
	else \
		echo "Showing logs for container: $(CONTAINER)"; \
		docker-compose logs -f $(CONTAINER); \
	fi
