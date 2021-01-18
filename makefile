VENV_NAME=.venv
PYTHON=${VENV_NAME}/bin/python3

new:
	brew install python
	brew install mongodb-community
	brew install mongodb-database-tools
	make venv

venv:
	rm -rf $(VENV_NAME)
	python3 -m venv $(VENV_NAME)
	make update

update:
	brew upgrade python
	brew upgrade mongodb-community
	brew upgrade mongodb-database-tools
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -U -r requirements.txt

db_recover:
	@echo "Recover MongoDB"
	mongod --dbpath ./db --repair --directoryperdb

db_stop:
	@echo "Stop MongoDB"
	pkill -x mongod

test:
	pytest poptimizer -v --cov=poptimizer --cov-report=term-missing --cov-report=xml --setup-show

lint:
	mypy poptimizer
	flake8 poptimizer
