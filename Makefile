# AI Racing Car Training - Makefile
# ================================

PYTHON = python
VENV = source venv/bin/activate &&

# Default values for headless training
CARS ?= 100
GENERATIONS ?= 500
TICKS ?= 1800
BEST ?= 10
MAP ?= DefaultSettings/DefaultRace.csv
BRAIN ?= AIbrain_best_sofar
SAVE ?= headless.npz
LOAD ?= 

.PHONY: help run headless headless-quick headless-long train-custom load clean list-brains list-saves list-maps

# Default target
help:
	@echo "AI Racing Car Training - Available Commands"
	@echo "============================================"
	@echo ""
	@echo "GUI Mode:"
	@echo "  make run              - Run the game with GUI"
	@echo ""
	@echo "Headless Training:"
	@echo "  make headless         - Run headless training (default settings)"
	@echo "  make headless-quick   - Quick test run (20 cars, 10 generations)"
	@echo "  make headless-long    - Long training run (100 cars, 500 generations)"
	@echo ""
	@echo "Custom Training:"
	@echo "  make train-custom CARS=50 GENERATIONS=100 TICKS=1200 BEST=10 SAVE=my_model.npz"
	@echo "  make load LOAD=existing.npz GENERATIONS=50 SAVE=improved.npz"
	@echo ""
	@echo "Utilities:"
	@echo "  make list-brains      - List available AI brain modules"
	@echo "  make list-saves       - List saved models"
	@echo "  make list-maps        - List available maps"
	@echo "  make clean            - Remove Python cache files"
	@echo ""
	@echo "Variables (override with VAR=value):"
	@echo "  CARS=$(CARS)  GENERATIONS=$(GENERATIONS)  TICKS=$(TICKS)  BEST=$(BEST)"
	@echo "  MAP=$(MAP)"
	@echo "  BRAIN=$(BRAIN)"
	@echo "  SAVE=$(SAVE)  LOAD=$(LOAD)"

# Run GUI mode
run:
	$(VENV) $(PYTHON) main.py

# Headless training with default settings
headless:
	$(VENV) $(PYTHON) headless_train.py \
		--map "$(MAP)" \
		--cars $(CARS) \
		--generations $(GENERATIONS) \
		--ticks $(TICKS) \
		--best $(BEST) \
		--brain "$(BRAIN)" \
		--save "$(SAVE)"

# Quick test run
headless-quick:
	$(VENV) $(PYTHON) headless_train.py \
		--cars 20 \
		--generations 10 \
		--ticks 600 \
		--best 5 \
		--save quick_test.npz

# Long training run
headless-long:
	$(VENV) $(PYTHON) headless_train.py \
		--cars 100 \
		--generations 500 \
		--ticks 1800 \
		--best 20 \
		--save long_training.npz

# Custom training with all parameters
train-custom:
	$(VENV) $(PYTHON) headless_train.py \
		--map "$(MAP)" \
		--cars $(CARS) \
		--generations $(GENERATIONS) \
		--ticks $(TICKS) \
		--best $(BEST) \
		--brain "$(BRAIN)" \
		--save "$(SAVE)"

# Load and continue training
load:
ifdef LOAD
	$(VENV) $(PYTHON) headless_train.py \
		--map "$(MAP)" \
		--cars $(CARS) \
		--generations $(GENERATIONS) \
		--ticks $(TICKS) \
		--best $(BEST) \
		--brain "$(BRAIN)" \
		--load "$(LOAD)" \
		--save "$(SAVE)"
else
	@echo "Error: LOAD variable required. Usage: make load LOAD=model.npz"
	@exit 1
endif

# List available brain modules
list-brains:
	@echo "Available AI Brain Modules:"
	@echo "---------------------------"
	@ls -1 AI_engines/AIbrain_*.py 2>/dev/null | sed 's/AI_engines\//  /g' | sed 's/\.py//g'

# List saved models
list-saves:
	@echo "Saved Models (UserData/SAVES/):"
	@echo "-------------------------------"
	@ls -lh UserData/SAVES/*.npz 2>/dev/null | awk '{print "  " $$9 " (" $$5 ")"}'  || echo "  No saved models found"

# List available maps
list-maps:
	@echo "Available Maps:"
	@echo "---------------"
	@echo "  Default:"
	@ls -1 DefaultSettings/*.csv 2>/dev/null | sed 's/^/    /g'
	@echo "  User Maps:"
	@ls -1 UserData/*.csv 2>/dev/null | sed 's/^/    /g' || echo "    No user maps found"

# Clean Python cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned Python cache files"
