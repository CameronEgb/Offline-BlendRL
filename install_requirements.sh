#!/bin/bash
# Install requirements and setup virtual environment

echo "Creating virtual environment..."
python3.11 -m venv venv

echo "activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing base requirements from requirements.txt..."
pip install -r requirements.txt

echo "Installing internal packages (neumann, nsfr, nudge)..."
pip install -e neumann/
pip install -e nsfr/
pip install -e nudge/

echo "Setup complete. You can now run experiments."
