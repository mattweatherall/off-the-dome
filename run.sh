#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# Create directories if they don't exist
mkdir -p data/sample_papers db

# Download sample papers if the directory is empty
if [ -z "$(ls -A data/sample_papers)" ]; then
  echo "Downloading sample papers..."
  python download_samples.py
fi

# Run the Streamlit app
streamlit run app.py
