# Physician Note Taker

A  pipeline to extract, analyze, and generate structured medical notes from doctor-patient conversations.

## Features
- ✅ Medical entity extraction (symptoms, diagnosis, treatment)
- ✅ Speaker-aware parsing (Doctor vs Patient)
- ✅ BioBERT & SciSpaCy support
- ✅ Patient sentiment & intent analysis
- ✅ SOAP note generation with LLM support and fallback

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt

