# Insurance Data Analysis System

A multi-agent system for analyzing insurance data using LangChain and LangGraph. The system processes Excel/HTML files containing insurance data and provides intelligent analysis through specialized agents.

## Features

- Document processing for Excel and HTML files
- Vector storage using PGVector
- Multi-agent analysis system with:
  - Trend Analysis Agent
  - Comparison Analysis Agent
  - Summary Agent
- Intelligent agent coordination through supervisor

## Prerequisites

- Python 3.10 or higher
- PostgreSQL with pgvector extension
- OpenAI API key
- Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies using Poetry (recommended):
```bash
poetry install
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
PGVECTOR_CONNECTION_STRING=postgresql+psycopg://user:pass@host:port/dbname
```

## Usage

Run the analysis through `main.py`:

```bash
python main.py --input-path /path/to/data --query "your analysis query"
```

### Arguments

- `--input-path`: Path to Excel/HTML file or directory containing files (required)
- `--query`: Analysis query to run (default: "Analyze documents and tell any interesting findings, insights, or highlights you can find.")
- `--debug`: Enable debug logging

### Example Queries

```bash
# Analyze trends
python main.py --input-path ./data --query "What are the main trends in insurance prices over time?"

# Compare periods
python main.py --input-path ./data --query "Compare insurance costs between 2018 and 2019"

# Get summary
python main.py --input-path ./data --query "Summarize the key findings from the data"
```

## Architecture

1. Document Processor: Loads and processes files
2. Vector Store: Stores document embeddings for efficient retrieval
3. Analysis System:
   - Supervisor coordinates multiple specialized agents
   - Each agent focuses on specific analysis types
   - Results are combined for comprehensive insights

## Troubleshooting

1. If you see connection errors, verify your PostgreSQL connection string
2. For memory issues with large files, try processing files individually
3. API errors usually indicate invalid or missing API keys