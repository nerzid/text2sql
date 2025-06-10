# Text2SQL Application

A FastAPI application that converts natural language to SQL queries and detects AI-generated text.

## Features

- Text to SQL conversion
- AI-generated text detection
- Feedback collection for model improvement

## How to Run

For is_ai_generated endpoint:

```bash
curl -X POST http://localhost:8000/is_ai_generated -H "Content-Type: application/json" -d "{\"text\": \"What is the format for South Australia?\"}"
```

For text2sql endpoint:

```bash
curl -X POST http://localhost:8000/text2sql -H "Content-Type: application/json" -d "{\"question\": \"Tell me what the notes are for South Australia\"}"
```

For preprocess_text endpoint (needs LLM service setup):

```bash
curl -X POST http://localhost:8000/preprocess_text -H "Content-Type: application/json" -d "{\"text\": \"who has the most goal?\"}"
```

## Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional)

### Option 1: Local Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nerzid/text2sql.git
   cd text2sql
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a `.env` file):

   ```
   SERVICE_HOST=localhost
   SERVICE_PORT=8000
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0
   LLM_PROVIDER=lm_studio
   LLM_MODEL=qwen3-4b
   LLM_API_URL=http://localhost:1234/v1
   DATA_PATH=data
   ```

5. Run the application:

   ```bash
   python run.py
   ```

6. Access the API at <http://localhost:8000>

### Option 2: Using Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/nerzid/text2sql.git
   cd text2sql
   ```

2. Run the Docker container:

   ```bash
   docker build -t text2sql-app .
   docker run -p 8000:8000 text2sql-app
   ```

3. Access the API at <http://localhost:8000>

### Option 3: Using Docker Image from GitHub Container Registry

```bash
docker pull ghcr.io/nerzid/text2sql:latest
docker run -p 8000:8000 ghcr.io/nerzid/text2sql:latest
```

## Development

### Setup

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python run.py
   ```

## API Endpoints

- `POST /is_ai_generated`: Detect if text is AI-generated
- `POST /text2sql`: Convert natural language to SQL
- `POST /preprocess`: Preprocess text for SQL conversion
- `POST /feedback`: Submit feedback for model improvement
- `GET /health`: Health check endpoint

## Testing

### For windows

```bash
 set PYTHONPATH=. && pytest -v
```

### For Linux/MacOS

```bash
 PYTHONPATH=. pytest -v
```
