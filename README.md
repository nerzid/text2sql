# Text2SQL Application

A FastAPI application that converts natural language to SQL queries and detects AI-generated text.

## Features

- Text to SQL conversion
- AI-generated text detection
- Feedback collection for model improvement

## Preface

My laptop is on the low-end in terms of GPU power (VRAM, operation speed, etc.), therefore, the solutions I provide are meant to be just prototypes that work, but not necessarily perform good enough for production.
In addition, since I don’t know the tech stack used in your ecosystem (e.g., cloud provider, LLM provider, version control system, etc.), I made the following assumptions:

- You use kubernetes for docker container orchestration, therefore I didn’t add any docker-compose files.
- The project has access to the internet and has enough storage capacity to download and store files.
- You use Redis for caching/queueing and PostgreSQL for storing data. The /feedback endpoint sends data to redis which is assumed to be handled by you and sent to PostgreSQL. Since I don’t have this setup on my laptop, I couldn't test if this pipeline works. However, it is the ideal setup that considers scaling as well.
- The docker container or the project has access to GPU to run models. Ideally, there should be REST API endpoints for each model for serving LM services. For example, for LLMs, the best library for this (also the one I use in my workplace) is vLLM.
- I used LM studio to serve qwen3-4B for /preprocess endpoint. If you want to use another provider (e.g., ollama), you can do so by changing `LLM_PROVIDER=“ollama”` in the .env file in the root directory. Don’t forget to export env file, otherwise the project will use the default provider, i.e., lmstudio (More on providers can be found here)
- I used ChromaDB to vectorize headers from the WikiSQL dataset and keep them in the local environment as sqlite. Ideally, if you use kubernetes, ChromaDB should be centralized and have REST API access. Otherwise, ChromaDB should be kept in the server with a volume written in the docker-compose file.

## Best Practices Followed

- I used Hugging Face as a version control system for the models, which can be accessed [on my Hugging Face page](https://huggingface.co/nerzid)
- I used FastAPI to serve the endpoints asynchronously  
- I followed FastAPI’s best practices on organizing the project folders which allows for better scalability with its modular approach  
- I added dockerization option for the project which can be connected to kubernetes  
- I added CI/CD (see .github\workflows\ci.yml)  
- I used Redis enqueue feedbacks which prevents the services from slowing down or freezing when there is high usage  
- I prepares unit tests for the most important functions  
- I added docstring for all functions  
- I added exception handling rigorously  
- I did a literature survey for the given tasks before attempting to implement my own, and used the existing approaches as base lines  

## Project Structure

project/  
├── app/  
│ ├── core/  
│ │ ├── config.py # Application settings  
│ │ ├── constants.py # Constants like prompt templates  
│ │ └── dependencies.py # Dependency injection (Redis)  
│ ├── chromadb/  
│ │ └── client.py # ChromaDB client for vector storage  
│ ├── llm/  
│ │ ├── config.py # LLM configuration  
│ │ ├── predictors.py # DSPy predictors  
│ │ └── signatures.py # DSPy model signatures  
│ ├── schemas/  
│ │ └── base_models.py # Pydantic models for API  
│ ├── services/  
│ │ ├── ai_detector.py # AI-generated text detection  
│ │ └── text_to_sql.py # Text-to-SQL conversion  
│ └── main.py # FastAPI application  
├── notebooks/  
│ ├── approach1.ipynb # AI detection approach 1  
│ └── approach2.ipynb # AI detection approach 2  
├── tests/  
│ └── test_main.py # API tests  
├── improve_lm_performance/  
│ └── Dockerfile # For model retraining  
├── static/ # Static files  
├── .github/workflows/  
│ └── ci.yml # CI pipeline  
├── Dockerfile # Main application Dockerfile  
├── requirements.txt # Python dependencies  
├── run.py # Application entry point  
└── README.md # Project documentation  

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

## Acknowledgements

- [WikiSQL dataset](https://github.com/salesforce/WikiSQL) by Salesforce Research was used for training and evaluating the text-to-SQL models.
