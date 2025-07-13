# VLLM Wiki Generator

This project generates a wiki for a given codebase using a VLLM model. It analyzes the code and creates a structured wiki with summaries and code snippets for each file.

## Features

-   **Code Analysis:** Analyzes source code files to generate summaries and documentation.
-   **VLLM Integration:** Uses a VLLM model to generate human-like summaries of the code.
-   **Text Chunking:** Handles large files by splitting them into smaller, manageable chunks.
-   **Configurable:** Key parameters, such as the VLLM model and chunking settings, are configurable.
-   **FastAPI Backend:** Provides a simple and efficient FastAPI backend for generating wikis.
-   **Zip Export:** Exports the generated wiki as a convenient zip file.

## Requirements

-   Python 3.8+
-   VLLM
-   FastAPI
-   Uvicorn
-   Tiktoken
-   python-dotenv

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/vllm-wiki-generator.git
    cd vllm-wiki-generator
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up the configuration:**

    Create a `.env` file in the root of the project and add the following variables:

    ```
    VLLM_MODEL=facebook/opt-125m
    CHUNK_SIZE=2048
    OVERLAP=256
    ```

    You can change the `VLLM_MODEL` to any other model supported by VLLM.

## Usage

1.  **Start the FastAPI server:**

    ```bash
    uvicorn main:app --reload
    ```

2.  **Send a request to the `/generate-wiki/` endpoint:**

    You can use a tool like `curl` or Postman to send a POST request to the `/generate-wiki/` endpoint. The request body should be a JSON object with the path to the local repository.

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"path": "/path/to/your/repo"}' http://127.0.0.1:8000/generate-wiki/ --output wiki.zip
    ```

    This will generate a `wiki.zip` file containing the generated wiki.

## Docker

To run the application in a Docker container, follow these steps:

1.  **Build the Docker image:**

    ```bash
    docker build -t vllm-wiki-generator .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -p 8000:8000 vllm-wiki-generator
    ```

    The application will be available at `http://localhost:8000`.

## Project Structure

```
vllm-wiki-generator/
├── .env
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
└── src/
    ├── file_processor.py
    ├── processed_projects.py
    ├── rag_pipeline.py
    ├── text_chunker.py
    ├── vllm_client.py
    └── wiki_generator.py
```

-   **`.env`**: Configuration file for the project.
    -   `VLLM_MODEL`: The name of the VLLM model to use.
    -   `VLLM_API_BASE`: The base URL of the VLLM API.
    -   `VLLM_API_KEY`: Your VLLM API key.
    -   `EMBEDDING_MODEL`: The name of the embedding model to use.
    -   `JINA_API_KEY`: Your Jina API key.
    -   `CHUNK_SIZE`: The chunk size for text splitting.
    -   `OVERLAP`: The overlap for text splitting.
-   **`Dockerfile`**: Dockerfile for building the application image.
-   **`main.py`**: FastAPI application for generating the wiki.
-   **`README.md`**: This file.
-   **`requirements.txt`**: Python dependencies for the project.
-   **`src/`**: Source code for the project.
    -   **`file_processor.py`**: Module for finding and reading code files.
    -   **`processed_projects.py`**: Module for managing processed projects.
    -   **`rag_pipeline.py`**: Module for the RAG pipeline.
    -   **`text_chunker.py`**: Module for splitting text into chunks.
    -   **`vllm_client.py`**: Module for interacting with the VLLM API.
    -   **`wiki_generator.py`**: Module for generating the wiki.
```
