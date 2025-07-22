# DeepWiki

DeepWiki is a powerful tool that takes a repository as input and generates a structured wiki using LLMs. It supports various LLM providers and can handle large repositories with ease.

## Features

*   **Structured Wiki Generation**: Automatically generates a structured wiki from a code repository.
*   **Multiple LLM Providers**: Supports Google, OpenAI, Azure AI, Ollama, and custom vLLM services.
*   **Large Repository Handling**: Efficiently processes large repositories without token limitations.
*   **Configurable and Extensible**: All settings, keys, and endpoints are fully configurable via `.env` files.
*   **Secure and Private**: Supports secured, remote vLLM and embedding services.

## Getting Started

### Prerequisites

*   Python 3.9 or higher
*   Node.js and npm

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/example/deepwiki.git
    cd deepwiki
    ```
2.  Install the backend dependencies:
    ```bash
    pip install -r api/requirements.txt
    ```
3.  Install the frontend dependencies:
    ```bash
    npm install
    ```

### Configuration

1.  Create a `.env` file in the root directory of the project.
2.  Add the following environment variables to the `.env` file:

    ```env
    # Server Configuration
    PORT=8001
    SERVER_BASE_URL=http://127.0.0.1:8001
    PYTHON_BACKEND_HOST=http://127.0.0.1:8001

    # LLM Provider Configuration
    # Uncomment the provider you want to use and fill in the required values.

    # Google (default)
    # GOOGLE_API_KEY=your_google_api_key

    # OpenAI
    # OPENAI_API_KEY=your_openai_api_key
    # OPENAI_API_BASE_URL=https://api.openai.com/v1

    # Azure AI
    # AZURE_API_KEY=your_azure_api_key
    # AZURE_API_BASE=your_azure_api_base

    # Ollama
    # OLLAMA_API_BASE_URL=http://localhost:11434

    # Custom vLLM
    # VLLM_API_BASE_URL=https://myvllm.com/qwen3-14b/v1
    # VLLM_API_KEY=your_vllm_api_key
    # VLLM_MODEL_NAME=/app/models/Qwen3-14B-FP8
    ```

### Running the Application

1.  Start the backend server:
    ```bash
    python api/main.py
    ```
2.  Start the frontend development server:
    ```bash
    npm run dev
    ```
3.  Open your browser and navigate to `http://localhost:3000`.

## How It Works

### Large Repository Handling

The system handles large repositories by splitting large files into smaller chunks. This is done in the `api/data_pipeline.py` file, where the `read_all_documents` function reads all the files in the repository and the `chunk_document` function splits the large files into smaller chunks. The size of the chunks is determined by the `MAX_EMBEDDING_TOKENS` constant, which is set to 8192 tokens.

### Switching Between LLM Providers

To switch between LLM providers, you need to set the appropriate environment variables in the `.env` file. The system will automatically detect the configured provider and use the corresponding client. For example, to use the custom vLLM service, you would set the `VLLM_API_BASE_URL`, `VLLM_API_KEY`, and `VLLM_MODEL_NAME` environment variables.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
