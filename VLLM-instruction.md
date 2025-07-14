# Using DeepWiki with vLLM

DeepWiki can integrate with [vLLM](https://github.com/vllm-project/vllm) instances that expose an OpenAI-compatible API endpoint. This allows you to leverage vLLM's fast inference capabilities for generating documentation.

## Prerequisites

1.  **Running vLLM Instance:** You need to have a vLLM server running and accessible. It should be configured to serve one or more models via an OpenAI-compatible API.
    *   Example vLLM server startup command (this is a basic example, consult vLLM documentation for more advanced configurations):
        ```bash
        python -m vllm.entrypoints.openai.api_server \
            --model <your_model_name_or_path> \
            --host 0.0.0.0 \
            --port 8000
        # Add --api-key YOUR_API_KEY if you want to secure your endpoint
        ```
    *   `<your_model_name_or_path>` could be something like `mistralai/Mistral-7B-Instruct-v0.1`.

2.  **Network Accessibility:** The machine running DeepWiki must be able to make HTTP requests to your vLLM server's API endpoint.

## Configuration

You need to configure DeepWiki using environment variables to connect to your vLLM instance. These can be set in a `.env` file in the root of the DeepWiki project.

### Environment Variables

*   `VLLM_API_BASE_URL`: **(Required)** The full base URL of your vLLM server's OpenAI-compatible API. This typically includes `/v1`.
    *   Example: `VLLM_API_BASE_URL=http://localhost:8000/v1`
    *   Example (if served on a different machine/port): `VLLM_API_BASE_URL=http://192.168.1.100:8000/v1`
    *   Example (with HTTPS and custom domain): `VLLM_API_BASE_URL=https://your-vllm.example.com/v1`

*   `VLLM_MODEL_NAME`: (Recommended) The model identifier that DeepWiki should request from your vLLM server. This should match a model that your vLLM instance is serving. If your vLLM server is serving multiple models, this tells DeepWiki which one to use. If the vLLM server only serves one model, this name might still be used in API calls depending on the server's strictness.
    *   Example: `VLLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1`

*   `VLLM_API_KEY`: (Optional) If your vLLM API endpoint is protected by an API key, set this variable. If your endpoint does not require an API key, you can omit this variable or leave it blank.
    *   Example: `VLLM_API_KEY=your-secret-vllm-api-key`

### Example `.env` configuration:

```env
# --- vLLM Configuration ---
VLLM_API_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
# VLLM_API_KEY=your-secret-vllm-api-key # Uncomment and set if your vLLM requires an API key

# --- Other DeepWiki configurations (PORT, Ollama, OpenAI, etc.) ---
# PORT=8001
# OPENAI_API_KEY=sk-xxxxxxxxxx
# OLLAMA_HOST=http://localhost:11434
```

### Modifying `generator.json`

To use the vLLM provider, you must add its configuration to the `providers` section in `api/config/generator.json`.

```json
{
  "default_provider": "google", // Or your preferred default
  "providers": {
    // ... other providers ...
    "vllm": {
      "client_class": "OpenAIClient",
      "default_model": "${VLLM_MODEL_NAME:-default-vllm-model}", // Uses env var or a fallback
      "supportsCustomModel": true,
      "models": {
        // This entry allows setting specific parameters for the model named by VLLM_MODEL_NAME
        // The key here should match what VLLM_MODEL_NAME provides.
        "${VLLM_MODEL_NAME:-default-vllm-model}": {
          "max_context_tokens": 40960, // Example for a model with a ~40k context window
          "temperature": 0.7,
          "top_p": 0.8
        }
      }
    }
    // ... other providers ...
  }
}
```

## Usage

1.  Ensure your vLLM server is running and configured as described above.
2.  Set the environment variables in your `.env` file.
3.  Add the `vllm` provider block to `api/config/generator.json`.
4.  Start the DeepWiki backend.
5.  When using the DeepWiki interface or API, select "vllm" as the provider and choose the appropriate model (which should correspond to your `VLLM_MODEL_NAME`).

DeepWiki will then direct generation requests to your vLLM instance.
