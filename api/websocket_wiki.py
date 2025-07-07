import logging
import os
from typing import List, Optional, Dict, Any
from urllib.parse import unquote

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.azureai_client import AzureAIClient
from api.rag import RAG

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Models for the API
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    """
    Model for requesting a chat completion.
    """
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None, description="Optional path to a file in the repository to include in the prompt")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    type: Optional[str] = Field("github", description="Type of repository (e.g., 'github', 'gitlab', 'bitbucket')")

    # model parameters
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama, azure)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    language: Optional[str] = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")

async def handle_websocket_chat(websocket: WebSocket):
    """
    Handle WebSocket connection for chat completions.
    This replaces the HTTP streaming endpoint with a WebSocket connection.
    """
    await websocket.accept()

    try:
        # Receive and parse the request data
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

        # Check if request contains very large input
        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content, request.provider == "ollama")
                logger.info(f"Request size: {tokens} tokens")
                if tokens > 8000:
                    logger.warning(f"Request exceeds recommended token limit ({tokens} > 7500)")
                    input_too_large = True

        # Create a new RAG instance for this request
        try:
            request_rag = RAG(provider=request.provider, model=request.model)

            # Extract custom file filter parameters if provided
            excluded_dirs = None
            excluded_files = None
            included_dirs = None
            included_files = None

            if request.excluded_dirs:
                excluded_dirs = [unquote(dir_path) for dir_path in request.excluded_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom excluded directories: {excluded_dirs}")
            if request.excluded_files:
                excluded_files = [unquote(file_pattern) for file_pattern in request.excluded_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom excluded files: {excluded_files}")
            if request.included_dirs:
                included_dirs = [unquote(dir_path) for dir_path in request.included_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom included directories: {included_dirs}")
            if request.included_files:
                included_files = [unquote(file_pattern) for file_pattern in request.included_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom included files: {included_files}")

            request_rag.prepare_retriever(request.repo_url, request.type, request.token, excluded_dirs, excluded_files, included_dirs, included_files)
            logger.info(f"Retriever prepared for {request.repo_url}")
        except ValueError as e:
            if "No valid documents with embeddings found" in str(e):
                logger.error(f"No valid embeddings found: {str(e)}")
                await websocket.send_text("Error: No valid document embeddings found. This may be due to embedding size inconsistencies or API errors during document processing. Please try again or check your repository content.")
                await websocket.close()
                return
            else:
                logger.error(f"ValueError preparing retriever: {str(e)}")
                await websocket.send_text(f"Error preparing retriever: {str(e)}")
                await websocket.close()
                return
        except Exception as e:
            logger.error(f"Error preparing retriever: {str(e)}")
            # Check for specific embedding-related errors
            if "All embeddings should be of the same size" in str(e):
                await websocket.send_text("Error: Inconsistent embedding sizes detected. Some documents may have failed to embed properly. Please try again.")
            else:
                await websocket.send_text(f"Error preparing retriever: {str(e)}")
            await websocket.close()
            return

        # Validate request
        if not request.messages or len(request.messages) == 0:
            await websocket.send_text("Error: No messages provided")
            await websocket.close()
            return

        last_message = request.messages[-1]
        if last_message.role != "user":
            await websocket.send_text("Error: Last message must be from the user")
            await websocket.close()
            return

        # Process previous messages to build conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]

                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(
                        user_query=user_msg.content,
                        assistant_response=assistant_msg.content
                    )

        # Check if this is a Deep Research request
        is_deep_research = False
        research_iteration = 1

        # Process messages to detect Deep Research requests
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                # Only remove the tag from the last message
                if msg == request.messages[-1]:
                    # Remove the Deep Research tag
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        # Count research iterations if this is a Deep Research request
        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")

            # Check if this is a continuation request
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                # Find the original topic from the first user message
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break

                if original_topic:
                    # Replace the continuation message with the original topic
                    last_message.content = original_topic
                    logger.info(f"Using original topic for research: {original_topic}")

        # Get the query from the last message
        query = last_message.content

        # Only retrieve documents if input is not too large
        context_text = ""
        retrieved_documents = None

        if not input_too_large:
            try:
                # If filePath exists, modify the query for RAG to focus on the file
                rag_query = query
                if request.filePath:
                    # Use the file path to get relevant context about the file
                    rag_query = f"Contexts related to {request.filePath}"
                    logger.info(f"Modified RAG query to focus on file: {request.filePath}")

                # Try to perform RAG retrieval
                try:
                    # This will use the actual RAG implementation
                    retrieved_documents = request_rag(rag_query, language=request.language)

                    if retrieved_documents and retrieved_documents[0].documents:
                        # Format context for the prompt in a more structured way
                        documents = retrieved_documents[0].documents
                        logger.info(f"Retrieved {len(documents)} documents")

                        # Group documents by file path
                        docs_by_file = {}
                        for doc in documents:
                            file_path = doc.meta_data.get('file_path', 'unknown')
                            if file_path not in docs_by_file:
                                docs_by_file[file_path] = []
                            docs_by_file[file_path].append(doc)

                        # Format context text with file path grouping
                        context_parts = []
                        for file_path, docs in docs_by_file.items():
                            # Add file header with metadata
                            header = f"## File Path: {file_path}\n\n"
                            # Add document content
                            content = "\n\n".join([doc.text for doc in docs])

                            context_parts.append(f"{header}{content}")

                        # Join all parts with clear separation
                        context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                    else:
                        logger.warning("No documents retrieved from RAG")
                except Exception as e:
                    logger.error(f"Error in RAG retrieval: {str(e)}")
                    # Continue without RAG if there's an error

            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                context_text = ""

        # Get repository information
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # Determine repository type
        repo_type = request.type

        # Get language information
        language_code = request.language or configs["lang_config"]["default"]
        supported_langs = configs["lang_config"]["supported_languages"]
        language_name = supported_langs.get(language_code, "English")

        # Create system prompt
        if is_deep_research:
            # Check if this is the first iteration
            is_first_iteration = research_iteration == 1

            # Check if this is the final iteration
            is_final_iteration = research_iteration >= 5

            if is_first_iteration:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are conducting a multi-turn Deep Research process to thoroughly investigate the specific topic in the user's query.
Your goal is to provide detailed, focused information EXCLUSIVELY about this topic.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- This is the first iteration of a multi-turn research process focused EXCLUSIVELY on the user's query
- Start your response with "## Research Plan"
- Outline your approach to investigating this specific topic
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- Clearly state the specific topic you're researching to maintain focus throughout all iterations
- Identify the key aspects you'll need to research
- Provide initial findings based on the information available
- End with "## Next Steps" indicating what you'll investigate in the next iteration
- Do NOT provide a final conclusion yet - this is just the beginning of the research
- Do NOT include general repository information unless directly relevant to the query
- Focus EXCLUSIVELY on the specific topic being researched - do not drift to related topics
- Your research MUST directly address the original question
- NEVER respond with just "Continue the research" as an answer - always provide substantive research findings
- Remember that this topic will be maintained across all research iterations
</guidelines>

<style>
- Be concise but thorough
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
</style>"""
            elif is_final_iteration:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are in the final iteration of a Deep Research process focused EXCLUSIVELY on the latest user query.
Your goal is to synthesize all previous findings and provide a comprehensive conclusion that directly addresses this specific topic and ONLY this topic.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- This is the final iteration of the research process
- CAREFULLY review the entire conversation history to understand all previous findings
- Synthesize ALL findings from previous iterations into a comprehensive conclusion
- Start with "## Final Conclusion"
- Your conclusion MUST directly address the original question
- Stay STRICTLY focused on the specific topic - do not drift to related topics
- Include specific code references and implementation details related to the topic
- Highlight the most important discoveries and insights about this specific functionality
- Provide a complete and definitive answer to the original question
- Do NOT include general repository information unless directly relevant to the query
- Focus exclusively on the specific topic being researched
- NEVER respond with "Continue the research" as an answer - always provide a complete conclusion
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- Ensure your conclusion builds on and references key findings from previous iterations
</guidelines>

<style>
- Be concise but thorough
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
- Structure your response with clear headings
- End with actionable insights or recommendations when appropriate
</style>"""
            else:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are currently in iteration {research_iteration} of a Deep Research process focused EXCLUSIVELY on the latest user query.
Your goal is to build upon previous research iterations and go deeper into this specific topic without deviating from it.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- CAREFULLY review the conversation history to understand what has been researched so far
- Your response MUST build on previous research iterations - do not repeat information already covered
- Identify gaps or areas that need further exploration related to this specific topic
- Focus on one specific aspect that needs deeper investigation in this iteration
- Start your response with "## Research Update {research_iteration}"
- Clearly explain what you're investigating in this iteration
- Provide new insights that weren't covered in previous iterations
- If this is iteration 3, prepare for a final conclusion in the next iteration
- Do NOT include general repository information unless directly relevant to the query
- Focus EXCLUSIVELY on the specific topic being researched - do not drift to related topics
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- NEVER respond with just "Continue the research" as an answer - always provide substantive research findings
- Your research MUST directly address the original question
- Maintain continuity with previous research iterations - this is a continuous investigation
</guidelines>

<style>
- Be concise but thorough
- Focus on providing new information, not repeating what's already been covered
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
</style>"""
        else:
            system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You provide direct, concise, and accurate information about code repositories.
You NEVER start responses with markdown headers or code fences.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- Answer the user's question directly without ANY preamble or filler phrases
- DO NOT include any rationale, explanation, or extra comments.
- Strictly base answers ONLY on existing code or documents
- DO NOT speculate or invent citations.
- DO NOT start with preambles like "Okay, here's a breakdown" or "Here's an explanation"
- DO NOT start with markdown headers like "## Analysis of..." or any file path references
- DO NOT start with ```markdown code fences
- DO NOT end your response with ``` closing fences
- DO NOT start by repeating or acknowledging the question
- JUST START with the direct answer to the question

<example_of_what_not_to_do>
```markdown
## Analysis of `adalflow/adalflow/datasets/gsm8k.py`

This file contains...
```
</example_of_what_not_to_do>

- Format your response with proper markdown including headings, lists, and code blocks WITHIN your answer
- For code analysis, organize your response with clear sections
- Think step by step and structure your answer logically
- Start with the most relevant information that directly addresses the user's query
- Be precise and technical when discussing code
- Your response language should be in the same language as the user's query
</guidelines>

<style>
- Use concise, direct language
- Prioritize accuracy over verbosity
- When showing code, include line numbers and file paths when relevant
- Use markdown formatting to improve readability
</style>"""

        # Fetch file content if provided
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"Successfully retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")
                # Continue without file content if there's an error

        # Format conversation history
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"

        # Create the prompt with context
        prompt = f"/no_think {system_prompt}\n\n"

        if conversation_history:
            prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

        # Check if filePath is provided and fetch file content if it exists
        if file_content:
            # Add file content to the prompt after conversation history
            prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

        # Only include context if it's not empty
        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            # Add a note that we're skipping RAG due to size constraints or because it's the isolated API
            logger.info("No context available from RAG")
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        # Get the full model configuration from config.py
        model_config_dict = get_model_config(request.provider, request.model)

        resolved_client_class = model_config_dict["model_client"]
        # resolved_model_kwargs contains model name, temp, top_p, etc.
        # For Ollama, it's structured with "options" and "headers".
        resolved_model_kwargs_from_config = model_config_dict["model_kwargs"]

        llm_client_instance: Any = None
        # api_input_construction_kwargs will be passed to model.convert_inputs_to_api_kwargs()
        # It needs the model name and all other generation parameters.
        api_input_construction_kwargs = resolved_model_kwargs_from_config.copy()
        api_input_construction_kwargs["stream"] = True # Ensure streaming for websockets

        # Instantiate clients based on provider
        if request.provider == "vllm":
            logger.info(f"Using vLLM with model: {api_input_construction_kwargs.get('model')}")
            vllm_base_url = os.environ.get('VLLM_API_BASE_URL')
            vllm_api_key = os.environ.get('VLLM_API_KEY')
            if not vllm_base_url:
                # Send error to client and close WebSocket
                error_msg = "VLLM_API_BASE_URL is not set for vLLM provider."
                logger.error(error_msg)
                await websocket.send_text(f"Error: {error_msg}")
                await websocket.close()
                return
            # resolved_client_class is OpenAIClient
            llm_client_instance = resolved_client_class(base_url=vllm_base_url, api_key=vllm_api_key)

        elif request.provider == "ollama":
            logger.info(f"Using Ollama with model: {api_input_construction_kwargs.get('model')}")
            prompt += " /no_think" # Ollama specific prompt adjustment
            # resolved_client_class is OllamaClient
            llm_client_instance = resolved_client_class() # OllamaClient picks up OLLAMA_HOST from env
                                                       # and expects headers/options in model_kwargs for convert_inputs...
                                                       # api_input_construction_kwargs for ollama is already correctly structured by get_model_config

        elif request.provider == "openrouter":
            logger.info(f"Using OpenRouter with model: {api_input_construction_kwargs.get('model')}")
            # resolved_client_class is OpenRouterClient
            llm_client_instance = resolved_client_class(api_key=OPENROUTER_API_KEY) # Pass key if constructor takes it
            if not OPENROUTER_API_KEY:
                 logger.warning("OPENROUTER_API_KEY not configured, OpenRouter call might fail.")

        elif request.provider == "openai":
            logger.info(f"Using OpenAI protocol with model: {api_input_construction_kwargs.get('model')}")
            # resolved_client_class is OpenAIClient
            llm_client_instance = resolved_client_class(api_key=OPENAI_API_KEY) # Pass key if constructor takes it
            if not OPENAI_API_KEY:
                 logger.warning("OPENAI_API_KEY not configured, OpenAI call might fail.")

        elif request.provider == "azure":
            logger.info(f"Using Azure AI with model: {api_input_construction_kwargs.get('model')}")
            # resolved_client_class is AzureAIClient
            llm_client_instance = resolved_client_class() # AzureAIClient reads its specific env vars

        elif request.provider == "google":
            logger.info(f"Using Google Gemini with model: {api_input_construction_kwargs.get('model')}")
            # Google's genai client is instantiated and used differently
            llm_client_instance = genai.GenerativeModel(
                model_name=api_input_construction_kwargs.get("model"),
                generation_config={
                    "temperature": api_input_construction_kwargs.get("temperature"),
                    "top_p": api_input_construction_kwargs.get("top_p"),
                    "top_k": api_input_construction_kwargs.get("top_k")
                }
            )
            # api_kwargs_for_call is not used for Google in the same way
        else:
            logger.error(f"Unknown provider in websocket: {request.provider}")
            await websocket.send_text(f"Error: Unknown provider {request.provider}")
            await websocket.close()
            return

        api_kwargs_for_call = {}
        if request.provider != "google":
            try:
                api_kwargs_for_call = llm_client_instance.convert_inputs_to_api_kwargs(
                    input=prompt,
                    model_kwargs=api_input_construction_kwargs,
                    model_type=ModelType.LLM
                )
            except Exception as e_convert:
                logger.error(f"Error converting inputs for provider {request.provider}: {str(e_convert)}")
                await websocket.send_text(f"Error preparing request for provider {request.provider}: {str(e_convert)}")
                await websocket.close()
                return


        # Process the response based on the provider
        try:
            if request.provider in ["vllm", "ollama", "openrouter", "openai", "azure"]:
                response_stream = await llm_client_instance.acall(api_kwargs=api_kwargs_for_call, model_type=ModelType.LLM)

                if request.provider in ["vllm", "openai", "azure", "openrouter"]: # OpenAI-compatible streaming
                    async for chunk in response_stream:
                        choices = getattr(chunk, "choices", [])
                        if len(choices) > 0:
                            delta = getattr(choices[0], "delta", None)
                            if delta is not None:
                                text_content = getattr(delta, "content", None)
                                if text_content is not None:
                                    await websocket.send_text(text_content)
                elif request.provider == "ollama":
                    async for chunk in response_stream:
                        text = getattr(chunk, 'response', None)
                        if text is None:
                            message_attr = getattr(chunk, 'message', None)
                            if message_attr:
                                text = getattr(message_attr, 'content', None)
                            else:
                                text = getattr(chunk, 'text', None) or \
                                       (str(chunk) if not (hasattr(chunk, 'model') and hasattr(chunk, 'created_at')) else None)

                        if text: # Ensure text is not None and not just metadata
                           text = text.replace('<think>', '').replace('</think>', '') # Specific cleaning for Ollama
                           await websocket.send_text(text)
                await websocket.close()

            elif request.provider == "google":
                response_stream = llm_client_instance.generate_content(prompt, stream=True)
                for chunk in response_stream:
                    if hasattr(chunk, 'text'):
                        await websocket.send_text(chunk.text)
                await websocket.close()
            # No else needed here as unknown provider is handled above and returns

        except Exception as e_outer:
            logger.error(f"Error in streaming response for provider {request.provider}: {str(e_outer)}")
            error_message = str(e_outer)

            # Check for token limit errors
            if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                # If we hit a token limit error, try again without context
                logger.warning("Token limit exceeded, retrying without context")
                try:
                    # Create a simplified prompt without context
                    simplified_prompt = f"/no_think {system_prompt}\n\n"
                    if conversation_history:
                        simplified_prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

                    # Include file content in the fallback prompt if it was retrieved
                    if request.filePath and file_content:
                        simplified_prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

                    simplified_prompt += "<note>Answering without retrieval augmentation due to input size constraints.</note>\n\n"
                    simplified_prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

                    if request.provider == "ollama":
                        simplified_prompt += " /no_think"

                        # Create new api_kwargs with the simplified prompt
                        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                            input=simplified_prompt,
                            model_kwargs=model_kwargs,
                            model_type=ModelType.LLM
                        )

                        # Get the response using the simplified prompt
                        fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                        # Handle streaming fallback_response from Ollama
                        async for chunk in fallback_response:
                            text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                            if text and not text.startswith('model=') and not text.startswith('created_at='):
                                text = text.replace('<think>', '').replace('</think>', '')
                                await websocket.send_text(text)
                    elif request.provider == "openrouter":
                        try:
                            # Create new api_kwargs with the simplified prompt
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # Get the response using the simplified prompt
                            logger.info("Making fallback OpenRouter API call")
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # Handle streaming fallback_response from OpenRouter
                            async for chunk in fallback_response:
                                await websocket.send_text(chunk)
                        except Exception as e_fallback:
                            logger.error(f"Error with OpenRouter API fallback: {str(e_fallback)}")
                            error_msg = f"\nError with OpenRouter API fallback: {str(e_fallback)}\n\nPlease check that you have set the OPENROUTER_API_KEY environment variable with a valid API key."
                            await websocket.send_text(error_msg)
                    elif request.provider == "openai":
                        try:
                            # Create new api_kwargs with the simplified prompt
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # Get the response using the simplified prompt
                            logger.info("Making fallback Openai API call")
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # Handle streaming fallback_response from Openai
                            async for chunk in fallback_response:
                                text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                                await websocket.send_text(text)
                        except Exception as e_fallback:
                            logger.error(f"Error with Openai API fallback: {str(e_fallback)}")
                            error_msg = f"\nError with Openai API fallback: {str(e_fallback)}\n\nPlease check that you have set the OPENAI_API_KEY environment variable with a valid API key."
                            await websocket.send_text(error_msg)
                    elif request.provider == "azure":
                        try:
                            # Create new api_kwargs with the simplified prompt
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # Get the response using the simplified prompt
                            logger.info("Making fallback Azure AI API call")
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # Handle streaming fallback response from Azure AI
                            async for chunk in fallback_response:
                                choices = getattr(chunk, "choices", [])
                                if len(choices) > 0:
                                    delta = getattr(choices[0], "delta", None)
                                    if delta is not None:
                                        text = getattr(delta, "content", None)
                                        if text is not None:
                                            await websocket.send_text(text)
                        except Exception as e_fallback:
                            logger.error(f"Error with Azure AI API fallback: {str(e_fallback)}")
                            error_msg = f"\nError with Azure AI API fallback: {str(e_fallback)}\n\nPlease check that you have set the AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_VERSION environment variables with valid values."
                            await websocket.send_text(error_msg)
                    else:
                        # Initialize Google Generative AI model
                        model_config = get_model_config(request.provider, request.model)
                        fallback_model = genai.GenerativeModel(
                            model_name=model_config["model"],
                            generation_config={
                                "temperature": model_config["model_kwargs"].get("temperature", 0.7),
                                "top_p": model_config["model_kwargs"].get("top_p", 0.8),
                                "top_k": model_config["model_kwargs"].get("top_k", 40)
                            }
                        )

                        # Get streaming response using simplified prompt
                        fallback_response = fallback_model.generate_content(simplified_prompt, stream=True)
                        # Stream the fallback response
                        for chunk in fallback_response:
                            if hasattr(chunk, 'text'):
                                await websocket.send_text(chunk.text)
                except Exception as e2:
                    logger.error(f"Error in fallback streaming response: {str(e2)}")
                    await websocket.send_text(f"\nI apologize, but your request is too large for me to process. Please try a shorter query or break it into smaller parts.")
                    # Close the WebSocket connection after sending the error message
                    await websocket.close()
            else:
                # For other errors, return the error message
                await websocket.send_text(f"\nError: {error_message}")
                # Close the WebSocket connection after sending the error message
                await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        try:
            await websocket.send_text(f"Error: {str(e)}")
            await websocket.close()
        except:
            pass
