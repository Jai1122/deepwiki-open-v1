import logging
from typing import List, Optional, Dict, Any
from urllib.parse import unquote

from api.rag import RAG
from api.data_pipeline import get_file_content

logger = logging.getLogger(__name__)

def prepare_rag_retriever(repo_url: str, repo_type: str, token: Optional[str],
                          excluded_dirs: Optional[str], excluded_files: Optional[str],
                          included_dirs: Optional[str], included_files: Optional[str],
                          provider: str, model: Optional[str]) -> RAG:
    """
    Prepare the RAG retriever for a repository.
    """
    try:
        request_rag = RAG(provider=provider, model=model)

        # Extract custom file filter parameters if provided
        _excluded_dirs = None
        _excluded_files = None
        _included_dirs = None
        _included_files = None

        if excluded_dirs:
            _excluded_dirs = [unquote(dir_path) for dir_path in excluded_dirs.split('\\n') if dir_path.strip()]
            logger.info(f"Using custom excluded directories: {_excluded_dirs}")
        if excluded_files:
            _excluded_files = [unquote(file_pattern) for file_pattern in excluded_files.split('\\n') if file_pattern.strip()]
            logger.info(f"Using custom excluded files: {_excluded_files}")
        if included_dirs:
            _included_dirs = [unquote(dir_path) for dir_path in included_dirs.split('\\n') if dir_path.strip()]
            logger.info(f"Using custom included directories: {_included_dirs}")
        if included_files:
            _included_files = [unquote(file_pattern) for file_pattern in included_files.split('\\n') if file_pattern.strip()]
            logger.info(f"Using custom included files: {_included_files}")

        request_rag.prepare_retriever(repo_url, repo_type, token, _excluded_dirs, _excluded_files, _included_dirs, _included_files)
        logger.info(f"Retriever prepared for {repo_url}")
        return request_rag
    except ValueError as e:
        if "No valid documents with embeddings found" in str(e):
            logger.error(f"No valid embeddings found: {str(e)}")
            raise
        else:
            logger.error(f"ValueError preparing retriever: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Error preparing retriever: {str(e)}")
        if "All embeddings should be of the same size" in str(e):
            raise
        else:
            raise

def get_rag_context(rag: RAG, query: str, file_path: Optional[str], language: str) -> str:
    """
    Get the context from the RAG retriever.
    """
    context_text = ""
    try:
        # If filePath exists, modify the query for RAG to focus on the file
        rag_query = query
        if file_path:
            # Use the file path to get relevant context about the file
            rag_query = f"Contexts related to {file_path}"
            logger.info(f"Modified RAG query to focus on file: {file_path}")

        # Try to perform RAG retrieval
        try:
            # This will use the actual RAG implementation
            retrieved_documents = rag(rag_query, language=language)

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
                    header = f"## File Path: {file_path}\\n\\n"
                    # Add document content
                    content = "\\n\\n".join([doc.text for doc in docs])

                    context_parts.append(f"{header}{content}")

                # Join all parts with clear separation
                context_text = "\\n\\n" + "-" * 10 + "\\n\\n".join(context_parts)
            else:
                logger.warning("No documents retrieved from RAG")
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {str(e)}")
            # Continue without RAG if there's an error

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")

    return context_text

async def handle_google_provider(model, prompt, websocket):
    """
    Handle the Google provider.
    """
    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if hasattr(chunk, 'text'):
                await websocket.send_text(chunk.text)
    except Exception as e:
        logger.error(f"Error with Google provider: {str(e)}")
        await websocket.send_text(f"\\nError with Google provider: {str(e)}")

async def handle_ollama_provider(model, api_kwargs, websocket):
    """
    Handle the Ollama provider.
    """
    try:
        response = await model.acall(api_kwargs=api_kwargs, model_type="llm")
        async for chunk in response:
            text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
            if text and not text.startswith('model=') and not text.startswith('created_at='):
                text = text.replace('<think>', '').replace('</think>', '')
                await websocket.send_text(text)
    except Exception as e:
        logger.error(f"Error with Ollama provider: {str(e)}")
        await websocket.send_text(f"\\nError with Ollama provider: {str(e)}")

async def handle_openrouter_provider(model, api_kwargs, websocket):
    """
    Handle the OpenRouter provider.
    """
    try:
        response = await model.acall(api_kwargs=api_kwargs, model_type="llm")
        async for chunk in response:
            await websocket.send_text(chunk)
    except Exception as e:
        logger.error(f"Error with OpenRouter provider: {str(e)}")
        await websocket.send_text(f"\\nError with OpenRouter provider: {str(e)}")

async def handle_openai_provider(model, api_kwargs, websocket):
    """
    Handle the OpenAI provider.
    """
    try:
        response = await model.acall(api_kwargs=api_kwargs, model_type="llm")
        async for chunk in response:
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                delta = getattr(choices[0], "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text is not None:
                        await websocket.send_text(text)
    except Exception as e:
        logger.error(f"Error with OpenAI provider: {str(e)}")
        await websocket.send_text(f"\\nError with OpenAI provider: {str(e)}")

async def handle_azure_provider(model, api_kwargs, websocket):
    """
    Handle the Azure provider.
    """
    try:
        response = await model.acall(api_kwargs=api_kwargs, model_type="llm")
        async for chunk in response:
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                delta = getattr(choices[0], "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text is not None:
                        await websocket.send_text(text)
    except Exception as e:
        logger.error(f"Error with Azure provider: {str(e)}")
        await websocket.send_text(f"\\nError with Azure provider: {str(e)}")

async def handle_vllm_provider(model, api_kwargs, websocket):
    """
    Handle the vLLM provider.
    """
    try:
        response = await model.acall(api_kwargs=api_kwargs, model_type="llm")
        async for chunk in response:
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                delta = getattr(choices[0], "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text is not None:
                        await websocket.send_text(text)
    except Exception as e:
        logger.error(f"Error with vLLM provider: {str(e)}")
        await websocket.send_text(f"\\nError with vLLM provider: {str(e)}")

def get_system_prompt(repo_type: str, repo_url: str, repo_name: str, language_name: str, is_deep_research: bool, research_iteration: int) -> str:
    """
    Get the system prompt for the chat completion.
    """
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
    return system_prompt
