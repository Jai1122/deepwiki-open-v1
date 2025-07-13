import os
from typing import List, Dict
from .vllm_client import VLLMClient
from .text_chunker import chunk_text

class WikiGenerator:
    """
    Generates a structured wiki for a given codebase.
    """

    def __init__(self, vllm_client: VLLMClient):
        """
        Initializes the WikiGenerator.

        Args:
            vllm_client (VLLMClient): The VLLM client to use for generating summaries.
        """
        self.vllm_client = vllm_client

import os
from .vllm_client import VLLMClient
from .text_chunker import chunk_text

class WikiGenerator:
    """
    Generates a structured wiki for a given codebase.
    """

    def __init__(self, vllm_client: VLLMClient):
        """
        Initializes the WikiGenerator.

        Args:
            vllm_client (VLLMClient): The VLLM client to use for generating summaries.
        """
        self.vllm_client = vllm_client
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 2048))
        self.overlap = int(os.getenv("OVERLAP", 256))

    def generate_summary(self, file_path: str, file_content: str) -> str:
        """
        Generates a summary for a single file.

        Args:
            file_path (str): The path to the file.
            file_content (str): The content of the file.

        Returns:
            str: The generated summary.
        """
        chunks = chunk_text(file_content, chunk_size=self.chunk_size, overlap=self.overlap)

        summaries = []
        for chunk in chunks:
            prompt = f"Summarize the following code from the file '{file_path}':\n\n```\n{chunk}\n```"
            summary = self.vllm_client.generate([prompt])[0]
            summaries.append(summary)

        if len(summaries) == 1:
            return summaries[0]
        else:
            combined_summary_prompt = f"Combine the following summaries for the file '{file_path}' into a single, coherent summary:\n\n"
            for i, summary in enumerate(summaries):
                combined_summary_prompt += f"Summary {i+1}:\n{summary}\n\n"

            final_summary = self.vllm_client.generate([combined_summary_prompt])[0]
            return final_summary

    def generate_wiki(self, code_files: List[str]) -> Dict[str, str]:
        """
        Generates a wiki for a list of code files.

        Args:
            code_files (List[str]): A list of paths to the code files.

        Returns:
            Dict[str, str]: A dictionary where the keys are the file paths and the values are the generated wiki pages.
        """
        wiki_pages = {}
        for file_path in code_files:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()

            summary = self.generate_summary(file_path, file_content)

            wiki_page = f"# {os.path.basename(file_path)}\n\n"
            wiki_page += f"**File Path:** `{file_path}`\n\n"
            wiki_page += "## Summary\n\n"
            wiki_page += f"{summary}\n\n"
            wiki_page += "## Code\n\n"
            wiki_page += f"```\n{file_content}\n```"

            wiki_pages[file_path] = wiki_page

        return wiki_pages

    def create_table_of_contents(self, wiki_pages: Dict[str, str]) -> str:
        """
        Creates a table of contents for the wiki.

        Args:
            wiki_pages (Dict[str, str]): A dictionary of wiki pages.

        Returns:
            str: The table of contents in Markdown format.
        """
        toc = "# Table of Contents\n\n"
        for file_path in sorted(wiki_pages.keys()):
            file_name = os.path.basename(file_path)
            link = f"./{file_path.replace('/', '_')}.md"
            toc += f"- [{file_name}]({link})\n"
        return toc
