import os
import zipfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.file_processor import find_code_files
from src.vllm_client import VLLMClient
from src.wiki_generator import WikiGenerator
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class RepoInfo(BaseModel):
    path: str

import uuid
from src.processed_projects import ProcessedProjectsDB
from fastapi.staticfiles import StaticFiles
from src.rag_pipeline import RAGPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class RepoInfo(BaseModel):
    path: str

class QAInfo(BaseModel):
    repo_path: str
    query: str

processed_projects_db = ProcessedProjectsDB()
rag_pipeline = RAGPipeline()

@app.post("/generate-wiki/")
async def generate_wiki(repo_info: RepoInfo):
    """
    Generates a wiki for a local repository and returns it as a zip file.
    """
    repo_path = repo_info.path
    if not os.path.isdir(repo_path):
        raise HTTPException(status_code=400, detail="Invalid repository path.")

    try:
        # Add project to the database
        project_id = str(uuid.uuid4())
        project_info = {"id": project_id, "path": repo_path, "status": "processing"}
        processed_projects_db.add_project(project_info)

        # Initialize VLLM client
        model_name = os.getenv("VLLM_MODEL", "facebook/opt-125m")
        vllm_client = VLLMClient(model=model_name)

        # Find code files
        code_files = find_code_files(repo_path)

        # Generate wiki
        wiki_generator = WikiGenerator(vllm_client)
        wiki_pages = wiki_generator.generate_wiki(code_files)
        toc = wiki_generator.create_table_of_contents(wiki_pages)

        # Create a temporary directory to store the wiki files
        temp_dir = "temp_wiki"
        os.makedirs(temp_dir, exist_ok=True)

        # Write the table of contents to a file
        with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(toc)

        # Write each wiki page to a file
        for file_path, content in wiki_pages.items():
            file_name = file_path.replace('/', '_') + ".md"
            with open(os.path.join(temp_dir, file_name), "w", encoding="utf-8") as f:
                f.write(content)

        # Create a zip file of the wiki
        zip_file_path = "wiki.zip"
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

        # Clean up the temporary directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

        # Update project status
        project_info["status"] = "completed"
        processed_projects_db.delete_project(project_id)
        processed_projects_db.add_project(project_info)


        return FileResponse(zip_file_path, media_type='application/zip', filename='wiki.zip')

    except Exception as e:
        # Update project status
        project_info["status"] = "failed"
        processed_projects_db.delete_project(project_id)
        processed_projects_db.add_project(project_info)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processed-projects/")
async def get_processed_projects():
    """
    Returns a list of all processed projects.
    """
    return processed_projects_db.get_all_projects()

@app.delete("/processed-projects/{project_id}")
async def delete_project(project_id: str):
    """
    Deletes a project from the processed projects list.
    """
    processed_projects_db.delete_project(project_id)
    return {"message": "Project deleted successfully."}

@app.post("/export-markdown/")
async def export_markdown(repo_info: RepoInfo):
    """
    Generates a wiki for a local repository and returns it as a single Markdown file.
    """
    repo_path = repo_info.path
    if not os.path.isdir(repo_path):
        raise HTTPException(status_code=400, detail="Invalid repository path.")

    try:
        # Initialize VLLM client
        model_name = os.getenv("VLLM_MODEL", "facebook/opt-125m")
        vllm_client = VLLMClient(model=model_name)

        # Find code files
        code_files = find_code_files(repo_path)

        # Generate wiki
        wiki_generator = WikiGenerator(vllm_client)
        wiki_pages = wiki_generator.generate_wiki(code_files)
        toc = wiki_generator.create_table_of_contents(wiki_pages)

        # Combine all wiki pages into a single Markdown file
        markdown_content = toc
        for file_path, content in wiki_pages.items():
            markdown_content += "\n\n" + content

        # Create a temporary file to store the Markdown content
        temp_file_path = "wiki.md"
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return FileResponse(temp_file_path, media_type='text/markdown', filename='wiki.md')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa/")
async def qa(qa_info: QAInfo):
    """
    Answers a question about a repository.
    """
    repo_path = qa_info.repo_path
    query = qa_info.query
    if not os.path.isdir(repo_path):
        raise HTTPException(status_code=400, detail="Invalid repository path.")

    try:
        answer = rag_pipeline.ask(repo_path, query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
