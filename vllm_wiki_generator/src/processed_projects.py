import json
import os
from typing import List, Dict, Any

class ProcessedProjectsDB:
    """
    A simple JSON-based database for storing processed projects.
    """

    def __init__(self, db_path: str = "processed_projects.json"):
        """
        Initializes the ProcessedProjectsDB.

        Args:
            db_path (str, optional): The path to the database file. Defaults to "processed_projects.json".
        """
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump([], f)

    def get_all_projects(self) -> List[Dict[str, Any]]:
        """
        Retrieves all projects from the database.

        Returns:
            List[Dict[str, Any]]: A list of all projects.
        """
        with open(self.db_path, "r") as f:
            return json.load(f)

    def add_project(self, project_info: Dict[str, Any]) -> None:
        """
        Adds a new project to the database.

        Args:
            project_info (Dict[str, Any]): The project information to add.
        """
        projects = self.get_all_projects()
        projects.append(project_info)
        with open(self.db_path, "w") as f:
            json.dump(projects, f, indent=4)

    def delete_project(self, project_id: str) -> None:
        """
        Deletes a project from the database.

        Args:
            project_id (str): The ID of the project to delete.
        """
        projects = self.get_all_projects()
        updated_projects = [p for p in projects if p.get("id") != project_id]
        with open(self.db_path, "w") as f:
            json.dump(updated_projects, f, indent=4)
