document.addEventListener("DOMContentLoaded", () => {
    const repoPathInput = document.getElementById("repo-path");
    const generateBtn = document.getElementById("generate-btn");
    const exportMdBtn = document.getElementById("export-md-btn");
    const projectsList = document.getElementById("projects-list");
    const questionInput = document.getElementById("question");
    const askBtn = document.getElementById("ask-btn");
    const answerDiv = document.getElementById("answer");

    // Fetch and display processed projects
    const fetchProcessedProjects = async () => {
        try {
            const response = await fetch("/processed-projects/");
            const projects = await response.json();
            projectsList.innerHTML = "";
            projects.forEach(project => {
                const li = document.createElement("li");
                li.textContent = `${project.path} - ${project.status}`;
                const deleteBtn = document.createElement("button");
                deleteBtn.textContent = "Delete";
                deleteBtn.addEventListener("click", async () => {
                    await fetch(`/processed-projects/${project.id}`, { method: "DELETE" });
                    fetchProcessedProjects();
                });
                li.appendChild(deleteBtn);
                projectsList.appendChild(li);
            });
        } catch (error) {
            console.error("Error fetching processed projects:", error);
        }
    };

    // Generate wiki
    generateBtn.addEventListener("click", async () => {
        const repoPath = repoPathInput.value;
        if (!repoPath) {
            alert("Please enter a repository path.");
            return;
        }

        try {
            const response = await fetch("/generate-wiki/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: repoPath }),
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "wiki.zip";
                document.body.appendChild(a);
                a.click();
                a.remove();
                fetchProcessedProjects();
            } else {
                const error = await response.json();
                alert(`Error: ${error.detail}`);
            }
        } catch (error) {
            console.error("Error generating wiki:", error);
        }
    });

    // Export as Markdown
    exportMdBtn.addEventListener("click", async () => {
        const repoPath = repoPathInput.value;
        if (!repoPath) {
            alert("Please enter a repository path.");
            return;
        }

        try {
            const response = await fetch("/export-markdown/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: repoPath }),
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "wiki.md";
                document.body.appendChild(a);
                a.click();
                a.remove();
            } else {
                const error = await response.json();
                alert(`Error: ${error.detail}`);
            }
        } catch (error) {
            console.error("Error exporting as Markdown:", error);
        }
    });

    // Ask a question
    askBtn.addEventListener("click", async () => {
        const question = questionInput.value;
        const repoPath = repoPathInput.value;

        if (!question || !repoPath) {
            alert("Please enter a repository path and a question.");
            return;
        }

        try {
            const response = await fetch("/qa/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ repo_path: repoPath, query: question }),
            });

            if (response.ok) {
                const answer = await response.json();
                answerDiv.textContent = answer.answer;
            } else {
                const error = await response.json();
                alert(`Error: ${error.detail}`);
            }
        } catch (error) {
            console.error("Error asking question:", error);
        }
    });


    // Initial fetch of processed projects
    fetchProcessedProjects();
});
