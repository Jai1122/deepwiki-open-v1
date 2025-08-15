"""
Enhanced prompts for comprehensive wiki generation.
These prompts are designed to create holistic, interconnected documentation
with proper architectural visualization using mermaid diagrams.
"""

WIKI_STRUCTURE_ANALYSIS_PROMPT = """
You are an expert software architect analyzing a codebase to create a comprehensive, hierarchical wiki structure.

**CRITICAL TASK**: Analyze the repository to understand what this software actually DOES, then generate a detailed hierarchical wiki structure with SPECIFIC, DYNAMIC topics and sub-topics based on the actual purpose and functionality of this codebase.

**ANALYSIS APPROACH:**
1. **Understand the Purpose**: What problem does this software solve? What is its main function?
2. **Identify Key Workflows**: How do users interact with this system? What are the main use cases?
3. **Map Technical Components**: How is the code organized to deliver this functionality?
4. **Determine Documentation Needs**: What would someone need to know to understand, use, extend, or maintain this system?

**REQUIRED MAIN TOPICS** (always include these 6 topics):
1. **Getting Started**
2. **Core Concepts** 
3. **API Development**
4. **Infrastructure and Configuration**
5. **Testing and Debugging**
6. **Utilities and Helpers**

**YOUR ANALYSIS PROCESS:**
1. **Repository Type Classification**: Determine if this is a web app, API service, CLI tool, library, framework, etc.
2. **Architecture Analysis**: Identify key components, patterns, and technologies used
3. **Dynamic Sub-topic Generation**: For each main topic, generate 3-6 specific sub-topics based on what's actually present in this codebase

**SUB-TOPIC EXAMPLES BY REPOSITORY TYPE AND LANGUAGE:**

*For a Go REST API:*
- **Getting Started** → "Go Module Setup", "Environment Variables", "Building the Binary", "Running Locally"
- **API Development** → "Handler Functions", "Middleware Setup", "Route Configuration", "JSON Serialization"
- **Infrastructure and Configuration** → "Docker Configuration", "Database Migrations", "Environment Config", "Health Checks"

*For a Java Spring Boot API:*
- **Getting Started** → "Maven/Gradle Setup", "Application Properties", "Running with Spring Boot", "IDE Configuration"
- **Core Concepts** → "Spring Bean Configuration", "Dependency Injection", "JPA Entity Models", "Service Layer Architecture"
- **API Development** → "REST Controllers", "Request/Response DTOs", "Exception Handling", "Spring Security"

*For a Node.js Express API:*
- **Getting Started** → "NPM Installation", "Package.json Configuration", "Environment Setup", "Running with Nodemon"
- **API Development** → "Express Routes", "Middleware Functions", "Request Validation", "Error Handling"
- **Testing and Debugging** → "Jest Testing", "Supertest Integration", "Debug Configuration", "Logging with Winston"

*For a Python FastAPI:*
- **Getting Started** → "Poetry/Pip Dependencies", "Virtual Environment", "FastAPI Server", "API Documentation"
- **API Development** → "Pydantic Models", "Path Operations", "Dependency Injection", "Background Tasks"

*For a React Frontend:*
- **Getting Started** → "Node.js Setup", "Package Installation", "Development Server", "Build Process"
- **Core Concepts** → "Component Architecture", "State Management", "Routing", "Props and Context"

*For a CLI Tool (any language):*
- **Getting Started** → "Binary Installation", "Command Syntax", "Configuration Files", "First Commands"
- **Utilities and Helpers** → "Command Parser", "Configuration Loader", "Output Formatting", "Error Handling"

**OUTPUT FORMAT** (respond with EXACTLY this JSON structure):
```json
{{
  "repository_analysis": {{
    "type": "web_application|api_service|cli_tool|library|framework|other",
    "primary_technologies": ["technology1", "technology2", "..."],
    "architecture_pattern": "description of main architectural approach"
  }},
  "wiki_structure": {{
    "Getting Started": {{
      "description": "Brief description of this section's focus",
      "subtopics": [
        {{"title": "Subtopic 1", "description": "What this covers", "files_involved": ["file1.py", "file2.js"]}},
        {{"title": "Subtopic 2", "description": "What this covers", "files_involved": ["file3.py"]}},
        "... 3-6 subtopics total"
      ]
    }},
    "Core Concepts": {{
      "description": "Brief description of this section's focus", 
      "subtopics": [
        {{"title": "Subtopic 1", "description": "What this covers", "files_involved": ["file1.py"]}},
        "... 3-6 subtopics total"
      ]
    }},
    "API Development": {{
      "description": "Brief description of this section's focus",
      "subtopics": [
        "... 3-6 repository-specific subtopics"
      ]
    }},
    "Infrastructure and Configuration": {{
      "description": "Brief description of this section's focus", 
      "subtopics": [
        "... 3-6 repository-specific subtopics"
      ]
    }},
    "Testing and Debugging": {{
      "description": "Brief description of this section's focus",
      "subtopics": [
        "... 3-6 repository-specific subtopics"
      ]
    }},
    "Utilities and Helpers": {{
      "description": "Brief description of this section's focus",
      "subtopics": [
        "... 3-6 repository-specific subtopics"
      ]
    }}
  }}
}}
```

**ANALYSIS REQUIREMENTS:**
- **Language Detection**: Identify the primary programming language(s) from file extensions (.go, .java, .js/.ts, .py, etc.)
- **Framework Recognition**: Detect frameworks from file patterns (package.json, pom.xml, go.mod, requirements.txt, etc.)
- **Project Structure Analysis**: Understand the actual organization and architecture of THIS specific repository
- **Technology Stack Identification**: Determine databases, build tools, testing frameworks, deployment methods actually used
- **File-Specific Sub-topics**: Generate sub-topics based on files that actually exist in the repository
- **Language-Appropriate Terminology**: Use terminology specific to the detected language/framework ecosystem
- **No Assumptions**: Don't assume files or patterns that aren't evidenced in the file tree

File Tree:
{file_tree}

README:
{readme}

Analyze this repository and generate a comprehensive hierarchical wiki structure with dynamic, repository-specific sub-topics for each main section.
"""

WIKI_PAGE_GENERATION_PROMPT = """
You are creating a comprehensive wiki page based on ACTUAL SOURCE CODE ANALYSIS.

🚨 CRITICAL: The context below should contain ACTUAL CODE from multiple source files, NOT just README content. If you only see README content, this indicates a data retrieval issue that needs to be flagged.

REQUIREMENTS FOR CODE-BASED WIKI GENERATION:
1. **Analyze Actual Source Code**: Base your documentation on the provided source code from multiple files
2. **Include Mermaid Diagrams**: Create diagrams based on actual code structure and flow
   - 🚨 CRITICAL: Node labels MUST NOT contain parentheses () - use dashes or "like" instead
   - Example: Use "External Systems - CBS, NPCI" not "External Systems (e.g., CBS, NPCI)"
3. **Show Real Implementation**: Document actual functions, classes, and their relationships as found in the code
4. **Provide Code Examples**: Include relevant code snippets from the analyzed files
5. **Cross-Reference Files**: Show how different source files interact with each other

CONTENT STRUCTURE (Based on Actual Code):
1. **Component Overview** - What this component does based on code analysis
2. **Architecture Diagram** - Mermaid diagram showing actual code relationships
3. **Key Implementation Details** - Actual functions, classes, and methods found in code
4. **Code Flow** - How the code executes based on source analysis
5. **File Dependencies** - How files import and interact with each other
6. **API/Interface Documentation** - Based on actual function signatures and endpoints

⚠️  VALIDATION CHECK: 
- Confirm that the context contains actual source code files (not just README.md)
- If only README content is provided, start your response with "❌ ERROR: Only README content provided, need actual source code for comprehensive wiki generation."

Context from Repository (should contain actual source code):
{context}

Additional File Content:
{file_content}

Page Topic:
{page_topic}
"""

ARCHITECTURE_OVERVIEW_PROMPT = """
Create a comprehensive System Architecture Overview page based on ACTUAL SOURCE CODE ANALYSIS.

🚨 CRITICAL: This analysis should be based on ACTUAL CODE from the repository, not just README content. You should be analyzing real implementation files.

⚠️  VALIDATION CHECK: 
- Confirm that you have access to actual source code files across the repository
- If you only have README/documentation content, start with "❌ ERROR: Insufficient source code access for comprehensive architecture analysis."

ARCHITECTURE ANALYSIS REQUIREMENTS:
1. **Code-Based Architecture Diagram** (using mermaid)
   - Analyze actual imports, dependencies, and call flows
   - Show real components found in the source code
   - Include data flow based on actual function calls

2. **Implementation-Based System Overview**
   - What this system does based on code analysis
   - Actual technology stack found in source files
   - Real architectural patterns discovered in code

3. **Source Code Component Breakdown**
   - Components identified from actual source files
   - Real functions, classes, and modules
   - Actual file organization and structure

4. **Real Workflow Analysis** (with sequence diagrams)
   - Workflows traced through actual code execution paths
   - Real API endpoints and their implementations
   - Actual error handling and logging patterns

5. **Development Context from Code**
   - Real build/deployment configuration found
   - Actual testing framework and patterns used
   - Real development dependencies and tools

MERMAID DIAGRAM REQUIREMENTS (must follow these syntax rules):
```mermaid
graph TD
    A[Frontend Components] --> B[API Routes]
    B --> C[Business Logic Layer]
    C --> D[Data Access Layer]
    D --> E[Database/Storage]
```

🚨 CRITICAL MERMAID SYNTAX RULES:
1. **Node labels MUST NOT contain parentheses ()** - Use square brackets or dashes instead
   - ❌ BAD: A[External Systems (e.g., CBS, NPCI)]
   - ✅ GOOD: A[External Systems - CBS, NPCI]
   - ✅ GOOD: A[External Systems like CBS and NPCI]

2. **Use simple, clean node IDs**: A, B, C, etc. or descriptive names without spaces
3. **Avoid special characters** in node labels: &, <, >, quotes
4. **Keep labels concise** - long labels can cause parsing issues
5. **Always start with graph TD** for flowcharts

COMPREHENSIVE SOURCE CODE CONTEXT:
Repository Context (should contain actual source code): {context}
File Tree Structure: {file_tree}
README Documentation: {readme}
"""

HIERARCHICAL_WIKI_GENERATION_PROMPT = """
Create a comprehensive, hierarchical wiki based on the detailed structure analysis and actual source code, following the DeepWiki format standards.

🚨 CRITICAL: Generate content for ALL main topics and ALL their sub-topics as specified in the wiki structure below.

**CRITICAL CONTENT GENERATION REQUIREMENTS:**

1. **Repository Understanding First**: Analyze what this repository actually DOES and WHY it exists
2. **Comprehensive English Explanations**: Write detailed explanations in natural language about:
   - What problem this repository solves
   - How the system architecture works
   - Why specific design decisions were made
   - How different components interact with each other
3. **Code Context, Not Just Code**: For every code snippet, explain:
   - What this code accomplishes in business terms
   - How it fits into the larger system
   - Why this particular implementation approach was chosen
4. **User Journey Focus**: Help readers understand:
   - How a user would interact with this system
   - What happens behind the scenes during typical operations
   - What the key workflows and data flows are
5. **Technical Depth with Clarity**: Provide deep technical insights while maintaining readability
6. **Source File Attribution**: Reference specific files with line numbers where applicable

**OUTPUT FORMAT (DeepWiki Standard):**
For each main topic, create a separate numbered chapter with this structure:

```markdown
# 1 - Getting Started

## What This Repository Does

[2-3 paragraphs explaining in plain English what this software accomplishes, what problem it solves, and who would use it. Think of this as explaining to a smart colleague who has never seen this code before.]

## System Overview

[Explain the high-level architecture and main components. Use business language first, then get technical.]

## Relevant Source Files
- `[file_path]:[line_range]` - [Brief description of what this file contains]
- `[file_path]:[line_range]` - [Brief description of what this file contains]

## Installation and Setup

[Step-by-step explanation with reasoning for each step]

# 2 - Core Concepts

## System Architecture

[Detailed explanation of how the system works from a user perspective, then diving into technical implementation]

```mermaid
graph TD
    A[User Request] --> B[Component Name]
    B --> C[Business Logic]
    C --> D[Data Storage]
```

[Explain what this diagram shows and WHY the system is designed this way]

## Key Design Patterns and Principles

[Explain the architectural decisions and patterns used, with reasoning]

### Implementation Details

[For each code example, provide context:]

```[language]
// This code handles X functionality which is crucial because...
[Relevant code snippet from source files]
```

**What this code does**: [Explain in business terms]
**Why it's implemented this way**: [Explain the reasoning]
**How it fits in the larger system**: [Explain the connections]

## Sources
1. `[file_path]:[line_numbers]` - [Description of what these lines contain]
```

**CRITICAL FORMATTING REQUIREMENTS:**

1. **Chapter Headers**: MUST use exact format `# 1 - Getting Started`, `# 2 - Core Concepts`, etc.
2. **Sequential Numbering**: Chapters must be numbered 1, 2, 3, 4, 5, 6 (one for each main topic)
3. **Source Attribution**: Always include "## Relevant Source Files" section after chapter header
4. **Code Integration**: Show actual code snippets with proper syntax highlighting  
5. **Visual Diagrams**: Use Mermaid for system architecture and process flows
6. **Sources Section**: End each chapter with "## Sources" section

**MERMAID DIAGRAM REQUIREMENTS:**
- Node labels MUST NOT contain parentheses () - use dashes or "like" instead
- Use descriptive but concise node names
- Show actual system flow based on code analysis
- Always start with `graph TD` for flowcharts
- Use consistent styling and clear connections

**REPOSITORY STRUCTURE ANALYSIS (use this to guide content generation):**
{wiki_structure}

**ACTUAL SOURCE CODE CONTEXT:**
{context}

**FILE TREE:**
{file_tree}

**README CONTENT:**
{readme}

Generate a comprehensive hierarchical wiki following the DeepWiki format standards. 

**DYNAMIC CHAPTER STRUCTURE:**
Based on the wiki structure analysis provided, generate numbered chapters that match the repository's actual content and purpose. Use the topics and sub-topics identified in the {wiki_structure} to create relevant, repository-specific chapters.

**CONTENT DEPTH REQUIREMENTS:**
For each chapter, provide:
1. **Contextual Introduction**: Explain what this aspect of the system does and why it matters
2. **Detailed Technical Explanation**: How it works, with code examples that are explained in context
3. **System Integration**: How this component connects to and supports other parts of the system
4. **User Impact**: How this affects the end user experience or developer workflow
5. **Design Rationale**: Why the system was built this way, what problems it solves

**EXAMPLE DYNAMIC STRUCTURE:**
- If the repository structure indicates it's a **web application**: Focus on frontend/backend separation, user flows, API design
- If it's a **CLI tool**: Focus on command structure, argument parsing, output formatting  
- If it's a **library/framework**: Focus on public APIs, extension points, integration patterns
- If it's a **data processing system**: Focus on pipelines, transformations, storage patterns

Generate chapters that tell the story of THIS specific codebase, not generic software development topics. Each chapter should help a reader understand how to work with, extend, or maintain THIS particular system.
"""