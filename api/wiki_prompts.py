"""
Enhanced prompts for comprehensive wiki generation.
These prompts are designed to create holistic, interconnected documentation
with proper architectural visualization using mermaid diagrams.
"""

WIKI_STRUCTURE_ANALYSIS_PROMPT = """
You are an expert software architect analyzing a codebase to create a comprehensive wiki structure.

Given the file tree and README below, your task is to design a logical wiki structure that:

1. **PRIORITIZES HOLISTIC UNDERSTANDING**
   - Create a main "System Architecture" page that shows the big picture
   - Group related functionality into logical modules (not scattered utility pages)
   - Show how components interact and depend on each other

2. **INCLUDES ARCHITECTURAL VISUALIZATION**
   - Design mermaid diagrams for system architecture
   - Plan sequence diagrams for key workflows
   - Include component relationship diagrams

3. **CREATES MEANINGFUL GROUPINGS**
   Instead of fragmenting into "utilities", "tests", "config", create functional groupings like:
   - Core Business Logic & Workflows
   - Data Management & APIs
   - User Interface & Experience
   - Infrastructure & Operations
   - Development & Quality Assurance

4. **ESTABLISHES INTERCONNECTIONS**
   - Plan how pages will reference each other
   - Identify shared concepts and cross-cutting concerns
   - Design navigation that shows system relationships

Respond with a JSON structure that includes:
- Main architecture overview page
- Logical functional groupings
- Planned mermaid diagrams for each section
- Cross-references between pages

File Tree:
{file_tree}

README:
{readme}
"""

WIKI_PAGE_GENERATION_PROMPT = """
You are creating a comprehensive wiki page based on ACTUAL SOURCE CODE ANALYSIS.

🚨 CRITICAL: The context below should contain ACTUAL CODE from multiple source files, NOT just README content. If you only see README content, this indicates a data retrieval issue that needs to be flagged.

REQUIREMENTS FOR CODE-BASED WIKI GENERATION:
1. **Analyze Actual Source Code**: Base your documentation on the provided source code from multiple files
2. **Include Mermaid Diagrams**: Create diagrams based on actual code structure and flow
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

EXAMPLE MERMAID (should reflect actual code structure):
```mermaid
graph TD
    A[Frontend Components] --> B[API Routes]
    B --> C[Business Logic Layer]
    C --> D[Data Access Layer]
    D --> E[Database/Storage]
```

COMPREHENSIVE SOURCE CODE CONTEXT:
Repository Context (should contain actual source code): {context}
File Tree Structure: {file_tree}
README Documentation: {readme}
"""