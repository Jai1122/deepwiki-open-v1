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
You are creating a comprehensive wiki page as part of a holistic documentation system.

CRITICAL REQUIREMENTS:
1. **Include Mermaid Diagrams**: ALWAYS include relevant mermaid diagrams (architecture, sequence, flowchart, or component diagrams)
2. **Show Relationships**: Explicitly reference and link to related components and pages
3. **Provide Context**: Explain not just what this component does, but how it fits into the larger system
4. **Use Logical Grouping**: Focus on functional cohesion rather than technical artifact separation

MERMAID DIAGRAM TYPES TO CONSIDER:
- System architecture: `graph TD` for showing component relationships
- Process flow: `sequenceDiagram` for showing interaction sequences
- Data flow: `flowchart LR` for showing data movement
- Class relationships: `classDiagram` for showing object relationships

CONTENT STRUCTURE:
1. **Overview & Purpose** - What this component/module does in the context of the larger system
2. **Architecture Diagram** - Mermaid diagram showing how this fits into the system
3. **Key Components** - Main classes, functions, or modules with their responsibilities
4. **Interactions** - How this component interacts with other parts of the system
5. **Implementation Details** - Important patterns, algorithms, or design decisions
6. **Related Pages** - Links to other relevant wiki pages

Remember: Create interconnected, comprehensive documentation that shows the forest, not just the trees.

Context:
{context}

File Content:
{file_content}

Page Topic:
{page_topic}
"""

ARCHITECTURE_OVERVIEW_PROMPT = """
Create a comprehensive System Architecture Overview page for this codebase.

This should be the main entry point to your wiki that provides:

1. **High-Level Architecture Diagram** (using mermaid)
   - Show major system components
   - Include external dependencies
   - Show data flow between components

2. **System Overview**
   - What this system does
   - Key architectural decisions
   - Technology stack and rationale

3. **Component Breakdown**
   - Brief description of each major component
   - Links to detailed pages for each component

4. **Key Workflows** (with sequence diagrams)
   - Most important user journeys or data processing flows
   - Cross-cutting concerns (logging, authentication, etc.)

5. **Development Context**
   - How to get started
   - Key development patterns
   - Testing approach

EXAMPLE MERMAID ARCHITECTURE:
```mermaid
graph TD
    A[User Interface] --> B[API Gateway]
    B --> C[Business Logic]
    C --> D[Data Layer]
    D --> E[Database]
    
    C --> F[External APIs]
    B --> G[Authentication]
    C --> H[Background Jobs]
```

Base your analysis on:
File Tree: {file_tree}
README: {readme}
Repository Context: {context}
"""