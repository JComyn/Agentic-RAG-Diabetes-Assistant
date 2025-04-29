# Agentic-RAG-Diabetes-Assistant

## Project Description
This repository contains the development of the TFG (Final Degree Project) titled **"Multi-Agent System based on RAG for Diabetes Management Assistance"**. The project aims to design and implement a multi-agent artificial intelligence system using Retrieval-Augmented Generation (RAG) to assist patients with diabetes management.

The system leverages LangGraph to create a cyclical process where agents collaborate to refine answers based on retrieved medical documents and web searches, ensuring comprehensive and reliable information delivery through a user-friendly Streamlit interface.

## Technologies Used
- **Programming Language**: Python
- **Frameworks and Libraries**:
    - Langchain & LangGraph: For building the agentic RAG pipeline.
    - Streamlit: For the user interface.
    - ChromaDB: For the vector store.
    - Tavily API: For web search capabilities.
- **Language Models**: Configurable via `.env` (e.g., OpenAI models)
- **Vector Database**: ChromaDB

## Repository Structure
```
.
├── .env                  # Environment variables (API keys, model names) - Create this file
├── .gitignore            # Git ignore file
├── main.py               # Main script (for testing purposes)
├── README.md             # Project description and instructions
├── requirements.txt      # Python dependencies
├── streamlit_app.py      # Main Streamlit application file
├── System flow diagram.png # Diagram illustrating the agent workflow
├── example.png           # Example of the Streamlit interface
├── chroma_db_diabetes/   # ChromaDB persistent storage (it is created in the first run)
├── diabetes_docs/        # Folder for storing PDF documents for RAG - Create this folder
├── evaluation/           # Evaluation scripts and results
│   ├── evaluation.py     # Script for evaluating the system
│   └── GroundTruth-PreguntasRespuestas.csv # Ground truth data for evaluation
└── src/                  # Source code for the agent, graph, components, etc.
    ├── agent.py
    ├── components.py
    ├── config.py
    ├── graph.py
    └── indexing.py
```

## System Flow
![System Flow Diagram](System%20flow%20diagram.png)
*(Diagram illustrating the interaction between the user, agents, RAG, and tools)*

## User Interface
![Streamlit Interface](example.png)
*(Example of the Streamlit interface for user interaction)*

## Installation and Configuration

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/JComyn/Agentic-RAG-Diabetes-Assistant.git>
    cd Agentic-RAG-Diabetes-Assistant
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory and add your API keys and desired model configuration:
    ```dotenv
    GITHUB_TOKEN="ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    TAVILY_API_KEY="tvly_XXXXXXXXXXXXXXXX"
    MISTRAL_API_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    Add any other necessary environment variables here.
    ```
    *Note: Replace placeholders with your actual keys.*

5.  **Prepare Documents:**
    Create a folder named `diabetes_docs` in the project root directory. Place all relevant PDF documents (clinical guides, books, articles, etc.) inside this folder. The system will index these documents on the first run if the vector database is empty.

## Usage

1.  **Ensure Setup:** Make sure you have completed the Installation and Configuration steps.
2.  **Run the Streamlit Application:**
    Open your terminal in the project root directory (with the virtual environment activated) and run:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  **Interact:** The application will open in your web browser. If it's the first time and you've added documents to `diabetes_docs`, it will index them automatically. You can then start asking questions about diabetes management in the chat interface.

## Contact
Author: **Javier Comyn Rodríguez**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/javier-comyn-rodriguez)

---
This project was developed as a Final Degree Project (TFG) for the Bachelor's Degree in Mathematics and Computer Science at ETSIINF, UPM.