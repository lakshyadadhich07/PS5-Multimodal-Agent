# PS5 — Multimodal Q&A Pro 🤖

This is the capstone project for the Multimodal Agent module. It is a full hybrid agent capable of understanding documents, searching the web, querying Wikipedia, and analyzing images, all within a single unified interface.

## Features
- **Intelligent Tool Routing**: The LangGraph ReAct agent intelligently routes queries to the correct tools.
- **Document Q&A**: Upload PDFs, which are indexed into a local ChromaDB vector store.
- **Image Studio**: Upload images which are analyzed using Groq Vision capabilities.
- **Web & Wikipedia Search**: Agent automatically degrades gracefully and searches the web or Wikipedia if information is missing from documents.
- **Reasoning Trace**: The interface displays a collapsible trace of the agent's exact tool calls and intermediate thoughts.
- **Sleek Gradio Interface**: A beautiful, single-page dashboard grouping all functionality.

## Requirements
- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/lakshyadadhich07/PS5-Multimodal-Agent.git
cd PS5-Multimodal-Agent
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Groq API Key:
```env
GROQ_API_KEY="your_api_key_here"
```

## Running the Application
To start the Gradio web interface locally:
```bash
python app.py
```
The application will automatically open in your default web browser at `http://127.0.0.1:7860/`.

## Deployment to Hugging Face Spaces
This app is HF Spaces ready! Simply push the code to a new Gradio space and add your `GROQ_API_KEY` to the repository secrets. The app will build and run without any hardcoded credentials.

# Screenshots

## 1. Home Interface
Main dashboard of the Multimodal Q&A Pro application showing the PDF upload section and chat interface.

![alt text](<screenshots/Screenshot 2026-07-08 185753.png>)

---

## 2. PDF Document Q&A
The agent retrieves relevant information from uploaded PDF documents using ChromaDB and generates context-aware answers.

![alt text](<screenshots/Screenshot 2026-07-08 190104.png>)
---

## 3. Image Analysis
The agent analyzes uploaded images using Groq Vision and provides detailed descriptions or answers image-related questions.

![alt text](<screenshots/Screenshot 2026-07-08 185811.png>)

![alt text](<screenshots/Screenshot 2026-07-08 190322.png>)

---

## 4. Web Search
When the required information is unavailable in uploaded documents, the agent automatically performs a web search to provide up-to-date information.

![alt text](<screenshots/Screenshot 2026-07-08 190239.png>)
