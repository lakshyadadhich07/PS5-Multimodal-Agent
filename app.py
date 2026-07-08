import gradio as gr
from langgraph_agent import run_hybrid_agent
from pdf_processor import process_uploaded_pdf
import os

# Global state for image path (simple approach for capstone)
CURRENT_IMAGE_PATH = None


# ==========================================
# PDF Upload Handler
# ==========================================

def handle_pdf_upload(uploaded_pdf):
    if uploaded_pdf is None:
        return "❌ Please upload a PDF."
    return process_uploaded_pdf(uploaded_pdf.name)


# ==========================================
# Image Upload Handler
# ==========================================

def handle_image_upload(uploaded_image):
    global CURRENT_IMAGE_PATH
    if uploaded_image is None:
        CURRENT_IMAGE_PATH = None
        return "❌ Please upload an image."
    CURRENT_IMAGE_PATH = uploaded_image
    return f"✅ Image uploaded successfully: {os.path.basename(uploaded_image)}"


# ==========================================
# Chat Handler
# ==========================================

def handle_chat(user_message, history):
    if not user_message.strip():
        return "Please enter a question."
        
    global CURRENT_IMAGE_PATH
    
    # Run the hybrid agent
    response, trace = run_hybrid_agent(user_message, image_path=CURRENT_IMAGE_PATH)
    
    # Format the final output with the reasoning trace inside an HTML details tag (Accordion)
    accordion_html = f"""
<details>
<summary>🔍 Reasoning Trace (Click to expand)</summary>
<pre style="white-space: pre-wrap; font-size: 12px; background-color: #f4f4f4; padding: 10px; border-radius: 5px; color: black;">
{trace}
</pre>
</details>
"""
    final_output = f"{response}\n\n{accordion_html}"
    return final_output


# ==========================================
# Gradio Application
# ==========================================

with gr.Blocks(title="HybridSight AI Pro") as demo:
    gr.Markdown("# 🤖 PS5 — Multimodal Q&A Pro")
    gr.Markdown("Full Hybrid Agent with Document Q&A, Web Search, Wikipedia, and Image Vision.")

    with gr.Row():
        
        # LEFT COLUMN: Uploads and Controls
        with gr.Column(scale=1):
            
            gr.Markdown("### 📄 Document Q&A")
            gr.Markdown("Upload a PDF to index it into the ChromaDB vector store.")
            uploaded_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_pdf_button = gr.Button("Index PDF", variant="primary")
            pdf_status = gr.Textbox(label="Processing Status", interactive=False)
            
            process_pdf_button.click(
                fn=handle_pdf_upload,
                inputs=uploaded_pdf,
                outputs=pdf_status
            )
            
            gr.Markdown("---")
            
            gr.Markdown("### 🖼️ Image Studio")
            gr.Markdown("Upload an image for the agent to analyze in the Hybrid Chat.")
            uploaded_image = gr.Image(label="Upload Image", type="filepath")
            process_image_button = gr.Button("Set Active Image", variant="primary")
            image_status = gr.Textbox(label="Image Status", interactive=False)
            
            process_image_button.click(
                fn=handle_image_upload,
                inputs=uploaded_image,
                outputs=image_status
            )
            
        # RIGHT COLUMN: Chat Interface
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Hybrid Chat")
            gr.Markdown("Ask anything! The agent will intelligently route to Document Search, Web Search, Wikipedia, or Image Analysis.")
            gr.ChatInterface(
                fn=handle_chat,
                chatbot=gr.Chatbot(height=700),
                textbox=gr.Textbox(placeholder="Ask your question here...")
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())