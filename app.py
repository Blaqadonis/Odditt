"""Odditt Gradio app -- entry point.

Assembles the pieces from odditt/ (config, model loading, DocChatbot) into the same Gradio
interface the notebook built in Sections 9-10, and launches it.

    pip install -r requirements.txt
    python app.py
"""
import logging

import gradio as gr

from odditt.chatbot import DocChatbot
from odditt.config import CONFIG
from odditt.model_loader import load_embeddings, load_llm

logging.getLogger().setLevel(logging.ERROR)

CUSTOM_CSS = '''
.gradio-container {
    background: radial-gradient(circle at top left, #1a1a2e 0%, #0f0f1a 65%) !important;
}

.odditt-header { text-align: center; padding: 18px 12px 6px 12px; }
.odditt-header h1 {
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2rem;
    margin-bottom: 4px;
}
.odditt-header p { color: #9ca3af; font-size: 0.95rem; }

.fade-in { animation: fadeInUp 0.5s ease both; }
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

.glow-btn { background: linear-gradient(90deg, #7c3aed, #06b6d4) !important; border: none !important; }
button { transition: transform 0.15s ease, box-shadow 0.15s ease !important; }
button:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(124, 58, 237, 0.35); }

.odditt-footer {
    text-align: center;
    padding: 18px 12px;
    margin-top: 10px;
    border-top: 1px solid #2a2a3d;
    color: #6b7280;
    font-size: 0.8rem;
}
.logo-placeholder {
    display: inline-block;
    width: 44px; height: 44px; line-height: 44px;
    border-radius: 10px;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    color: white; font-weight: 700; margin-bottom: 6px;
}
'''


def build_demo(chatbot: DocChatbot) -> gr.Blocks:
    with gr.Blocks(title=CONFIG["app_title"]) as demo:
        gr.HTML(f'''
        <div class="odditt-header">
          <h1>{CONFIG["app_title"]}</h1>
          <p>Upload any PDF and ask questions — every answer ships with a grounding score and a source-page screenshot.</p>
        </div>
        ''')

        with gr.Row():
            chat_history = gr.Chatbot(value=[], elem_id="chatbot", elem_classes=["fade-in"], height=380)

        with gr.Row():
            with gr.Column(scale=8):
                query = gr.Textbox(show_label=False, placeholder="Ask a question about your document...", container=False)
            with gr.Column(scale=1):
                submit_btn = gr.Button("Send", elem_classes=["glow-btn"])
            with gr.Column(scale=1):
                pdf_files = gr.Files(label="📁 Upload PDF(s)", file_types=[".pdf"])

        with gr.Row():
            grounding_box = gr.Markdown(
                "_No evidence yet — ask a question to see the source page(s) here._",
                elem_classes=["fade-in"],
            )

        with gr.Row():
            evidence_gallery = gr.Gallery(
                label="📄 Evidence — source page(s)", columns=3, height=260, elem_classes=["fade-in"]
            )

        with gr.Row():
            image_box = gr.Image(label="Document preview (first page of upload)")

        with gr.Row():
            clear_btn = gr.Button("🔄 Reset session (clear documents + history)")

        gr.HTML('''
        <div class="odditt-footer">
          <div class="logo-placeholder">OD</div>
          <p>Prototype for internal audit-workflow testing — not an official company tool.<br/>
          Replace the placeholder mark above with your organization's authorized logo asset if needed.</p>
        </div>
        ''')

        submit_btn.click(
            chatbot.process_pdfs,
            inputs=[pdf_files, query, chat_history],
            outputs=[chat_history, evidence_gallery, grounding_box, query],
        )
        query.submit(
            chatbot.process_pdfs,
            inputs=[pdf_files, query, chat_history],
            outputs=[chat_history, evidence_gallery, grounding_box, query],
        )
        pdf_files.change(chatbot.render_file, inputs=[pdf_files], outputs=[image_box])

        clear_btn.click(
            chatbot.reset,
            outputs=[chat_history, evidence_gallery, grounding_box, pdf_files, image_box],
        )

    return demo


def main():
    embeddings = load_embeddings(CONFIG)
    _model, _tokenizer, _pipeline, llm = load_llm(CONFIG["llm_model"], CONFIG)
    print(f"Loaded embedding model: {CONFIG['embedding_model']}")
    print(f"Loaded LLM: {CONFIG['llm_model']}")

    chatbot = DocChatbot(CONFIG, embeddings, llm)
    demo = build_demo(chatbot)
    demo.launch(share=False, debug=True)


if __name__ == "__main__":
    main()
