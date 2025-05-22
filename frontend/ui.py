import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from src.rag_pipeline import ask_question

chat_history = []

def chatbot_interface(user_input, history):
    global chat_history
    response = ask_question(user_input)

    answer = response["answer"]
    
    chat_history.append((user_input, answer))


    return chat_history, chat_history

with gr.Blocks() as demo:
    gr.Markdown("## 💬 RAG Chatbot (LangChain + CPU)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your question here...", show_label=False).style(container=False)

    clear = gr.Button("🧹 Clear All Chat ")

    msg.submit(chatbot_interface, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()