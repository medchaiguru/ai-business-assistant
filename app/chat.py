import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/query"

# Maintain chat history
chat_history = []

def ask_ai(user_message):
    global chat_history
    # Call FastAPI RAG endpoint
    try:
        response = requests.post(API_URL, json={"question": user_message})
        data = response.json()
    except Exception as e:
        return chat_history + [(user_message, f"Error: {str(e)}")]

    answer = data["answer"]
    sources = data.get("sources", [])
    sources_text = "\n".join([f"- {s}" for s in sources])
    final_answer = f"{answer}\n\nSources:\n{sources_text}" if sources else answer

    # Append to chat history
    chat_history.append((user_message, final_answer))
    return chat_history

# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 💬 AI Customer Support Chat")
    chatbot = gr.Chatbot(label="Chat with AI", height=400)
    msg = gr.Textbox(placeholder="Type your question here...", lines=2)
    submit = gr.Button("Send")

    submit.click(
        ask_ai,
        inputs=msg,
        outputs=chatbot
    )
    # Optional: Press Enter to send message
    msg.submit(ask_ai, inputs=msg, outputs=chatbot)

demo.launch()
