import gradio as gr
import requests
import time

from app.config import settings


# Metrics tracking
total_tokens = 0
total_calls = 0
total_time = 0.0
budget_used = 0.0

def ask_ai(user_message, history):
    """Send the query to FastAPI and return the model response."""
    global total_tokens, total_calls, total_time, budget_used

    start = time.time()
    #res = requests.post(settings.QUERY_URL, json={"question": user_message}).json()
    res= {"answer": "test", "sources": []}
    end = time.time()

    answer = res.get("answer", "No response")
    tokens = res.get("tokens", 150)  # can come from backend
    duration = end - start

    # Update metrics
    total_calls += 1
    total_tokens += tokens
    total_time += duration
    budget_used += (tokens / 1000) * 0.002  # example pricing
    avg_time = total_time / total_calls

    # Updated stats panel
    metadata_info = f"""
    ### 📊 Stats
    - **Tokens used:** {total_tokens}
    - **Avg response time:** {avg_time:.2f}s
    - **Budget used:** ${budget_used:.4f}
    """

    # Append chat history
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "ai", "content": answer},
    ]
    return history, metadata_info


def render_chat(history):
    """Simple formatted chat output."""
    chat = ""
    for msg in history:
        if msg["role"] == "user":
            chat += f"🧍: {msg['content']}\n\n"
        else:
            chat += f"🥊: {msg['content']}\n\n"
    return chat


def update_chat(user_message, history):
    history, metadata = ask_ai(user_message, history)
    chat_display = render_chat(history)
    return chat_display, metadata, history, ""


# --- Themed UI ---
with gr.Blocks() as demo:
    gr.Markdown("## <center>🥊 Welcome to Our Muay Thai Gym</center>")
    with gr.Row():
        with gr.Column(scale=2):
            pass
        with gr.Column(scale=1):
            chatbox = gr.Textbox(
                label="Conversation",
                lines=15,
                interactive=False,
                placeholder="Chat will appear here...",
                elem_classes="chat-message"
            )

            user_input = gr.Textbox(
                placeholder="Ask your question...",
                label=""
            )
            send = gr.Button("Send")

        with gr.Column(scale=2):
            metadata_box = gr.Markdown(
                "### 📊 Stats\n- **Tokens used:** 0\n- **Avg response time:** 0s\n- **Budget used:** $0.0000"
            )

    history_state = gr.State([])

    send.click(
        update_chat,
        inputs=[user_input, history_state],
        outputs=[chatbox, metadata_box, history_state, user_input],
    )
    user_input.submit(
        update_chat,
        inputs=[user_input, history_state],
        outputs=[chatbox, metadata_box, history_state, user_input],
    )

demo.launch()
