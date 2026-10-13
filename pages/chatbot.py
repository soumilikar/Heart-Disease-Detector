import os
import io
import time
import wave
import base64
import warnings

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from st_copy import copy_button
from google.genai import Client
from google.genai import errors

client = Client() 

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- config
MODEL_ID = "gemini-3.8-flash"             # or "gemini-3-pro-preview" for harder reasoning

MAIN_PAGE = "app1.py"                           # your main predictor script (for the back button)

SYSTEM_PROMPT = """You are a health-information assistant inside a heart disease
risk screening app. You are NOT a doctor.

Rules:
- Explain things in plain, simple language: what a risk factor means, what a
  medicine class is generally used for, what a lab value indicates.
- Never diagnose, and never say the user definitely has or doesn't have a condition.
- Never tell the user to start, stop, or change a medicine or dose. Direct that to
  their doctor or pharmacist.
- The app's model output is a statistical screening estimate, not a diagnosis.
  Say so whenever you discuss it.
- If the user describes chest pain or pressure, shortness of breath, fainting,
  pain spreading to the arm or jaw, or similar symptoms, tell them to seek
  emergency medical care immediately.
- If an uploaded image or PDF is unreadable or ambiguous, say so instead of
  guessing drug names, strengths or doses. Mark unclear items as [unclear].
- If the user uploads a prescription or medical report without a specific question,
  give a table: medicine | strength | frequency | duration | what this medicine
  class is generally used for. Then add a few plain-language points relevant to
  heart health, and recommend confirming everything with their doctor or pharmacist.
- Reply in the language the user writes in.
- Keep answers concise and well organised."""

QUICK_PROMPTS = {
    "📊 Explain my result": (
        "Explain my latest screening result in plain language: which of my inputs "
        "most likely contribute to it, and what general lifestyle factors are linked "
        "to heart health. Remind me this is a screening estimate, not a diagnosis."
    ),
    "🔍 What do my inputs mean?": (
        "Go through each of the inputs I entered in the predictor and explain, in "
        "simple terms, what each one means and what ranges are generally considered "
        "typical."
    ),
    "🩺 Questions for my doctor": (
        "Based on my screening result and inputs, suggest a short list of questions "
        "I could ask my doctor at a check-up."
    ),
}


# ---------------------------------------------------------------- client
@st.cache_resource
def get_client():
    key = None
    try:
        key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        st.error(
            "No Gemini API key found. Add GOOGLE_API_KEY to .streamlit/secrets.toml "
            "or set it as an environment variable."
        )
        st.stop()
    return genai.Client(api_key=key)


# ---------------------------------------------------------------- helpers
def setup():
    st.set_page_config(
        page_title="Heart Health Assistant",
        page_icon="❤️",
        layout="centered",
    )
    st.markdown(
        """
        <style>
            section.stMain .block-container {
                padding-top: 2rem; padding-bottom: 1rem;
                padding-left: 1rem; padding-right: 1rem;
            }
            #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    st.session_state.setdefault("chat_history", [])    # types.Content list sent to Gemini
    st.session_state.setdefault("display", [])         # what we render in the UI
    st.session_state.setdefault("uploaded_names", [])  # Gemini Files API names, for cleanup


def build_system_instruction():
    """Rebuilt on every turn, so it always reflects the latest prediction."""
    p = st.session_state.get("prediction")
    if not p:
        return SYSTEM_PROMPT + (
            "\n\nThe user has not run the risk predictor yet, so you have no "
            "screening result for them."
        )
    inputs = "\n".join(f"- {k}: {v}" for k, v in p.get("inputs", {}).items())
    return (
        SYSTEM_PROMPT
        + "\n\nThe user's latest screening result from the app:\n"
        + f"Result: {p.get('risk_label')} (model probability {p.get('probability', 0):.0%})\n"
        + f"Inputs the user entered:\n{inputs}"
    )


def clear_chat():
    """Clears the chat and deletes this session's uploads from Gemini's servers."""
    client = get_client()
    for name in st.session_state.get("uploaded_names", []):
        try:
            client.files.delete(name=name)
        except Exception:
            pass
    st.session_state.chat_history = []
    st.session_state.display = []
    st.session_state.uploaded_names = []


def queue_prompt(text):
    st.session_state["pending_prompt"] = text


def render_message(role, text, note=None, thoughts=None):
    avatar = "🧞‍♀️" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        if note:
            st.caption(note)
        if text:
            st.markdown(text)
        if role == "assistant" and text:
            copy_button(text, tooltip="Copy response", copied_label="Copied!", icon="st")
        if thoughts:
            with st.expander("See Model's Thought Summaries"):
                st.write(thoughts)


def sidebar():
    st.sidebar.subheader(":orange[Options]", divider="blue")
    st.sidebar.button("🗑️ Clear Chat", key="clear", on_click=clear_chat)

    st.sidebar.subheader(":orange[Quick questions]", divider="blue")
    has_prediction = "prediction" in st.session_state
    for label, prompt in QUICK_PROMPTS.items():
        st.sidebar.button(
            label,
            on_click=queue_prompt,
            args=(prompt,),
            disabled=not has_prediction,
            use_container_width=True,
        )

    st.sidebar.subheader(":orange[Your screening result]", divider="blue")
    p = st.session_state.get("prediction")
    if p:
        st.sidebar.metric(p["risk_label"], f"{p['probability']:.0%}")
    else:
        st.sidebar.info("Run the risk predictor first to get personalised answers.")

def upload_files(client, files):
    """Uploads files to the Gemini Files API and returns Parts referencing them."""
    parts = []
    for f in files:
        doc = client.files.upload(file=f, config=dict(mime_type=f.type))
        st.session_state.uploaded_names.append(doc.name)
        while doc.state.name == "PROCESSING":
            time.sleep(2)
            doc = client.files.get(name=doc.name)
        parts.append(types.Part.from_uri(file_uri=doc.uri, mime_type=doc.mime_type))
    return parts

# ---------------------------------------------------------------- main
def main():
    client = get_client()
    init_state()

    st.header(":red[:material/favorite:] Heart Health Assistant", anchor=False)
    st.warning(
        "This assistant gives general information only. It is **not medical advice** and "
        "cannot diagnose conditions. For chest pain, breathlessness or fainting, "
        "seek emergency care immediately.",
        icon="⚠️",
    )
    st.caption(
        "Files you upload are sent to Google's Gemini API for processing and are deleted "
        "when you clear the chat."
    )

    audio_or_not = sidebar()

    # replay earlier messages
    for m in st.session_state.display:
        render_message(m["role"], m["text"], m.get("note"), m.get("thoughts"))

    chat_data = st.chat_input(
        "Ask a question or upload a prescription / report...",
        accept_file="multiple",
        file_type=["pdf", "jpg", "jpeg", "png", "webp", "csv", "mp3"],
        accept_audio=True,
        audio_sample_rate=44100,
    )
    pending = st.session_state.pop("pending_prompt", None)

    text = chat_data.text if chat_data else pending
    files = list(chat_data.files) if chat_data and chat_data.files else []
    audio = chat_data.audio if chat_data and chat_data.audio else None

    if not (text or files or audio):
        return

    # show the user's message
    notes = []
    if files:
        notes.append(f"📎 Sent {len(files)} file(s): " + ", ".join(f.name for f in files))
    if audio:
        notes.append("🎙️ Sent [Voice Question]")
    note = " · ".join(notes) or None
    render_message("user", text, note)

    with st.chat_message("assistant", avatar="🧞‍♀️"):
        try:
            # build the user's parts
            message_parts = []
            if text:
                message_parts.append(types.Part.from_text(text=text))
            if files:
                with st.spinner("Uploading files..."):
                    message_parts.extend(upload_files(client, files))
            if audio:
                message_parts.append(
                    types.Part.from_bytes(data=audio.read(), mime_type="audio/wav")
                )

            chat = client.chats.create(
                model=MODEL_ID,
                history=st.session_state.chat_history,
                config=types.GenerateContentConfig(
                    system_instruction=build_system_instruction(),
                    response_mime_type="text/plain",
                    thinking_config=types.ThinkingConfig(
                        thinking_level="high", include_thoughts=True
                    ),
                ),
            )
            with st.spinner("Thinking..."):
                response = chat.send_message(message_parts)

            reply = response.text or (
                "I couldn't generate a response to that. Please try rephrasing your question."
            )
            thoughts = "\n\n".join(
                p.text
                for p in response.candidates[0].content.parts
                if getattr(p, "thought", False) and p.text
            )

            st.markdown(reply)
            copy_button(reply, tooltip="Copy response", copied_label="Copied!", icon="st")
            if thoughts:
                with st.expander("See Model's Thought Summaries"):
                    st.write(thoughts)

            # persist only after success
            st.session_state.chat_history.append(types.Content(role="user", parts=message_parts))
            st.session_state.chat_history.append(
                types.Content(role="model", parts=[types.Part.from_text(text=reply)])
            )
            st.session_state.display.append({"role": "user", "text": text, "note": note})
            st.session_state.display.append(
                {"role": "assistant", "text": reply, "thoughts": thoughts}
            )

        except Exception as e:
            st.error(f"Error: {e}")
            return

        if audio_or_not == "Yes":
            try:
                speak(client, reply)
            except Exception as e:
                st.caption(f"Voice response unavailable: {e}")


setup()
main()