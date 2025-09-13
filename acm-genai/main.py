"""
Streamlit ChatBot with Persona Selection and Memory
- Uses Hugging Face DialoGPT (fallback to cpu if no GPU)
- Keeps conversation across messages using session_state
- Personas: RoastBot, ShakespeareBot, Emoji Translator, and Neutral

Run:
    pip install -r requirements.txt
    streamlit run streamlit_chatbot.py

requirements.txt (suggested):
streamlit
transformers
torch
emoji

This is a single-file Streamlit app (not a notebook).
"""

import streamlit as st
from datetime import datetime
import random

# Lazy imports and installation helper
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation, pipeline
except Exception:
    st.write("Installing required packages. Please wait — this may take a minute.")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "emoji", "--quiet"]) 
    from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation, pipeline

try:
    import emoji
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "emoji", "--quiet"]) 
    import emoji

# -----------------------
# Config / load model
# -----------------------
MODEL_NAME = "microsoft/DialoGPT-medium"
@st.cache_resource(show_spinner=False)
def load_conversational_pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    conv_pipe = pipeline("conversational", model=model, tokenizer=tok)
    return conv_pipe

conv_pipeline = load_conversational_pipeline()

# -----------------------
# Utility helpers
# -----------------------

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Few roast templates
ROAST_TEMPLATES = [
    "Oh look, {u}. If I wanted to hear from someone like you, I'd ask my Wi-Fi to drop connection.",
    "{u}, you're like a software update. Whenever I see you, I think 'Not now.'",
    "If wit were RAM, {u}, you'd still be buffering.",
    "You're proof that evolution can go in reverse, {u}.",
]

def make_roast(user_text):
    t = random.choice(ROAST_TEMPLATES)
    short_user = (user_text[:30] + '...') if len(user_text) > 30 else user_text
    return t.format(u=short_user)

# Very small, imperfect 'Shakespeare-izer' (for demo only)
SHAKESPEARE_REPLACEMENTS = [
    (" you ", " thou "),
    (" are ", " art "),
    (" my ", " mine "),
    (" is ", " isst "),
    (" don't ", " doest not "),
    (" do not ", " doest not "),
    (" hello", " well met"),
]

def to_shakespeare(text):
    t = " " + text.lower() + " "
    for a, b in SHAKESPEARE_REPLACEMENTS:
        t = t.replace(a, b)
    t = t.strip()
    # Capitalize first letter
    if len(t) > 0:
        t = t[0].upper() + t[1:]
    # Add a sprinkle of archaic endings
    if not t.endswith(('.', '!', '?')):
        t += ", prithee."
    return t

# Emoji translator (simple common words map)
EMOJI_MAP = {
    "happy": ":smile:",
    "sad": ":pensive:",
    "love": ":heart:",
    "fire": ":fire:",
    "ok": ":ok_hand:",
    "yes": ":white_check_mark:",
    "no": ":x:",
    "cat": ":cat:",
    "dog": ":dog:",
    "hello": ":wave:",
    "thanks": ":pray:",
}

def to_emoji(text):
    words = text.split()
    out = []
    for w in words:
        key = w.lower().strip('.,!?')
        if key in EMOJI_MAP:
            out.append(emoji.emojize(EMOJI_MAP[key], language='alias'))
        else:
            # try to break into characters for emphasis
            out.append(w)
    return ' '.join(out)

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Persona ChatBot", layout="wide")
st.title("Persona ChatBot — Streamlit + Hugging Face")
st.write("A simple conversational bot with memory and selectable personas. Conversation persists while the app runs.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    persona = st.selectbox("Choose a persona:", ["Neutral", "RoastBot", "ShakespeareBot", "Emoji Translator"])
    max_history = st.slider("Max history turns to keep (affects context)", 1, 20, 6)
    if st.button("Clear conversation"):
        st.session_state.clear()
        st.experimental_rerun()

# Initialize session state for history and conversation object
if 'history' not in st.session_state:
    st.session_state['history'] = []  # list of (role, text, time)
if 'conv' not in st.session_state:
    st.session_state['conv'] = None  # transformers.Conversation

# Display chat history
chat_container = st.container()
with chat_container:
    for turn in st.session_state['history']:
        role, text, tstamp = turn
        if role == 'user':
            st.markdown(f"**You** <span style='color:gray;font-size:12px'> {tstamp}</span>", unsafe_allow_html=True)
            st.write(text)
        else:
            st.markdown(f"**Bot ({persona})** <span style='color:gray;font-size:12px'> {tstamp}</span>", unsafe_allow_html=True)
            st.markdown(text)

# Input area
st.markdown("---")
user_input = st.text_input("Type your message and press Enter:")

if user_input:
    # Append user message to history
    st.session_state['history'].append(('user', user_input, timestamp()))

    # Keep history length manageable
    recent = [h for h in st.session_state['history'] if h[0] in ('user','bot')]
    # Build persona-instructed user message to the model
    persona_instruction = ''
    if persona == 'RoastBot':
        persona_instruction = "You are RoastBot. Always answer with a witty, sarcastic roast aimed playfully at the user.\n"
    elif persona == 'ShakespeareBot':
        persona_instruction = "You are ShakespeareBot. Answer in Early Modern English, using poetic and archaic phrasing.\n"
    elif persona == 'Emoji Translator':
        persona_instruction = "You are Emoji Translator Bot. Concisely translate the user's message into emoji-rich text.\n"

    # Combine last few turns into a single prompt for better context
    turns_to_include = [t for t in st.session_state['history'] if t[0] in ('user','bot')][-max_history*2:]
    prompt_parts = [persona_instruction]
    for role, text, _ in turns_to_include:
        if role == 'user':
            prompt_parts.append(f"User: {text}\n")
        else:
            prompt_parts.append(f"Bot: {text}\n")
    prompt_parts.append(f"User: {user_input}\nBot:")
    model_input = "\n".join(prompt_parts)

    # Use transformers conversational pipeline
    conv = Conversation(model_input)
    st.session_state['conv'] = conv
    with st.spinner("Generating..."):
        try:
            output_conv = conv_pipeline(conv)
            bot_raw = output_conv.generated_responses[-1] if output_conv.generated_responses else "I have nothing to say."
        except Exception as e:
            bot_raw = "(model error) " + str(e)

    # Post-process according to persona if needed
    if persona == 'RoastBot':
        roast_prefix = make_roast(user_input)
        bot_display = f"**{roast_prefix}**  \n\n{bot_raw}"
    elif persona == 'ShakespeareBot':
        # We'll try to transform the model output further
        bot_display = to_shakespeare(bot_raw)
    elif persona == 'Emoji Translator':
        # prefer to emoji-translate the user_input rather than model output
        bot_display = to_emoji(user_input)
    else:
        bot_display = bot_raw

    # Append bot response to history
    st.session_state['history'].append(('bot', bot_display, timestamp()))

    # Rerun to refresh chat display
    st.experimental_rerun()

# Footer / usage notes
st.markdown("---")
st.markdown("**Notes:** This demo uses the `microsoft/DialoGPT-medium` model via Hugging Face transformers.\n\nIf the model or dependencies take long to download the first time, please be patient. The conversation persists while the Streamlit app runs.\n")

