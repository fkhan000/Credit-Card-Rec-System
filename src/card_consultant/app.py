import streamlit as st
import warnings
from agent import CreditCardAgent
from dotenv import load_dotenv
import os
import json
from io import BytesIO


warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

st.set_page_config(page_title="JPMC Credit Card Consultant", layout="wide", page_icon="💳")

st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background-color: #0a2540 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        section[data-testid="stSidebar"] {
            min-width: 350px !important;
            max-width: 100% !important;
            resize: horizontal;
            overflow: auto;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: white;
            color: black;
            padding: 1rem;
        }
        button[data-testid="baseButton"][aria-label^="Talk with an Agent"] {
            background-color: #005eb8;
            color: white;
            border: none;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
        }


        .scroll-container {
            max-height: 75vh;
            overflow-y: scroll;
            padding-right: 1rem;
            margin-top: 0;
        }
        .chat-title {
            font-size: 2.5rem;
            color: white;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid white;
            padding-bottom: 0.5rem;
            text-align: center;
            background: linear-gradient(to right, #0a2540, #1e3c72);
            border-radius: 8px;
            padding: 1rem;
        }
        .sidebar-title {
            font-size: calc(1.5rem + 0.8vw);
            margin-bottom: 0.5rem;
            color: black;
            text-align: left;
            font-weight: bold;
        }
        .card-block {
            border: 1px solid #e0e0e0;
            background-color: #f7f9fc;
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .card-name {
            font-size: 1.25rem;
            font-weight: bold;
            color: #005eb8;
            margin-bottom: 0.5rem;
            text-align: center;
            border-bottom: 1px solid #d9d9d9;
            padding-bottom: 0.5rem;
        }
        .card-benefit {
            position: relative;
            padding-left: 2rem;
            margin-bottom: 0.75rem;
            font-size: 1rem;
            font-weight: 500;
            color: #2d2d2d;
            font-family: 'Segoe UI', sans-serif;
            line-height: 1.6;
            letter-spacing: 0.2px;
        }

        .card-benefit::before {
            content: "✔";
            position: absolute;
            left: 0;
            top: 0;
            font-size: 1rem;
            color: #0072ce;
            font-weight: bold;
            transform: translateY(2px);
        }
            
        .apply-btn, .agent-btn {
            background-color: #005eb8;
            color: white;
            border: none;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            cursor: pointer;
        }
        .stChatMessage.user, .stChatMessage.assistant {
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            max-width: 65%;
            text-align: left;
            word-wrap: break-word;
        }
        .stChatMessage.user {
            background-color: #d1eaff;
            border-radius: 20px 20px 0 20px;
            margin-left: auto;
            margin-right: 1rem;
            color: black;
        }
        .stChatMessage.assistant {
            background-color: #ffffff;
            border-radius: 20px 20px 20px 0;
            margin-left: 1rem;
            margin-right: auto;
            color: #1e1e1e;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = CreditCardAgent(1647)
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

agent = st.session_state.agent

# Sidebar Card Recommendations
st.sidebar.markdown("<h2 class='sidebar-title'>Suggested Credit Cards</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>Welcome Amir 👋</p>", unsafe_allow_html=True)

with open(os.path.join("..", "..", "data", "credit_cards.json"), "r") as f:
    card_data = json.load(f)["credit_cards"]

for index in range(len(card_data)):
    card_data[index]["image"] = os.path.join("card_images", card_data[index]["image"])

st.sidebar.markdown('<div class="scroll-container">', unsafe_allow_html=True)
for index in [3, 2]:
    card = card_data[index]
    with st.sidebar.container():
        st.markdown("<div class='card-block'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-name'>{card['name']}</div>", unsafe_allow_html=True)
        st.image(card["image"], use_container_width=True)

        for b in card["benefits"]:
            st.markdown(f"<div class='card-benefit'>{b}</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<button class='apply-btn'>Apply Now</button>", unsafe_allow_html=True)
        with col2:
            if st.button("Talk with an Agent", key="talk_" + card["name"]):
                st.session_state.prefill = f"Can you tell me more about the {card['name']} card?"

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main Chat UI
st.markdown("""<div class='chat-title'>🤖 Credit Card Consultant Chatbot</div>""", unsafe_allow_html=True)

# Display previous chat
for message in agent.get_memory():
    role = "user" if message.type == "human" else "assistant"
    st.chat_message(role).markdown(message.content)

# Chat input
user_input = st.chat_input("Ask me about any credit card...")
if not user_input and st.session_state.prefill:
    user_input = st.session_state.prefill
    st.session_state.prefill = ""

if user_input:
    st.chat_message("user").markdown(user_input)
    output = agent.ask(user_input)

    if isinstance(output, dict):
        if "message" in output:
            st.chat_message("assistant").markdown(output["message"])
        else:
            st.chat_message("assistant").markdown("Here’s what I found:")

        if "image_bytes" in output:
            image_bytes = output["image_bytes"]
            st.image(BytesIO(image_bytes), use_column_width=True)

        if "category_totals" in output:
            st.markdown("### Category Totals")
            for category, amount in output["category_totals"].items():
                st.markdown(f"- **{category}**: ${amount:.2f}")

    else:
        st.chat_message("assistant").markdown(output)