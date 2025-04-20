import streamlit as st
from agent import CreditCardAgent

# Page layout
st.set_page_config(page_title="Credit Card Consultant", layout="wide")

# ✅ Robust session state initialization
if "agent" not in st.session_state:
    st.session_state.agent = CreditCardAgent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = st.session_state.agent

# Sidebar credit card list
st.sidebar.title("💳 Credit Cards")
card_data = [
    {"name": "Visa Gold", "image": "card_images/visa_gold.jpg"},
    {"name": "Amex Gold", "image": "card_images/amex_gold.png"},
    {"name": "Amex Platinum", "image": "card_images/amex_platinum.jpg"},
    {"name": "Visa Classic", "image": "card_images/visa_classic.png"},
    {"name": "Mastercard Gold", "image": "card_images/mastercard_gold.jpg"}
]

for card in card_data:
    st.sidebar.image(card["image"], width=150)
    st.sidebar.write(f"**{card['name']}**")
    st.sidebar.markdown("---")

# Chat UI
st.title("💬 Credit Card Consultant Chatbot")

for message in agent.get_memory():
    role = "user" if message.type == "human" else "assistant"
    st.chat_message(role).markdown(message.content)

user_input = st.chat_input("Ask me about any credit card...")

if user_input:
    st.chat_message("user").markdown(user_input)
    output = agent.ask(user_input)
    st.chat_message("assistant").markdown(output)
