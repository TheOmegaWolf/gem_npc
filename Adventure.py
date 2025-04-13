# === Adventure Game with LLM NPCs and Voice Interaction ===
import streamlit as st
import random
import speech_recognition as sr
import requests
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import os

# === API KEYS ===
GEMINI_API_KEY = "AIzaSyDDVJ75w6vj3x8HpbEVK22lKUUlHYmJd10"
OPENROUTER_API_KEY = "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"

# === Game State ===
if "location" not in st.session_state:
    st.session_state.location = "Forest"
if "npc" not in st.session_state:
    st.session_state.npc = "Elder Oak"
if "inventory" not in st.session_state:
    st.session_state.inventory = []
if "quest" not in st.session_state:
    st.session_state.quest = {"Find the Amulet": "Not Started"}
if "history" not in st.session_state:
    st.session_state.history = []
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Mistral"
if "npc_embeddings" not in st.session_state:
    st.session_state.npc_embeddings = {}

# === NPC Profiles ===
npc_profiles = {
    "Elder Oak": "You are Elder Oak, a wise forest spirit who speaks in ancient riddles and guides travelers.",
    "Blitz": "You are Blitz, a hyper goblin merchant who lies and hoards shiny things.",
    "Captain Mira": "You are Captain Mira, a battle-hardened space marine with dry wit and a tactical mind."
}

# === Speech Recognition ===
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        text = r.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Didn't catch that. Try again.")
        return ""
    except sr.RequestError:
        st.error("Speech recognition service error.")
        return ""

# === Handle Text-Based Conversations ===
def handle_text_input(user_input):
    npc_profile = st.session_state.npc_embeddings.get(st.session_state.npc, npc_profiles.get(st.session_state.npc, ""))
    full_prompt = f"{npc_profile}\nThe player is in {st.session_state.location} and says: {user_input}\nRespond as the NPC:"

    if st.session_state.model_choice == "Gemini":
        return chat_with_gemini(full_prompt)
    else:
        return chat_with_mistral(full_prompt, st.session_state.history)

# === Chat with Gemini API ===
def chat_with_gemini(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text

# === Chat with Mistral API via OpenRouter ===
def chat_with_mistral(prompt, history=[]):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = [{"role": "system", "content": "You are a helpful and immersive RPG NPC."}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "⚠️ LLM Error: " + response.text

# === File Processing ===
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages)

def process_uploaded_file(file):
    filename = file.name
    text = ""
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.type == "text/plain":
        text = str(file.read(), 'utf-8')
    else:
        st.error("Only PDF or text files are supported.")
        return

    # Handle NPC Upload
    if filename.startswith("npc_"):
        npc_name = filename.split("_")[1].split(".")[0]
        st.session_state.npc_embeddings[npc_name] = text
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings)
        st.session_state.npc_embeddings[f"{npc_name}_vector"] = vector_store
        st.success(f"🧙‍♂️ NPC '{npc_name}' added to the game!")
    
    # Handle Quest Upload
    elif filename.startswith("quest_"):
        quest_name = filename.split("_")[1].split(".")[0].replace("_", " ")
        st.session_state.quest[quest_name] = "Not Started"
        st.success(f"🗺️ Quest '{quest_name}' added to your log!")

# === Location and NPC Management ===
def change_location(new_location):
    if new_location != st.session_state.location:
        st.session_state.location = new_location
        update_npc_based_on_location()

def update_npc_based_on_location():
    if st.session_state.location == "Goblin Market":
        st.session_state.npc = "Blitz"
    elif st.session_state.location == "Space Station":
        st.session_state.npc = "Captain Mira"
    else:
        st.session_state.npc = "Elder Oak"

# === Inventory and Quest Management ===
def update_inventory_and_quest(user_input):
    if "amulet" in user_input.lower() and "Ancient Amulet" not in st.session_state.inventory:
        st.session_state.quest["Find the Amulet"] = "Completed"
        st.session_state.inventory.append("Ancient Amulet")
        return "\n🧿 You have found the Ancient Amulet!"
    return ""

# === Game UI Components ===
def display_game_ui():
    st.markdown(f"## 🤖 You meet {st.session_state.npc} in the {st.session_state.location}")

    text_input = st.text_input("💬 Type your message")

    if st.button("📤 Send Text"):
        if text_input:
            npc_reply = handle_text_input(text_input)
            npc_reply += update_inventory_and_quest(text_input)
            st.session_state.history.append((text_input, npc_reply))
            st.markdown(f"**{st.session_state.npc}:** {npc_reply}")

    if st.button("🎤 Talk (Voice)"):
        voice_input = listen()
        if voice_input:
            npc_reply = handle_text_input(voice_input)
            npc_reply += update_inventory_and_quest(voice_input)
            st.session_state.history.append((voice_input, npc_reply))
            st.markdown(f"**{st.session_state.npc}:** {npc_reply}")

    if st.session_state.history:
        st.markdown("### 💬 Conversation History")
        for user, npc in st.session_state.history[-5:]:
            st.markdown(f"**You:** {user}")
            st.markdown(f"**{st.session_state.npc}:** {npc}")

    st.sidebar.markdown("## 🧾 Quests")
    for quest, status in st.session_state.quest.items():
        st.sidebar.write(f"- {quest}: **{status}**")

    st.sidebar.markdown("## 🎒 Inventory")
    for item in st.session_state.inventory:
        st.sidebar.write(f"- {item}")

# === Main Flow ===
def main():
    locations = ["Forest", "Goblin Market", "Space Station"]
    st.sidebar.title("🌍 Navigation")
    new_location = st.sidebar.selectbox("Choose a location", locations)
    change_location(new_location)

    st.sidebar.markdown("## 🤖 Choose LLM")
    st.session_state.model_choice = st.sidebar.radio("Model", ["Mistral", "Gemini"])

    st.sidebar.markdown("## ⬆️ Upload NPC/Quest")
    uploaded_file = st.sidebar.file_uploader("Upload (start filename with 'npc_' or 'quest_')", type=["pdf", "txt"])
    if uploaded_file:
        process_uploaded_file(uploaded_file)

    display_game_ui()

# Run the Game
if __name__ == "__main__":
    main()