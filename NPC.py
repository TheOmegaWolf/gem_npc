import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import random
import json

# === Streamlit Setup ===
st.set_page_config(page_title="🎙️ Voice NPC Adventure", page_icon="🎮")
st.title("🎙️ Voice NPC Adventure Game")

# === Gemini Setup ===
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat(history=[])

# === NPC Profiles ===
npc_profiles = {
    "Elder Oak": "You are Elder Oak, a wise forest spirit who speaks in ancient riddles and guides travelers.",
    "Blitz": "You are Blitz, a hyper goblin merchant who lies and hoards shiny things.",
    "Captain Mira": "You are Captain Mira, a battle-hardened space marine with dry wit and a tactical mind."
}

# === NPC States ===
npc_states = {
    "Elder Oak": {"mood": "calm", "trust": 5},
    "Blitz": {"mood": "greedy", "trust": 2},
    "Captain Mira": {"mood": "serious", "trust": 4}
}

# === Initial Session State ===
def init_state():
    st.session_state.setdefault("location", "Forest")
    st.session_state.setdefault("npc", "Elder Oak")
    st.session_state.setdefault("inventory", [])
    st.session_state.setdefault("quest", {
        "Find the Amulet": "Not Started",
        "Defeat the Shadow Beast": "Not Started"
    })
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("player", {
        "name": "Adventurer",
        "class": "Rogue",
        "charisma": 7,
        "strength": 5
    })
    st.session_state.setdefault("event", None)
init_state()

# === Voice Recognition ===
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

# === Navigation ===
locations = ["Forest", "Goblin Market", "Space Station"]
st.sidebar.title("🌍 Navigation")
new_location = st.sidebar.selectbox("Choose a location", locations)

if new_location != st.session_state.location:
    st.session_state.location = new_location
    st.session_state.event = None
    if new_location == "Goblin Market":
        st.session_state.npc = "Blitz"
    elif new_location == "Space Station":
        st.session_state.npc = "Captain Mira"
    else:
        st.session_state.npc = "Elder Oak"

    # Random Event
    if random.random() < 0.3:
        st.session_state.event = random.choice([
            "Bandit Ambush", "Strange Lights", "Merchant Encounter"])

# === Game UI ===
npc_name = st.session_state.npc
npc_prompt = npc_profiles[npc_name]
st.markdown(f"## 🤖 You meet {npc_name} in the {st.session_state.location}")

# Event display
if st.session_state.event:
    st.warning(f"⚠️ Event: {st.session_state.event}")

# === Talk to NPC ===
if st.button("🎤 Talk (Voice)"):
    user_input = listen()
    if user_input:
        player_info = st.session_state.player
        trust = npc_states[npc_name]['trust']
        prompt = f"{npc_prompt}\nPlayer Info: {player_info}, NPC trust: {trust}\nThe player is in {st.session_state.location} and says: {user_input}\nRespond as the NPC:"
        response = chat.send_message(prompt)
        npc_reply = response.text

        # Quest logic
        if "amulet" in user_input.lower() and "Ancient Amulet" not in st.session_state.inventory:
            st.session_state.quest["Find the Amulet"] = "Completed"
            st.session_state.inventory.append("Ancient Amulet")
            npc_reply += "\n🧿 You have found the Ancient Amulet!"

        if "shadow beast" in user_input.lower() and st.session_state.quest["Find the Amulet"] == "Completed":
            st.session_state.quest["Defeat the Shadow Beast"] = "Completed"
            st.session_state.inventory.append("Shadow Beast Fang")
            npc_reply += "\n🐉 The Shadow Beast is defeated!"

        st.session_state.history.append((user_input, npc_reply))
        st.markdown(f"**{npc_name}:** {npc_reply}")

# === Conversation History ===
if st.session_state.history:
    st.markdown("### 💬 Conversation History")
    for user, npc in st.session_state.history[-5:]:
        st.markdown(f"**You:** {user}")
        st.markdown(f"**{npc_name}:** {npc}")

# === Player Stats ===
st.sidebar.markdown("## 🧙 Character")
for key, val in st.session_state.player.items():
    st.sidebar.write(f"- {key}: {val}")

# === Quests + Inventory ===
st.sidebar.markdown("## 🧾 Quests")
for quest, status in st.session_state.quest.items():
    st.sidebar.write(f"- {quest}: **{status}**")

st.sidebar.markdown("## 🎒 Inventory")
for item in st.session_state.inventory:
    st.sidebar.write(f"- {item}")

# === Dice Roll ===
if st.sidebar.button("🎲 Roll a Skill Check"):
    roll = random.randint(1, 20)
    st.sidebar.success(f"You rolled a {roll}")

# === Save / Load Game ===
st.sidebar.markdown("## 💾 Save / Load")
if st.sidebar.button("💾 Save Game"):
    with open("savegame.json", "w") as f:
        json.dump(dict(st.session_state), f, default=str)
    st.sidebar.success("Game saved!")

if st.sidebar.button("📂 Load Game"):
    try:
        with open("savegame.json", "r") as f:
            data = json.load(f)
            for key, val in data.items():
                st.session_state[key] = val
        st.sidebar.success("Game loaded!")
    except Exception as e:
        st.sidebar.error("Failed to load game.")