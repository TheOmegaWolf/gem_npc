import streamlit as st
from langchain_community.llms import HuggingFaceLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import faiss
import numpy as np
import os

# === Set up Streamlit interface ===
st.title('Text Adventure Game with Dynamic NPC Interactions')
st.write("Interact with the game world, complete quests, and talk to NPCs!")

# === Initialize Game State ===
if 'inventory' not in st.session_state:
    st.session_state.inventory = []
if 'quest' not in st.session_state:
    st.session_state.quest = {"Find the Ancient Amulet": "Not Started"}
if 'turns_taken' not in st.session_state:
    st.session_state.turns_taken = 0
if 'npc' not in st.session_state:
    st.session_state.npc = "Wise Owl"
if 'background' not in st.session_state:
    st.session_state.background = "Enchanted Forest"

# === Functions for managing NPCs and dynamic dialogues ===

# Loading the pre-trained language model
llm = HuggingFaceLLM.from_pretrained("gpt-3.5-turbo")

# Initialize LangChain Memory and Conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_chain = ConversationChain(llm=llm, memory=memory)

# Function to load NPC from a file
def load_npc_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        npc_info = {
            "name": lines[0].strip(),
            "description": lines[1].strip(),
            "dialogue": lines[2].strip()
        }
    return npc_info

# Load NPC and Backgrounds dynamically
npc_info = load_npc_from_file("new_npc.txt")
st.session_state.npc = npc_info["name"]
st.session_state.background = "Enchanted Forest"

# Function to interact with NPC
def interact_with_npc(user_input):
    npc_dialogue = npc_info["dialogue"]
    conversation_chain.predict(input=user_input)  # Run the user input through LangChain
    npc_response = conversation_chain.get_messages()[-1]["text"]
    return npc_response

# === Achievement system ===
def update_achievement_progress(user_input):
    # Increment the number of turns taken
    st.session_state.turns_taken += 1

    # Handle quest: Find the Ancient Amulet
    if "amulet" in user_input.lower():
        if "Ancient Amulet" not in st.session_state.inventory:
            if "First Magical Stone" not in st.session_state.inventory:
                return "\nYou need to find the three magical stones before finding the Amulet."
            
            if st.session_state.turns_taken > 10:
                st.session_state.quest["Find the Ancient Amulet"] = "Failed"
                return "\nYou have taken too long to find the Amulet! Quest failed."

            if "Second Magical Stone" not in st.session_state.inventory or "Third Magical Stone" not in st.session_state.inventory:
                return "\nYou must find all three magical stones first!"

            st.session_state.quest["Find the Ancient Amulet"] = "Completed"
            st.session_state.inventory.append("Ancient Amulet")
            return "\nYou have found the Ancient Amulet! Quest completed!"

    if all(status == "Completed" for status in st.session_state.quest.values()):
        npc_congratulation = f"🎉 Congratulations! You have completed all quests, {st.session_state.npc} is proud of you!"
        return npc_congratulation

    return ""

# === FAISS Setup for NPC Knowledge Base ===
def build_faiss_index():
    # Let's assume we load some pre-defined NPC knowledge in the form of text
    npc_knowledge = [
        "The Wise Owl knows about the ancient world and its secrets.",
        "The owl guards the entrance to the cave where the amulet is kept.",
        "The forest is full of magical creatures and hidden treasures.",
        "The owl will test your wisdom by offering riddles before helping you."
    ]

    # Convert knowledge into embeddings using HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    npc_embeddings = embeddings.embed_documents(npc_knowledge)

    # Build the FAISS index
    faiss_index = faiss.IndexFlatL2(len(npc_embeddings[0]))
    faiss_index.add(np.array(npc_embeddings).astype(np.float32))

    return faiss_index, npc_knowledge, embeddings

# Perform FAISS retrieval
def faiss_retrieve(query, faiss_index, npc_knowledge, embeddings):
    query_embedding = embeddings.embed_query(query)
    _, indices = faiss_index.search(np.array([query_embedding]).astype(np.float32), k=1)
    best_match_idx = indices[0][0]
    return npc_knowledge[best_match_idx]

# === Streamlit UI components ===
user_input = st.text_input("Enter your action (e.g., 'find the stone', 'solve the riddle'):")

if user_input:
    # Retrieve response from NPC using FAISS if necessary
    faiss_index, npc_knowledge, embeddings = build_faiss_index()
    faiss_response = faiss_retrieve(user_input, faiss_index, npc_knowledge, embeddings)

    # Combine the FAISS response with NPC's dynamic dialogue
    npc_dynamic_dialogue = interact_with_npc(user_input)

    # Combine FAISS and NPC response for a comprehensive interaction
    st.write(f"NPC (FAISS): {faiss_response}")
    st.write(f"NPC (Dynamic): {npc_dynamic_dialogue}")

    # Update progress on achievement
    result = update_achievement_progress(user_input)
    st.write(result)

# === Display current status ===
st.write(f"Current quest: {st.session_state.quest}")
st.write(f"Turns taken: {st.session_state.turns_taken}")
st.write(f"Inventory: {st.session_state.inventory}")
st.write(f"NPC: {st.session_state.npc}")
st.write(f"Background: {st.session_state.background}")

