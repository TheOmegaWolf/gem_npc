import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # Import the Gemini integration
import os
import json
import numpy as np
import random

# === Set up Streamlit interface ===
st.title('Text Adventure Game with Dynamic NPC Interactions')
st.write("Interact with the game world, complete quests, and talk to NPCs!")

# === Initialize Game State ===
if 'inventory' not in st.session_state:
    st.session_state.inventory = []
if 'quests' not in st.session_state:
    st.session_state.quests = {}
if 'completed_quests' not in st.session_state:
    st.session_state.completed_quests = []
if 'turns_taken' not in st.session_state:
    st.session_state.turns_taken = 0
if 'current_npc' not in st.session_state:
    st.session_state.current_npc = None
if 'npcs' not in st.session_state:
    st.session_state.npcs = {}
if 'current_location' not in st.session_state:
    st.session_state.current_location = "Enchanted Forest"
if 'game_map' not in st.session_state:
    st.session_state.game_map = {
        "Enchanted Forest": {
            "description": "A mystical forest with towering trees and shimmering leaves.",
            "connections": ["Village Square", "Ancient Ruins"],
            "npcs": []
        },
        "Village Square": {
            "description": "A bustling center of activity with shops and villagers.",
            "connections": ["Enchanted Forest", "Mysterious Cave"],
            "npcs": []
        },
        "Ancient Ruins": {
            "description": "Crumbling stone structures covered in strange symbols.",
            "connections": ["Enchanted Forest", "Mountain Pass"],
            "npcs": []
        },
        "Mysterious Cave": {
            "description": "A dark cave with glowing crystals embedded in the walls.",
            "connections": ["Village Square", "Underground Lake"],
            "npcs": []
        },
        "Mountain Pass": {
            "description": "A narrow path winding through tall, snow-capped mountains.",
            "connections": ["Ancient Ruins", "Dragon's Lair"],
            "npcs": []
        },
        "Underground Lake": {
            "description": "A vast subterranean lake with crystal-clear water.",
            "connections": ["Mysterious Cave"],
            "npcs": []
        },
        "Dragon's Lair": {
            "description": "A massive cavern filled with treasures and scorched walls.",
            "connections": ["Mountain Pass"],
            "npcs": []
        }
    }
if 'interaction_history' not in st.session_state:
    st.session_state.interaction_history = []
if 'npc_memories' not in st.session_state:
    st.session_state.npc_memories = {}

# === Configure the Gemini Model ===
# Set your API token for Google Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyDDVJ75w6vj3x8HpbEVK22lKUUlHYmJd10"  # Replace with your actual Gemini API key

# Initialize LangChain with Google Gemini
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Use the Gemini 1.5 Flash model
        temperature=0.7,
        max_output_tokens=512,
        top_p=0.95,
        top_k=40
    )

# Load LLM
llm = get_llm()

# Conversation template for NPCs
npc_template = PromptTemplate(
    input_variables=["npc_name", "npc_description", "npc_personality", "chat_history", "human_input"],
    template="""
    You are {npc_name}, {npc_description}. Your personality is {npc_personality}.
    
    Previous conversation:
    {chat_history}
    
    Human: {human_input}
    NPC {npc_name}:"""
)

# === Load NPCs from files ===
def load_npcs():
    # Check if NPC directory exists, if not create it
    if not os.path.exists("npcs"):
        os.makedirs("npcs")
        # Create default NPC file
        with open("npcs/wise_owl.txt", "w") as f:
            f.write("Wise Owl\n")
            f.write("An ancient owl with knowledge of the forest's secrets\n")
            f.write("Wise and mysterious, speaks in riddles\n")
            f.write("Enchanted Forest\n")
            f.write("I've been watching you from these branches. What knowledge do you seek, traveler?\n")
            f.write("The First Magical Stone can be found where water meets earth in perfect harmony.\n")
    
    npcs = {}
    for filename in os.listdir("npcs"):
        if filename.endswith(".txt"):
            npc_data = load_npc_from_file(os.path.join("npcs", filename))
            npcs[npc_data["name"]] = npc_data
            
            # Add NPC to the appropriate location
            if npc_data["location"] in st.session_state.game_map:
                if npc_data["name"] not in st.session_state.game_map[npc_data["location"]]["npcs"]:
                    st.session_state.game_map[npc_data["location"]]["npcs"].append(npc_data["name"])
    
    return npcs

def load_npc_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        npc_info = {
            "name": lines[0].strip(),
            "description": lines[1].strip(),
            "personality": lines[2].strip(),
            "location": lines[3].strip(),
            "greeting": lines[4].strip(),
            "quest_hint": lines[5].strip() if len(lines) > 5 else "",
        }
    return npc_info

# === Load quests from files ===
def load_quests():
    # Check if quests directory exists, if not create it
    if not os.path.exists("quests"):
        os.makedirs("quests")
        # Create default quest file
        with open("quests/ancient_amulet.txt", "w") as f:
            f.write("Find the Ancient Amulet\n")
            f.write("A powerful amulet is hidden somewhere in these lands. Collect the three magical stones to reveal its location.\n")
            f.write("3\n")  # Number of steps to complete
            f.write("Find the First Magical Stone|Wise Owl|Underground Lake|First Magical Stone\n")
            f.write("Find the Second Magical Stone|Forest Spirit|Ancient Ruins|Second Magical Stone\n")
            f.write("Find the Third Magical Stone|Village Elder|Dragon's Lair|Third Magical Stone\n")
            f.write("Ancient Amulet|You gained the power to speak with animals!\n")
    
    quests = {}
    for filename in os.listdir("quests"):
        if filename.endswith(".txt"):
            quest_data = load_quest_from_file(os.path.join("quests", filename))
            quests[quest_data["name"]] = quest_data
    
    return quests

def load_quest_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        quest_name = lines[0].strip()
        quest_description = lines[1].strip()
        num_steps = int(lines[2].strip())
        
        steps = []
        for i in range(3, 3 + num_steps):
            if i < len(lines):
                step_parts = lines[i].strip().split('|')
                steps.append({
                    "description": step_parts[0],
                    "npc": step_parts[1],
                    "location": step_parts[2],
                    "item": step_parts[3],
                    "completed": False
                })
        
        reward_parts = lines[3 + num_steps].strip().split('|')
        reward = {
            "item": reward_parts[0],
            "effect": reward_parts[1]
        }
        
        return {
            "name": quest_name,
            "description": quest_description,
            "steps": steps,
            "current_step": 0,
            "reward": reward,
            "completed": False
        }

# === FAISS Setup for NPC Knowledge Base ===
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_npc_knowledge_base(npc_info):
    embeddings = load_embeddings()
    
    # Base knowledge for the NPC
    npc_knowledge = [
        f"{npc_info['name']} is {npc_info['description']}",
        f"{npc_info['name']} has a {npc_info['personality']} personality",
        f"{npc_info['name']} can be found in {npc_info['location']}",
        f"{npc_info['name']} says: {npc_info['greeting']}",
    ]
    
    if npc_info["quest_hint"]:
        npc_knowledge.append(f"{npc_info['name']} knows: {npc_info['quest_hint']}")
    
    # Add general game world knowledge
    for location, details in st.session_state.game_map.items():
        npc_knowledge.append(f"The {location} is {details['description']}")
    
    # Create FAISS index
    vectorstore = FAISS.from_texts(npc_knowledge, embeddings)
    return vectorstore

# Function to interact with NPC using LangChain and FAISS
def interact_with_npc(npc_name, user_input):
    npc_info = st.session_state.npcs[npc_name]
    
    # Get relevant knowledge from FAISS
    vectorstore = build_npc_knowledge_base(npc_info)
    relevant_docs = vectorstore.similarity_search(user_input, k=2)
    relevant_knowledge = " ".join([doc.page_content for doc in relevant_docs])
    
    # Initialize or retrieve memory for this NPC
    if npc_name not in st.session_state.npc_memories:
        st.session_state.npc_memories[npc_name] = []
    
    # Format chat history from memory
    chat_history = ""
    for exchange in st.session_state.npc_memories[npc_name][-5:]:  # Only use last 5 exchanges
        chat_history += f"Human: {exchange['human']}\nNPC {npc_name}: {exchange['npc']}\n"
    
    # Create LLMChain for this interaction
    npc_chain = LLMChain(
        llm=llm,
        prompt=npc_template,
        verbose=False
    )
    
    # Run the chain with all required variables
    response = npc_chain.invoke({
        "npc_name": npc_info["name"],
        "npc_description": npc_info["description"] + ". " + relevant_knowledge,
        "npc_personality": npc_info["personality"],
        "chat_history": chat_history,
        "human_input": user_input
    })
    
    # Extract the response text
    npc_response = response["text"].strip()
    
    # Update NPC memory
    st.session_state.npc_memories[npc_name].append({
        "human": user_input,
        "npc": npc_response
    })
    
    return npc_response

# === Game functions (unchanged) ===

# Function to check if user input completes a quest step
def check_quest_progress(user_input, current_location):
    # Update the number of turns taken
    st.session_state.turns_taken += 1
    
    progress_updates = []
    
    for quest_name, quest in st.session_state.quests.items():
        if quest["completed"]:
            continue
        
        current_step_idx = quest["current_step"]
        if current_step_idx >= len(quest["steps"]):
            continue
            
        current_step = quest["steps"][current_step_idx]
        
        # Check if user is in the right location and talking to the right NPC
        if (current_location == current_step["location"] and 
            st.session_state.current_npc == current_step["npc"]):
            
            # Check if user input seems to be looking for the item
            item_keywords = current_step["item"].lower().split()
            user_input_lower = user_input.lower()
            
            if any(keyword in user_input_lower for keyword in ["find", "look", "search", "get", "take"]) and \
               any(keyword in user_input_lower for keyword in item_keywords):
                
                # Complete this step
                current_step["completed"] = True
                st.session_state.inventory.append(current_step["item"])
                
                progress_updates.append(f"🎉 You found the {current_step['item']}!")
                
                # Move to next step or complete quest
                quest["current_step"] += 1
                if quest["current_step"] >= len(quest["steps"]):
                    quest["completed"] = True
                    st.session_state.completed_quests.append(quest_name)
                    st.session_state.inventory.append(quest["reward"]["item"])
                    progress_updates.append(f"✨ Quest completed: {quest_name}")
                    progress_updates.append(f"🏆 Reward: {quest['reward']['item']} - {quest['reward']['effect']}")
                else:
                    progress_updates.append(f"📋 Next step: {quest['steps'][quest['current_step']]['description']}")
    
    return progress_updates

# Function to get available quests for the current NPC
def get_available_quests(npc_name):
    available_quests = []
    
    for quest_name, quest in st.session_state.quests.items():
        if quest["completed"]:
            continue
            
        current_step_idx = quest["current_step"]
        if current_step_idx < len(quest["steps"]):
            current_step = quest["steps"][current_step_idx]
            if current_step["npc"] == npc_name:
                available_quests.append({
                    "name": quest_name,
                    "description": quest["description"],
                    "current_step": current_step["description"]
                })
    
    return available_quests

# Function to move to a different location
def move_to_location(new_location):
    if new_location in st.session_state.game_map[st.session_state.current_location]["connections"]:
        st.session_state.current_location = new_location
        st.session_state.current_npc = None  # Reset current NPC when moving
        return f"You travel to {new_location}. {st.session_state.game_map[new_location]['description']}"
    else:
        return f"You cannot travel directly to {new_location} from here."

# Function to handle user commands
def process_command(user_input):
    command = user_input.lower().strip()
    
    # Movement commands
    if command.startswith("go to ") or command.startswith("move to "):
        location = command.split(" to ", 1)[1].strip()
        for valid_location in st.session_state.game_map[st.session_state.current_location]["connections"]:
            if location.lower() in valid_location.lower():
                return move_to_location(valid_location)
        return f"Cannot find a path to {location} from here."
    
    # Talk to NPC command
    elif command.startswith("talk to "):
        npc_name = command.split("talk to ", 1)[1].strip()
        for valid_npc in st.session_state.game_map[st.session_state.current_location]["npcs"]:
            if npc_name.lower() in valid_npc.lower():
                st.session_state.current_npc = valid_npc
                greeting = st.session_state.npcs[valid_npc]["greeting"]
                return f"{valid_npc}: {greeting}"
        return f"There is no one named {npc_name} here."
    
    # Inventory command
    elif command == "inventory" or command == "check inventory":
        if not st.session_state.inventory:
            return "Your inventory is empty."
        return f"Inventory: {', '.join(st.session_state.inventory)}"
    
    # Look around command
    elif command == "look" or command == "look around":
        npcs_here = st.session_state.game_map[st.session_state.current_location]["npcs"]
        connections = st.session_state.game_map[st.session_state.current_location]["connections"]
        
        response = f"You are in {st.session_state.current_location}. {st.session_state.game_map[st.session_state.current_location]['description']}\n\n"
        
        if npcs_here:
            response += f"NPCs here: {', '.join(npcs_here)}\n\n"
        else:
            response += "There is no one here.\n\n"
            
        response += f"You can go to: {', '.join(connections)}"
        return response
    
    # Quest command
    elif command == "quests" or command == "check quests":
        active_quests = {name: quest for name, quest in st.session_state.quests.items() if not quest["completed"]}
        
        if not active_quests:
            return "You have no active quests."
            
        response = "Active Quests:\n"
        for name, quest in active_quests.items():
            current_step = quest["steps"][quest["current_step"]] if quest["current_step"] < len(quest["steps"]) else None
            response += f"• {name}: {quest['description']}\n"
            if current_step:
                response += f"  Current step: {current_step['description']} (Talk to {current_step['npc']} in {current_step['location']})\n"
        
        return response
    
    # Help command
    elif command == "help":
        return """
        Available commands:
        - go to [location]: Move to a connected location
        - talk to [npc]: Start a conversation with an NPC
        - look/look around: See details about your current location
        - inventory/check inventory: See what items you're carrying
        - quests/check quests: See your current quest progress
        - help: Show this help message
        
        When talking to an NPC, you can ask about quests or items they might know about.
        """
    
    # NPC conversation or default action
    elif st.session_state.current_npc:
        # Check if this input advances any quests
        quest_updates = check_quest_progress(command, st.session_state.current_location)
        
        # Get NPC response
        npc_response = interact_with_npc(st.session_state.current_npc, command)
        
        # Combine quest updates and NPC response
        if quest_updates:
            return f"{st.session_state.current_npc}: {npc_response}\n\n" + "\n".join(quest_updates)
        else:
            return f"{st.session_state.current_npc}: {npc_response}"
    
    else:
        return "Try 'look around' to see your surroundings, or 'help' for a list of commands."

# === Function to add a new NPC from file ===
def add_new_npc_from_file(file_path):
    try:
        npc_data = load_npc_from_file(file_path)
        st.session_state.npcs[npc_data["name"]] = npc_data
        
        # Add NPC to the appropriate location
        if npc_data["location"] in st.session_state.game_map:
            if npc_data["name"] not in st.session_state.game_map[npc_data["location"]]["npcs"]:
                st.session_state.game_map[npc_data["location"]]["npcs"].append(npc_data["name"])
                return f"Added new NPC: {npc_data['name']} in {npc_data['location']}"
        else:
            return f"Error: Location {npc_data['location']} does not exist in the game map"
    except Exception as e:
        return f"Error adding NPC: {str(e)}"

# === Function to add a new quest from file ===
def add_new_quest_from_file(file_path):
    try:
        quest_data = load_quest_from_file(file_path)
        st.session_state.quests[quest_data["name"]] = quest_data
        return f"Added new quest: {quest_data['name']}"
    except Exception as e:
        return f"Error adding quest: {str(e)}"


# === Main Gameplay Loop ===

# Initialize game on first run
if 'game_initialized' not in st.session_state:
    st.session_state.npcs = load_npcs()
    st.session_state.quests = load_quests()
    st.session_state.game_initialized = True

# Sidebar - Admin Controls
with st.sidebar:
    st.header("Admin Controls")
    
    # Upload new NPC file
    st.subheader("Add New NPC")
    npc_file = st.file_uploader("Upload NPC file (new_npc.txt)", type="txt", key="npc_uploader")
    if npc_file is not None:
        # Save the uploaded file
        with open("new_npc.txt", "wb") as f:
            f.write(npc_file.getvalue())
        
        result = add_new_npc_from_file("new_npc.txt")
        st.success(result)
    
    # Upload new Quest file
    st.subheader("Add New Quest")
    quest_file = st.file_uploader("Upload Quest file (new_quest.txt)", type="txt", key="quest_uploader")
    if quest_file is not None:
        # Save the uploaded file
        with open("new_quest.txt", "wb") as f:
            f.write(quest_file.getvalue())
        
        result = add_new_quest_from_file("new_quest.txt")
        st.success(result)
    
    # Game information
    st.subheader("Game Info")
    st.write(f"Current Location: {st.session_state.current_location}")
    st.write(f"Current NPC: {st.session_state.current_npc or 'None'}")
    st.write(f"Turns Taken: {st.session_state.turns_taken}")
    
    # Reset game button
    if st.button("Reset Game"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Main game interface
col1, col2 = st.columns([2, 1])

with col1:
    # Game world display
    st.subheader(f"Location: {st.session_state.current_location}")
    st.write(st.session_state.game_map[st.session_state.current_location]["description"])
    
    # NPC conversation
    if st.session_state.current_npc:
        st.subheader(f"Talking to: {st.session_state.current_npc}")
        
        # Get available quests from current NPC
        available_quests = get_available_quests(st.session_state.current_npc)
        if available_quests:
            for quest in available_quests:
                st.info(f"Quest: {quest['name']} - {quest['current_step']}")
    
    # User input
    user_input = st.text_input("What would you like to do?")
    
    if user_input:
        response = process_command(user_input)
        
        # Add to interaction history
        st.session_state.interaction_history.append({
            "input": user_input,
            "response": response
        })
        
        # Modern approach
        st.query_params.update(input="")
    
    # Display interaction history (most recent first)
    st.subheader("Recent Interactions")
    for interaction in reversed(st.session_state.interaction_history[-10:]):
        # Fixed text area height to meet minimum requirement of 68 pixels
        for idx, interaction in enumerate(reversed(st.session_state.interaction_history[-10:])):
            st.text_area("You", interaction["input"], height=68, disabled=True, key=f"input_{idx}")
            st.text_area("Game", interaction["response"], height=100, disabled=True, key=f"response_{idx}")
            st.markdown("---")

with col2:
    # Quick actions
    st.subheader("Quick Actions")
    
    # Movement buttons
    st.write("Movement:")
    connections = st.session_state.game_map[st.session_state.current_location]["connections"]
    cols = st.columns(min(3, len(connections)))
    for i, location in enumerate(connections):
        if cols[i % len(cols)].button(f"Go to {location}"):
            response = move_to_location(location)
            st.session_state.interaction_history.append({
                "input": f"go to {location}",
                "response": response
            })
            st.experimental_rerun()
    
    # Talk to NPCs buttons
    npcs_here = st.session_state.game_map[st.session_state.current_location]["npcs"]
    if npcs_here:
        st.write("Talk to:")
        npc_cols = st.columns(min(2, len(npcs_here)))
        for i, npc in enumerate(npcs_here):
            if npc_cols[i % len(npc_cols)].button(f"Talk to {npc}"):
                st.session_state.current_npc = npc
                response = f"{npc}: {st.session_state.npcs[npc]['greeting']}"
                st.session_state.interaction_history.append({
                    "input": f"talk to {npc}",
                    "response": response
                })
                st.experimental_rerun()
    
    # Common actions
    st.write("Actions:")
    actions_col1, actions_col2 = st.columns(2)
    
    if actions_col1.button("Look Around"):
        response = process_command("look around")
        st.session_state.interaction_history.append({
            "input": "look around",
            "response": response
        })
        st.experimental_rerun()
        
    if actions_col2.button("Inventory"):
        response = process_command("inventory")
        st.session_state.interaction_history.append({
            "input": "inventory",
            "response": response
        })
        st.experimental_rerun()
        
    if actions_col1.button("Quests"):
        response = process_command("quests")
        st.session_state.interaction_history.append({
            "input": "quests",
            "response": response
        })
        st.experimental_rerun()
        
    if actions_col2.button("Help"):
        response = process_command("help")
        st.session_state.interaction_history.append({
            "input": "help",
            "response": response
        })
        st.experimental_rerun()
    
    # Display inventory
    st.subheader("Inventory")
    if not st.session_state.inventory:
        st.write("Your inventory is empty")
    else:
        for item in st.session_state.inventory:
            st.write(f"• {item}")
    
    # Display quests
    st.subheader("Active Quests")
    active_quests = {name: quest for name, quest in st.session_state.quests.items() if not quest["completed"]}
    if not active_quests:
        st.write("No active quests")
    else:
        for name, quest in active_quests.items():
            progress = quest["current_step"] / len(quest["steps"]) * 100
            st.write(f"• {name}")
            st.progress(int(progress))
    
    # Display completed quests
    if st.session_state.completed_quests:
        st.subheader("Completed Quests")
        for quest_name in st.session_state.completed_quests:
            st.write(f"✅ {quest_name}")