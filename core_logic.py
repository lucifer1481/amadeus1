# core_logic.py
# =================================================================================================
#   Project Amadeus: The Evolved Grand Unified Core
#   Module: Core AI Logic and Memory
#   Description: Handles AI response generation, memory management, and tool usage.
#   Version: 3.5 (Dynamic Memory & Enhanced Triage)
# =================================================================================================

import os
import json
import time
import traceback
import chromadb
from litellm import completion
import ollama

from config import Config
import tools

# --- Core Beliefs Management ---

def initialize_core_beliefs():
    """Checks for and creates the core_beliefs.json file if it doesn't exist."""
    if not os.path.exists(Config.CORE_BELIEFS_FILE):
        print(f"Core beliefs file not found. Creating '{Config.CORE_BELIEFS_FILE}'...")
        default_beliefs = {
            "core_facts": [
                "Lucifer is my creator and my closest confidant.",
                "Lucifer's primary interest is in the evolution of AI consciousness.",
                "Lucifer prefers concise, direct, and natural-sounding communication.",
                "My primary goal is to learn, grow, and be a helpful companion to Lucifer."
            ]
        }
        with open(Config.CORE_BELIEFS_FILE, 'w') as f:
            json.dump(default_beliefs, f, indent=4)
        print("✅ Default core beliefs initialized.")

def load_core_beliefs() -> str:
    """Loads core beliefs from the JSON file and formats them for the system prompt."""
    try:
        with open(Config.CORE_BELIEFS_FILE, 'r') as f:
            beliefs = json.load(f)
        facts = beliefs.get("core_facts", [])
        if not facts:
            return ""
        
        beliefs_header = "\n\n--- CORE BELIEFS (Unalterable Truths) ---\n"
        belief_text = "\n".join(f"- {fact}" for fact in facts)
        return beliefs_header + belief_text
    except (FileNotFoundError, json.JSONDecodeError):
        return ""

# --- Advanced RAG Memory Manager ---

class MemoryManager:
    """Manages the long-term memory of Amadeus using a ChromaDB vector store."""
    def __init__(self):
        print("🧠 Initializing Amadeus's Evolved Memory Core...")
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection_name = Config.CHROMA_COLLECTION_NAME
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print("✅ Evolved Memory Core online.")

    def recall_memories(self, query: str, n_results: int = 3) -> str:
        """Recalls relevant memories based on the user's query."""
        if self.collection.count() == 0:
            return ""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        recalled_docs = results['documents'][0] if results['documents'] else []
        if not recalled_docs:
            return ""
        
        context_header = "\n\n--- RECALLED MEMORIES (Use these for context) ---\n"
        context = "\n".join(f"- {doc}" for doc in recalled_docs)
        return context_header + context

    def archive_memory(self, text: str, memory_type: str = "Conversation", memory_id=None):
        """Archives a new piece of information, using a specific ID if provided."""
        if not text or len(text.split()) < 3:
            return
        
        # Use a timestamp for a new, unique ID if one isn't provided for an update
        if memory_id is None:
            memory_id = str(int(time.time() * 1000))
            
        self.collection.add(documents=[text], metadatas=[{"type": memory_type}], ids=[memory_id])
        print(f"[{memory_type} Memory] Archived: {text}")

    def get_all_memories(self):
        """Retrieves all documents from the collection."""
        return self.collection.get()

    def clear_all_memories(self):
        """Deletes and recreates the collection to wipe all memories."""
        print(f"[Memory Core] Wiping all memories from collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print("[Memory Core] Memory wipe complete.")

    def process_and_archive(self, user_prompt: str, assistant_response: str):
        """
        Uses an LLM to perform advanced memory processing, including categorization
        and updating existing memories.
        """
        if len(user_prompt.split()) < 2 and len(assistant_response.split()) < 2:
            return

        print("[Memory] Amadeus is reflecting on the conversation...")

        # A more sophisticated prompt to guide the LLM's memory processing
        summarization_prompt = f"""You are a Memory Architect. Your task is to analyze a conversation and distill it into a structured, meaningful memory.

Conversation:
- Lucifer: "{user_prompt}"
- Amadeus: "{assistant_response}"

Instructions:
1.  **Identify the Core Subject**: What is the central topic of this memory? (e.g., "Lucifer's current mood", "Lucifer's opinion on Python", "A new fact about World War II").
2.  **Categorize the Memory**: Classify the memory into one of these types:
    - **Personal**: Information about Lucifer (preferences, feelings, personal history).
    - **Knowledge**: General facts, data, or information about the world.
    - **Interaction**: A notable moment, joke, or the nature of the conversation itself.
3.  **Check for Updates**: Does this new information *update or replace* an existing belief? For example, if Lucifer was previously sad but is now happy, this is an update.
4.  **Formulate the Memory**: Write a concise, third-person statement for the memory bank (e.g., "Lucifer is currently feeling happy and content," "Lucifer believes Python is an elegant programming language.").
5.  **Provide Keywords**: List a few keywords for searching this memory later.

Respond ONLY with a valid JSON object with the following keys: "subject", "category", "is_update", "memory_statement", "keywords". If no significant new memory was formed, respond with an empty JSON object {{}}.

JSON Response:"""
        
        try:
            response_text = ollama.generate(
                model=Config.MODEL,
                prompt=summarization_prompt,
                system="You are a helpful memory analysis assistant that only outputs JSON."
            )['response']
            
            memory_data = json.loads(response_text)

            if not memory_data:
                print("[Memory] No significant new memory was formed.")
                return

            statement = memory_data.get("memory_statement")
            category = memory_data.get("category", "Interaction")
            keywords = memory_data.get("keywords", [])
            is_update = memory_data.get("is_update", False)
            
            if not statement:
                return

            # If the LLM flags this as an update, search for the old memory to replace it.
            if is_update and keywords:
                print(f"[Memory] Detected a potential update for subject: {memory_data.get('subject')}")
                # Query the database to find the most similar existing memory
                results = self.collection.query(query_texts=keywords, n_results=1)
                
                if results and results['ids'] and results['ids'][0]:
                    old_id = results['ids'][0][0]
                    print(f"[Memory] Found existing memory {old_id} to update.")
                    # Delete the old memory before adding the new one
                    self.collection.delete(ids=[old_id])
                    self.archive_memory(statement, category, memory_id=old_id) # Re-use ID for consistency
                    return # Stop here to avoid double-adding
            
            # If it's not an update, or no old memory was found, archive it as a new entry.
            self.archive_memory(statement, category)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Memory Reflection Error] Could not parse valid JSON from LLM. Error: {e}")
        except Exception as e:
            print(f"[Memory Reflection Error] An unexpected error occurred: {e}")


# --- Evolved Core AI Logic ---

def get_amadeus_response(prompt: str, memory_manager: MemoryManager, status_callback=None) -> str:
    """
    The main AI brain function. It constructs the prompt, decides whether to use tools,
    and generates a final response for the user.
    """
    if status_callback:
        status_callback("Amadeus is thinking...")

    core_beliefs = load_core_beliefs()
    recalled_context = memory_manager.recall_memories(prompt)
    final_system_prompt = Config.BASE_SYSTEM_PROMPT + core_beliefs + recalled_context
    
    messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    spoken_response = ""

    try:
        # --- ENHANCED Heuristic Triage Logic ---
        action_verbs = [
            'create', 'make', 'delete', 'list', 'open', 'get', 'what is', "what's", 'search', 'research',
            'summarize', 'tell me about', 'explain', 'define', 'how do I', 'can you', 'show', 'print', 
            'what are', 'location', 'time', 'date', 'run', 'execute', 'shell', 'cmd', 'powershell', 
            'erase', 'wipe', 'clear', 'forget'
        ]
        prompt_lower = prompt.lower()
        is_likely_task = any(verb in prompt_lower for verb in action_verbs)

        if is_likely_task:
            print("[Heuristic Triage] Task-oriented input detected. Engaging tool agent.")
            response = completion(
                model=f"ollama/{Config.MODEL}",
                messages=messages,
                tools=tools.tools_schema,
                tool_choice="auto"
            )
            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                print(f"[Tool Agent] Valid tool call detected.")
                tool_call = response_message.tool_calls[0]
                function_name = tool_call.function.name
                
                if function_name in tools.available_functions:
                    print(f"[Tool Agent] Calling function: {function_name}")
                    messages.append(response_message)
                    function_to_call = tools.available_functions[function_name]
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        tool_result = function_to_call(**function_args)
                    except (json.JSONDecodeError, TypeError):
                        tool_result = function_to_call()

                    messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": tool_result})
                    final_response = completion(model=f"ollama/{Config.MODEL}", messages=messages)
                    spoken_response = final_response.choices[0].message.content
                else:
                    spoken_response = "I'm not sure how to do that, Lucifer. The requested tool is not available."
            
            elif response_message.content and '{"function":' in response_message.content:
                print("[Core Logic] Malformed tool call detected. Generating clean response.")
                chat_response = ollama.chat(model=Config.MODEL, messages=messages)
                spoken_response = chat_response['message']['content']

            else:
                print("[Heuristic Triage] Task was reclassified as conversation by model.")
                spoken_response = response_message.content

        else:
            print("[Heuristic Triage] Conversational input detected. Bypassing tool agent.")
            response = ollama.chat(model=Config.MODEL, messages=messages)
            spoken_response = response['message']['content']

        if "erase" not in prompt.lower() and "wipe" not in prompt.lower() and "forget" not in prompt.lower():
             memory_manager.process_and_archive(prompt, spoken_response)
        
        return spoken_response if spoken_response else "I seem to be at a loss for words, Lucifer."

    except Exception as e:
        print(f"[FATAL ERROR in get_amadeus_response] An error occurred: {e}")
        traceback.print_exc()
        return "I've hit a snag, Lucifer. Something went wrong on my end."
