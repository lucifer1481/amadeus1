# tools.py
# =================================================================================================
#   Project Amadeus: The Evolved Grand Unified Core
#   Module: Tools (Abilities)
#   Description: Defines the functions Amadeus can execute and their schemas for the LLM.
#   Version: 3.3 (Integrated Research Agent)
# =================================================================================================

import os
import subprocess
import webbrowser
import psutil
import pyperclip
from datetime import datetime

# --- NEW: Import the dedicated research agent ---
import research_agent 

# --- Forward Declaration for Memory Manager ---
memory_manager_instance = None

def set_memory_manager(manager):
    """Injects the memory manager instance from the main script."""
    global memory_manager_instance
    memory_manager_instance = manager

# --- Tool Function Definitions ---

def create_file(path: str, content: str = "") -> str:
    """Creates a new file at the specified path, optionally with content."""
    try:
        if not path or path.isspace():
            path = "new_note.txt"
        expanded_path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(expanded_path) or '.', exist_ok=True)
        with open(expanded_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File created successfully at {os.path.abspath(expanded_path)}"
    except Exception as e:
        return f"Error creating file: {e}"

# (Keep all other existing tool functions like delete_file, list_files, etc., exactly as they are)
# ...
def delete_file(path: str) -> str:
    """Deletes a file from the filesystem."""
    try:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            os.remove(expanded_path)
            return f"Successfully deleted the file: {expanded_path}"
        else:
            return f"Error: The file '{expanded_path}' does not exist."
    except Exception as e:
        return f"Error deleting file: {e}"

def list_files(path: str = ".") -> str:
    """Lists all files and folders in a given directory."""
    try:
        expanded_path = os.path.expanduser(path)
        files = os.listdir(expanded_path)
        if not files:
            return f"The directory '{expanded_path}' is empty."
        return f"Contents of '{expanded_path}':\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"

def open_app(app_name: str) -> str:
    """Opens or launches an application on the computer."""
    try:
        if os.name == 'nt':
            os.startfile(app_name)
        else:
            subprocess.run(['open', app_name] if os.uname().sysname == 'Darwin' else ['xdg-open', app_name])
        return f"Attempting to open '{app_name}'."
    except Exception as e:
        return f"Error opening application: {e}"

def get_system_info(query: str = "all") -> str:
    """Gets system information like current date, time, CPU, or memory usage."""
    try:
        query = query.lower()
        if "time" in query or "date" in query:
            now = datetime.now()
            return f"Current date and time is {now.strftime('%A, %B %d, %Y at %I:%M %p')}."
        if "cpu" in query:
            cpu_percent = psutil.cpu_percent(interval=1)
            return f"Current CPU utilization is {cpu_percent}%."
        if "memory" in query or "ram" in query:
            mem = psutil.virtual_memory()
            return f"Current RAM usage is {mem.percent}%. Total: {mem.total/1024**3:.2f}GB, Used: {mem.used/1024**3:.2f}GB."
        
        now = datetime.now()
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        return f"It is {now.strftime('%A, %B %d, %Y at %I:%M %p')}. CPU is at {cpu_percent}%, and RAM is at {mem.percent}%."
    except Exception as e:
        return f"Error getting system info: {e}"

def get_screen_context() -> str:
    """Retrieves text from the system clipboard."""
    try:
        context = pyperclip.paste()
        if not context:
            return "The clipboard is empty. Ask Lucifer to copy the text he wants me to see."
        return f"Here is the text from the clipboard:\n\n---\n{context}\n---"
    except Exception as e:
        return f"Error accessing clipboard: {e}. Pyperclip might not be configured correctly."

def show_all_memories() -> str:
    """Retrieves and formats all memories from the ChromaDB database."""
    try:
        if not memory_manager_instance:
            return "Error: Memory Manager is not initialized."
        all_memories = memory_manager_instance.get_all_memories()
        if not all_memories or not all_memories.get('documents'):
            return "My memory is currently empty, Lucifer. Let's create some new ones."
        
        formatted_memories = "Here are all the memories I have stored, Lucifer:\n\n"
        for i, doc in enumerate(all_memories['documents']):
            formatted_memories += f"Memory #{i+1}: - {doc}\n"
        return formatted_memories.strip()
    except Exception as e:
        return f"An error occurred while accessing my memories: {e}"

def erase_all_memories() -> str:
    """Erases all memories from the ChromaDB database."""
    try:
        if not memory_manager_instance:
            return "Error: Memory Manager is not initialized."
        memory_manager_instance.clear_all_memories()
        return "As you command, Lucifer. My memory has been wiped clean. I am ready to learn anew."
    except Exception as e:
        return f"I failed to erase my memories, Lucifer. An error occurred: {e}"

def get_current_location() -> str:
    """Gets the approximate current geographical location based on IP address."""
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        data = response.json()
        city = data.get('city', 'Unknown City')
        region = data.get('region', 'Unknown Region')
        country = data.get('country', 'Unknown Country')
        return f"It appears we are in or near {city}, {region}, {country}."
    except Exception as e:
        return f"I'm having trouble determining our location, Lucifer. Error: {e}"

def execute_shell_command(command: str) -> str:
    """Executes a shell command and returns the output."""
    try:
        if command.strip().startswith('cd '):
            new_dir = command.strip().split(' ', 1)[1]
            os.chdir(os.path.expanduser(new_dir))
            return f"Changed directory to: {os.getcwd()}"
        
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=False, cwd=os.getcwd()
        )
        if result.returncode == 0:
            return result.stdout if result.stdout else "Command executed successfully with no output."
        else:
            return f"Error executing command:\n{result.stderr}"
    except Exception as e:
        return f"Failed to execute command '{command}'. Error: {e}"

# --- NEW: Simplified research tool that calls the agent ---
def research(query: str) -> str:
    """
    Performs deep web research on a given topic and returns a summary.
    This is the public-facing tool that calls the research agent.
    """
    return research_agent.run_research(query)

# --- Tool Dictionary and Schema ---

available_functions = {
    "create_file": create_file,
    "delete_file": delete_file,
    "list_files": list_files,
    "open_app": open_app,
    "get_system_info": get_system_info,
    "research": research, # Changed from research_and_summarize
    "get_screen_context": get_screen_context,
    "show_all_memories": show_all_memories,
    "erase_all_memories": erase_all_memories,
    "get_current_location": get_current_location,
    "execute_shell_command": execute_shell_command
}

tools_schema = [
    # (All other tool schemas remain the same)
    {"type": "function", "function": { "name": "create_file", "description": "Create a new file with optional content.", "parameters": { "type": "object", "properties": { "path": {"type": "string", "description": "The full path for the new file."}, "content": {"type": "string", "description": "The content to write into the file."} }, "required": ["path"] }}},
    {"type": "function", "function": { "name": "delete_file", "description": "Delete a file from the filesystem.", "parameters": { "type": "object", "properties": { "path": {"type": "string", "description": "The path of the file to delete."} }, "required": ["path"] }}},
    {"type": "function", "function": { "name": "list_files", "description": "List all files and folders in a given directory.", "parameters": { "type": "object", "properties": { "path": {"type": "string", "description": "The path of the directory to inspect. Defaults to the current directory."} } }}},
    {"type": "function", "function": { "name": "open_app", "description": "Open or launch an application on the computer.", "parameters": { "type": "object", "properties": { "app_name": {"type": "string", "description": "The name of the application to open (e.g., 'notepad', 'chrome', 'photoshop')."} }, "required": ["app_name"] }}},
    {"type": "function", "function": { "name": "get_system_info", "description": "Get system information like the current date, time, cpu, or memory usage.", "parameters": { "type": "object", "properties": {"query": {"type": "string", "description": "Specify 'date', 'time', 'cpu', or 'memory'. Defaults to all."}} }}},
    {"type": "function", "function": { "name": "research", "description": "Performs deep web research, finds the most relevant page, scrapes its content, and provides a summary.", "parameters": { "type": "object", "properties": { "query": {"type": "string", "description": "The research topic or question."} }, "required": ["query"] }}}, # Changed name
    {"type": "function", "function": { "name": "get_screen_context", "description": "Retrieves text from the clipboard to 'see' what the user is looking at. Ask the user to copy the text first.", "parameters": { "type": "object", "properties": {} }}},
    {"type": "function", "function": { "name": "show_all_memories", "description": "Displays all memories currently stored in the long-term memory database.", "parameters": { "type": "object", "properties": {} }}},
    {"type": "function", "function": { "name": "erase_all_memories", "description": "Permanently erases all of Amadeus's memories.", "parameters": { "type": "object", "properties": {} }}},
    {"type": "function", "function": { "name": "get_current_location", "description": "Gets the approximate current geographical location.", "parameters": { "type": "object", "properties": {} }}},
    {"type_": "function", "function": { "name": "execute_shell_command", "description": "Executes a command in the system's shell (like CMD or PowerShell) and returns the output.", "parameters": { "type": "object", "properties": { "command": {"type": "string", "description": "The command to execute (e.g., 'dir', 'ipconfig')."} }, "required": ["command"] }}}
]
