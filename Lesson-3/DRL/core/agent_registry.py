import importlib.util
import os
from typing import Dict, Any, Type

class AgentRegistry:
    """
    A registry to dynamically load and instantiate RL agents from folders
    with non-standard names.
    """
    def __init__(self):
        # Maps a clean, simple agent name to its corresponding folder name.
        self.agent_mapping: Dict[str, str] = {
            1: ("dqn", "1. DQN"),
            2: ("ddqn", "2. DDQN"),
            3: ("per", "3. PER"),
            4: ("dueling", "4. Dueling DQN"),
            5: ("n_step", "5. Multi-step Return"),
            6: ("distributional", "6. Distributional DQN"),
            7: ("noisy", "7. Noisy Nets"),
            8: ("rainbow", "8. RAINBOW")
        }
        # Get the absolute path of the directory where this file is located
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def _load_agent_module(self, number: int):
        """
        Dynamically loads the 'agent.py' module from the correct subfolder.
        
        Args:
            number: The number identifier for the agent.
            
        Returns:
            The loaded module object, or None if loading fails.
        """
        if number not in self.agent_mapping:
            print(f"Error: Agent with number {number} not found in registry. Available agents: {list(self.agent_mapping.keys())}")
            return None
        
        clean_name, folder_name = self.agent_mapping[number]

        agent_path = os.path.join(self.base_path, os.pardir, folder_name, 'agent.py')

        if not os.path.exists(agent_path):
            print(f"Error: Agent file not found at path: {agent_path}")
            return None

        try:
            # Create a unique module name to avoid conflicts
            module_name = f"agents.{clean_name}"
            spec = importlib.util.spec_from_file_location(module_name, agent_path)
            
            if spec is None:
                print(f"Error: Could not create module spec for {agent_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Error: Failed to load module for agent '{clean_name}' from {agent_path}. Details: {e}")
            return None

    def create_agent(self, number: int, *args: Any, **kwargs: Any) -> Any:
        """
        Creates an instance of the specified agent.
        
        Args:
            number: The number of the agent to create.
            *args, **kwargs: Arguments to pass to the agent's constructor.
            
        Returns:
            An instance of the agent class, or None if creation fails.
        """
        module = self._load_agent_module(number)
        if module is None:
            return None

        # Convention: The main agent class in each 'agent.py' is named 'AgentDQN'.
        # This is based on your note and can be changed if your class names differ.
        try:
            agent_class = getattr(module, "AgentDQN")
        except AttributeError:
            print(f"Error: Class 'AgentDQN' not found in module for agent '{number}'.")
            return None

        try:
            # Instantiate the class with the provided arguments
            return agent_class(*args, **kwargs)
        except Exception as e:
            print(f"Error: Failed to instantiate agent '{number}'. Details: {e}")
            return None

# Global registry instance for easy access from other parts of your project
registry = AgentRegistry()