import streamlit as st
import os
from typing import List, Union, Dict
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import ToolsRenderer, render_text_description_and_args
from langchain.agents.format_scratchpad import format_xml
import json
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from llm_agent.config import get_model, get_tools, AVAILABLE_MODELS, AVAILABLE_TOOLS
from utils.file_utils import get_combined_file_contents, count_tokens, get_file_tree
from streamlit_tree_select import tree_select
from llm_agent.token_cost_tracker import TokenCostTracker


class LLMAgent:
    def __init__(self, model_id, aws_profile):
        self.aws_profile = aws_profile
        self.current_model_index = AVAILABLE_MODELS.index(model_id)
        self.model = self._get_model_with_fallback()
        self.token_tracker = TokenCostTracker(model_id)

    def _get_model_with_fallback(self):
        while self.current_model_index < len(AVAILABLE_MODELS):
            try:
                model = get_model(AVAILABLE_MODELS[self.current_model_index], self.aws_profile)
                return model
            except Exception as e:
                if 'ThrottlingException' in str(e):
                    st.warning(f'Model {AVAILABLE_MODELS[self.current_model_index]} is throttled. Trying next model...')
                    self.current_model_index += 1
                else:
                    raise e
        raise Exception('All models are throttled. Please try again later.')

    def run(self, user_prompt: str, chat_history: List[Union[HumanMessage, AIMessage]], context: str = None):
        try:
            system_message = """You are a helpful AI assistant. If provided with code context, 
            you can help analyze and discuss it. Always format code blocks with appropriate 
            language tags."""
            
            messages = [HumanMessage(content=system_message)]    
            
            if context:
                messages.append(HumanMessage(content=f"Context:\n{context}"))
            
            # Add chat history from the parameter
            if chat_history:
                messages.extend(chat_history)
            
            # Add current user message
            # current_user_message = HumanMessage(content=user_prompt) # already in chat history
            # messages.append(current_user_message)
                        
            # Count input tokens
            input_text = system_message
            if context:
                input_text += context
            if chat_history:
                input_text += " ".join([msg.content for msg in chat_history])
            input_text += user_prompt

            input_tokens = count_tokens(input_text)
            self.token_tracker.add_input_tokens(input_tokens)

            # Create a container for streaming output
            container = st.empty()
            response_text = ""
            
            for chunk in self.model.stream(messages):
                if chunk.content:
                    response_text += chunk.content
                    # Update the container with accumulated text
                    container.markdown(response_text + "▌")

            # Final update without cursor
            container.markdown(response_text)

            # Count output tokens
            output_tokens = count_tokens(response_text)
            self.token_tracker.add_output_tokens(output_tokens)

            return response_text
        except ValueError as e:
            if 'ExpiredTokenException' in str(e):
                error_message = """
                ### 🔑 AWS Session Token has expired

                Please refresh your AWS credentials using one of these methods:

                1. If using AWS CLI profile:
                ```bash
                aws sso login --profile your-profile-name
                ```

                2. If using environment variables, get new credentials and set:
                ```bash
                export AWS_ACCESS_KEY_ID=your_access_key
                export AWS_SECRET_ACCESS_KEY=your_secret_key
                export AWS_SESSION_TOKEN=your_session_token
                ```

                Then restart the Streamlit application.
                """
                st.error(error_message)
                return "ExpiredTokenException"
            elif 'ThrottlingException' in str(e):
                st.warning(f'Model {AVAILABLE_MODELS[self.current_model_index]} is throttled. Trying next model...')
                self.current_model_index += 1
                if self.current_model_index >= len(AVAILABLE_MODELS):
                    raise Exception('All models are throttled. Please try again later.')
                self.model = self._get_model_with_fallback()
                return self.run(user_prompt, chat_history, context)
            else:
                raise e
        except Exception as e:
            raise e

def truncate_name(name: str, max_length: int = 40) -> str:
    """Truncate filename if it's too long"""
    if len(name) <= max_length:
        return name
    return f"{name[:max_length-3]}..."

def render_file_tree(folder_path: str) -> Dict:
    file_tree = get_file_tree(folder_path)

    def build_nodes(node, path=""):
        local_nodes = []
        for name, content in sorted(node.items(), key=lambda x: (x[1] is not None, x[0])):
            full_path = os.path.join(path, name)
            display_name = truncate_name(name)
            if content is None:  # File
                local_nodes.append({
                    "label": display_name,
                    "value": full_path,
                    "title": name
                })
            else:  # Folder
                children = build_nodes(content, full_path)
                local_nodes.append({
                    "label": display_name,
                    "value": full_path,
                    "children": children,
                    "title": name
                })
        return local_nodes

    root_name = os.path.basename(folder_path)
    nodes = [{
        "label": root_name,
        "value": folder_path,
        "children": build_nodes(file_tree, folder_path),
        "title": root_name
    }]
    return nodes


def update_selected_files():
    st.session_state.selected_files = [
        file for file in st.session_state.selected_files
        if any(file.startswith(folder) for folder in st.session_state.folder_paths)
    ]

def save_conversation(messages, token_tracker, name=None):
    """Save conversation to a JSON file."""
    if not name:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    conversations_dir = "conversations"
    os.makedirs(conversations_dir, exist_ok=True)

    data = {
        'messages': messages,
        'token_data': token_tracker.to_dict()
    }

    filepath = os.path.join(conversations_dir, f"{name}.json")
    with open(filepath, "w") as f:
        json.dump(data, f)

def load_conversations():
    """Load all saved conversations from the conversations directory."""
    conversations_dir = "conversations"
    if not os.path.exists(conversations_dir):
        return {}
    
    conversations = {}
    for filename in os.listdir(conversations_dir):
        if filename.endswith(".json"):
            with open(os.path.join(conversations_dir, filename), "r") as f:
                name = filename[:-5]  # Remove .json
                conversations[name] = json.load(f)
    return conversations

def delete_conversation(name: str):
    conversations_dir = "conversations"
    filepath = os.path.join(conversations_dir, f"{name}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def update_token_counts():
    """Calculate and display the token counts for selected files, chat history, and total."""
    # Calculate token count for selected files
    if st.session_state.selected_files:
        selected_file_contents = get_combined_file_contents([f for f in st.session_state.selected_files if os.path.isfile(f)])
        file_token_count = count_tokens(selected_file_contents)
    else:
        file_token_count = 0  # No files selected

    st.markdown(f"**Tokens from files:** {file_token_count}")

    # if agent is not None and agent.chat_history:
    #     # Calculate token count for chat history
    #     chat_history_token_count = count_tokens(" ".join([msg.content for msg in agent.chat_history])) if agent.chat_history else 0
    #     st.markdown(f"**Tokens from chat history:** {chat_history_token_count}")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.message_placeholder = container.empty()
        self.token_count = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.token_count += 1
        self.text += token
        self.message_placeholder.markdown(f"{self.text}▌\n\n*Tokens: {self.token_count}*")

    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends running."""
        self.message_placeholder.markdown(f"{self.text}\n\n*Total Tokens: {self.token_count}*")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM errors."""
        self.message_placeholder.error(f"Error: {str(error)}")


def main():
    st.set_page_config(page_title="Chitti", page_icon=":speech_balloon:", layout="wide")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "aws_profile" not in st.session_state:
        st.session_state.aws_profile = ""
    if "model_id" not in st.session_state:
        st.session_state.model_id = AVAILABLE_MODELS[0]
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "folder_paths" not in st.session_state:
        st.session_state.folder_paths = []

    # Main chat interface
    # st.title("🐧 CHITTI")
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap" rel="stylesheet">
    <style>
    .custom-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        font-family: 'Dancing Script', cursive; /* Modern cursive font */
    }
    </style>
    <div class="custom-title">🐧 Chitti</div>
    """, unsafe_allow_html=True)

    # Tree select hover for truncated titles TODO: move to utils or style section
    st.markdown("""
    <style>
    .streamlit-tree-select span {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 180px;
        display: inline-block;
        position: relative;
    }

    .streamlit-tree-select span:hover::after {
        content: attr(title);
        position: absolute;
        left: 0;
        top: 100%;
        background: white;
        padding: 2px 6px;
        border: 1px solid #ddd;
        border-radius: 3px;
        z-index: 1000;
        white-space: normal;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize token tracker in session state if not exists
    if "token_tracker" not in st.session_state:
        st.session_state.token_tracker = TokenCostTracker(st.session_state.model_id)

    # Initialize agent
    agent = LLMAgent(st.session_state.model_id, st.session_state.aws_profile)
    agent.token_tracker = st.session_state.token_tracker  # Use the session state tracker

    # Sidebar
    with st.sidebar:
        st.header(":gear: Settings")
        st.session_state.aws_profile = st.text_input("AWS Profile", value=st.session_state.aws_profile)
        st.session_state.model_id = st.selectbox("Select Model", AVAILABLE_MODELS,
                                                index=AVAILABLE_MODELS.index(st.session_state.model_id))

        # File Tree Section
        st.header(":file_folder: Code Context")
        new_folder_path = st.text_input("Enter folder path")
        if st.button("Add Folder"):
            if new_folder_path and os.path.isdir(new_folder_path) and new_folder_path not in st.session_state.folder_paths:
                st.session_state.folder_paths.append(new_folder_path)
                update_selected_files()
                # Calculate token count after updating selected files
                if st.session_state.selected_files:
                    update_token_counts()
                    # selected_file_contents = get_combined_file_contents([f for f in st.session_state.selected_files if os.path.isfile(f)])
                    # token_count = count_tokens(selected_file_contents)
                    # st.markdown(f"**Total tokens in context:** {token_count}")

            elif not os.path.isdir(new_folder_path):
                st.error("Invalid folder path")

        # File tree with scrollable container
        file_tree_container = st.container()
        with file_tree_container:
            for i, folder_path in enumerate(st.session_state.folder_paths):
                st.markdown(f"**{os.path.basename(folder_path)}**")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(folder_path)
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.folder_paths.pop(i)
                        update_selected_files()
                        st.rerun()

                # Render file tree in scrollable container
                with st.expander("View folder structure", expanded=True):
                    folder_nodes = render_file_tree(folder_path)
                    selected = tree_select(folder_nodes, key=f"tree_{i}")
                    folder_files = set(selected['checked'])
                    st.session_state.selected_files = list(
                        (set(st.session_state.selected_files) - set(f for f in st.session_state.selected_files if f.startswith(folder_path)))
                        | folder_files
                    )
        
        # Calculate token count only for selected files (not folders)
        if st.session_state.selected_files:
            update_token_counts()
            # selected_file_contents = get_combined_file_contents([f for f in st.session_state.selected_files if os.path.isfile(f)])
            # token_count = count_tokens(selected_file_contents)
            # st.markdown(f"**Total tokens in context:** {token_count}")


        # Conversation History Section
        st.header("💾 Saved Conversations")
        conversations = load_conversations()

        with st.expander("View Saved Conversations", expanded=False):
            for conv_name in conversations:
                col1, col2 = st.columns([4,1])
                with col1:
                    display_name = truncate_name(conv_name)
                    if st.button(display_name, key=f"load_{conv_name}"):
                        conv_data = conversations[conv_name]
                        st.session_state.messages = conv_data['messages']
                        st.session_state.token_tracker = TokenCostTracker.from_dict(
                            conv_data.get('token_data', {}),
                            st.session_state.model_id
                        )
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{conv_name}"):
                        if delete_conversation(conv_name):
                            st.success(f"Deleted {conv_name}")
                            st.rerun()
                        else:
                            st.error("Failed to delete conversation")

        # Save current conversation
        if st.session_state.messages:
            save_name = st.text_input("Save conversation as:", key="save_name")
            if st.button("Save Conversation"):
                if save_name:
                    save_conversation(st.session_state.messages, st.session_state.token_tracker, save_name)
                    st.success(f"Conversation saved as {save_name}")

    # Main chat container with scrollbar
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to talk about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            context = get_combined_file_contents(st.session_state.selected_files) if st.session_state.selected_files else None
            chat_history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]
            response = agent.run(prompt, chat_history, context)
            # st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Initialize token tracker in session state
    if "token_tracker" not in st.session_state:
        st.session_state.token_tracker = TokenCostTracker()

    # In sidebar, add token and cost tracking
    with st.sidebar:
        st.header("💰 Session Costs")

        # Display file context tokens
        if st.session_state.selected_files:
            selected_file_contents = get_combined_file_contents([f for f in st.session_state.selected_files if os.path.isfile(f)])
            file_token_count = count_tokens(selected_file_contents)
            st.markdown(f"**Context Tokens:** {file_token_count:,}")

        # Display chat history tokens and costs
        costs = st.session_state.token_tracker.calculate_costs()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Input Tokens:**")
            st.markdown(f"{costs['input_tokens']:,}")
            st.markdown("**Output Tokens:**")
            st.markdown(f"{costs['output_tokens']:,}")

        with col2:
            st.markdown("**Input Cost:**")
            st.markdown(f"${costs['input_cost']:.3f}")
            st.markdown("**Output Cost:**")
            st.markdown(f"${costs['output_cost']:.3f}")

        st.markdown("**Total Cost:**")
        st.markdown(f"${costs['total_cost']:.3f}")

if __name__ == "__main__":
    main()
