import streamlit as st
import os
from typing import List, Union, Dict
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import ToolsRenderer, render_text_description_and_args
from langchain.agents.format_scratchpad import format_xml
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler

from llm_agent.config import get_model, get_tools, AVAILABLE_MODELS, AVAILABLE_TOOLS
from utils.file_utils import get_combined_file_contents, count_tokens, get_file_tree
from streamlit_tree_select import tree_select


class LLMAgent:
    def __init__(self, model_id, aws_profile):
        self.aws_profile = aws_profile
        self.current_model_index = AVAILABLE_MODELS.index(model_id)
        self.model = self._get_model_with_fallback()
        self.tools = get_tools(st.session_state.selected_tools)
        self.model.bind_tools(self.tools)
        self.system_prompt = self._create_system_prompt()
        self.agent = self._create_agent()
        self.agent_executor = self._create_agent_executor()

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

    def _create_system_prompt(self):
        AGENT_PROMPT = """You are an excellent coder. Help the user with their coding tasks. 
        You are optionally given the entire codebase of the user in your context. 
        It is in the format like below where first line has the File path and then the content follows.

        File: <filepath>
        <Content of the file>.

        Now below is the current codebase of the user:
        {codebase}
        
        **ONLY** use tools when necessary. Think step by step if it is needed to meet the given objective.
        You have access to the following tools:
        {tools}

        ALWAYS wrap code blocks with backticks and language tag. Give Filename in the format File: <filename>.
        For example, \n\nFile: <filename>\n\n```python\n

        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
        Valid "action" values: "Final Answer", {tool_names}.
        DON'T invent actions other than given ones.
        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{
          "action": $TOOL_NAME,
          "action_input": $INPUT
        }}
        ```
        
        Follow this format:
        
        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)

        Thought: I know what to respond
        Action:
        ```
        {{
          "action": "Final Answer",
          "action_input": "Final response to human"
        }}
        ```

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools ONLY if necessary.
        Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""

        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", AGENT_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{prompt} {agent_scratchpad}"
                         "(reminder to respond in a JSON blob no matter what)"
                         "(reminder that there is codebase in the system prompt that contains file paths & their contents)"),
            ]
        )

        tools_renderer: ToolsRenderer = render_text_description_and_args
        system_prompt = system_prompt.partial(
            tools=tools_renderer(list(self.tools)),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        return system_prompt

    def _create_agent(self):
        return create_structured_chat_agent(
            self.model,
            tools=self.tools,
            prompt=self.system_prompt,
            stop_sequence=True
        )

    def _create_agent_executor(self):
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

    def run(self, user_prompt: str, chat_history: List[Union[HumanMessage, AIMessage]], context: str):
        while True:
            try:
                agent_input = {
                    "prompt": user_prompt,
                    "chat_history": chat_history,
                    "codebase": context,
                    "agent_scratchpad": format_xml([])  # Initialize with empty scratchpad
                }
                st_callback = StreamlitCallbackHandler(st.container())
                response = self.agent_executor.invoke(agent_input, {"callbacks": [st_callback]})
                return response["output"]
            except Exception as e:
                if 'ThrottlingException' in str(e):
                    st.warning(f'Model {AVAILABLE_MODELS[self.current_model_index]} is throttled. Trying next model...')
                    self.current_model_index += 1
                    if self.current_model_index >= len(AVAILABLE_MODELS):
                        raise Exception('All models are throttled. Please try again later.')
                    # Get new model and recreate agent
                    self.model = self._get_model_with_fallback()
                    self.model.bind_tools(self.tools)
                    self.agent = self._create_agent()
                    self.agent_executor = self._create_agent_executor()
                else:
                    raise e

def render_file_tree(folder_path: str) -> Dict:
    file_tree = get_file_tree(folder_path)

    def build_nodes(node, path=""):
        local_nodes = []
        for name, content in sorted(node.items(), key=lambda x: (x[1] is not None, x[0])):
            full_path = os.path.join(path, name)
            if content is None:  # File
                local_nodes.append({"label": name, "value": full_path})
            else:  # Folder
                children = build_nodes(content, full_path)
                local_nodes.append({"label": name, "value": full_path, "children": children})
        return local_nodes

    root_name = os.path.basename(folder_path)
    nodes = [{
        "label": root_name,
        "value": folder_path,
        "children": build_nodes(file_tree, folder_path)
    }]
    return nodes


def update_selected_files():
    st.session_state.selected_files = [
        file for file in st.session_state.selected_files
        if any(file.startswith(folder) for folder in st.session_state.folder_paths)
    ]


def main():
    st.set_page_config(page_title="LLM Agent", page_icon=":robot_face:", layout="wide")

    # Initialize session state variables
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "aws_profile" not in st.session_state:
        st.session_state.aws_profile = ""
    if "model_id" not in st.session_state:
        st.session_state.model_id = AVAILABLE_MODELS[0]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "folder_paths" not in st.session_state:
        st.session_state.folder_paths = []
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = ['tavily_search']

    # Main chat interface
    st.title("🐧 CHITTI")

    # Sidebar for settings and file selection
    with st.sidebar:
        st.header(":gear: Settings")

        # AWS Profile selection
        st.session_state.aws_profile = st.text_input("AWS Profile", value=st.session_state.aws_profile)

        # Model selection
        st.session_state.model_id = st.selectbox("Select Model", AVAILABLE_MODELS,
                                                 index=AVAILABLE_MODELS.index(st.session_state.model_id))

        st.header(':wrench: Tool Selection')
        st.session_state.selected_tools = st.multiselect(
            'Select Tools',
            AVAILABLE_TOOLS,
            default=[]
        )

        st.header(":file_folder: Context Selection")

        # Multiple folder paths input
        new_folder_path = st.text_input("Enter folder path")
        if st.button("Add Folder"):
            if new_folder_path and os.path.isdir(new_folder_path) and new_folder_path not in st.session_state.folder_paths:
                st.session_state.folder_paths.append(new_folder_path)
                update_selected_files()
            elif not os.path.isdir(new_folder_path):
                st.error("Invalid folder path. Please enter a valid directory path.")

        # Display folders, allow removal, and show file tree for each
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

            # Render file tree for this folder
            folder_nodes = render_file_tree(folder_path)

            # Display file tree for this folder
            with st.expander("View folder structure", expanded=False):
                selected = tree_select(folder_nodes, key=f"tree_{i}")
                # Update selected files for this folder
                folder_files = set(selected['checked'])
                st.session_state.selected_files = list(
                    (set(st.session_state.selected_files) - set(f for f in st.session_state.selected_files if f.startswith(folder_path)))
                    | folder_files
                )

            st.markdown("---")  # Add a separator between folders

        # Calculate token count only for selected files (not folders)
        if st.session_state.selected_files:
            selected_file_contents = get_combined_file_contents([f for f in st.session_state.selected_files if os.path.isfile(f)])
            token_count = count_tokens(selected_file_contents)
            st.markdown(f"**Total tokens in context:** {token_count}")

    # Initialize or reinitialize the agent with current settings
    agent = LLMAgent(st.session_state.model_id, st.session_state.aws_profile)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            context = get_combined_file_contents(st.session_state.selected_files)
            chat_history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]
            response = agent.run(prompt, chat_history, context)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
