import getpass
import os
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain.tools import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults

AVAILABLE_MODELS = [
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0"
]

AVAILABLE_TOOLS = ['tavily_search']

def get_model(model_id, aws_profile=None):
    return ChatBedrock(
        model_id=model_id,
        model_kwargs={"max_tokens": 8192, "temperature": 0.3, "top_k": 15},
        credentials_profile_name=aws_profile if aws_profile else None,
        config=Config(read_timeout=900)
    )


def get_tools(selected_tools):
    tools = []

    if 'tavily_search' in selected_tools:
        if not os.environ.get('TAVILY_API_KEY'):
            os.environ['TAVILY_API_KEY'] = getpass.getpass('Tavily API key:\n')

        search_tool = TavilySearchResults(
            max_results=5,
            search_depth='advanced',
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )
        tools.append(search_tool)

    return tools
