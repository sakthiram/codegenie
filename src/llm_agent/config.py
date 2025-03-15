import getpass
import os
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain.tools import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults

AVAILABLE_MODELS = [
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0"
]

MODEL_CONFIGS = {
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
        "supports_thinking": True,
        "normal_mode": {
            "max_tokens": 8192,
            "temperature": 0.3,
            "top_k": 15
        },
        "thinking_mode": {
            "max_tokens": 64000,
            "thinking_tokens": 32000
        }
    },
    "default": {
        "supports_thinking": False,
        "normal_mode": {
            "max_tokens": 8192,
            "temperature": 0.3,
            "top_k": 15
        }
    }
}

MODEL_PRICING = {
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_cost_per_1k": 0.003,
        "output_cost_per_1k": 0.015
    },
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input_cost_per_1k": 0.001,
        "output_cost_per_1k": 0.005
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input_cost_per_1k": 0.003,
        "output_cost_per_1k": 0.015
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "input_cost_per_1k": 0.015,
        "output_cost_per_1k": 0.075
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.00125
    },
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "input_cost_per_1k": 0.003,
        "output_cost_per_1k": 0.015
    }
}

AVAILABLE_TOOLS = ['tavily_search']

def get_model(model_id, aws_profile=None, thinking_mode=False):
    model_config = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS["default"])

    if thinking_mode and model_config["supports_thinking"]:
        # Thinking mode configuration
        model_kwargs = {
            "max_tokens": model_config["thinking_mode"]["max_tokens"],
            "thinking": {
                "type": "enabled",
                "budget_tokens": model_config["thinking_mode"]["thinking_tokens"]
            }
        }
    else:
        # Normal mode configuration
        model_kwargs = {
            "max_tokens": model_config["normal_mode"]["max_tokens"],
            "temperature": model_config["normal_mode"]["temperature"],
            "top_k": model_config["normal_mode"]["top_k"]
        }
    return ChatBedrock(
        model_id=model_id,
        model_kwargs=model_kwargs,
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
