from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain.tools import Tool

AVAILABLE_MODELS = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0"
]


def get_model(model_id, aws_profile=None):
    return ChatBedrock(
        model_id=model_id,
        model_kwargs={"max_tokens": 8192, "temperature": 0.3, "top_k": 15},
        credentials_profile_name=aws_profile if aws_profile else None,
        config=Config(read_timeout=900)
    )


def get_tools():
    return []
