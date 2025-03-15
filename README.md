# codegenie
AI agent for assisting developers

## Features
- Local filetree selection for context addition
- Tool addition support
- Bedrock models support
- Using langchain's structured chat agent to constrain agent outputs for multi agent or agent tool collaboration
- Streamlit callback used to stream intermediate steps

### Claude 3.7 extended thinking support
References:
- https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models
- https://docs.anthropic.com/en/docs/about-claude/models/all-models#model-comparison


## Development
### Building & Running
1. `cd codegenie`
2. `poetry install`
3. `poetry run chitti`

### Publishing


## Command History (just for tracking)
1. `poetry add streamlit@latest langchain@latest langchain-community@latest langgraph@latest tiktoken@latest botocore@latest boto3@latest langchain_aws@latest streamlit_tree_select@latest`
2. 