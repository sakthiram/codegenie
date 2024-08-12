import subprocess

def codegenie():
    subprocess.run(["streamlit", "run", "src/llm_agent/agent.py"])
