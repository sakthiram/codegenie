import os
import fnmatch
from typing import List, Dict
from langchain_community.document_loaders import TextLoader
import tiktoken

def load_file_contents(file_path: str) -> str:
    try:
        docs = TextLoader(file_path).load()
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""

def get_combined_file_contents(files: List[str]) -> str:
    combined_contents = ""
    for file in files:
        content = load_file_contents(file)
        combined_contents += f"File: {file}\n{content}\n\n"
    return combined_contents

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def parse_gitignore(gitignore_path: str) -> List[str]:
    if not os.path.exists(gitignore_path):
        return []

    with open(gitignore_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def should_ignore(path: str, ignore_patterns: List[str], root: str) -> bool:
    rel_path = os.path.relpath(path, root)
    return any(fnmatch.fnmatch("/" + rel_path + "/", pattern) for pattern in ignore_patterns)

def get_file_tree(folder_path: str) -> Dict:
    gitignore_path = os.path.join(folder_path, '.gitignore')
    ignore_patterns = parse_gitignore(gitignore_path)
    ignore_patterns.append('/.git/')

    file_tree = {}
    for root, dirs, files in os.walk(folder_path):
        # Remove ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), ignore_patterns, folder_path)]

        current_dict = file_tree
        path_parts = os.path.relpath(root, folder_path).split(os.sep)
        for part in path_parts:
            if part == '.':
                continue
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        for file in files:
            if not should_ignore(os.path.join(root, file), ignore_patterns, folder_path):
                current_dict[file] = None

    return file_tree
