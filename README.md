# pilgrim

👷 WIP: TQA package using a LLM to answer user prompts about databases.

## Installation

```bash
# as soon as it's published on pypi
pip install pilgrim
```

## Usage

Create a .env file with the following:

```
DB_URI=sqlite:///path/to/your/database.db
OPENAI_API_KEY=your_openai_api_key
```

It's best to try this out first in a jupyter notebook. This is a baseline model that can answer questions about a database.

```python
from IPython.display import Image, display

import gradio as gr
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from pilgrim.graphs import baseline_graph
from pilgrim.utils import draw_mermaid_diagram

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
db = SQLDatabase.from_uri(os.environ["DB_URI"])

graph = baseline_graph(llm, db)

display(Image(draw_mermaid_diagram(graph)))

def chat(message: str, history):
    human_message = HumanMessage(content=message)
    result = graph.invoke({"messages": history + [human_message]})
    print("-------NEW HISTORY-------")
    for msg in result["messages"]:

        # Determine message type
        if isinstance(msg, AIMessage):
            msg_type = "ASSISTANT"
            if msg.tool_calls:
                msg.content = f"Tool calls: {[tool['name'] for tool in msg.tool_calls]}"
        elif isinstance(msg, HumanMessage):
            msg_type = "HUMAN"
        elif isinstance(msg, ToolMessage):
            msg_type = "TOOL"
        else:
            msg_type = "UNKNOWN"

        print(f"{msg_type}: {msg.content}\n-----")

    # For the Gradio interface, format the response with the message type
    last_msg = result["messages"][-1]
    return last_msg.content

gr.ChatInterface(fn=chat, type="messages").launch()
```
