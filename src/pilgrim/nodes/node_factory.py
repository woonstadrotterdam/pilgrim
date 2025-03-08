from typing import Any, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState


def create_tool_node(tool: BaseTool) -> Callable[[MessagesState], dict[str, Any]]:
    """
    Creates a tool node for use in a LangGraph.

    Args:
        tool (BaseTool): The tool to use

    Returns:
        Callable[[MessagesState], dict[str, Any]]: A function that can be used as a node in a LangGraph
    """

    def tool_node(state: MessagesState) -> dict[str, Any]:
        # Get the last message content as input to the tool
        last_message = state["messages"][-1]

        # Extract content from the message
        tool_input = last_message.content

        # Ensure tool_input is a string
        if not isinstance(tool_input, str):
            tool_input = str(tool_input)

        # Run the tool
        tool_output = tool.invoke(tool_input)

        # Add the tool output as a new message
        new_messages = state["messages"] + [AIMessage(content=str(tool_output))]

        # Return updates to the state
        return {"messages": new_messages}

    return tool_node


def create_llm_node(
    llm: BaseChatModel,
    system_message_content: str | None = None,
) -> Callable[[MessagesState], dict[str, Any]]:
    """
    Creates a configurable LLM node for use in a LangGraph.

    Args:
        llm (BaseChatModel): The language model to use
        system_message_content (str | None): Content for the system message

    Returns:
        Callable[[MessagesState], dict[str, Any]]: A function that can be used as a node in a LangGraph
    """

    def llm_node(state: MessagesState) -> dict[str, Any]:
        if system_message_content:
            # Add the system message to the beginning of the messages list
            messages = [SystemMessage(content=system_message_content)] + state[
                "messages"
            ]
        else:
            messages = state["messages"]

        # Return the LLM's response
        return {"messages": [llm.invoke(messages)]}

    return llm_node


def create_llm_tool_node(
    llm: BaseChatModel,
    tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
    system_message_content: str | None = None,
) -> Callable[[MessagesState], dict[str, Any]]:
    """
    Creates a configurable LLM tool node for use in a LangGraph.

    Args:
        llm (BaseChatModel): The language model to use
        tools (Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool]): The tools to use
        system_message_content (str | None): Content for the system message

    Returns:
        Callable[[MessagesState], dict[str, Any]]: A function that can be used as a node in a LangGraph
    """

    def llm_tool_node(state: MessagesState) -> dict[str, Any]:
        if system_message_content:
            # Add the system message to the beginning of the messages list
            messages = [SystemMessage(content=system_message_content)] + state[
                "messages"
            ]
        else:
            messages = state["messages"]
        llm_with_tools = llm.bind_tools(tools)
        return {"messages": [llm_with_tools.invoke(messages)]}

    return llm_tool_node
