from typing import Literal
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from pilgrim.nodes.node_factory import create_llm_tool_node, create_llm_node


def _create_baseline_base_graph(db: SQLDatabase, llm: BaseChatModel) -> StateGraph:
    """
    Creates the base components for a SQL database ReAct agent.

    Args:
        db (SQLDatabase): The database to interact with
        llm (BaseChatModel): The LLM to use

    Returns:
        StateGraph: The initialized graph with basic nodes
    """
    # Create SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Get the tools we need
    tool_dict = {tool.name: tool for tool in tools}
    list_tables_tool = tool_dict["sql_db_list_tables"]
    get_schema_tool = tool_dict["sql_db_schema"]
    query_checker_tool = tool_dict["sql_db_query_checker"]
    execute_query_tool = tool_dict["sql_db_query"]

    tools_list = [
        list_tables_tool,
        get_schema_tool,
        query_checker_tool,
        execute_query_tool,
    ]

    # create custom tool nodes
    llm_query_tool_node = create_llm_tool_node(
        llm=llm,
        tools=tools_list,
        # source: langchain-ai/sql-agent-system-prompt
        system_message_content="""
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer. "
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            To start you should ALWAYS look at the tables in the database to see what you can query.
            Do NOT skip this step.
            Then you should query the schema of the most relevant tables.
        """,
    )

    # Initialize graph builder
    graph_builder = StateGraph(MessagesState)

    # Add base nodes
    graph_builder.add_node("llm_with_sql_tools", llm_query_tool_node)
    graph_builder.add_node("tools", ToolNode(tools_list))

    # Set entry point
    graph_builder.set_entry_point("llm_with_sql_tools")

    return graph_builder


def baseline_graph(db: SQLDatabase, llm: BaseChatModel) -> StateGraph:
    """
    A langgraph graph that uses a LLM to interact with a SQL database using tools in a ReAct agent architecture.
    This version does not include an explanation node.

    Args:
        db (SQLDatabase): The database to interact with
        llm (BaseChatModel): The LLM to use

    Returns:
        StateGraph: The graph
    """
    graph_builder = _create_baseline_base_graph(db, llm)

    # Add edges
    graph_builder.add_conditional_edges("llm_with_sql_tools", tools_condition)
    graph_builder.add_edge("tools", "llm_with_sql_tools")

    return graph_builder


def baseline_graph_with_explainer(
    db: SQLDatabase, llm: BaseChatModel, explain_llm: BaseChatModel | None = None
) -> StateGraph:
    """
    A langgraph graph that uses a LLM to interact with a SQL database using tools in a ReAct agent architecture.
    This version includes an explanation node that always explains the agent's actions when tools are used.

    Args:
        db (SQLDatabase): The database to interact with
        llm (BaseChatModel): The LLM to use to answer the user's question
        explain_llm (BaseChatModel | None): The LLM to use for explaining the agent's actions. If None, the same LLM will be used.

    Returns:
        StateGraph: The graph
    """
    if not explain_llm:
        explain_llm = llm

    graph_builder = _create_baseline_base_graph(db, llm)

    # Create explaining LLM node
    explaining_llm_node = create_llm_node(
        llm=explain_llm,
        system_message_content="""
            Your task is to explain to the user which steps the agent took to answer the user's last question if tools were used.
            Start with the question and the answer.
            You should explain the steps in a way that is easy to understand, if tools were used.
            You should show every query that was executed.
            You should explain the results of the query and the answer if tools were used.

            Always reiterate the answer to the user's question.
        """,
    )

    # Add explanation node
    graph_builder.add_node("explaining_llm", explaining_llm_node)

    # Create custom router function
    def custom_router(state: MessagesState) -> Literal["tools", "explaining_llm"]:
        result = tools_condition(state)
        if result == "tools":
            return "tools"
        else:
            # For any other result (including "__end__"), route to explaining_llm
            return "explaining_llm"

    # Add conditional edges
    graph_builder.add_conditional_edges(
        "llm_with_sql_tools",
        custom_router,
        {"tools": "tools", "explaining_llm": "explaining_llm"},
    )

    # Add remaining edges
    graph_builder.add_edge("tools", "llm_with_sql_tools")
    graph_builder.add_edge("explaining_llm", "__end__")

    return graph_builder
