from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from pilgrim.nodes.node_factory import create_llm_tool_node


def baseline_graph(db: SQLDatabase, llm: BaseChatModel) -> CompiledStateGraph:
    """
    A langgraph graph that uses a LLM to interact with a SQL database using the following tools in a ReAct agent architecture:
    - sql_db_list_tables
    - sql_db_schema
    - sql_db_query_checker
    - sql_db_query

    Args:
        db (SQLDatabase): The database to interact with
        llm (BaseChatModel): The LLM to use

    Returns:
        CompiledStateGraph: The compiled graph
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

    # create custom tool nodes
    llm_query_tool_node = create_llm_tool_node(
        llm=llm,
        tools=[
            list_tables_tool,
            get_schema_tool,
            query_checker_tool,
            execute_query_tool,
        ],
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

    # add nodes
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("llm_with_sql_tools", llm_query_tool_node)
    graph_builder.add_node(
        "tools",
        ToolNode(
            [list_tables_tool, get_schema_tool, query_checker_tool, execute_query_tool]
        ),
    )

    # set entry point
    graph_builder.set_entry_point("llm_with_sql_tools")

    # add edges
    graph_builder.add_conditional_edges("llm_with_sql_tools", tools_condition)
    graph_builder.add_edge("tools", "llm_with_sql_tools")

    # compile graph
    graph = graph_builder.compile()

    return graph
