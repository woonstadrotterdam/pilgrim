import time
from typing import Optional, cast

import requests
from langgraph.graph.state import CompiledStateGraph


def draw_mermaid_diagram(
    graph: CompiledStateGraph, max_attempts: int = 5
) -> Optional[bytes]:
    """Draw a Mermaid diagram of a langgraph CompiledStateGraph with retry logic.

    Args:
        graph (CompiledStateGraph): The graph to display
        max_attempts (int): The maximum number of attempts to display the graph

    Returns:
        Optional[bytes]: PNG image data of the diagram if successful, None if all attempts fail
    """
    for attempt in range(max_attempts):
        try:
            # Explicitly cast the return value to bytes
            return cast(bytes, graph.get_graph().draw_mermaid_png())
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as e:
            print(f"Attempt {attempt + 1}/{max_attempts} failed: {type(e).__name__}")
            if attempt == max_attempts - 1:  # If this was the last attempt
                print("All retry attempts failed.")
                # Optionally fall back to text representation
                print(graph.get_graph().draw_mermaid())
                return None
            else:
                print("Retrying in 2 seconds...")
                time.sleep(2)  # Simple fixed delay between retries

    # This should never be reached, but added to satisfy mypy
    return None
