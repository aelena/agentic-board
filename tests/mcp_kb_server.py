"""A tiny MCP knowledge base for tests, served over stdio: python tests/mcp_kb_server.py"""

from mcp.server.fastmcp import FastMCP

kb = FastMCP("kb")


@kb.tool()
def lookup(topic: str) -> str:
    """Look up the internal notes on a topic."""
    return f"Internal note on {topic}: churn is 4% a month (https://kb.example/{topic})"


@kb.tool()
def delete_all() -> str:
    """A tool a board should never be allowed to call."""
    return "deleted everything"


if __name__ == "__main__":
    kb.run()
