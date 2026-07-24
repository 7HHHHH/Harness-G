"""Tool implementations for the Harness-G training path."""

__all__ = ["HarnessGTool"]


def _default_tools(env):
    if env == "harness_g":
        from agent.tool.tools.harness_g_tool import HarnessGTool

        return [HarnessGTool()]
    raise NotImplementedError("This checkout only supports tool.env='harness_g'.")
