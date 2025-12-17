import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class RetailSalesAgentNb6FunctionConfig(FunctionBaseConfig, name="retail_sales_agent_nb6"):
    """
    NAT function template. Please update the description.
    """
    prefix: str = Field(default="Echo:", description="Prefix to add before the echoed text.")


@register_function(config_type=RetailSalesAgentNb6FunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def retail_sales_agent_nb6_function(config: RetailSalesAgentNb6FunctionConfig, builder: Builder):
    """
    Registers a function (addressable via `retail_sales_agent_nb6` in the configuration).
    This registration ensures a static mapping of the function type, `retail_sales_agent_nb6`, to the `RetailSalesAgentNb6FunctionConfig` configuration object.

    Args:
        config (RetailSalesAgentNb6FunctionConfig): The configuration for the function.
        builder (Builder): The builder object.

    Returns:
        FunctionInfo: The function info object for the function.
    """

    # Define the function that will be registered.
    async def _echo(text: str) -> str:
        """
        Takes a text input and echoes back with a pre-defined prefix.

        Args:
            text (str): The text to echo back.

        Returns:
            str: The text with the prefix.
        """
        return f"{config.prefix} {text}"

    # The callable is wrapped in a FunctionInfo object.
    # The description parameter is used to describe the function.
    yield FunctionInfo.from_fn(_echo, description=_echo.__doc__)