import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class GettingStartedFunctionConfig(FunctionBaseConfig, name="getting_started"):
    """
    NAT function template. Please update the description.
    """
    prefix: str = Field(default="Echo:", description="Prefix to add before the echoed text.")


@register_function(config_type=GettingStartedFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def getting_started_function(config: GettingStartedFunctionConfig, builder: Builder):
    """
    Registers a function (addressable via `getting_started` in the configuration).
    This registration ensures a static mapping of the function type, `getting_started`, to the `GettingStartedFunctionConfig` configuration object.

    Args:
        config (GettingStartedFunctionConfig): The configuration for the function.
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