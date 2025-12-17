from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GetSalesPerDayConfig(FunctionBaseConfig, name="get_sales_per_day"):
    """Get total sales across all products per day."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=GetSalesPerDayConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def sales_per_day_function(config: GetSalesPerDayConfig, builder: Builder):
    """Get total sales across all products per day."""
    import pandas as pd

    df = pd.read_csv(config.data_path)
    df['Product'] = df["Product"].apply(lambda x: x.lower())

    async def _get_sales_per_day(date: str, product: str) -> str:
        """
        Calculate total sales data across all products for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            product: Product name

        Returns:
            String message with the total sales for the day
        """
        if date == "None":
            return "Please provide a date in YYYY-MM-DD format."
        total_revenue = df[(df['Date'] == date) & (df['Product'] == product)]['Revenue'].sum()
        total_units_sold = df[(df['Date'] == date) & (df['Product'] == product)]['UnitsSold'].sum()

        return f"Total revenue for {date} is {total_revenue} and total units sold is {total_units_sold}"

    yield FunctionInfo.from_fn(
        _get_sales_per_day,
        description=_get_sales_per_day.__doc__)
