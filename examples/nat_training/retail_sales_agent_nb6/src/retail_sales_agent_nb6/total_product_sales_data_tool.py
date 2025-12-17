from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GetTotalProductSalesDataConfig(FunctionBaseConfig, name="get_total_product_sales_data"):
    """Get total sales data by product."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=GetTotalProductSalesDataConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def get_total_product_sales_data_function(config: GetTotalProductSalesDataConfig, _builder: Builder):
    """Get total sales data for a specific product."""
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _get_total_product_sales_data(product_name: str) -> str:
        """
        Retrieve total sales data for a specific product.

        Args:
            product_name: Name of the product

        Returns:
            String message containing total sales data
        """
        df['Product'] = df["Product"].apply(lambda x: x.lower())
        revenue = df[df['Product'] == product_name]['Revenue'].sum()
        units_sold = df[df['Product'] == product_name]['UnitsSold'].sum()

        return f"Revenue for {product_name} are {revenue} and total units sold are {units_sold}"

    yield FunctionInfo.from_fn(
        _get_total_product_sales_data,
        description=_get_total_product_sales_data.__doc__)
