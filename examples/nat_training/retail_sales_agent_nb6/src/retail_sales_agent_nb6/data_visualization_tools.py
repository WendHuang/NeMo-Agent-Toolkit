from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class PlotSalesTrendForStoresConfig(FunctionBaseConfig, name="plot_sales_trend_for_stores"):
    """Plot sales trend for a specific store."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=PlotSalesTrendForStoresConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def plot_sales_trend_for_stores_function(config: PlotSalesTrendForStoresConfig, _builder: Builder):
    """Create a visualization of sales trends over time."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _plot_sales_trend_for_stores(store_id: str) -> str:
        if store_id not in df["StoreID"].unique():
            data = df
            title = "Sales Trend for All Stores"
        else:
            data = df[df["StoreID"] == store_id]
            title = f"Sales Trend for Store {store_id}"

        plt.figure(figsize=(10, 5))
        trend = data.groupby("Date")["Revenue"].sum()
        trend.plot(title=title)
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.tight_layout()
        plt.savefig("sales_trend.png")

        return "Sales trend plot saved to sales_trend.png"

    yield FunctionInfo.from_fn(
        _plot_sales_trend_for_stores,
        description=(
            "This tool can be used to plot the sales trend for a specific store or all stores. "
            "It takes in a store ID creates and saves an image of a plot of the revenue trend for that store."))


class PlotAndCompareRevenueAcrossStoresConfig(FunctionBaseConfig, name="plot_and_compare_revenue_across_stores"):
    """Plot and compare revenue across stores."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=PlotAndCompareRevenueAcrossStoresConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def plot_revenue_across_stores_function(config: PlotAndCompareRevenueAcrossStoresConfig, _builder: Builder):
    """Create a visualization comparing sales trends between stores."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _plot_revenue_across_stores(arg: str) -> str:
        pivot = df.pivot_table(index="Date", columns="StoreID", values="Revenue", aggfunc="sum")
        pivot.plot(figsize=(12, 6), title="Revenue Trends Across Stores")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.legend(title="StoreID")
        plt.tight_layout()
        plt.savefig("revenue_across_stores.png")

        return "Revenue trends across stores plot saved to revenue_across_stores.png"

    yield FunctionInfo.from_fn(
        _plot_revenue_across_stores,
        description=(
            "This tool can be used to plot and compare the revenue trends across stores. Use this tool only if the "
            "user asks for a comparison of revenue trends across stores."
            "It takes in a single string as input (which is ignored) and creates and saves an image of a plot of the revenue trends across stores."
        ))


class PlotAverageDailyRevenueConfig(FunctionBaseConfig, name="plot_average_daily_revenue"):
    """Plot average daily revenue for stores and products."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=PlotAverageDailyRevenueConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def plot_average_daily_revenue_function(config: PlotAverageDailyRevenueConfig, _builder: Builder):
    """Create a bar chart showing average daily revenue by day of week."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _plot_average_daily_revenue(arg: str) -> str:
        daily_revenue = df.groupby(["StoreID", "Product", "Date"])["Revenue"].sum().reset_index()

        avg_daily_revenue = daily_revenue.groupby(["StoreID", "Product"])["Revenue"].mean().unstack()

        avg_daily_revenue.plot(kind="bar", figsize=(12, 6), title="Average Daily Revenue per Store by Product")
        plt.ylabel("Average Revenue")
        plt.xlabel("Store ID")
        plt.xticks(rotation=0)
        plt.legend(title="Product", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("average_daily_revenue.png")

        return "Average daily revenue plot saved to average_daily_revenue.png"

    yield FunctionInfo.from_fn(
        _plot_average_daily_revenue,
        description=("This tool can be used to plot the average daily revenue for stores and products "
                     "It takes in a single string as input and creates and saves an image of a grouped bar chart "
                     "of the average daily revenue"))
