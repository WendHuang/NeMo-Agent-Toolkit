from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class DetectOutliersIQRConfig(FunctionBaseConfig, name="detect_outliers_iqr"):
    """Detect outliers in sales data using IQR method."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=DetectOutliersIQRConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def detect_outliers_iqr_function(config: DetectOutliersIQRConfig, _builder: Builder):
    """Detect outliers in sales data using the Interquartile Range (IQR) method."""
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _detect_outliers_iqr(metric: str) -> str:
        """
        Detect outliers in retail data using the IQR method.

        Args:
            metric: Specific metric to check for outliers

        Returns:
            Dictionary containing outlier analysis results
        """
        if metric == "None":
            column = "Revenue"
        else:
            column = metric

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)]

        return f"Outliers in {column} are {outliers.to_dict('records')}"

    yield FunctionInfo.from_fn(
        _detect_outliers_iqr,
        description=_detect_outliers_iqr.__doc__)
