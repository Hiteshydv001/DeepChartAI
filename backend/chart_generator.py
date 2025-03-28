# backend/chart_generator.py
import plotly.express as px
import logging

logger = logging.getLogger(__name__)


def create_chart(chart_type: str, df, x_axis: str, y_axis: str) -> dict:
    """Generates a Plotly Express figure based on the given chart type and data.

    Args:
        chart_type: Type of chart (line, bar, pie, scatter, heatmap).
        df: Pandas DataFrame containing the data.
        x_axis: Column name for the x-axis.
        y_axis: Column name for the y-axis.

    Returns:
        A dictionary representing the Plotly Express figure.  Can be directly serialized to JSON.

    Raises:
        ValueError: If the chart type is unsupported or if there is an error during chart generation.
    """
    try:
        if chart_type == "line":
            fig = px.line(df, x=x_axis, y=y_axis, title="Line Chart")
            fig.update_layout(
                xaxis_title=x_axis, yaxis_title=y_axis, xaxis_tickangle=-45
            )

        elif chart_type == "bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title="Bar Chart")
            fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)

        elif chart_type == "pie":
            fig = px.pie(df, names=x_axis, values=y_axis, title="Pie Chart")

        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title="Scatter Plot")
            fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)

        elif chart_type == "heatmap":
            fig = px.density_heatmap(df, x=x_axis, y=y_axis, title="Heatmap")
            fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)

        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        fig_dict = fig.to_dict()  # Convert to dict before returning

        logger.debug(f"Generated {chart_type} chart successfully.")  # Use debug level
        return fig_dict

    except Exception as e:
        logger.error(f"Error generating chart: {e}", exc_info=True) # include traceback
        raise ValueError("Failed to generate chart.") from e  # Raise a ValueError for routes.py to handle