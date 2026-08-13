from __future__ import annotations

from io import BytesIO
from datetime import datetime
from typing import Any
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
    Table,
    TableStyle,
    Image
)
from reportlab.platypus import Image
from PIL import Image as PILImage
import plotly.io as pio

from core.execution.result_analyzer import analyze_result
from core.visualization.chart_selector import select_chart, ChartType
from core.visualization.chart_builders.bar_chart import build_bar_chart
from core.visualization.chart_builders.line_chart import build_line_chart
from core.visualization.chart_builders.pie_chart import build_pie_chart
from core.visualization.chart_builders.scatter_chart import build_scatter_chart
from core.visualization.chart_builders.histogram_chart import build_histogram


def _safe_text(value: Any) -> str:
    """Convert any value into safe printable text."""
    if value is None:
        return ""

    return str(value)


def _paragraph_text(value: Any) -> str:
    """Escape text so it is safe inside a ReportLab Paragraph."""
    return escape(_safe_text(value)).replace("\n", "<br/>")


def _get_query_data(query_result: Any):
    """
    Extract columns and rows from the project's QueryResult object.

    Returns:
        tuple[list, list]: columns and rows
    """
    if query_result is None:
        return [], []

    columns = getattr(query_result, "columns", None)
    rows = getattr(query_result, "rows", None)

    if columns is None or rows is None:
        return [], []

    columns = list(columns)
    rows = list(rows)

    return columns, rows


def _build_query_table(
    query_result: Any,
    available_width: float,
):
    """
    Build a readable ReportLab table from QueryResult.

    The table:
    - wraps long text
    - repeats headers on every page
    - supports multiple pages
    - automatically adjusts column widths
    """

    columns, rows = _get_query_data(query_result)

    if not columns:
        return None

    # ---------------------------------------------------------------
    # Styles for table cells
    # ---------------------------------------------------------------

    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=6.5,
        leading=8,
        textColor=colors.white,
        alignment=TA_CENTER,
    )

    cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=6,
        leading=7.5,
        wordWrap="CJK",
    )

    # ---------------------------------------------------------------
    # Convert headers into Paragraphs
    # ---------------------------------------------------------------

    header_row = [
        Paragraph(_paragraph_text(column), header_style)
        for column in columns
    ]

    table_data = [header_row]

    # ---------------------------------------------------------------
    # Convert each row
    # ---------------------------------------------------------------

    for row in rows:

        if isinstance(row, dict):
            values = [
                row.get(column, "")
                for column in columns
            ]
        else:
            values = list(row)

            # Make sure every row has the same number of cells
            if len(values) < len(columns):
                values.extend([""] * (len(columns) - len(values)))

            values = values[: len(columns)]

        table_data.append(
            [
                Paragraph(_paragraph_text(value), cell_style)
                for value in values
            ]
        )

    # ---------------------------------------------------------------
    # Calculate reasonable column widths
    # ---------------------------------------------------------------

    column_count = len(columns)

    # Default equal width.
    column_widths = [
        available_width / column_count
        for _ in columns
    ]

    # Give some columns slightly more room based on their names.
    for index, column in enumerate(columns):

        name = _safe_text(column).lower()

        if any(
            keyword in name
            for keyword in [
                "address",
                "description",
                "email",
                "company",
            ]
        ):
            column_widths[index] *= 1.35

        elif any(
            keyword in name
            for keyword in [
                "id",
                "postal",
                "zip",
            ]
        ):
            column_widths[index] *= 0.75

    # Normalize widths so total width remains exactly available_width.
    total_width = sum(column_widths)

    if total_width > 0:
        scale = available_width / total_width
        column_widths = [
            width * scale
            for width in column_widths
        ]

    # ---------------------------------------------------------------
    # Create table
    # ---------------------------------------------------------------

    table = Table(
        table_data,
        colWidths=column_widths,
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )

    # ---------------------------------------------------------------
    # Table styling
    # ---------------------------------------------------------------

    table.setStyle(
        TableStyle(
            [
                # Header
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2F3E46"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                # Grid
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#B0B0B0"),
                ),

                # Padding
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),

                # Alignment
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
            ]
        )
    )

    return table

def _build_pdf_chart(query_result: Any):
    """
    Recreate the appropriate Plotly chart from QueryResult.

    The Plotly Figure is created temporarily for PDF generation.
    It is NOT stored in the database.
    """
    if query_result is None:
        return None

    try:
        shape = analyze_result(query_result)
        selection = select_chart(shape)

        if selection.chart_type == ChartType.NONE:
            return None

        if selection.chart_type == ChartType.BAR:
            return build_bar_chart(selection)

        if selection.chart_type == ChartType.LINE:
            return build_line_chart(selection)

        if selection.chart_type == ChartType.PIE:
            return build_pie_chart(selection)

        if selection.chart_type == ChartType.SCATTER:
            return build_scatter_chart(selection)

        if selection.chart_type == ChartType.HISTOGRAM:
            return build_histogram(selection)

    except Exception:
        return None

    return None

def create_conversation_pdf(chat_history: list[Any]) -> bytes:
    """
    Generate a readable PDF containing the current Streamlit conversation.

    Includes:
    - User questions
    - Assistant responses
    - Generated SQL
    - Query results as proper tables
    - Validation errors
    - Other assistant errors
    """

    buffer = BytesIO()

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Use LANDSCAPE because SQL results can have many columns.
    # ---------------------------------------------------------------

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="AI Data Analyst Conversation",
        author="AI Data Analyst",
    )

    styles = getSampleStyleSheet()

    # ---------------------------------------------------------------
    # Styles
    # ---------------------------------------------------------------

    title_style = ParagraphStyle(
        "PDFTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "PDFSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=18,
    )

    user_style = ParagraphStyle(
        "UserMessage",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    assistant_style = ParagraphStyle(
        "AssistantMessage",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading3"],
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=5,
    )

    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        leftIndent=5,
        rightIndent=5,
        spaceAfter=8,
    )

    story = []

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------

    story.append(
        Paragraph(
            "AI Data Analyst",
            title_style,
        )
    )

    generated_at = datetime.now().strftime(
        "%d-%m-%Y %H:%M"
    )

    story.append(
        Paragraph(
            f"Conversation Export · Generated on {generated_at}",
            subtitle_style,
        )
    )

    story.append(Spacer(1, 5))

    # ---------------------------------------------------------------
    # Conversation
    # ---------------------------------------------------------------

    if not chat_history:

        story.append(
            Paragraph(
                "No conversation messages available.",
                assistant_style,
            )
        )

    for index, message in enumerate(
        chat_history,
        start=1,
    ):

        role = getattr(
            message,
            "role",
            "",
        )

        content = _safe_text(
            getattr(
                message,
                "content",
                "",
            )
        )

        # ===========================================================
        # USER MESSAGE
        # ===========================================================

        if role == "user":

            story.append(
                Paragraph(
                    f"<b>User</b> — Message {index}",
                    heading_style,
                )
            )

            story.append(
                Paragraph(
                    _paragraph_text(content),
                    user_style,
                )
            )

        # ===========================================================
        # ASSISTANT MESSAGE
        # ===========================================================

        else:

            story.append(
                Paragraph(
                    f"<b>AI Data Analyst</b> — Response {index}",
                    heading_style,
                )
            )

            story.append(
                Paragraph(
                    _paragraph_text(content),
                    assistant_style,
                )
            )

            # -------------------------------------------------------
            # Generated SQL
            # -------------------------------------------------------

            sql = getattr(
                message,
                "sql",
                None,
            )

            if sql:

                story.append(
                    Paragraph(
                        "Generated SQL",
                        heading_style,
                    )
                )

                story.append(
                    Preformatted(
                        _safe_text(sql),
                        code_style,
                    )
                )

            # -------------------------------------------------------
            # Validation Error
            # -------------------------------------------------------

            validation_error = getattr(
                message,
                "validation_error",
                None,
            )

            if validation_error:

                story.append(
                    Paragraph(
                        "Validation Error",
                        heading_style,
                    )
                )

                story.append(
                    Paragraph(
                        _paragraph_text(
                            validation_error
                        ),
                        assistant_style,
                    )
                )

            # -------------------------------------------------------
            # Query Result
            # -------------------------------------------------------

            query_result = getattr(
                message,
                "query_result",
                None,
            )

            if query_result is not None:

                columns, rows = _get_query_data(
                    query_result
                )

                if columns:

                    story.append(
                        Paragraph(
                            "Query Result",
                            heading_style,
                        )
                    )

                    # Landscape A4 usable width:
                    #
                    # 297mm - 12mm - 12mm = 273mm
                    #
                    available_width = (
                        landscape(A4)[0]
                        - (24 * mm)
                    )

                    table = _build_query_table(
                        query_result,
                        available_width,
                    )

                    if table is not None:

                        story.append(table)

                        story.append(
                            Spacer(
                                1,
                                8,
                            )
                        )

                else:

                    result_text = _safe_text(
                        query_result
                    )

                    if result_text:

                        story.append(
                            Paragraph(
                                "Query Result",
                                heading_style,
                            )
                        )

                        story.append(
                            Preformatted(
                                result_text,
                                code_style,
                            )
                        )
            
                
            # -------------------------------------------------------
            # Business Insights
            # -------------------------------------------------------

            insight = getattr(
                message,
                "insight",
                None,
            )

            if insight is not None and not getattr(
                insight,
                "is_empty",
                False,
            ):

                story.append(
                    Paragraph(
                        "Business Insights",
                        heading_style,
                    )
                )

                summary = getattr(
                    insight,
                    "summary",
                    None,
                )

                if summary:
                    story.append(
                        Paragraph(
                            f"<b>Summary:</b> "
                            f"{_paragraph_text(summary)}",
                            assistant_style,
                        )
                    )

                key_trends = getattr(
                    insight,
                    "key_trends",
                    [],
                )

                if key_trends:

                    story.append(
                        Paragraph(
                            "<b>Key Trends</b>",
                            assistant_style,
                        )
                    )

                    for trend in key_trends:
                        story.append(
                            Paragraph(
                                f"• {_paragraph_text(trend)}",
                                assistant_style,
                            )
                        )

                outliers = getattr(
                    insight,
                    "outliers",
                    [],
                )

                if outliers:

                    story.append(
                        Paragraph(
                            "<b>Outliers</b>",
                            assistant_style,
                        )
                    )

                    for outlier in outliers:
                        story.append(
                            Paragraph(
                                f"• {_paragraph_text(outlier)}",
                                assistant_style,
                            )
                        )

                important_metrics = getattr(
                    insight,
                    "important_metrics",
                    [],
                )

                if important_metrics:

                    story.append(
                        Paragraph(
                            "<b>Important Metrics</b>",
                            assistant_style,
                        )
                    )

                    for metric in important_metrics:
                        story.append(
                            Paragraph(
                                f"• {_paragraph_text(metric)}",
                                assistant_style,
                            )
                        )

                follow_up_questions = getattr(
                    insight,
                    "follow_up_questions",
                    [],
                )

                if follow_up_questions:

                    story.append(
                        Paragraph(
                            "<b>Suggested Follow-up Questions</b>",
                            assistant_style,
                        )
                    )

                    for question in follow_up_questions:
                        story.append(
                            Paragraph(
                                f"• {_paragraph_text(question)}",
                                assistant_style,
                            )
                        )

                story.append(
                    Spacer(
                        1,
                        8,
                    )
                )
            
            # -------------------------------------------------------
            # Actual Chart
            # -------------------------------------------------------

            if query_result is not None:

                try:
                    fig = _build_pdf_chart(query_result)

                    if fig is not None:

                        image_bytes = fig.to_image(
                            format="png",
                            width=1200,
                            height=600,
                            scale=2,
                        )

                        image_buffer = BytesIO(image_bytes)

                        story.append(
                            Paragraph(
                                "Chart",
                                heading_style,
                            )
                        )

                        story.append(
                            Image(
                                image_buffer,
                                width=240 * mm,
                                height=120 * mm,
                            )
                        )

                        story.append(
                            Spacer(
                                1,
                                8,
                            )
                        )

                except Exception as exc:
                        story.append(
                        Paragraph(
                        f"Chart rendering error: {_paragraph_text(exc)}",
                            assistant_style,
                            )
                            )
             # -------------------------------------------------------
            # Chart Information
            # -------------------------------------------------------

            chart_metadata = getattr(
                message,
                "chart_metadata",
                None,
            )

            if chart_metadata:

                story.append(
                    Paragraph(
                        "Chart",
                        heading_style,
                    )
                )

                chart_type = chart_metadata.get(
                    "chart_type"
                )

                chart_title = chart_metadata.get(
                    "title"
                )

                chart_description = chart_metadata.get(
                    "description"
                )

                x_column = chart_metadata.get(
                    "x_column"
                )

                y_columns = chart_metadata.get(
                    "y_columns",
                    [],
                )

                if chart_title:
                    story.append(
                        Paragraph(
                            f"<b>Title:</b> "
                            f"{_paragraph_text(chart_title)}",
                            assistant_style,
                        )
                    )

                if chart_type:
                    story.append(
                        Paragraph(
                            f"<b>Chart Type:</b> "
                            f"{_paragraph_text(chart_type)}",
                            assistant_style,
                        )
                    )

                if chart_description:
                    story.append(
                        Paragraph(
                            f"<b>Description:</b> "
                            f"{_paragraph_text(chart_description)}",
                            assistant_style,
                        )
                    )

                if x_column:
                    story.append(
                        Paragraph(
                            f"<b>X-axis:</b> "
                            f"{_paragraph_text(x_column)}",
                            assistant_style,
                        )
                    )

                if y_columns:
                    story.append(
                        Paragraph(
                            f"<b>Y-axis:</b> "
                            f"{_paragraph_text(', '.join(y_columns))}",
                            assistant_style,
                        )
                    )

                story.append(
                    Spacer(
                        1,
                        8,
                    )
                )
            
            
            # -------------------------------------------------------
            # Error
            # -------------------------------------------------------

            error = getattr(
                message,
                "error",
                None,
            )

            if error:

                story.append(
                    Paragraph(
                        "Error",
                        heading_style,
                    )
                )

                story.append(
                    Paragraph(
                        _paragraph_text(error),
                        assistant_style,
                    )
                )

        story.append(
            Spacer(
                1,
                6,
            )
        )

    # ---------------------------------------------------------------
    # Build PDF
    # ---------------------------------------------------------------

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()

