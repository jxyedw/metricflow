from __future__ import annotations

import os
from typing import Collection

from metricflow_semantic_interfaces.enum_extension import assert_values_exhausted
from metricflow_semantic_interfaces.type_enums.date_part import DatePart
from metricflow_semantic_interfaces.type_enums.time_granularity import TimeGranularity
from metricflow_semantics.errors.error_classes import UnsupportedEngineFeatureError
from metricflow_semantics.sql.sql_bind_parameters import SqlBindParameterSet
from metricflow_semantics.sql.sql_exprs import (
    SqlAddTimeExpression,
    SqlArithmeticExpression,
    SqlArithmeticOperator,
    SqlExtractExpression,
    SqlIntegerExpression,
    SqlPercentileExpression,
    SqlPercentileFunctionType,
    SqlSubtractTimeIntervalExpression,
)
from typing_extensions import override

from metricflow.protocols.sql_client import SqlEngine
from metricflow.sql.render.expr_renderer import (
    DefaultSqlExpressionRenderer,
    SqlExpressionRenderer,
    SqlExpressionRenderResult,
)
from metricflow.sql.render.sql_plan_renderer import DefaultSqlPlanRenderer, SqlPlanRenderResult
from metricflow.sql.sql_ctas_node import SqlCreateTableAsNode


class DorisSqlExpressionRenderer(DefaultSqlExpressionRenderer):
    """Expression renderer for Apache Doris.

    Doris speaks the MySQL wire protocol, but its SQL dialect differs from MySQL and from the
    ANSI dialect assumed by ``DefaultSqlExpressionRenderer`` in a few places that MetricFlow
    actually emits:

    - ``TIMESTAMP`` is not a queryable date-time type; use ``DATETIME``.
    - ``DATEADD`` does not exist; use ``DATE_ADD`` / ``DATE_SUB`` with ``INTERVAL``.
    - ``EXTRACT(DOW ...)`` follows MySQL and returns 1=Sunday..7=Saturday, while MetricFlow
      assumes ISO (1=Monday..7=Sunday).
    - Only ``PERCENTILE_APPROX`` is available for percentile metrics; the exact variants are
      not supported.
    """

    sql_engine = SqlEngine.DORIS

    @property
    @override
    def timestamp_data_type(self) -> str:
        """Doris uses DATETIME rather than TIMESTAMP as its primary date-time type."""
        return "DATETIME"

    @property
    @override
    def supported_percentile_function_types(self) -> Collection[SqlPercentileFunctionType]:
        return {SqlPercentileFunctionType.APPROXIMATE_CONTINUOUS}

    @override
    def render_date_part(self, date_part: DatePart) -> str:
        """Doris has no ``isodow``; DOW is converted to ISO in ``visit_extract_expr``."""
        if date_part is DatePart.DOW:
            return "DOW"
        return date_part.value

    @override
    def visit_extract_expr(self, node: SqlExtractExpression) -> SqlExpressionRenderResult:
        """Convert Doris' Sunday-first DOW into the ISO Monday-first convention."""
        extract_rendering_result = super().visit_extract_expr(node)
        if node.date_part is not DatePart.DOW:
            return extract_rendering_result

        extract_stmt = extract_rendering_result.sql
        return SqlExpressionRenderResult(
            sql=f"IF({extract_stmt} = 1, 7, {extract_stmt} - 1)",
            bind_parameter_set=extract_rendering_result.bind_parameter_set,
        )

    @override
    def visit_subtract_time_interval_expr(self, node: SqlSubtractTimeIntervalExpression) -> SqlExpressionRenderResult:
        """Render ``DATE_SUB(<arg>, INTERVAL <count> <unit>)``."""
        arg_rendered = node.arg.accept(self)

        count = node.count
        granularity = node.granularity
        if granularity is TimeGranularity.QUARTER:
            granularity = TimeGranularity.MONTH
            count *= 3

        return SqlExpressionRenderResult(
            sql=f"DATE_SUB({arg_rendered.sql}, INTERVAL {count} {granularity.value})",
            bind_parameter_set=arg_rendered.bind_parameter_set,
        )

    @override
    def visit_add_time_expr(self, node: SqlAddTimeExpression) -> SqlExpressionRenderResult:
        """Render ``DATE_ADD(<arg>, INTERVAL <count_expr> <unit>)``."""
        granularity = node.granularity
        count_expr = node.count_expr
        if granularity is TimeGranularity.QUARTER:
            granularity = TimeGranularity.MONTH
            count_expr = SqlArithmeticExpression.create(
                left_expr=node.count_expr,
                operator=SqlArithmeticOperator.MULTIPLY,
                right_expr=SqlIntegerExpression.create(3),
            )

        arg_rendered = node.arg.accept(self)
        count_rendered = count_expr.accept(self)

        return SqlExpressionRenderResult(
            sql=f"DATE_ADD({arg_rendered.sql}, INTERVAL {count_rendered.sql} {granularity.value})",
            bind_parameter_set=SqlBindParameterSet.merge_iterable(
                (arg_rendered.bind_parameter_set, count_rendered.bind_parameter_set)
            ),
        )

    @override
    def visit_percentile_expr(self, node: SqlPercentileExpression) -> SqlExpressionRenderResult:
        """Render ``PERCENTILE_APPROX``; exact percentile variants are unavailable in Doris."""
        arg_rendered = self.render_sql_expr(node.order_by_arg)
        percentile = node.percentile_args.percentile

        if node.percentile_args.function_type is SqlPercentileFunctionType.APPROXIMATE_CONTINUOUS:
            return SqlExpressionRenderResult(
                sql=f"PERCENTILE_APPROX({arg_rendered.sql}, {percentile})",
                bind_parameter_set=arg_rendered.bind_parameter_set,
            )
        elif (
            node.percentile_args.function_type is SqlPercentileFunctionType.APPROXIMATE_DISCRETE
            or node.percentile_args.function_type is SqlPercentileFunctionType.CONTINUOUS
            or node.percentile_args.function_type is SqlPercentileFunctionType.DISCRETE
        ):
            raise UnsupportedEngineFeatureError(
                "Only approximate continuous percentile aggregations are supported for Doris. Set "
                + "use_approximate_percentile to true and disable use_discrete_percentile in all percentile "
                + "simple-metrics."
            )
        else:
            assert_values_exhausted(node.percentile_args.function_type)


class DorisSqlPlanRenderer(DefaultSqlPlanRenderer):
    """Plan renderer for Apache Doris.

    A bare ``CREATE TABLE ... AS SELECT`` fails on a single-BE cluster because Doris defaults
    ``replication_num`` to 3. Set ``MF_DORIS_CTAS_PROPERTIES`` (for example
    ``"replication_num" = "1"``) to append a ``PROPERTIES`` clause. Left unset, the statement is
    rendered like the ANSI default, so the caller keeps full control over cluster-specific DDL.
    """

    EXPR_RENDERER = DorisSqlExpressionRenderer()

    @property
    @override
    def expr_renderer(self) -> SqlExpressionRenderer:
        return self.EXPR_RENDERER

    @override
    def visit_create_table_as_node(self, node: SqlCreateTableAsNode) -> SqlPlanRenderResult:
        result = super().visit_create_table_as_node(node)
        ctas_properties = os.environ.get("MF_DORIS_CTAS_PROPERTIES")
        if not ctas_properties:
            return result

        create_prefix = f"CREATE {node.sql_table.table_type.value.upper()} {node.sql_table.sql}"
        replacement_prefix = f"{create_prefix} PROPERTIES({ctas_properties})"
        if not result.sql.startswith(create_prefix):
            raise RuntimeError(f"Unexpected CTAS rendering, cannot inject properties: {result.sql[:200]}")
        return SqlPlanRenderResult(
            sql=replacement_prefix + result.sql[len(create_prefix) :],
            bind_parameter_set=result.bind_parameter_set,
        )