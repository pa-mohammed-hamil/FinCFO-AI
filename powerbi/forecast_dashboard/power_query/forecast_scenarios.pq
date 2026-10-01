let
    // ============================================================
    // FinCo AI — Forecast Scenarios
    // File: forecast_scenarios.pq
    // Purpose:
    //   Generate scenario-based financial forecasts from
    //   the canonical ForecastResults dataset.
    //
    // Scenarios:
    //   Base
    //   Optimistic
    //   Pessimistic
    //   Stress
    //
    // Expected source query/table:
    //   ForecastResults
    // ============================================================

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    ScenarioStartDate = #date(2022, 1, 1),
    ScenarioEndDate   = #date(2030, 12, 31),

    // Scenario adjustment assumptions
    BaseAdjustment        = 0.00,
    OptimisticAdjustment  = 0.10,
    PessimisticAdjustment = -0.10,
    StressAdjustment      = -0.20,

    // Confidence adjustments
    BaseConfidence        = 0.00,
    OptimisticConfidence  = 0.05,
    PessimisticConfidence = -0.05,
    StressConfidence      = -0.15,

    // ------------------------------------------------------------
    // Source
    // ------------------------------------------------------------
    Source =
        try ForecastResults
        otherwise
            #table(
                type table[
                    ForecastDate = date,
                    Company = text,
                    BusinessUnit = text,
                    Metric = text,
                    Actual = number,
                    Forecast = number,
                    LowerBound = number,
                    UpperBound = number,
                    Confidence = number,
                    Model = text,
                    ModelVersion = text
                ],
                {}
            ),

    // ------------------------------------------------------------
    // Standardize column names
    // ------------------------------------------------------------
    StandardizedColumns =
        Table.TransformColumnNames(
            Source,
            each Text.Trim(_)
        ),

    // ------------------------------------------------------------
    // Ensure required columns exist
    // ------------------------------------------------------------
    RequiredColumns = {
        "ForecastDate",
        "Company",
        "BusinessUnit",
        "Metric",
        "Actual",
        "Forecast",
        "LowerBound",
        "UpperBound",
        "Confidence",
        "Model",
        "ModelVersion"
    },

    AddMissingColumns =
        List.Accumulate(
            RequiredColumns,
            StandardizedColumns,
            (state, columnName) =>
                if List.Contains(Table.ColumnNames(state), columnName)
                then state
                else
                    Table.AddColumn(
                        state,
                        columnName,
                        each null
                    )
        ),

    // ------------------------------------------------------------
    // Select required columns
    // ------------------------------------------------------------
    SelectedColumns =
        Table.SelectColumns(
            AddMissingColumns,
            RequiredColumns,
            MissingField.UseNull
        ),

    // ------------------------------------------------------------
    // Data types
    // ------------------------------------------------------------
    Typed =
        Table.TransformColumnTypes(
            SelectedColumns,
            {
                {"ForecastDate", type date},
                {"Company", type text},
                {"BusinessUnit", type text},
                {"Metric", type text},
                {"Actual", type number},
                {"Forecast", type number},
                {"LowerBound", type number},
                {"UpperBound", type number},
                {"Confidence", type number},
                {"Model", type text},
                {"ModelVersion", type text}
            }
        ),

    // ------------------------------------------------------------
    // Clean text fields
    // ------------------------------------------------------------
    CleanText =
        Table.TransformColumns(
            Typed,
            {
                {
                    "Company",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                },
                {
                    "BusinessUnit",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                },
                {
                    "Metric",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                },
                {
                    "Model",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                },
                {
                    "ModelVersion",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                }
            }
        ),

    // ------------------------------------------------------------
    // Clean numeric values
    // ------------------------------------------------------------
    CleanNumbers =
        Table.TransformColumns(
            CleanText,
            {
                {
                    "Forecast",
                    each
                        if _ = null or not Value.Is(_, type number)
                        then 0
                        else _,
                    type number
                },
                {
                    "LowerBound",
                    each
                        if _ = null or not Value.Is(_, type number)
                        then 0
                        else _,
                    type number
                },
                {
                    "UpperBound",
                    each
                        if _ = null or not Value.Is(_, type number)
                        then 0
                        else _,
                    type number
                },
                {
                    "Confidence",
                    each
                        if _ = null
                        then 0
                        else Number.Max(
                            0,
                            Number.Min(1, Number.From(_))
                        ),
                    type number
                }
            }
        ),

    // ------------------------------------------------------------
    // Date filter
    // ------------------------------------------------------------
    FilteredDates =
        Table.SelectRows(
            CleanNumbers,
            each
                [ForecastDate] <> null
                and [ForecastDate] >= ScenarioStartDate
                and [ForecastDate] <= ScenarioEndDate
        ),

    // ------------------------------------------------------------
    // Add date attributes
    // ------------------------------------------------------------
    AddYear =
        Table.AddColumn(
            FilteredDates,
            "Year",
            each Date.Year([ForecastDate]),
            Int64.Type
        ),

    AddQuarterNumber =
        Table.AddColumn(
            AddYear,
            "QuarterNumber",
            each Date.QuarterOfYear([ForecastDate]),
            Int64.Type
        ),

    AddQuarter =
        Table.AddColumn(
            AddQuarterNumber,
            "Quarter",
            each "Q" & Text.From([QuarterNumber]),
            type text
        ),

    AddMonthNumber =
        Table.AddColumn(
            AddQuarter,
            "MonthNumber",
            each Date.Month([ForecastDate]),
            Int64.Type
        ),

    AddMonth =
        Table.AddColumn(
            AddMonthNumber,
            "Month",
            each Date.ToText([ForecastDate], "yyyy-MM"),
            type text
        ),

    AddMonthName =
        Table.AddColumn(
            AddMonth,
            "MonthName",
            each Date.MonthName([ForecastDate]),
            type text
        ),

    AddMonthStart =
        Table.AddColumn(
            AddMonthName,
            "MonthStart",
            each Date.StartOfMonth([ForecastDate]),
            type date
        ),

    // ------------------------------------------------------------
    // Create scenario parameter table
    // ------------------------------------------------------------
    ScenarioTable =
        #table(
            type table[
                Scenario = text,
                ScenarioOrder = Int64.Type,
                Adjustment = number,
                ConfidenceAdjustment = number
            ],
            {
                {
                    "Base",
                    1,
                    BaseAdjustment,
                    BaseConfidence
                },
                {
                    "Optimistic",
                    2,
                    OptimisticAdjustment,
                    OptimisticConfidence
                },
                {
                    "Pessimistic",
                    3,
                    PessimisticAdjustment,
                    PessimisticConfidence
                },
                {
                    "Stress",
                    4,
                    StressAdjustment,
                    StressConfidence
                }
            }
        ),

    // ------------------------------------------------------------
    // Cross join forecast records with scenarios
    // ------------------------------------------------------------
    AddScenarioTable =
        Table.AddColumn(
            AddMonthStart,
            "ScenarioTable",
            each ScenarioTable
        ),

    ExpandedScenarios =
        Table.ExpandTableColumn(
            AddScenarioTable,
            "ScenarioTable",
            {
                "Scenario",
                "ScenarioOrder",
                "Adjustment",
                "ConfidenceAdjustment"
            },
            {
                "Scenario",
                "ScenarioOrder",
                "ScenarioAdjustment",
                "ConfidenceAdjustment"
            }
        ),

    // ------------------------------------------------------------
    // Scenario Forecast
    // ------------------------------------------------------------
    AddScenarioForecast =
        Table.AddColumn(
            ExpandedScenarios,
            "ScenarioForecast",
            each
                [Forecast] * (1 + [ScenarioAdjustment]),
            type number
        ),

    // ------------------------------------------------------------
    // Scenario Lower Bound
    // ------------------------------------------------------------
    AddScenarioLowerBound =
        Table.AddColumn(
            AddScenarioForecast,
            "ScenarioLowerBound",
            each
                [LowerBound] * (1 + [ScenarioAdjustment]),
            type number
        ),

    // ------------------------------------------------------------
    // Scenario Upper Bound
    // ------------------------------------------------------------
    AddScenarioUpperBound =
        Table.AddColumn(
            AddScenarioLowerBound,
            "ScenarioUpperBound",
            each
                [UpperBound] * (1 + [ScenarioAdjustment]),
            type number
        ),

    // ------------------------------------------------------------
    // Scenario Confidence
    // ------------------------------------------------------------
    AddScenarioConfidence =
        Table.AddColumn(
            AddScenarioUpperBound,
            "ScenarioConfidence",
            each
                Number.Max(
                    0,
                    Number.Min(
                        1,
                        [Confidence] + [ConfidenceAdjustment]
                    )
                ),
            type number
        ),

    // ------------------------------------------------------------
    // Scenario Change
    // ------------------------------------------------------------
    AddScenarioChange =
        Table.AddColumn(
            AddScenarioConfidence,
            "ScenarioChange",
            each
                [ScenarioForecast] - [Forecast],
            type number
        ),

    AddScenarioChangePct =
        Table.AddColumn(
            AddScenarioChange,
            "ScenarioChangePct",
            each
                if [Forecast] = null or [Forecast] = 0
                then null
                else
                    [ScenarioChange] / Number.Abs([Forecast]),
            type number
        ),

    // ------------------------------------------------------------
    // Forecast variance versus actual
    // ------------------------------------------------------------
    AddScenarioVariance =
        Table.AddColumn(
            AddScenarioChangePct,
            "ScenarioVariance",
            each
                if [Actual] = null
                then null
                else
                    [Actual] - [ScenarioForecast],
            type number
        ),

    AddScenarioVariancePct =
        Table.AddColumn(
            AddScenarioVariance,
            "ScenarioVariancePct",
            each
                if [Actual] = null or [Actual] = 0
                then null
                else
                    ([ScenarioForecast] - [Actual])
                    / Number.Abs([Actual]),
            type number
        ),

    // ------------------------------------------------------------
    // Scenario band width
    // ------------------------------------------------------------
    AddBandWidth =
        Table.AddColumn(
            AddScenarioVariancePct,
            "ScenarioBandWidth",
            each
                [ScenarioUpperBound] - [ScenarioLowerBound],
            type number
        ),

    AddBandWidthPct =
        Table.AddColumn(
            AddBandWidth,
            "ScenarioBandWidthPct",
            each
                if [ScenarioForecast] = null
                    or [ScenarioForecast] = 0
                then null
                else
                    [ScenarioBandWidth]
                    / Number.Abs([ScenarioForecast]),
            type number
        ),

    // ------------------------------------------------------------
    // Confidence risk
    // ------------------------------------------------------------
    AddConfidenceRisk =
        Table.AddColumn(
            AddBandWidthPct,
            "ConfidenceRisk",
            each
                if [ScenarioConfidence] < 0.40 then 40
                else if [ScenarioConfidence] < 0.60 then 30
                else if [ScenarioConfidence] < 0.75 then 20
                else if [ScenarioConfidence] < 0.90 then 10
                else 0,
            Int64.Type
        ),

    // ------------------------------------------------------------
    // Volatility risk
    // ------------------------------------------------------------
    AddVolatilityRisk =
        Table.AddColumn(
            AddConfidenceRisk,
            "VolatilityRisk",
            each
                if [ScenarioBandWidthPct] = null then 0
                else if [ScenarioBandWidthPct] >= 0.50 then 40
                else if [ScenarioBandWidthPct] >= 0.30 then 30
                else if [ScenarioBandWidthPct] >= 0.15 then 20
                else if [ScenarioBandWidthPct] >= 0.05 then 10
                else 0,
            Int64.Type
        ),

    // ------------------------------------------------------------
    // Scenario downside risk
    // ------------------------------------------------------------
    AddDownsideRisk =
        Table.AddColumn(
            AddVolatilityRisk,
            "DownsideRisk",
            each
                if [ScenarioAdjustment] <= -0.20 then 40
                else if [ScenarioAdjustment] <= -0.10 then 30
                else if [ScenarioAdjustment] < 0 then 20
                else 0,
            Int64.Type
        ),

    // ------------------------------------------------------------
    // Confidence-adjusted scenario risk
    // ------------------------------------------------------------
    AddScenarioRiskScore =
        Table.AddColumn(
            AddDownsideRisk,
            "ScenarioRiskScore",
            each
                Number.Min(
                    100,
                    [ConfidenceRisk]
                    + [VolatilityRisk]
                    + [DownsideRisk]
                ),
            Int64.Type
        ),

    // ------------------------------------------------------------
    // Risk level
    // ------------------------------------------------------------
    AddScenarioRiskLevel =
        Table.AddColumn(
            AddScenarioRiskScore,
            "ScenarioRiskLevel",
            each
                if [ScenarioRiskScore] >= 75 then "CRITICAL"
                else if [ScenarioRiskScore] >= 50 then "HIGH"
                else if [ScenarioRiskScore] >= 25 then "MEDIUM"
                else "LOW",
            type text
        ),

    // ------------------------------------------------------------
    // Scenario classification
    // ------------------------------------------------------------
    AddScenarioType =
        Table.AddColumn(
            AddScenarioRiskLevel,
            "ScenarioType",
            each
                if [Scenario] = "Base"
                then "Expected"
                else if [Scenario] = "Optimistic"
                then "Upside"
                else if [Scenario] = "Pessimistic"
                then "Downside"
                else if [Scenario] = "Stress"
                then "Severe Downside"
                else "Other",
            type text
        ),

    // ------------------------------------------------------------
    // Scenario direction
    // ------------------------------------------------------------
    AddScenarioDirection =
        Table.AddColumn(
            AddScenarioType,
            "ScenarioDirection",
            each
                if [ScenarioAdjustment] > 0
                then "UP"
                else if [ScenarioAdjustment] < 0
                then "DOWN"
                else "NEUTRAL",
            type text
        ),

    // ------------------------------------------------------------
    // Scenario status
    // ------------------------------------------------------------
    AddScenarioStatus =
        Table.AddColumn(
            AddScenarioDirection,
            "ScenarioStatus",
            each
                if [Scenario] = "Stress"
                    and [ScenarioRiskLevel] = "CRITICAL"
                then "Stress Warning"
                else if [Scenario] = "Stress"
                then "Stress Scenario"
                else if [Scenario] = "Pessimistic"
                    and [ScenarioRiskLevel] = "HIGH"
                then "Downside Warning"
                else if [Scenario] = "Optimistic"
                then "Upside Opportunity"
                else if [Scenario] = "Base"
                then "Base Forecast"
                else "Scenario Review",
            type text
        ),

    // ------------------------------------------------------------
    // Alert flag
    // ------------------------------------------------------------
    AddAlertFlag =
        Table.AddColumn(
            AddScenarioStatus,
            "AlertFlag",
            each
                if
                    [ScenarioRiskLevel] = "CRITICAL"
                    or [ScenarioRiskLevel] = "HIGH"
                then "YES"
                else "NO",
            type text
        ),

    // ------------------------------------------------------------
    // Alert type
    // ------------------------------------------------------------
    AddAlertType =
        Table.AddColumn(
            AddAlertFlag,
            "AlertType",
            each
                if [ScenarioRiskLevel] = "CRITICAL"
                then "Critical Scenario Risk"
                else if [ScenarioRiskLevel] = "HIGH"
                then "High Scenario Risk"
                else if [Scenario] = "Stress"
                then "Stress Scenario Monitoring"
                else null,
            type text
        ),

    // ------------------------------------------------------------
    // Recommended action
    // ------------------------------------------------------------
    AddRecommendedAction =
        Table.AddColumn(
            AddAlertType,
            "RecommendedAction",
            each
                if [ScenarioRiskLevel] = "CRITICAL"
                then
                    "Immediate management review, stress testing, and mitigation planning"
                else if [ScenarioRiskLevel] = "HIGH"
                then
                    "Review downside drivers and execute mitigation actions"
                else if [Scenario] = "Optimistic"
                then
                    "Evaluate capacity, investment, and growth opportunities"
                else if [Scenario] = "Pessimistic"
                then
                    "Review downside assumptions and prepare contingency actions"
                else if [Scenario] = "Stress"
                then
                    "Run severe stress test and validate liquidity resilience"
                else
                    "Continue monitoring base-case assumptions",
            type text
        ),

    // ------------------------------------------------------------
    // Metric category
    // ------------------------------------------------------------
    AddMetricCategory =
        Table.AddColumn(
            AddRecommendedAction,
            "MetricCategory",
            each
                let
                    MetricText =
                        Text.Lower(
                            if [Metric] = null
                            then ""
                            else [Metric]
                        )
                in
                    if Text.Contains(MetricText, "revenue")
                    then "Revenue"
                    else if Text.Contains(MetricText, "profit")
                    then "Profitability"
                    else if Text.Contains(MetricText, "cash")
                    then "Cash Flow"
                    else if Text.Contains(MetricText, "expense")
                    then "Expenses"
                    else if Text.Contains(MetricText, "liquidity")
                    then "Liquidity"
                    else "Other",
            type text
        ),

    // ------------------------------------------------------------
    // Scenario ID
    // ------------------------------------------------------------
    AddScenarioID =
        Table.AddColumn(
            AddMetricCategory,
            "ScenarioID",
            each
                Text.Combine(
                    {
                        Date.ToText(
                            [ForecastDate],
                            "yyyyMMdd"
                        ),
                        [Company],
                        [BusinessUnit],
                        [Metric],
                        [Scenario],
                        [Model],
                        [ModelVersion]
                    },
                    "|"
                ),
            type text
        ),

    // ------------------------------------------------------------
    // Scenario label
    // ------------------------------------------------------------
    AddScenarioLabel =
        Table.AddColumn(
            AddScenarioID,
            "ScenarioLabel",
            each
                [Company]
                & " | "
                & [BusinessUnit]
                & " | "
                & [Metric]
                & " | "
                & [Scenario],
            type text
        ),

    // ------------------------------------------------------------
    // What-if adjustment percentage
    // ------------------------------------------------------------
    AddWhatIfAdjustment =
        Table.AddColumn(
            AddScenarioLabel,
            "WhatIfAdjustmentPct",
            each [ScenarioAdjustment],
            type number
        ),

    // ------------------------------------------------------------
    // Decision priority
    // ------------------------------------------------------------
    AddDecisionPriority =
        Table.AddColumn(
            AddWhatIfAdjustment,
            "DecisionPriority",
            each
                if [ScenarioRiskLevel] = "CRITICAL"
                then 1
                else if [ScenarioRiskLevel] = "HIGH"
                then 2
                else if [ScenarioRiskLevel] = "MEDIUM"
                then 3
                else 4,
            Int64.Type
        ),

    // ------------------------------------------------------------
    // Scenario ranking
    // ------------------------------------------------------------
    AddScenarioRank =
        Table.AddColumn(
            AddDecisionPriority,
            "ScenarioRank",
            each [ScenarioOrder],
            Int64.Type
        ),

    // ------------------------------------------------------------
    // Final sort
    // ------------------------------------------------------------
    Sorted =
        Table.Sort(
            AddScenarioRank,
            {
                {"ForecastDate", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending},
                {"Metric", Order.Ascending},
                {"ScenarioOrder", Order.Ascending}
            }
        ),

    // ------------------------------------------------------------
    // Reorder important columns
    // ------------------------------------------------------------
    Reordered =
        Table.ReorderColumns(
            Sorted,
            {
                "ScenarioID",
                "ForecastDate",
                "Year",
                "QuarterNumber",
                "Quarter",
                "MonthNumber",
                "Month",
                "MonthName",
                "MonthStart",
                "Company",
                "BusinessUnit",
                "Metric",
                "MetricCategory",
                "Scenario",
                "ScenarioType",
                "ScenarioOrder",
                "ScenarioRank",
                "ScenarioDirection",
                "ScenarioAdjustment",
                "WhatIfAdjustmentPct",
                "Actual",
                "Forecast",
                "ScenarioForecast",
                "LowerBound",
                "UpperBound",
                "ScenarioLowerBound",
                "ScenarioUpperBound",
                "ScenarioBandWidth",
                "ScenarioBandWidthPct",
                "ScenarioChange",
                "ScenarioChangePct",
                "ScenarioVariance",
                "ScenarioVariancePct",
                "Confidence",
                "ScenarioConfidence",
                "ConfidenceAdjustment",
                "ConfidenceRisk",
                "VolatilityRisk",
                "DownsideRisk",
                "ScenarioRiskScore",
                "ScenarioRiskLevel",
                "ScenarioStatus",
                "AlertFlag",
                "AlertType",
                "DecisionPriority",
                "RecommendedAction",
                "ScenarioLabel",
                "Model",
                "ModelVersion"
            },
            MissingField.UseNull
        ),

    // ------------------------------------------------------------
    // Buffer final result
    // ------------------------------------------------------------
    Final =
        Table.Buffer(Reordered)

in
    Final