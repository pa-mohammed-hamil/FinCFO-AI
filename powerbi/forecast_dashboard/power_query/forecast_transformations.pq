let
    // ============================================================
    // FinCo AI — Forecast Transformations
    // File: forecast_transformations.pq
    //
    // Purpose:
    //   Standardize, clean, validate, enrich, and prepare
    //   raw financial forecast data for downstream:
    //
    //   ForecastResults
    //   ForecastScenarios
    //   Forecast Dashboard
    //   Forecast Risk Engine
    //   What-If Analysis
    //   Agentic Decision Engine
    //
    // Expected source:
    //   ForecastData
    //
    // Supported source columns:
    //   ForecastDate
    //   Company
    //   BusinessUnit
    //   Metric
    //   Actual
    //   Forecast
    //   LowerBound
    //   UpperBound
    //   Confidence
    //   Model
    //   ModelVersion
    // ============================================================


    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------

    ForecastStartDate =
        #date(2022, 1, 1),

    ForecastEndDate =
        #date(2030, 12, 31),


    // ------------------------------------------------------------
    // Source
    // ------------------------------------------------------------

    Source =
        try ForecastData
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
    // Normalize column names
    // ------------------------------------------------------------

    NormalizeColumnNames =
        Table.TransformColumnNames(
            Source,
            each
                Text.Trim(
                    Text.Clean(
                        Text.From(_)
                    )
                )
        ),


    // ------------------------------------------------------------
    // Required columns
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


    // ------------------------------------------------------------
    // Add missing columns
    // ------------------------------------------------------------

    AddMissingColumns =
        List.Accumulate(
            RequiredColumns,
            NormalizeColumnNames,
            (CurrentTable, ColumnName) =>
                if List.Contains(
                    Table.ColumnNames(CurrentTable),
                    ColumnName
                )
                then
                    CurrentTable
                else
                    Table.AddColumn(
                        CurrentTable,
                        ColumnName,
                        each null
                    )
        ),


    // ------------------------------------------------------------
    // Select required columns
    // ------------------------------------------------------------

    SelectRequiredColumns =
        Table.SelectColumns(
            AddMissingColumns,
            RequiredColumns,
            MissingField.UseNull
        ),


    // ------------------------------------------------------------
    // Set data types
    // ------------------------------------------------------------

    SetDataTypes =
        Table.TransformColumnTypes(
            SelectRequiredColumns,
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
    // Clean text columns
    // ------------------------------------------------------------

    CleanText =
        Table.TransformColumns(
            SetDataTypes,
            {
                {
                    "Company",
                    each
                        if _ = null
                        then "Unknown"
                        else
                            Text.Trim(
                                Text.Clean(
                                    Text.From(_)
                                )
                            ),
                    type text
                },
                {
                    "BusinessUnit",
                    each
                        if _ = null
                        then "Unknown"
                        else
                            Text.Trim(
                                Text.Clean(
                                    Text.From(_)
                                )
                            ),
                    type text
                },
                {
                    "Metric",
                    each
                        if _ = null
                        then "Unknown"
                        else
                            Text.Trim(
                                Text.Clean(
                                    Text.From(_)
                                )
                            ),
                    type text
                },
                {
                    "Model",
                    each
                        if _ = null
                        then "Unknown"
                        else
                            Text.Trim(
                                Text.Clean(
                                    Text.From(_)
                                )
                            ),
                    type text
                },
                {
                    "ModelVersion",
                    each
                        if _ = null
                        then "Unknown"
                        else
                            Text.Trim(
                                Text.Clean(
                                    Text.From(_)
                                )
                            ),
                    type text
                }
            }
        ),


    // ------------------------------------------------------------
    // Clean numeric columns
    // ------------------------------------------------------------

    CleanNumericValues =
        Table.TransformColumns(
            CleanText,
            {
                {
                    "Actual",
                    each
                        if _ = null
                        then null
                        else
                            try Number.From(_)
                            otherwise null,
                    type number
                },
                {
                    "Forecast",
                    each
                        if _ = null
                        then null
                        else
                            try Number.From(_)
                            otherwise null,
                    type number
                },
                {
                    "LowerBound",
                    each
                        if _ = null
                        then null
                        else
                            try Number.From(_)
                            otherwise null,
                    type number
                },
                {
                    "UpperBound",
                    each
                        if _ = null
                        then null
                        else
                            try Number.From(_)
                            otherwise null,
                    type number
                },
                {
                    "Confidence",
                    each
                        if _ = null
                        then null
                        else
                            try Number.From(_)
                            otherwise null,
                    type number
                }
            }
        ),


    // ------------------------------------------------------------
    // Filter valid dates
    // ------------------------------------------------------------

    FilterValidDates =
        Table.SelectRows(
            CleanNumericValues,
            each
                [ForecastDate] <> null
                and [ForecastDate] >= ForecastStartDate
                and [ForecastDate] <= ForecastEndDate
        ),


    // ------------------------------------------------------------
    // Normalize confidence
    // ------------------------------------------------------------

    NormalizeConfidence =
        Table.TransformColumns(
            FilterValidDates,
            {
                {
                    "Confidence",
                    each
                        if _ = null
                        then null
                        else
                            Number.Max(
                                0,
                                Number.Min(
                                    1,
                                    _
                                )
                            ),
                    type number
                }
            }
        ),


    // ------------------------------------------------------------
    // Normalize forecast bounds
    // ------------------------------------------------------------

    NormalizeLowerBound =
        Table.AddColumn(
            NormalizeConfidence,
            "NormalizedLowerBound",
            each
                if [LowerBound] = null
                then [Forecast]
                else [LowerBound],
            type number
        ),


    NormalizeUpperBound =
        Table.AddColumn(
            NormalizeLowerBound,
            "NormalizedUpperBound",
            each
                if [UpperBound] = null
                then [Forecast]
                else [UpperBound],
            type number
        ),


    // ------------------------------------------------------------
    // Correct reversed confidence intervals
    // ------------------------------------------------------------

    CorrectLowerBound =
        Table.AddColumn(
            NormalizeUpperBound,
            "FinalLowerBound",
            each
                if [NormalizedLowerBound] = null
                then null
                else
                    Number.Min(
                        [NormalizedLowerBound],
                        [NormalizedUpperBound]
                    ),
            type number
        ),


    CorrectUpperBound =
        Table.AddColumn(
            CorrectLowerBound,
            "FinalUpperBound",
            each
                if [NormalizedUpperBound] = null
                then null
                else
                    Number.Max(
                        [NormalizedLowerBound],
                        [NormalizedUpperBound]
                    ),
            type number
        ),


    // ------------------------------------------------------------
    // Date dimensions
    // ------------------------------------------------------------

    AddYear =
        Table.AddColumn(
            CorrectUpperBound,
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
            each
                "Q"
                & Text.From([QuarterNumber]),
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
            each
                Date.ToText(
                    [ForecastDate],
                    "yyyy-MM"
                ),
            type text
        ),


    AddMonthName =
        Table.AddColumn(
            AddMonth,
            "MonthName",
            each
                Date.MonthName(
                    [ForecastDate]
                ),
            type text
        ),


    AddMonthStart =
        Table.AddColumn(
            AddMonthName,
            "MonthStart",
            each
                Date.StartOfMonth(
                    [ForecastDate]
                ),
            type date
        ),


    AddMonthYear =
        Table.AddColumn(
            AddMonthStart,
            "MonthYear",
            each
                Date.ToText(
                    [ForecastDate],
                    "MMM yyyy"
                ),
            type text
        ),


    AddPeriodKey =
        Table.AddColumn(
            AddMonthYear,
            "PeriodKey",
            each
                Date.ToText(
                    [ForecastDate],
                    "yyyyMM"
                ),
            type text
        ),


    // ------------------------------------------------------------
    // Metric normalization
    // ------------------------------------------------------------

    AddNormalizedMetric =
        Table.AddColumn(
            AddPeriodKey,
            "NormalizedMetric",
            each
                let
                    MetricText =
                        Text.Lower(
                            Text.Trim(
                                if [Metric] = null
                                then ""
                                else [Metric]
                            )
                        )
                in
                    if
                        Text.Contains(
                            MetricText,
                            "revenue"
                        )
                    then
                        "Revenue"

                    else if
                        Text.Contains(
                            MetricText,
                            "sales"
                        )
                    then
                        "Revenue"

                    else if
                        Text.Contains(
                            MetricText,
                            "gross profit"
                        )
                    then
                        "Gross Profit"

                    else if
                        Text.Contains(
                            MetricText,
                            "operating profit"
                        )
                    then
                        "Operating Profit"

                    else if
                        Text.Contains(
                            MetricText,
                            "net profit"
                        )
                    then
                        "Net Profit"

                    else if
                        Text.Contains(
                            MetricText,
                            "profit"
                        )
                    then
                        "Profit"

                    else if
                        Text.Contains(
                            MetricText,
                            "cash flow"
                        )
                    then
                        "Cash Flow"

                    else if
                        Text.Contains(
                            MetricText,
                            "cash"
                        )
                    then
                        "Cash Flow"

                    else if
                        Text.Contains(
                            MetricText,
                            "expense"
                        )
                    then
                        "Expenses"

                    else if
                        Text.Contains(
                            MetricText,
                            "cost"
                        )
                    then
                        "Expenses"

                    else if
                        Text.Contains(
                            MetricText,
                            "liquidity"
                        )
                    then
                        "Liquidity"

                    else
                        Text.Proper(
                            [Metric]
                        ),
            type text
        ),


    // ------------------------------------------------------------
    // Metric category
    // ------------------------------------------------------------

    AddMetricCategory =
        Table.AddColumn(
            AddNormalizedMetric,
            "MetricCategory",
            each
                if [NormalizedMetric] = "Revenue"
                then "Revenue"

                else if
                    [NormalizedMetric]
                    = "Gross Profit"
                    or
                    [NormalizedMetric]
                    = "Operating Profit"
                    or
                    [NormalizedMetric]
                    = "Net Profit"
                    or
                    [NormalizedMetric]
                    = "Profit"
                then "Profitability"

                else if
                    [NormalizedMetric]
                    = "Cash Flow"
                then "Cash Flow"

                else if
                    [NormalizedMetric]
                    = "Expenses"
                then "Expenses"

                else if
                    [NormalizedMetric]
                    = "Liquidity"
                then "Liquidity"

                else "Other",
            type text
        ),


    // ------------------------------------------------------------
    // Forecast horizon classification
    // ------------------------------------------------------------

    CurrentDate =
        Date.From(
            DateTime.LocalNow()
        ),


    AddForecastHorizon =
        Table.AddColumn(
            AddMetricCategory,
            "ForecastHorizon",
            each
                if [ForecastDate] < CurrentDate
                then "Historical"

                else if
                    [ForecastDate] = CurrentDate
                then "Current"

                else if
                    Duration.Days(
                        [ForecastDate]
                        - CurrentDate
                    ) <= 30
                then "Near Term"

                else if
                    Duration.Days(
                        [ForecastDate]
                        - CurrentDate
                    ) <= 90
                then "Short Term"

                else if
                    Duration.Days(
                        [ForecastDate]
                        - CurrentDate
                    ) <= 365
                then "Medium Term"

                else "Long Term",
            type text
        ),


    // ------------------------------------------------------------
    // Actual / forecast classification
    // ------------------------------------------------------------

    AddForecastType =
        Table.AddColumn(
            AddForecastHorizon,
            "ForecastType",
            each
                if [Actual] <> null
                then "Actual + Forecast Evaluation"
                else "Future Forecast",
            type text
        ),


    // ------------------------------------------------------------
    // Forecast error
    // ------------------------------------------------------------

    AddForecastError =
        Table.AddColumn(
            AddForecastType,
            "ForecastError",
            each
                if [Actual] = null
                or [Forecast] = null
                then null
                else
                    [Actual] - [Forecast],
            type number
        ),


    AddAbsoluteError =
        Table.AddColumn(
            AddForecastError,
            "AbsoluteError",
            each
                if [ForecastError] = null
                then null
                else
                    Number.Abs(
                        [ForecastError]
                    ),
            type number
        ),


    AddSquaredError =
        Table.AddColumn(
            AddAbsoluteError,
            "SquaredError",
            each
                if [ForecastError] = null
                then null
                else
                    Number.Power(
                        [ForecastError],
                        2
                    ),
            type number
        ),


    // ------------------------------------------------------------
    // Percentage error
    // ------------------------------------------------------------

    AddAPE =
        Table.AddColumn(
            AddSquaredError,
            "APE",
            each
                if
                    [Actual] = null
                    or [Actual] = 0
                    or [Forecast] = null
                then null
                else
                    Number.Abs(
                        [Actual] - [Forecast]
                    )
                    /
                    Number.Abs(
                        [Actual]
                    ),
            type number
        ),


    AddPercentError =
        Table.AddColumn(
            AddAPE,
            "PercentError",
            each
                if
                    [Actual] = null
                    or [Actual] = 0
                    or [Forecast] = null
                then null
                else
                    ([Forecast] - [Actual])
                    /
                    Number.Abs(
                        [Actual]
                    ),
            type number
        ),


    // ------------------------------------------------------------
    // Forecast accuracy
    // ------------------------------------------------------------

    AddForecastAccuracy =
        Table.AddColumn(
            AddPercentError,
            "ForecastAccuracy",
            each
                if [APE] = null
                then null
                else
                    Number.Max(
                        0,
                        1 - [APE]
                    ),
            type number
        ),


    // ------------------------------------------------------------
    // Forecast direction
    // ------------------------------------------------------------

    AddForecastDirection =
        Table.AddColumn(
            AddForecastAccuracy,
            "ForecastDirection",
            each
                if [Forecast] = null
                then "UNKNOWN"

                else if
                    [Actual] <> null
                    and [Forecast] > [Actual]
                then "UP"

                else if
                    [Actual] <> null
                    and [Forecast] < [Actual]
                then "DOWN"

                else "NEUTRAL",
            type text
        ),


    // ------------------------------------------------------------
    // Forecast change
    // ------------------------------------------------------------

    AddForecastChange =
        Table.AddColumn(
            AddForecastDirection,
            "ForecastChange",
            each
                if
                    [Actual] = null
                    or [Forecast] = null
                then null
                else
                    [Forecast] - [Actual],
            type number
        ),


    AddForecastChangePct =
        Table.AddColumn(
            AddForecastChange,
            "ForecastChangePct",
            each
                if
                    [Actual] = null
                    or [Actual] = 0
                    or [Forecast] = null
                then null
                else
                    ([Forecast] - [Actual])
                    /
                    Number.Abs(
                        [Actual]
                    ),
            type number
        ),


    // ------------------------------------------------------------
    // Confidence interval calculations
    // ------------------------------------------------------------

    AddConfidenceBandWidth =
        Table.AddColumn(
            AddForecastChangePct,
            "ConfidenceBandWidth",
            each
                if
                    [FinalUpperBound] = null
                    or [FinalLowerBound] = null
                then null
                else
                    [FinalUpperBound]
                    - [FinalLowerBound],
            type number
        ),


    AddConfidenceBandWidthPct =
        Table.AddColumn(
            AddConfidenceBandWidth,
            "ConfidenceBandWidthPct",
            each
                if
                    [Forecast] = null
                    or [Forecast] = 0
                    or [ConfidenceBandWidth] = null
                then null
                else
                    [ConfidenceBandWidth]
                    /
                    Number.Abs(
                        [Forecast]
                    ),
            type number
        ),


    // ------------------------------------------------------------
    // Confidence classification
    // ------------------------------------------------------------

    AddConfidenceLevel =
        Table.AddColumn(
            AddConfidenceBandWidthPct,
            "ConfidenceLevel",
            each
                if [Confidence] = null
                then "Unknown"

                else if [Confidence] >= 0.90
                then "Very High"

                else if [Confidence] >= 0.75
                then "High"

                else if [Confidence] >= 0.60
                then "Medium"

                else if [Confidence] >= 0.40
                then "Low"

                else "Very Low",
            type text
        ),


    // ------------------------------------------------------------
    // Confidence risk
    // ------------------------------------------------------------

    AddConfidenceRisk =
        Table.AddColumn(
            AddConfidenceLevel,
            "ConfidenceRisk",
            each
                if [Confidence] = null
                then 40

                else if [Confidence] < 0.40
                then 40

                else if [Confidence] < 0.60
                then 30

                else if [Confidence] < 0.75
                then 20

                else if [Confidence] < 0.90
                then 10

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
                if [ConfidenceBandWidthPct] = null
                then 0

                else if
                    [ConfidenceBandWidthPct] >= 0.50
                then 40

                else if
                    [ConfidenceBandWidthPct] >= 0.30
                then 30

                else if
                    [ConfidenceBandWidthPct] >= 0.15
                then 20

                else if
                    [ConfidenceBandWidthPct] >= 0.05
                then 10

                else 0,
            Int64.Type
        ),


    // ------------------------------------------------------------
    // Decline risk
    // ------------------------------------------------------------

    AddDeclineRisk =
        Table.AddColumn(
            AddVolatilityRisk,
            "DeclineRisk",
            each
                if [ForecastChangePct] = null
                then 0

                else if
                    [ForecastChangePct] <= -0.30
                then 40

                else if
                    [ForecastChangePct] <= -0.20
                then 30

                else if
                    [ForecastChangePct] <= -0.10
                then 20

                else if
                    [ForecastChangePct] < 0
                then 10

                else 0,
            Int64.Type
        ),


    // ------------------------------------------------------------
    // Composite forecast risk score
    // ------------------------------------------------------------

    AddForecastRiskScore =
        Table.AddColumn(
            AddDeclineRisk,
            "ForecastRiskScore",
            each
                Number.Min(
                    100,
                    [ConfidenceRisk]
                    + [VolatilityRisk]
                    + [DeclineRisk]
                ),
            Int64.Type
        ),


    // ------------------------------------------------------------
    // Forecast risk level
    // ------------------------------------------------------------

    AddForecastRiskLevel =
        Table.AddColumn(
            AddForecastRiskScore,
            "ForecastRiskLevel",
            each
                if [ForecastRiskScore] >= 75
                then "CRITICAL"

                else if [ForecastRiskScore] >= 50
                then "HIGH"

                else if [ForecastRiskScore] >= 25
                then "MEDIUM"

                else "LOW",
            type text
        ),


    // ------------------------------------------------------------
    // Forecast status
    // ------------------------------------------------------------

    AddForecastStatus =
        Table.AddColumn(
            AddForecastRiskLevel,
            "ForecastStatus",
            each
                if [Actual] = null
                then "Future Forecast"

                else if
                    [ForecastRiskLevel] = "CRITICAL"
                then "Critical"

                else if
                    [ForecastRiskLevel] = "HIGH"
                then "High Risk"

                else if
                    [ForecastAccuracy] >= 0.90
                then "Accurate"

                else if
                    [ForecastAccuracy] >= 0.75
                then "Acceptable"

                else "Needs Review",
            type text
        ),


    // ------------------------------------------------------------
    // Model quality
    // ------------------------------------------------------------

    AddModelQuality =
        Table.AddColumn(
            AddForecastStatus,
            "ModelQuality",
            each
                if [Actual] = null
                then "Not Evaluated"

                else if
                    [ForecastAccuracy] >= 0.90
                then "Excellent"

                else if
                    [ForecastAccuracy] >= 0.80
                then "Good"

                else if
                    [ForecastAccuracy] >= 0.70
                then "Acceptable"

                else "Needs Improvement",
            type text
        ),


    // ------------------------------------------------------------
    // Forecast alert
    // ------------------------------------------------------------

    AddAlertFlag =
        Table.AddColumn(
            AddModelQuality,
            "AlertFlag",
            each
                if
                    [ForecastRiskLevel] = "CRITICAL"
                    or
                    [ForecastRiskLevel] = "HIGH"
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
                if
                    [ForecastRiskLevel]
                    = "CRITICAL"
                then
                    "Severe Forecast Risk"

                else if
                    [ForecastRiskLevel]
                    = "HIGH"
                then
                    "High Forecast Risk"

                else if
                    [ForecastRiskLevel]
                    = "MEDIUM"
                then
                    "Forecast Monitoring"

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
                if
                    [ForecastRiskLevel]
                    = "CRITICAL"
                then
                    "Immediate management review and scenario analysis"

                else if
                    [ForecastRiskLevel]
                    = "HIGH"
                then
                    "Investigate forecast drivers and execute mitigation plan"

                else if
                    [ForecastRiskLevel]
                    = "MEDIUM"
                then
                    "Monitor forecast and review underlying assumptions"

                else
                    "Continue normal monitoring",
            type text
        ),


    // ------------------------------------------------------------
    // Data quality checks
    // ------------------------------------------------------------

    AddDateQuality =
        Table.AddColumn(
            AddRecommendedAction,
            "DateQuality",
            each
                if [ForecastDate] = null
                then "Invalid Date"
                else "Valid",
            type text
        ),


    AddForecastQuality =
        Table.AddColumn(
            AddDateQuality,
            "ForecastQuality",
            each
                if [Forecast] = null
                then "Missing Forecast"

                else if
                    [Forecast] = 0
                then "Zero Forecast"

                else "Valid Forecast",
            type text
        ),


    AddConfidenceQuality =
        Table.AddColumn(
            AddForecastQuality,
            "ConfidenceQuality",
            each
                if [Confidence] = null
                then "Missing Confidence"

                else if
                    [Confidence] < 0.40
                then "Low Confidence"

                else "Valid Confidence",
            type text
        ),


    // ------------------------------------------------------------
    // Overall data quality
    // ------------------------------------------------------------

    AddDataQualityStatus =
        Table.AddColumn(
            AddConfidenceQuality,
            "DataQualityStatus",
            each
                if
                    [DateQuality] = "Invalid Date"
                    or
                    [ForecastQuality]
                    = "Missing Forecast"
                then
                    "INVALID"

                else if
                    [ConfidenceQuality]
                    = "Missing Confidence"
                then
                    "WARNING"

                else
                    "VALID",
            type text
        ),


    // ------------------------------------------------------------
    // Forecast eligibility
    // ------------------------------------------------------------

    AddForecastEligibility =
        Table.AddColumn(
            AddDataQualityStatus,
            "ForecastEligible",
            each
                if
                    [DataQualityStatus]
                    = "INVALID"
                then "NO"

                else if
                    [Forecast] = null
                then "NO"

                else if
                    [ForecastDate] = null
                then "NO"

                else "YES",
            type text
        ),


    // ------------------------------------------------------------
    // Decision priority
    // ------------------------------------------------------------

    AddDecisionPriority =
        Table.AddColumn(
            AddForecastEligibility,
            "DecisionPriority",
            each
                if
                    [ForecastRiskLevel]
                    = "CRITICAL"
                then 1

                else if
                    [ForecastRiskLevel]
                    = "HIGH"
                then 2

                else if
                    [ForecastRiskLevel]
                    = "MEDIUM"
                then 3

                else 4,
            Int64.Type
        ),


    // ------------------------------------------------------------
    // Forecast ID
    // ------------------------------------------------------------

    AddForecastID =
        Table.AddColumn(
            AddDecisionPriority,
            "ForecastID",
            each
                Text.Combine(
                    {
                        Date.ToText(
                            [ForecastDate],
                            "yyyyMMdd"
                        ),
                        [Company],
                        [BusinessUnit],
                        [NormalizedMetric],
                        [Model],
                        [ModelVersion]
                    },
                    "|"
                ),
            type text
        ),


    // ------------------------------------------------------------
    // Forecast label
    // ------------------------------------------------------------

    AddForecastLabel =
        Table.AddColumn(
            AddForecastID,
            "ForecastLabel",
            each
                [Company]
                & " | "
                & [BusinessUnit]
                & " | "
                & [NormalizedMetric],
            type text
        ),


    // ------------------------------------------------------------
    // Final cleanup
    // ------------------------------------------------------------

    RemoveTemporaryColumns =
        Table.RemoveColumns(
            AddForecastLabel,
            {
                "NormalizedLowerBound",
                "NormalizedUpperBound"
            },
            MissingField.Ignore
        ),


    RenameFinalBounds =
        Table.RenameColumns(
            RemoveTemporaryColumns,
            {
                {
                    "FinalLowerBound",
                    "LowerBound"
                },
                {
                    "FinalUpperBound",
                    "UpperBound"
                }
            },
            MissingField.Ignore
        ),


    // ------------------------------------------------------------
    // Remove duplicate forecast records
    // ------------------------------------------------------------

    RemoveDuplicates =
        Table.Distinct(
            RenameFinalBounds,
            {
                "ForecastID"
            }
        ),


    // ------------------------------------------------------------
    // Final column order
    // ------------------------------------------------------------

    ReorderedColumns =
        Table.ReorderColumns(
            RemoveDuplicates,
            {
                "ForecastID",
                "ForecastDate",
                "Year",
                "QuarterNumber",
                "Quarter",
                "MonthNumber",
                "Month",
                "MonthName",
                "MonthYear",
                "MonthStart",
                "PeriodKey",

                "Company",
                "BusinessUnit",
                "Metric",
                "NormalizedMetric",
                "MetricCategory",

                "Actual",
                "Forecast",
                "LowerBound",
                "UpperBound",

                "ForecastError",
                "AbsoluteError",
                "SquaredError",
                "APE",
                "PercentError",
                "ForecastAccuracy",

                "ForecastDirection",
                "ForecastChange",
                "ForecastChangePct",

                "Confidence",
                "ConfidenceLevel",
                "ConfidenceBandWidth",
                "ConfidenceBandWidthPct",

                "ForecastHorizon",
                "ForecastType",

                "ConfidenceRisk",
                "VolatilityRisk",
                "DeclineRisk",
                "ForecastRiskScore",
                "ForecastRiskLevel",

                "ForecastStatus",
                "ModelQuality",

                "AlertFlag",
                "AlertType",
                "DecisionPriority",
                "RecommendedAction",

                "DateQuality",
                "ForecastQuality",
                "ConfidenceQuality",
                "DataQualityStatus",
                "ForecastEligible",

                "ForecastLabel",
                "Model",
                "ModelVersion"
            },
            MissingField.UseNull
        ),


    // ------------------------------------------------------------
    // Sort
    // ------------------------------------------------------------

    Sorted =
        Table.Sort(
            ReorderedColumns,
            {
                {
                    "ForecastDate",
                    Order.Ascending
                },
                {
                    "Company",
                    Order.Ascending
                },
                {
                    "BusinessUnit",
                    Order.Ascending
                },
                {
                    "NormalizedMetric",
                    Order.Ascending
                }
            }
        ),


    // ------------------------------------------------------------
    // Buffer final result
    // ------------------------------------------------------------

    Final =
        Table.Buffer(
            Sorted
        )

in
    Final