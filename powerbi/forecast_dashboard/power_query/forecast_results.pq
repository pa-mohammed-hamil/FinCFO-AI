let
    // ============================================================
    // FINCO AI
    // FORECAST RESULTS
    // Power Query / M
    // ============================================================

    // ============================================================
    // 1. PARAMETERS
    // ============================================================

    ForecastStartDate =
        #date(2022, 1, 1),

    ForecastEndDate =
        #date(2030, 12, 31),

    // ============================================================
    // 2. SOURCE
    //
    // Expected table:
    //
    // ForecastResults
    //
    // Columns:
    // ForecastDate
    // Company
    // BusinessUnit
    // Metric
    // Actual
    // Forecast
    // LowerBound
    // UpperBound
    // Confidence
    // Model
    // ModelVersion
    // ============================================================

    Source =
        try
            Excel.CurrentWorkbook()
                {[Name = "ForecastResults"]}[Content]
        otherwise
            #table(
                {
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
                {}
            ),

    // ============================================================
    // 3. STANDARDIZE COLUMN NAMES
    // ============================================================

    StandardizedColumns =
        Table.TransformColumnNames(
            Source,
            each Text.Trim(_)
        ),

    // ============================================================
    // 4. DATA TYPES
    // ============================================================

    Typed =
        Table.TransformColumnTypes(
            StandardizedColumns,
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
            },
            "en-US"
        ),

    // ============================================================
    // 5. VALID DATE RANGE
    // ============================================================

    ValidDates =
        Table.SelectRows(
            Typed,
            each
                [ForecastDate] <> null
                and [ForecastDate] >= ForecastStartDate
                and [ForecastDate] <= ForecastEndDate
        ),

    // ============================================================
    // 6. CLEAN TEXT
    // ============================================================

    CleanText =
        Table.TransformColumns(
            ValidDates,
            {
                {
                    "Company",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(_),
                    type text
                },
                {
                    "BusinessUnit",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(_),
                    type text
                },
                {
                    "Metric",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Proper(Text.Trim(_)),
                    type text
                },
                {
                    "Model",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(_),
                    type text
                },
                {
                    "ModelVersion",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(_),
                    type text
                }
            }
        ),

    // ============================================================
    // 7. NULL NUMERIC VALUES
    //
    // Keep Actual null when it represents a future period.
    // Forecast and confidence-related values receive safe defaults.
    // ============================================================

    CleanForecastValues =
        Table.ReplaceValue(
            CleanText,
            null,
            0,
            Replacer.ReplaceValue,
            {
                "Forecast",
                "LowerBound",
                "UpperBound"
            }
        ),

    CleanConfidence =
        Table.TransformColumns(
            CleanForecastValues,
            {
                {
                    "Confidence",
                    each
                        if _ = null
                        then 0
                        else Number.Max(
                            0,
                            Number.Min(1, _)
                        ),
                    type number
                }
            }
        ),

    // ============================================================
    // 8. YEAR
    // ============================================================

    AddYear =
        Table.AddColumn(
            CleanConfidence,
            "Year",
            each Date.Year([ForecastDate]),
            Int64.Type
        ),

    // ============================================================
    // 9. QUARTER NUMBER
    // ============================================================

    AddQuarterNumber =
        Table.AddColumn(
            AddYear,
            "QuarterNumber",
            each Date.QuarterOfYear([ForecastDate]),
            Int64.Type
        ),

    // ============================================================
    // 10. QUARTER
    // ============================================================

    AddQuarter =
        Table.AddColumn(
            AddQuarterNumber,
            "Quarter",
            each
                "Q" &
                Number.ToText([QuarterNumber]),
            type text
        ),

    // ============================================================
    // 11. MONTH NUMBER
    // ============================================================

    AddMonthNumber =
        Table.AddColumn(
            AddQuarter,
            "MonthNumber",
            each Date.Month([ForecastDate]),
            Int64.Type
        ),

    // ============================================================
    // 12. MONTH
    // ============================================================

    AddMonth =
        Table.AddColumn(
            AddMonthNumber,
            "Month",
            each Date.ToText(
                [ForecastDate],
                "MMM"
            ),
            type text
        ),

    // ============================================================
    // 13. MONTH NAME
    // ============================================================

    AddMonthName =
        Table.AddColumn(
            AddMonth,
            "MonthName",
            each Date.ToText(
                [ForecastDate],
                "MMMM"
            ),
            type text
        ),

    // ============================================================
    // 14. MONTH-YEAR
    // ============================================================

    AddMonthYear =
        Table.AddColumn(
            AddMonthName,
            "MonthYear",
            each Date.ToText(
                [ForecastDate],
                "MMM yyyy"
            ),
            type text
        ),

    // ============================================================
    // 15. PERIOD KEY
    // ============================================================

    AddPeriodKey =
        Table.AddColumn(
            AddMonthYear,
            "PeriodKey",
            each
                Date.Year([ForecastDate]) * 100
                +
                Date.Month([ForecastDate]),
            Int64.Type
        ),

    // ============================================================
    // 16. MONTH START
    // ============================================================

    AddMonthStart =
        Table.AddColumn(
            AddPeriodKey,
            "MonthStart",
            each Date.StartOfMonth(
                [ForecastDate]
            ),
            type date
        ),

    // ============================================================
    // 17. FORECAST ERROR
    //
    // Actual - Forecast
    // ============================================================

    AddForecastError =
        Table.AddColumn(
            AddMonthStart,
            "ForecastError",
            each
                if [Actual] = null
                then null
                else
                    [Actual] - [Forecast],
            type number
        ),

    // ============================================================
    // 18. ABSOLUTE ERROR
    // ============================================================

    AddAbsoluteError =
        Table.AddColumn(
            AddForecastError,
            "AbsoluteError",
            each
                if [ForecastError] = null
                then null
                else Number.Abs(
                    [ForecastError]
                ),
            type number
        ),

    // ============================================================
    // 19. SQUARED ERROR
    // ============================================================

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

    // ============================================================
    // 20. APE
    //
    // Absolute Percentage Error
    // ============================================================

    AddAPE =
        Table.AddColumn(
            AddSquaredError,
            "APE",
            each
                if [Actual] = null
                    or [Actual] = 0
                then null
                else
                    Number.Abs(
                        [Actual] - [Forecast]
                    )
                    /
                    Number.Abs([Actual]),
            Percentage.Type
        ),

    // ============================================================
    // 21. SIGNED PERCENT ERROR
    // ============================================================

    AddPercentError =
        Table.AddColumn(
            AddAPE,
            "PercentError",
            each
                if [Actual] = null
                    or [Actual] = 0
                then null
                else
                    ([Forecast] - [Actual])
                    /
                    Number.Abs([Actual]),
            Percentage.Type
        ),

    // ============================================================
    // 22. FORECAST ACCURACY
    // ============================================================

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
            Percentage.Type
        ),

    // ============================================================
    // 23. CONFIDENCE BAND
    // ============================================================

    AddConfidenceBand =
        Table.AddColumn(
            AddForecastAccuracy,
            "ConfidenceBand",
            each
                if [Confidence] >= 0.90
                then "Very High"
                else if [Confidence] >= 0.75
                then "High"
                else if [Confidence] >= 0.60
                then "Medium"
                else if [Confidence] >= 0.40
                then "Low"
                else
                    "Very Low",
            type text
        ),

    // ============================================================
    // 24. FORECAST DIRECTION
    // ============================================================

    AddForecastDirection =
        Table.AddColumn(
            AddConfidenceBand,
            "ForecastDirection",
            each
                if [Actual] = null
                then
                    "Future"
                else if [Forecast] > [Actual]
                then
                    "Increase"
                else if [Forecast] < [Actual]
                then
                    "Decrease"
                else
                    "Stable",
            type text
        ),

    // ============================================================
    // 25. FORECAST CHANGE
    // ============================================================

    AddForecastChange =
        Table.AddColumn(
            AddForecastDirection,
            "ForecastChange",
            each
                if [Actual] = null
                then null
                else
                    [Forecast] - [Actual],
            type number
        ),

    // ============================================================
    // 26. FORECAST CHANGE %
    // ============================================================

    AddForecastChangePct =
        Table.AddColumn(
            AddForecastChange,
            "ForecastChangePct",
            each
                if [Actual] = null
                    or [Actual] = 0
                then null
                else
                    ([Forecast] - [Actual])
                    /
                    Number.Abs([Actual]),
            Percentage.Type
        ),

    // ============================================================
    // 27. FORECAST BAND WIDTH
    // ============================================================

    AddBandWidth =
        Table.AddColumn(
            AddForecastChangePct,
            "ForecastBandWidth",
            each
                [UpperBound] - [LowerBound],
            type number
        ),

    // ============================================================
    // 28. BAND WIDTH %
    // ============================================================

    AddBandWidthPct =
        Table.AddColumn(
            AddBandWidth,
            "ForecastBandWidthPct",
            each
                if [Forecast] = 0
                then null
                else
                    ([UpperBound] - [LowerBound])
                    /
                    Number.Abs([Forecast]),
            Percentage.Type
        ),

    // ============================================================
    // 29. MODEL CONFIDENCE RISK
    // ============================================================

    AddConfidenceRisk =
        Table.AddColumn(
            AddBandWidthPct,
            "ConfidenceRisk",
            each
                if [Confidence] < 0.40
                then 40
                else if [Confidence] < 0.60
                then 30
                else if [Confidence] < 0.75
                then 20
                else if [Confidence] < 0.90
                then 10
                else
                    0,
            Int64.Type
        ),

    // ============================================================
    // 30. FORECAST VOLATILITY RISK
    // ============================================================

    AddVolatilityRisk =
        Table.AddColumn(
            AddConfidenceRisk,
            "VolatilityRisk",
            each
                if [ForecastBandWidthPct] = null
                then 0
                else if [ForecastBandWidthPct] >= 0.50
                then 40
                else if [ForecastBandWidthPct] >= 0.30
                then 30
                else if [ForecastBandWidthPct] >= 0.15
                then 20
                else if [ForecastBandWidthPct] >= 0.05
                then 10
                else
                    0,
            Int64.Type
        ),

    // ============================================================
    // 31. FORECAST DECLINE RISK
    // ============================================================

    AddDeclineRisk =
        Table.AddColumn(
            AddVolatilityRisk,
            "DeclineRisk",
            each
                if [ForecastChangePct] = null
                then 0
                else if [ForecastChangePct] <= -0.30
                then 40
                else if [ForecastChangePct] <= -0.20
                then 30
                else if [ForecastChangePct] <= -0.10
                then 20
                else if [ForecastChangePct] < 0
                then 10
                else
                    0,
            Int64.Type
        ),

    // ============================================================
    // 32. COMBINED FORECAST RISK SCORE
    // ============================================================

    AddForecastRiskScore =
        Table.AddColumn(
            AddDeclineRisk,
            "ForecastRiskScore",
            each
                Number.Min(
                    100,
                    [ConfidenceRisk]
                    +
                    [VolatilityRisk]
                    +
                    [DeclineRisk]
                ),
            Int64.Type
        ),

    // ============================================================
    // 33. FORECAST RISK LEVEL
    // ============================================================

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
                else
                    "LOW",
            type text
        ),

    // ============================================================
    // 34. FORECAST STATUS
    // ============================================================

    AddForecastStatus =
        Table.AddColumn(
            AddForecastRiskLevel,
            "ForecastStatus",
            each
                if [Actual] = null
                then "Future Forecast"
                else if [ForecastRiskLevel] = "CRITICAL"
                then "Critical"
                else if [ForecastRiskLevel] = "HIGH"
                then "High Risk"
                else if [ForecastAccuracy] <> null
                    and [ForecastAccuracy] >= 0.90
                then "Accurate"
                else if [ForecastAccuracy] <> null
                    and [ForecastAccuracy] >= 0.75
                then "Acceptable"
                else
                    "Needs Review",
            type text
        ),

    // ============================================================
    // 35. MODEL QUALITY
    // ============================================================

    AddModelQuality =
        Table.AddColumn(
            AddForecastStatus,
            "ModelQuality",
            each
                if [Actual] = null
                then "Not Evaluated"
                else if [ForecastAccuracy] >= 0.90
                then "Excellent"
                else if [ForecastAccuracy] >= 0.80
                then "Good"
                else if [ForecastAccuracy] >= 0.70
                then "Acceptable"
                else
                    "Needs Improvement",
            type text
        ),

    // ============================================================
    // 36. ALERT FLAG
    // ============================================================

    AddAlertFlag =
        Table.AddColumn(
            AddModelQuality,
            "AlertFlag",
            each
                if [ForecastRiskLevel] = "CRITICAL"
                    or [ForecastRiskLevel] = "HIGH"
                then true
                else false,
            type logical
        ),

    // ============================================================
    // 37. ALERT TYPE
    // ============================================================

    AddAlertType =
        Table.AddColumn(
            AddAlertFlag,
            "AlertType",
            each
                if [ForecastRiskLevel] = "CRITICAL"
                then
                    "Severe Forecast Risk"
                else if [ForecastRiskLevel] = "HIGH"
                then
                    "High Forecast Risk"
                else if [ForecastRiskLevel] = "MEDIUM"
                then
                    "Forecast Monitoring"
                else
                    null,
            type text
        ),

    // ============================================================
    // 38. RECOMMENDED ACTION
    // ============================================================

    AddRecommendedAction =
        Table.AddColumn(
            AddAlertType,
            "RecommendedAction",
            each
                if [ForecastRiskLevel] = "CRITICAL"
                then
                    "Immediate management review and scenario analysis"
                else if [ForecastRiskLevel] = "HIGH"
                then
                    "Investigate forecast drivers and execute mitigation plan"
                else if [ForecastRiskLevel] = "MEDIUM"
                then
                    "Monitor forecast and review underlying assumptions"
                else
                    "Continue normal monitoring",
            type text
        ),

    // ============================================================
    // 39. FUTURE / HISTORICAL FLAG
    // ============================================================

    AddForecastType =
        Table.AddColumn(
            AddRecommendedAction,
            "ForecastType",
            each
                if [Actual] = null
                then "Forecast"
                else "Actual",
            type text
        ),

    // ============================================================
    // 40. METRIC CATEGORY
    // ============================================================

    AddMetricCategory =
        Table.AddColumn(
            AddForecastType,
            "MetricCategory",
            each
                if Text.Contains(
                    Text.Lower([Metric]),
                    "revenue"
                )
                then "Revenue"

                else if Text.Contains(
                    Text.Lower([Metric]),
                    "profit"
                )
                then "Profitability"

                else if Text.Contains(
                    Text.Lower([Metric]),
                    "cash"
                )
                then "Cash Flow"

                else if Text.Contains(
                    Text.Lower([Metric]),
                    "expense"
                )
                then "Expenses"

                else if Text.Contains(
                    Text.Lower([Metric]),
                    "liquidity"
                )
                then "Liquidity"

                else
                    "Other",
            type text
        ),

    // ============================================================
    // 41. FORECAST ID
    // ============================================================

    AddForecastID =
        Table.AddColumn(
            AddMetricCategory,
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
                        [Metric],
                        [Model],
                        [ModelVersion]
                    },
                    "|"
                ),
            type text
        ),

    // ============================================================
    // 42. FINAL SORT
    // ============================================================

    Sorted =
        Table.Sort(
            AddForecastID,
            {
                {"ForecastDate", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending},
                {"Metric", Order.Ascending}
            }
        ),

    // ============================================================
    // 43. FINAL OUTPUT
    // ============================================================

    Final =
        Table.Buffer(Sorted)

in
    Final