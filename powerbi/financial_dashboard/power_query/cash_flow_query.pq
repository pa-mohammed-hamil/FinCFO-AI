// ============================================================
// FINCO AI — CASH FLOW QUERY
// File: cash_flow_query.pq
//
// PURPOSE
//   Cash-flow transformation layer for Financial Dashboard.
//
// OUTPUT
//   CashFlow
//
// SOURCE
//   HistoricalFinancials
//
// EXPECTED SOURCE COLUMNS
//   Date
//   Company
//   BusinessUnit
//   Revenue
//   OperatingCashFlow
//   CapitalExpenditure
//   FreeCashFlow
//   CashAndEquivalents
//   CurrentLiabilities
//   TotalAssets
//   Debt
//
// ============================================================

let

    // ========================================================
    // 1. PARAMETERS
    // ========================================================

    StartDate =
        #date(2015, 1, 1),

    EndDate =
        #date(2030, 12, 31),


    // ========================================================
    // 2. SOURCE
    // ========================================================

    Source =
        try HistoricalFinancials
        otherwise
            #table(
                {
                    "Date",
                    "Company",
                    "BusinessUnit",
                    "Revenue",
                    "OperatingCashFlow",
                    "CapitalExpenditure",
                    "FreeCashFlow",
                    "CashAndEquivalents",
                    "CurrentLiabilities",
                    "TotalAssets",
                    "Debt"
                },
                {}
            ),


    // ========================================================
    // 3. NORMALIZE COLUMN NAMES
    // ========================================================

    NormalizeColumns =
        Table.TransformColumnNames(
            Source,
            each Text.Trim(_)
        ),


    // ========================================================
    // 4. REQUIRED COLUMNS
    // ========================================================

    RequiredColumns =
        {
            "Date",
            "Company",
            "BusinessUnit",
            "Revenue",
            "OperatingCashFlow",
            "CapitalExpenditure",
            "FreeCashFlow",
            "CashAndEquivalents",
            "CurrentLiabilities",
            "TotalAssets",
            "Debt"
        },


    AddMissingColumns =
        List.Accumulate(
            RequiredColumns,
            NormalizeColumns,
            (state, columnName) =>
                if List.Contains(
                    Table.ColumnNames(state),
                    columnName
                )
                then
                    state
                else
                    Table.AddColumn(
                        state,
                        columnName,
                        each null
                    )
        ),


    // ========================================================
    // 5. DATA TYPES
    // ========================================================

    Typed =
        Table.TransformColumnTypes(
            AddMissingColumns,
            {
                {"Date", type date},
                {"Company", type text},
                {"BusinessUnit", type text},

                {"Revenue", type number},
                {"OperatingCashFlow", type number},
                {"CapitalExpenditure", type number},
                {"FreeCashFlow", type number},
                {"CashAndEquivalents", type number},
                {"CurrentLiabilities", type number},
                {"TotalAssets", type number},
                {"Debt", type number}
            }
        ),


    // ========================================================
    // 6. CLEAN TEXT
    // ========================================================

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
                }
            }
        ),


    // ========================================================
    // 7. CLEAN NUMERIC DATA
    // ========================================================

    NumericColumns =
        {
            "Revenue",
            "OperatingCashFlow",
            "CapitalExpenditure",
            "FreeCashFlow",
            "CashAndEquivalents",
            "CurrentLiabilities",
            "TotalAssets",
            "Debt"
        },


    CleanNumbers =
        Table.TransformColumns(
            CleanText,
            List.Transform(
                NumericColumns,
                (columnName) =>
                    {
                        columnName,
                        each
                            try Number.From(_)
                            otherwise null,
                        type number
                    }
            )
        ),


    // ========================================================
    // 8. VALID DATE FILTER
    // ========================================================

    ValidDates =
        Table.SelectRows(
            CleanNumbers,
            each
                [Date] <> null
                and [Date] >= StartDate
                and [Date] <= EndDate
        ),


    // ========================================================
    // 9. DATE ATTRIBUTES
    // ========================================================

    AddYear =
        Table.AddColumn(
            ValidDates,
            "Year",
            each Date.Year([Date]),
            Int64.Type
        ),


    AddQuarterNumber =
        Table.AddColumn(
            AddYear,
            "QuarterNumber",
            each Date.QuarterOfYear([Date]),
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
            each Date.Month([Date]),
            Int64.Type
        ),


    AddMonth =
        Table.AddColumn(
            AddMonthNumber,
            "Month",
            each Date.MonthName([Date]),
            type text
        ),


    AddMonthShort =
        Table.AddColumn(
            AddMonth,
            "MonthShort",
            each Date.ToText([Date], "MMM"),
            type text
        ),


    AddMonthStart =
        Table.AddColumn(
            AddMonthShort,
            "MonthStart",
            each Date.StartOfMonth([Date]),
            type date
        ),


    AddYearMonth =
        Table.AddColumn(
            AddMonthStart,
            "YearMonth",
            each Date.ToText([Date], "yyyy-MM"),
            type text
        ),


    AddYearMonthKey =
        Table.AddColumn(
            AddYearMonth,
            "YearMonthKey",
            each
                [Year] * 100
                    + [MonthNumber],
            Int64.Type
        ),


    // ========================================================
    // 10. NORMALIZE CAPITAL EXPENDITURE
    //
    // If CapEx is supplied as a positive outflow, it is
    // converted into a negative cash-flow representation.
    // ========================================================

    AddNormalizedCapEx =
        Table.AddColumn(
            AddYearMonthKey,
            "CapitalExpenditureNormalized",
            each
                if [CapitalExpenditure] = null
                then null
                else
                    -Number.Abs(
                        [CapitalExpenditure]
                    ),
            type number
        ),


    // ========================================================
    // 11. FREE CASH FLOW
    //
    // FCF = Operating Cash Flow + negative CapEx
    // ========================================================

    AddCalculatedFCF =
        Table.AddColumn(
            AddNormalizedCapEx,
            "FreeCashFlowCalculated",
            each
                if [OperatingCashFlow] = null
                    or [CapitalExpenditureNormalized] = null
                then
                    if [FreeCashFlow] <> null
                    then [FreeCashFlow]
                    else null
                else
                    [OperatingCashFlow]
                    + [CapitalExpenditureNormalized],
            type number
        ),


    // ========================================================
    // 12. FINAL FREE CASH FLOW
    // ========================================================

    RemoveOldFCF =
        Table.RemoveColumns(
            AddCalculatedFCF,
            {"FreeCashFlow"}
        ),


    RenameFCF =
        Table.RenameColumns(
            RemoveOldFCF,
            {
                {
                    "FreeCashFlowCalculated",
                    "FreeCashFlow"
                }
            }
        ),


    // ========================================================
    // 13. OPERATING CASH FLOW MARGIN
    // ========================================================

    AddOCFMargin =
        Table.AddColumn(
            RenameFCF,
            "OperatingCashFlowMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [OperatingCashFlow]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 14. FREE CASH FLOW MARGIN
    // ========================================================

    AddFCFMargin =
        Table.AddColumn(
            AddOCFMargin,
            "FreeCashFlowMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [FreeCashFlow]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 15. CAPEX TO REVENUE
    // ========================================================

    AddCapExToRevenue =
        Table.AddColumn(
            AddFCFMargin,
            "CapExToRevenue",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    Number.Abs(
                        [CapitalExpenditure]
                    )
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 16. OCF TO CURRENT LIABILITIES
    // ========================================================

    AddOCFToCurrentLiabilities =
        Table.AddColumn(
            AddCapExToRevenue,
            "OperatingCashFlowToCurrentLiabilities",
            each
                if [CurrentLiabilities] = null
                    or [CurrentLiabilities] = 0
                then null
                else
                    [OperatingCashFlow]
                    / [CurrentLiabilities],
            type number
        ),


    // ========================================================
    // 17. CASH TO ASSETS
    // ========================================================

    AddCashToAssets =
        Table.AddColumn(
            AddOCFToCurrentLiabilities,
            "CashToAssetRatio",
            each
                if [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else
                    [CashAndEquivalents]
                    / [TotalAssets],
            type number
        ),


    // ========================================================
    // 18. CASH TO CURRENT LIABILITIES
    // ========================================================

    AddCashCoverage =
        Table.AddColumn(
            AddCashToAssets,
            "CashCoverageRatio",
            each
                if [CurrentLiabilities] = null
                    or [CurrentLiabilities] = 0
                then null
                else
                    [CashAndEquivalents]
                    / [CurrentLiabilities],
            type number
        ),


    // ========================================================
    // 19. FREE CASH FLOW STATUS
    // ========================================================

    AddFCFStatus =
        Table.AddColumn(
            AddCashCoverage,
            "FreeCashFlowStatus",
            each
                if [FreeCashFlow] = null
                then "Unknown"
                else if [FreeCashFlow] < 0
                then "Negative"
                else if [FreeCashFlowMargin] <> null
                    and [FreeCashFlowMargin] < 0.05
                then "Weak"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 20. OPERATING CASH FLOW STATUS
    // ========================================================

    AddOCFStatus =
        Table.AddColumn(
            AddFCFStatus,
            "OperatingCashFlowStatus",
            each
                if [OperatingCashFlow] = null
                then "Unknown"
                else if [OperatingCashFlow] < 0
                then "Negative"
                else if [OperatingCashFlowMargin] <> null
                    and [OperatingCashFlowMargin] < 0.05
                then "Weak"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 21. CAPEX STATUS
    // ========================================================

    AddCapExStatus =
        Table.AddColumn(
            AddOCFStatus,
            "CapitalExpenditureStatus",
            each
                if [CapitalExpenditure] = null
                then "Unknown"
                else if [CapitalExpenditure] < 0
                then "Review Sign"
                else
                    "Available",
            type text
        ),


    // ========================================================
    // 22. CASH FLOW RISK
    // ========================================================

    AddCashFlowRisk =
        Table.AddColumn(
            AddCapExStatus,
            "CashFlowRisk",
            each
                if [FreeCashFlow] = null
                then 20
                else if [FreeCashFlow] < 0
                then 40
                else if [FreeCashFlowMargin] <> null
                    and [FreeCashFlowMargin] < 0.05
                then 20
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 23. OPERATING CASH FLOW RISK
    // ========================================================

    AddOperatingCashFlowRisk =
        Table.AddColumn(
            AddCashFlowRisk,
            "OperatingCashFlowRisk",
            each
                if [OperatingCashFlow] = null
                then 20
                else if [OperatingCashFlow] < 0
                then 40
                else if [OperatingCashFlowMargin] <> null
                    and [OperatingCashFlowMargin] < 0.05
                then 15
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 24. LIQUIDITY RISK
    // ========================================================

    AddLiquidityRisk =
        Table.AddColumn(
            AddOperatingCashFlowRisk,
            "LiquidityRisk",
            each
                if [CashCoverageRatio] = null
                then 20
                else if [CashCoverageRatio] < 0.25
                then 40
                else if [CashCoverageRatio] < 0.50
                then 20
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 25. CAPEX RISK
    // ========================================================

    AddCapExRisk =
        Table.AddColumn(
            AddLiquidityRisk,
            "CapExRisk",
            each
                if [CapExToRevenue] = null
                then 10
                else if [CapExToRevenue] >= 0.30
                then 30
                else if [CapExToRevenue] >= 0.20
                then 20
                else if [CapExToRevenue] >= 0.10
                then 10
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 26. TOTAL CASH FLOW RISK
    // ========================================================

    AddCashFlowRiskScore =
        Table.AddColumn(
            AddCapExRisk,
            "CashFlowRiskScore",
            each
                Number.Min(
                    100,
                    [CashFlowRisk]
                    + [OperatingCashFlowRisk]
                    + [LiquidityRisk]
                    + [CapExRisk]
                ),
            type number
        ),


    // ========================================================
    // 27. RISK LEVEL
    // ========================================================

    AddCashFlowRiskLevel =
        Table.AddColumn(
            AddCashFlowRiskScore,
            "CashFlowRiskLevel",
            each
                if [CashFlowRiskScore] >= 75
                then "CRITICAL"
                else if [CashFlowRiskScore] >= 50
                then "HIGH"
                else if [CashFlowRiskScore] >= 25
                then "MEDIUM"
                else
                    "LOW",
            type text
        ),


    // ========================================================
    // 28. CASH FLOW HEALTH SCORE
    // ========================================================

    AddCashFlowHealthScore =
        Table.AddColumn(
            AddCashFlowRiskLevel,
            "CashFlowHealthScore",
            each
                Number.Max(
                    0,
                    100 - [CashFlowRiskScore]
                ),
            type number
        ),


    // ========================================================
    // 29. CASH FLOW HEALTH STATUS
    // ========================================================

    AddCashFlowHealthStatus =
        Table.AddColumn(
            AddCashFlowHealthScore,
            "CashFlowHealthStatus",
            each
                if [CashFlowHealthScore] >= 80
                then "Excellent"
                else if [CashFlowHealthScore] >= 65
                then "Healthy"
                else if [CashFlowHealthScore] >= 50
                then "Watch"
                else if [CashFlowHealthScore] >= 30
                then "Weak"
                else
                    "Critical",
            type text
        ),


    // ========================================================
    // 30. NEGATIVE CASH FLOW FLAG
    // ========================================================

    AddNegativeCashFlowFlag =
        Table.AddColumn(
            AddCashFlowHealthStatus,
            "NegativeCashFlowFlag",
            each
                [FreeCashFlow] <> null
                and [FreeCashFlow] < 0,
            type logical
        ),


    // ========================================================
    // 31. NEGATIVE OPERATING CASH FLOW FLAG
    // ========================================================

    AddNegativeOCFFlag =
        Table.AddColumn(
            AddNegativeCashFlowFlag,
            "NegativeOperatingCashFlowFlag",
            each
                [OperatingCashFlow] <> null
                and [OperatingCashFlow] < 0,
            type logical
        ),


    // ========================================================
    // 32. ALERT FLAG
    // ========================================================

    AddAlertFlag =
        Table.AddColumn(
            AddNegativeOCFFlag,
            "AlertFlag",
            each
                [CashFlowRiskLevel] = "CRITICAL"
                or [CashFlowRiskLevel] = "HIGH"
                or [NegativeCashFlowFlag]
                or [NegativeOperatingCashFlowFlag],
            type logical
        ),


    // ========================================================
    // 33. ALERT TYPE
    // ========================================================

    AddAlertType =
        Table.AddColumn(
            AddAlertFlag,
            "AlertType",
            each
                if [CashFlowRiskLevel] = "CRITICAL"
                then
                    "Critical Cash Flow Risk"
                else if [CashFlowRiskLevel] = "HIGH"
                then
                    "High Cash Flow Risk"
                else if [NegativeOperatingCashFlowFlag]
                then
                    "Negative Operating Cash Flow"
                else if [NegativeCashFlowFlag]
                then
                    "Negative Free Cash Flow"
                else if [CapExRisk] >= 30
                then
                    "High Capital Expenditure"
                else
                    null,
            type text
        ),


    // ========================================================
    // 34. RECOMMENDED ACTION
    // ========================================================

    AddRecommendedAction =
        Table.AddColumn(
            AddAlertType,
            "RecommendedAction",
            each
                if [CashFlowRiskLevel] = "CRITICAL"
                then
                    "Immediate liquidity review, cash preservation and management intervention"
                else if [CashFlowRiskLevel] = "HIGH"
                then
                    "Investigate cash-flow drivers and execute cash optimization plan"
                else if [NegativeOperatingCashFlowFlag]
                then
                    "Investigate operating cash burn and improve working-capital conversion"
                else if [NegativeCashFlowFlag]
                then
                    "Review capital allocation, CapEx and near-term liquidity requirements"
                else if [CapExRisk] >= 30
                then
                    "Review capital expenditure pipeline and investment return assumptions"
                else
                    "Continue normal cash-flow monitoring",
            type text
        ),


    // ========================================================
    // 35. DATA QUALITY
    // ========================================================

    AddDateQuality =
        Table.AddColumn(
            AddRecommendedAction,
            "DateQuality",
            each
                if [Date] = null
                then "Invalid"
                else "Valid",
            type text
        ),


    AddCashFlowDataQuality =
        Table.AddColumn(
            AddDateQuality,
            "CashFlowDataQuality",
            each
                if [OperatingCashFlow] = null
                    and [FreeCashFlow] = null
                then "Missing"
                else if [Revenue] = null
                then "Incomplete"
                else
                    "Available",
            type text
        ),


    AddCapExDataQuality =
        Table.AddColumn(
            AddCashFlowDataQuality,
            "CapExDataQuality",
            each
                if [CapitalExpenditure] = null
                then "Missing"
                else
                    "Available",
            type text
        ),


    AddDataQualityStatus =
        Table.AddColumn(
            AddCapExDataQuality,
            "DataQualityStatus",
            each
                if [DateQuality] = "Invalid"
                then "Invalid"
                else if [CashFlowDataQuality] = "Missing"
                then "Incomplete"
                else if [CapExDataQuality] = "Missing"
                then "Partial"
                else
                    "Valid",
            type text
        ),


    // ========================================================
    // 36. DECISION PRIORITY
    // ========================================================

    AddDecisionPriority =
        Table.AddColumn(
            AddDataQualityStatus,
            "DecisionPriority",
            each
                if [CashFlowRiskLevel] = "CRITICAL"
                then "P0"
                else if [CashFlowRiskLevel] = "HIGH"
                then "P1"
                else if [CashFlowRiskLevel] = "MEDIUM"
                then "P2"
                else
                    "P3",
            type text
        ),


    // ========================================================
    // 37. CASH FLOW RECORD ID
    // ========================================================

    AddRecordID =
        Table.AddColumn(
            AddDecisionPriority,
            "CashFlowRecordID",
            each
                Text.Combine(
                    {
                        Date.ToText(
                            [Date],
                            "yyyyMMdd"
                        ),
                        [Company],
                        [BusinessUnit]
                    },
                    "|"
                ),
            type text
        ),


    // ========================================================
    // 38. DISPLAY LABEL
    // ========================================================

    AddDisplayLabel =
        Table.AddColumn(
            AddRecordID,
            "CashFlowLabel",
            each
                [Company]
                & " | "
                & [BusinessUnit]
                & " | "
                & Date.ToText(
                    [Date],
                    "MMM yyyy"
                ),
            type text
        ),


    // ========================================================
    // 39. MANAGEMENT STATUS
    // ========================================================

    AddManagementStatus =
        Table.AddColumn(
            AddDisplayLabel,
            "ManagementStatus",
            each
                if [CashFlowRiskLevel] = "CRITICAL"
                then "Immediate Action"
                else if [CashFlowRiskLevel] = "HIGH"
                then "Management Review"
                else if [CashFlowRiskLevel] = "MEDIUM"
                then "Monitor"
                else
                    "Normal",
            type text
        ),


    // ========================================================
    // 40. SORT
    // ========================================================

    Sorted =
        Table.Sort(
            AddManagementStatus,
            {
                {"Date", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending}
            }
        ),


    // ========================================================
    // 41. REMOVE DUPLICATES
    // ========================================================

    Deduplicated =
        Table.Distinct(
            Sorted,
            {
                "CashFlowRecordID"
            }
        ),


    // ========================================================
    // 42. BUFFER
    // ========================================================

    Final =
        Table.Buffer(
            Deduplicated
        )

in
    Final