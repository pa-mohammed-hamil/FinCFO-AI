let
    // ============================================================
    // FinCo AI — Executive Dashboard
    // Consolidated Power Query
    //
    // Source:
    //     FinancialTransformations
    //
    // Output:
    //     ExecutiveDashboard
    //
    // Purpose:
    //     Executive-level financial performance, liquidity,
    //     cash-flow, leverage, risk, health, alerts and
    //     decision-support analytics.
    // ============================================================

    // ------------------------------------------------------------
    // 1. SOURCE
    // ------------------------------------------------------------

    Source = FinancialTransformations,

    // ------------------------------------------------------------
    // 2. REQUIRED COLUMNS
    // ------------------------------------------------------------

    RequiredColumns =
        {
            "Date",
            "Company",
            "BusinessUnit",
            "Revenue",
            "CostOfRevenue",
            "GrossProfit",
            "OperatingExpenses",
            "OperatingProfit",
            "InterestExpense",
            "TaxExpense",
            "NetProfit",
            "OperatingCashFlow",
            "CapitalExpenditure",
            "FreeCashFlow",
            "TotalAssets",
            "CurrentAssets",
            "CurrentLiabilities",
            "TotalLiabilities",
            "TotalEquity",
            "CashAndEquivalents",
            "AccountsReceivable",
            "Inventory",
            "AccountsPayable",
            "Debt",
            "BudgetRevenue",
            "BudgetExpenses",
            "FinancialRiskScore",
            "FinancialRiskLevel",
            "FinancialHealthScore",
            "FinancialHealthStatus",
            "AlertFlag",
            "AlertType",
            "RecommendedAction",
            "DataQualityStatus",
            "DecisionPriority"
        },

    ExistingColumns = Table.ColumnNames(Source),

    MissingColumns =
        List.Difference(
            RequiredColumns,
            ExistingColumns
        ),

    AddMissingColumns =
        List.Accumulate(
            MissingColumns,
            Source,
            (CurrentTable, ColumnName) =>
                Table.AddColumn(
                    CurrentTable,
                    ColumnName,
                    each null
                )
        ),

    // ------------------------------------------------------------
    // 3. SELECT
    // ------------------------------------------------------------

    Selected =
        Table.SelectColumns(
            AddMissingColumns,
            RequiredColumns,
            MissingField.UseNull
        ),

    // ------------------------------------------------------------
    // 4. DATA TYPES
    // ------------------------------------------------------------

    Typed =
        Table.TransformColumnTypes(
            Selected,
            {
                {"Date", type date},

                {"Company", type text},
                {"BusinessUnit", type text},

                {"Revenue", type number},
                {"CostOfRevenue", type number},
                {"GrossProfit", type number},
                {"OperatingExpenses", type number},
                {"OperatingProfit", type number},
                {"InterestExpense", type number},
                {"TaxExpense", type number},
                {"NetProfit", type number},

                {"OperatingCashFlow", type number},
                {"CapitalExpenditure", type number},
                {"FreeCashFlow", type number},

                {"TotalAssets", type number},
                {"CurrentAssets", type number},
                {"CurrentLiabilities", type number},
                {"TotalLiabilities", type number},
                {"TotalEquity", type number},

                {"CashAndEquivalents", type number},
                {"AccountsReceivable", type number},
                {"Inventory", type number},
                {"AccountsPayable", type number},
                {"Debt", type number},

                {"BudgetRevenue", type number},
                {"BudgetExpenses", type number},

                {"FinancialRiskScore", type number},
                {"FinancialRiskLevel", type text},
                {"FinancialHealthScore", type number},
                {"FinancialHealthStatus", type text},

                {"AlertFlag", type text},
                {"AlertType", type text},
                {"RecommendedAction", type text},
                {"DataQualityStatus", type text},
                {"DecisionPriority", type text}
            }
        ),

    // ------------------------------------------------------------
    // 5. TEXT CLEANING
    // ------------------------------------------------------------

    CleanText =
        Table.TransformColumns(
            Typed,
            {
                {
                    "Company",
                    each
                        if _ = null or Text.Trim(Text.From(_)) = ""
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                },

                {
                    "BusinessUnit",
                    each
                        if _ = null or Text.Trim(Text.From(_)) = ""
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                },

                {
                    "FinancialRiskLevel",
                    each
                        if _ = null
                        then "UNKNOWN"
                        else Text.Upper(Text.Trim(Text.From(_))),
                    type text
                },

                {
                    "FinancialHealthStatus",
                    each
                        if _ = null
                        then "UNKNOWN"
                        else Text.Trim(Text.From(_)),
                    type text
                },

                {
                    "AlertFlag",
                    each
                        if _ = null
                        then "No"
                        else Text.Trim(Text.From(_)),
                    type text
                },

                {
                    "AlertType",
                    each
                        if _ = null
                        then "None"
                        else Text.Trim(Text.From(_)),
                    type text
                },

                {
                    "RecommendedAction",
                    each
                        if _ = null
                        then "No action required"
                        else Text.Trim(Text.From(_)),
                    type text
                },

                {
                    "DataQualityStatus",
                    each
                        if _ = null
                        then "UNKNOWN"
                        else Text.Trim(Text.From(_)),
                    type text
                },

                {
                    "DecisionPriority",
                    each
                        if _ = null
                        then "P3 - Normal"
                        else Text.Trim(Text.From(_)),
                    type text
                }
            }
        ),

    // ------------------------------------------------------------
    // 6. DATE FILTER
    // ------------------------------------------------------------

    FilterValidDates =
        Table.SelectRows(
            CleanText,
            each
                [Date] <> null
                and [Date] >= #date(2015, 1, 1)
                and [Date] <= #date(2030, 12, 31)
        ),

    // ------------------------------------------------------------
    // 7. FINANCIAL FALLBACK CALCULATIONS
    // ------------------------------------------------------------

    AddGrossProfit =
        Table.AddColumn(
            FilterValidDates,
            "ExecutiveGrossProfit",
            each
                if [GrossProfit] <> null
                then [GrossProfit]
                else
                    if [Revenue] <> null
                       and [CostOfRevenue] <> null
                    then [Revenue] - [CostOfRevenue]
                    else null,
            type number
        ),

    AddOperatingProfit =
        Table.AddColumn(
            AddGrossProfit,
            "ExecutiveOperatingProfit",
            each
                if [OperatingProfit] <> null
                then [OperatingProfit]
                else
                    if [ExecutiveGrossProfit] <> null
                       and [OperatingExpenses] <> null
                    then
                        [ExecutiveGrossProfit]
                        - [OperatingExpenses]
                    else null,
            type number
        ),

    AddNetProfit =
        Table.AddColumn(
            AddOperatingProfit,
            "ExecutiveNetProfit",
            each
                if [NetProfit] <> null
                then [NetProfit]
                else
                    if [ExecutiveOperatingProfit] <> null
                       and [InterestExpense] <> null
                       and [TaxExpense] <> null
                    then
                        [ExecutiveOperatingProfit]
                        - [InterestExpense]
                        - [TaxExpense]
                    else null,
            type number
        ),

    AddFreeCashFlow =
        Table.AddColumn(
            AddNetProfit,
            "ExecutiveFreeCashFlow",
            each
                if [FreeCashFlow] <> null
                then [FreeCashFlow]
                else
                    if [OperatingCashFlow] <> null
                       and [CapitalExpenditure] <> null
                    then
                        [OperatingCashFlow]
                        + [CapitalExpenditure]
                    else null,
            type number
        ),

    // ------------------------------------------------------------
    // 8. DATE ATTRIBUTES
    // ------------------------------------------------------------

    AddYear =
        Table.AddColumn(
            AddFreeCashFlow,
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
            each "Q" & Number.ToText([QuarterNumber]),
            type text
        ),

    AddMonthNumber =
        Table.AddColumn(
            AddQuarter,
            "MonthNumber",
            each Date.Month([Date]),
            Int64.Type
        ),

    AddMonthName =
        Table.AddColumn(
            AddMonthNumber,
            "MonthName",
            each Date.MonthName([Date]),
            type text
        ),

    AddMonthShort =
        Table.AddColumn(
            AddMonthName,
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

    // ------------------------------------------------------------
    // 9. PROFITABILITY METRICS
    // ------------------------------------------------------------

    AddGrossMargin =
        Table.AddColumn(
            AddYearMonthKey,
            "ExecutiveGrossMargin",
            each
                if [Revenue] = null
                   or [Revenue] = 0
                then null
                else
                    [ExecutiveGrossProfit]
                    / [Revenue],
            type number
        ),

    AddOperatingMargin =
        Table.AddColumn(
            AddGrossMargin,
            "ExecutiveOperatingMargin",
            each
                if [Revenue] = null
                   or [Revenue] = 0
                then null
                else
                    [ExecutiveOperatingProfit]
                    / [Revenue],
            type number
        ),

    AddNetMargin =
        Table.AddColumn(
            AddOperatingMargin,
            "ExecutiveNetMargin",
            each
                if [Revenue] = null
                   or [Revenue] = 0
                then null
                else
                    [ExecutiveNetProfit]
                    / [Revenue],
            type number
        ),

    // ------------------------------------------------------------
    // 10. CASH FLOW METRICS
    // ------------------------------------------------------------

    AddOCFMargin =
        Table.AddColumn(
            AddNetMargin,
            "ExecutiveOCFMargin",
            each
                if [Revenue] = null
                   or [Revenue] = 0
                then null
                else
                    [OperatingCashFlow]
                    / [Revenue],
            type number
        ),

    AddFCFMargin =
        Table.AddColumn(
            AddOCFMargin,
            "ExecutiveFCFMargin",
            each
                if [Revenue] = null
                   or [Revenue] = 0
                then null
                else
                    [ExecutiveFreeCashFlow]
                    / [Revenue],
            type number
        ),

    AddCapexPct =
        Table.AddColumn(
            AddFCFMargin,
            "ExecutiveCapexPctRevenue",
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

    // ------------------------------------------------------------
    // 11. LIQUIDITY
    // ------------------------------------------------------------

    AddWorkingCapital =
        Table.AddColumn(
            AddCapexPct,
            "ExecutiveWorkingCapital",
            each
                if [CurrentAssets] = null
                   or [CurrentLiabilities] = null
                then null
                else
                    [CurrentAssets]
                    - [CurrentLiabilities],
            type number
        ),

    AddCurrentRatio =
        Table.AddColumn(
            AddWorkingCapital,
            "ExecutiveCurrentRatio",
            each
                if [CurrentLiabilities] = null
                   or [CurrentLiabilities] = 0
                then null
                else
                    [CurrentAssets]
                    / [CurrentLiabilities],
            type number
        ),

    AddQuickRatio =
        Table.AddColumn(
            AddCurrentRatio,
            "ExecutiveQuickRatio",
            each
                if [CurrentLiabilities] = null
                   or [CurrentLiabilities] = 0
                then null
                else
                    (
                        [CurrentAssets]
                        - [Inventory]
                    )
                    / [CurrentLiabilities],
            type number
        ),

    AddCashRatio =
        Table.AddColumn(
            AddQuickRatio,
            "ExecutiveCashRatio",
            each
                if [CurrentLiabilities] = null
                   or [CurrentLiabilities] = 0
                then null
                else
                    [CashAndEquivalents]
                    / [CurrentLiabilities],
            type number
        ),

    AddCashCoverage =
        Table.AddColumn(
            AddCashRatio,
            "ExecutiveCashCoverage",
            each
                if [CurrentLiabilities] = null
                   or [CurrentLiabilities] = 0
                then null
                else
                    (
                        [CashAndEquivalents]
                        + List.Max(
                            {
                                0,
                                if [OperatingCashFlow] = null
                                then 0
                                else [OperatingCashFlow]
                            }
                        )
                    )
                    / [CurrentLiabilities],
            type number
        ),

    // ------------------------------------------------------------
    // 12. LEVERAGE
    // ------------------------------------------------------------

    AddDebtToAssets =
        Table.AddColumn(
            AddCashCoverage,
            "ExecutiveDebtToAssets",
            each
                if [TotalAssets] = null
                   or [TotalAssets] = 0
                then null
                else
                    [Debt]
                    / [TotalAssets],
            type number
        ),

    AddDebtToEquity =
        Table.AddColumn(
            AddDebtToAssets,
            "ExecutiveDebtToEquity",
            each
                if [TotalEquity] = null
                   or [TotalEquity] = 0
                then null
                else
                    [Debt]
                    / [TotalEquity],
            type number
        ),

    AddLiabilityToAssets =
        Table.AddColumn(
            AddDebtToEquity,
            "ExecutiveLiabilityToAssets",
            each
                if [TotalAssets] = null
                   or [TotalAssets] = 0
                then null
                else
                    [TotalLiabilities]
                    / [TotalAssets],
            type number
        ),

    // ------------------------------------------------------------
    // 13. RETURNS
    // ------------------------------------------------------------

    AddROA =
        Table.AddColumn(
            AddLiabilityToAssets,
            "ExecutiveROA",
            each
                if [TotalAssets] = null
                   or [TotalAssets] = 0
                then null
                else
                    [ExecutiveNetProfit]
                    / [TotalAssets],
            type number
        ),

    AddROE =
        Table.AddColumn(
            AddROA,
            "ExecutiveROE",
            each
                if [TotalEquity] = null
                   or [TotalEquity] = 0
                then null
                else
                    [ExecutiveNetProfit]
                    / [TotalEquity],
            type number
        ),

    // ------------------------------------------------------------
    // 14. BALANCE SHEET INTEGRITY
    // ------------------------------------------------------------

    AddBalanceDifference =
        Table.AddColumn(
            AddROE,
            "ExecutiveBalanceDifference",
            each
                if [TotalAssets] = null
                   or [TotalLiabilities] = null
                   or [TotalEquity] = null
                then null
                else
                    [TotalAssets]
                    - [TotalLiabilities]
                    - [TotalEquity],
            type number
        ),

    AddBalanceCheck =
        Table.AddColumn(
            AddBalanceDifference,
            "ExecutiveBalanceCheck",
            each
                if [ExecutiveBalanceDifference] = null
                then "Unknown"
                else if
                    Number.Abs(
                        [ExecutiveBalanceDifference]
                    ) <= 0.01
                then "Balanced"
                else "Out of Balance",
            type text
        ),

    // ------------------------------------------------------------
    // 15. EXECUTIVE RISK COMPONENTS
    // ------------------------------------------------------------

    AddProfitRisk =
        Table.AddColumn(
            AddBalanceCheck,
            "ExecutiveProfitRisk",
            each
                if [ExecutiveNetProfit] = null then 25
                else if [ExecutiveNetProfit] < 0 then 40
                else if
                    [ExecutiveNetMargin] <> null
                    and [ExecutiveNetMargin] < 0.05
                then 20
                else 0,
            Int64.Type
        ),

    AddLiquidityRisk =
        Table.AddColumn(
            AddProfitRisk,
            "ExecutiveLiquidityRisk",
            each
                if [ExecutiveCurrentRatio] = null then 25
                else if [ExecutiveCurrentRatio] < 0.75 then 40
                else if [ExecutiveCurrentRatio] < 1 then 30
                else if [ExecutiveCurrentRatio] < 1.5 then 20
                else if [ExecutiveCurrentRatio] < 2 then 10
                else 0,
            Int64.Type
        ),

    AddCashFlowRisk =
        Table.AddColumn(
            AddLiquidityRisk,
            "ExecutiveCashFlowRisk",
            each
                if [OperatingCashFlow] = null then 25
                else if [OperatingCashFlow] < 0 then 40
                else if
                    [ExecutiveOCFMargin] <> null
                    and [ExecutiveOCFMargin] < 0.05
                then 30
                else if
                    [ExecutiveFCFMargin] <> null
                    and [ExecutiveFCFMargin] < 0
                then 30
                else 0,
            Int64.Type
        ),

    AddLeverageRisk =
        Table.AddColumn(
            AddCashFlowRisk,
            "ExecutiveLeverageRisk",
            each
                if [ExecutiveDebtToAssets] = null then 20
                else if [ExecutiveDebtToAssets] >= 0.75 then 40
                else if [ExecutiveDebtToAssets] >= 0.60 then 30
                else if [ExecutiveDebtToAssets] >= 0.45 then 20
                else if [ExecutiveDebtToAssets] >= 0.30 then 10
                else 0,
            Int64.Type
        ),

    AddIntegrityRisk =
        Table.AddColumn(
            AddLeverageRisk,
            "ExecutiveIntegrityRisk",
            each
                if [ExecutiveBalanceCheck] = "Out of Balance"
                then 40
                else if [ExecutiveBalanceCheck] = "Unknown"
                then 15
                else 0,
            Int64.Type
        ),

    // ------------------------------------------------------------
    // 16. EXECUTIVE RISK SCORE
    // ------------------------------------------------------------

    AddCalculatedRiskScore =
        Table.AddColumn(
            AddIntegrityRisk,
            "ExecutiveCalculatedRiskScore",
            each
                List.Min(
                    {
                        100,
                        [ExecutiveProfitRisk]
                        + [ExecutiveLiquidityRisk]
                        + [ExecutiveCashFlowRisk]
                        + [ExecutiveLeverageRisk]
                        + [ExecutiveIntegrityRisk]
                    }
                ),
            Int64.Type
        ),

    AddCalculatedRiskLevel =
        Table.AddColumn(
            AddCalculatedRiskScore,
            "ExecutiveCalculatedRiskLevel",
            each
                if [ExecutiveCalculatedRiskScore] >= 75
                then "CRITICAL"
                else if [ExecutiveCalculatedRiskScore] >= 50
                then "HIGH"
                else if [ExecutiveCalculatedRiskScore] >= 25
                then "MEDIUM"
                else "LOW",
            type text
        ),

    AddCalculatedHealthScore =
        Table.AddColumn(
            AddCalculatedRiskLevel,
            "ExecutiveCalculatedHealthScore",
            each
                List.Max(
                    {
                        0,
                        100
                        - [ExecutiveCalculatedRiskScore]
                    }
                ),
            type number
        ),

    AddCalculatedHealthStatus =
        Table.AddColumn(
            AddCalculatedHealthScore,
            "ExecutiveCalculatedHealthStatus",
            each
                if [ExecutiveCalculatedHealthScore] >= 80
                then "Excellent"
                else if [ExecutiveCalculatedHealthScore] >= 65
                then "Healthy"
                else if [ExecutiveCalculatedHealthScore] >= 50
                then "Watch"
                else if [ExecutiveCalculatedHealthScore] >= 30
                then "Weak"
                else "Critical",
            type text
        ),

    // ------------------------------------------------------------
    // 17. EXECUTIVE STATUS
    // ------------------------------------------------------------

    AddExecutiveStatus =
        Table.AddColumn(
            AddCalculatedHealthStatus,
            "ExecutiveStatus",
            each
                if [ExecutiveCalculatedRiskLevel] = "CRITICAL"
                then "Critical"
                else if [ExecutiveCalculatedRiskLevel] = "HIGH"
                then "High Risk"
                else if [ExecutiveCalculatedRiskLevel] = "MEDIUM"
                then "Watch"
                else "Healthy",
            type text
        ),

    // ------------------------------------------------------------
    // 18. ALERTS
    // ------------------------------------------------------------

    AddExecutiveAlertFlag =
        Table.AddColumn(
            AddExecutiveStatus,
            "ExecutiveAlertFlag",
            each
                if
                    [ExecutiveCalculatedRiskScore] >= 50
                    or [ExecutiveNetProfit] < 0
                    or [ExecutiveCurrentRatio] < 1
                    or [OperatingCashFlow] < 0
                    or [ExecutiveBalanceCheck] = "Out of Balance"
                then "Yes"
                else "No",
            type text
        ),

    AddExecutiveAlertType =
        Table.AddColumn(
            AddExecutiveAlertFlag,
            "ExecutiveAlertType",
            each
                if [ExecutiveBalanceCheck] = "Out of Balance"
                then "Balance Sheet Integrity"
                else if [ExecutiveNetProfit] < 0
                then "Profitability"
                else if [OperatingCashFlow] < 0
                then "Negative Operating Cash Flow"
                else if [ExecutiveCurrentRatio] < 1
                then "Liquidity Stress"
                else if [ExecutiveCalculatedRiskScore] >= 75
                then "Critical Financial Risk"
                else if [ExecutiveCalculatedRiskScore] >= 50
                then "High Financial Risk"
                else if [ExecutiveCalculatedRiskScore] >= 25
                then "Medium Financial Risk"
                else "None",
            type text
        ),

    AddRecommendedAction =
        Table.AddColumn(
            AddExecutiveAlertType,
            "ExecutiveRecommendedAction",
            each
                if [ExecutiveBalanceCheck] = "Out of Balance"
                then "Investigate balance sheet integrity immediately"
                else if [ExecutiveNetProfit] < 0
                then "Review profitability and cost structure"
                else if [OperatingCashFlow] < 0
                then "Review operating cash generation and working capital"
                else if [ExecutiveCurrentRatio] < 1
                then "Strengthen short-term liquidity"
                else if [ExecutiveDebtToAssets] >= 0.75
                then "Review leverage and debt repayment capacity"
                else if [ExecutiveCalculatedRiskScore] >= 50
                then "Escalate financial risk for management review"
                else "Continue monitoring financial performance",
            type text
        ),

    // ------------------------------------------------------------
    // 19. DECISION PRIORITY
    // ------------------------------------------------------------

    AddDecisionPriority =
        Table.AddColumn(
            AddRecommendedAction,
            "ExecutiveDecisionPriority",
            each
                if [ExecutiveCalculatedRiskScore] >= 75
                then "P1 - Immediate"
                else if [ExecutiveCalculatedRiskScore] >= 50
                then "P2 - High"
                else if [ExecutiveCalculatedRiskScore] >= 25
                then "P3 - Monitor"
                else "P4 - Normal",
            type text
        ),

    // ------------------------------------------------------------
    // 20. MANAGEMENT STATUS
    // ------------------------------------------------------------

    AddManagementStatus =
        Table.AddColumn(
            AddDecisionPriority,
            "ExecutiveManagementStatus",
            each
                if [ExecutiveCalculatedRiskScore] >= 75
                then "Executive Intervention Required"
                else if [ExecutiveCalculatedRiskScore] >= 50
                then "Management Attention Required"
                else if [ExecutiveCalculatedRiskScore] >= 25
                then "Management Monitoring"
                else "Business as Usual",
            type text
        ),

    // ------------------------------------------------------------
    // 21. PERFORMANCE CATEGORY
    // ------------------------------------------------------------

    AddPerformanceCategory =
        Table.AddColumn(
            AddManagementStatus,
            "ExecutivePerformanceCategory",
            each
                if [ExecutiveNetMargin] = null
                then "Unknown"
                else if [ExecutiveNetMargin] >= 0.15
                then "Excellent"
                else if [ExecutiveNetMargin] >= 0.10
                then "Strong"
                else if [ExecutiveNetMargin] >= 0.05
                then "Stable"
                else if [ExecutiveNetMargin] >= 0
                then "Weak"
                else "Loss Making",
            type text
        ),

    // ------------------------------------------------------------
    // 22. DATA QUALITY
    // ------------------------------------------------------------

    AddExecutiveDataQuality =
        Table.AddColumn(
            AddPerformanceCategory,
            "ExecutiveDataQuality",
            each
                if
                    [Date] = null
                    or [Company] = null
                    or [Revenue] = null
                then "Incomplete"
                else if
                    [ExecutiveBalanceCheck] = "Unknown"
                then "Partial"
                else "Complete",
            type text
        ),

    // ------------------------------------------------------------
    // 23. EXECUTIVE RECORD ID
    // ------------------------------------------------------------

    AddRecordID =
        Table.AddColumn(
            AddExecutiveDataQuality,
            "ExecutiveRecordID",
            each
                Date.ToText(
                    [Date],
                    "yyyy-MM-dd"
                )
                & "|"
                & [Company]
                & "|"
                & [BusinessUnit],
            type text
        ),

    // ------------------------------------------------------------
    // 24. EXECUTIVE LABEL
    // ------------------------------------------------------------

    AddExecutiveLabel =
        Table.AddColumn(
            AddRecordID,
            "ExecutiveLabel",
            each
                [Company]
                & " | "
                & [BusinessUnit]
                & " | "
                & Date.ToText(
                    [Date],
                    "yyyy-MM-dd"
                ),
            type text
        ),

    // ------------------------------------------------------------
    // 25. FINAL RISK SCORE
    // ------------------------------------------------------------

    AddFinalRiskScore =
        Table.AddColumn(
            AddExecutiveLabel,
            "ExecutiveRiskScore",
            each
                if [FinancialRiskScore] <> null
                then [FinancialRiskScore]
                else [ExecutiveCalculatedRiskScore],
            type number
        ),

    AddFinalHealthScore =
        Table.AddColumn(
            AddFinalRiskScore,
            "ExecutiveHealthScore",
            each
                if [FinancialHealthScore] <> null
                then [FinancialHealthScore]
                else [ExecutiveCalculatedHealthScore],
            type number
        ),

    AddFinalRiskLevel =
        Table.AddColumn(
            AddFinalHealthScore,
            "ExecutiveRiskLevel",
            each
                if [FinancialRiskLevel] <> null
                   and [FinancialRiskLevel] <> "UNKNOWN"
                then [FinancialRiskLevel]
                else [ExecutiveCalculatedRiskLevel],
            type text
        ),

    // ------------------------------------------------------------
    // 26. DEDUPLICATION
    // ------------------------------------------------------------

    Deduplicated =
        Table.Distinct(
            AddFinalRiskLevel,
            {
                "ExecutiveRecordID"
            }
        ),

    // ------------------------------------------------------------
    // 27. SORT
    // ------------------------------------------------------------

    Sorted =
        Table.Sort(
            Deduplicated,
            {
                {"Date", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending}
            }
        ),

    // ------------------------------------------------------------
    // 28. BUFFER
    // ------------------------------------------------------------

    Final =
        Table.Buffer(Sorted)

in
    Final