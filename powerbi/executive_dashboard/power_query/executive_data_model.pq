let
    // ============================================================
    // FinCo AI
    // executive_data_model.pq
    //
    // Purpose:
    //   Consolidated executive-level analytical model.
    //
    // Inputs:
    //   IncomeStatement
    //   BalanceSheet
    //   CashFlow
    //   ForecastResults
    //   DimDate
    //   DimCompany
    //   DimBusinessUnit
    //
    // Output:
    //   ExecutiveDataModel
    // ============================================================


    // ============================================================
    // 1. PARAMETERS
    // ============================================================

    StartDate = #date(2015, 1, 1),
    EndDate   = #date(2030, 12, 31),


    // ============================================================
    // 2. SOURCE TABLES
    // ============================================================

    IncomeSource = IncomeStatement,
    BalanceSource = BalanceSheet,
    CashFlowSource = CashFlow,
    ForecastSource = ForecastResults,


    // ============================================================
    // 3. INCOME STATEMENT PREPARATION
    // ============================================================

    IncomeColumns =
        Table.SelectColumns(
            IncomeSource,
            {
                "Date",
                "Company",
                "BusinessUnit",
                "Revenue",
                "GrossProfit",
                "OperatingExpenses",
                "OperatingProfit",
                "InterestExpense",
                "TaxExpense",
                "NetProfit",
                "BudgetRevenue",
                "BudgetExpenses",
                "GrossMargin",
                "OperatingMargin",
                "NetMargin",
                "RevenueVariance",
                "RevenueVariancePct",
                "ExpenseVariance",
                "ExpenseVariancePct",
                "FinancialRiskScore",
                "FinancialRiskLevel",
                "FinancialHealthScore",
                "FinancialHealthStatus",
                "AlertFlag",
                "AlertType",
                "RecommendedAction",
                "DecisionPriority",
                "DataQualityStatus"
            },
            MissingField.UseNull
        ),

    IncomeTyped =
        Table.TransformColumnTypes(
            IncomeColumns,
            {
                {"Date", type date},
                {"Company", type text},
                {"BusinessUnit", type text},
                {"Revenue", type number},
                {"GrossProfit", type number},
                {"OperatingExpenses", type number},
                {"OperatingProfit", type number},
                {"InterestExpense", type number},
                {"TaxExpense", type number},
                {"NetProfit", type number},
                {"BudgetRevenue", type number},
                {"BudgetExpenses", type number},
                {"GrossMargin", type number},
                {"OperatingMargin", type number},
                {"NetMargin", type number},
                {"RevenueVariance", type number},
                {"RevenueVariancePct", type number},
                {"ExpenseVariance", type number},
                {"ExpenseVariancePct", type number},
                {"FinancialRiskScore", type number},
                {"FinancialRiskLevel", type text},
                {"FinancialHealthScore", type number},
                {"FinancialHealthStatus", type text},
                {"AlertFlag", type text},
                {"AlertType", type text},
                {"RecommendedAction", type text},
                {"DecisionPriority", type text},
                {"DataQualityStatus", type text}
            }
        ),

    IncomeFiltered =
        Table.SelectRows(
            IncomeTyped,
            each
                [Date] <> null
                and [Date] >= StartDate
                and [Date] <= EndDate
        ),


    // ============================================================
    // 4. BALANCE SHEET PREPARATION
    // ============================================================

    BalanceColumns =
        Table.SelectColumns(
            BalanceSource,
            {
                "Date",
                "Company",
                "BusinessUnit",
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
                "WorkingCapital",
                "CurrentRatio",
                "QuickRatio",
                "CashRatio",
                "DebtToAssetRatio",
                "DebtToEquity",
                "LiabilityToAssetRatio",
                "EquityToAssetRatio",
                "BalanceSheetDifference",
                "BalanceSheetCheck",
                "LiquidityRisk",
                "LeverageRisk",
                "WorkingCapitalRisk",
                "BalanceIntegrityRisk",
                "BalanceSheetRiskScore",
                "BalanceSheetRiskLevel",
                "BalanceSheetHealthScore",
                "BalanceSheetHealthStatus",
                "AlertFlag",
                "AlertType",
                "RecommendedAction",
                "DecisionPriority",
                "DataQualityStatus"
            },
            MissingField.UseNull
        ),

    BalanceTyped =
        Table.TransformColumnTypes(
            BalanceColumns,
            {
                {"Date", type date},
                {"Company", type text},
                {"BusinessUnit", type text},
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
                {"WorkingCapital", type number},
                {"CurrentRatio", type number},
                {"QuickRatio", type number},
                {"CashRatio", type number},
                {"DebtToAssetRatio", type number},
                {"DebtToEquity", type number},
                {"LiabilityToAssetRatio", type number},
                {"EquityToAssetRatio", type number},
                {"BalanceSheetDifference", type number},
                {"BalanceSheetCheck", type text},
                {"LiquidityRisk", type number},
                {"LeverageRisk", type number},
                {"WorkingCapitalRisk", type number},
                {"BalanceIntegrityRisk", type number},
                {"BalanceSheetRiskScore", type number},
                {"BalanceSheetRiskLevel", type text},
                {"BalanceSheetHealthScore", type number},
                {"BalanceSheetHealthStatus", type text},
                {"AlertFlag", type text},
                {"AlertType", type text},
                {"RecommendedAction", type text},
                {"DecisionPriority", type text},
                {"DataQualityStatus", type text}
            }
        ),

    BalanceFiltered =
        Table.SelectRows(
            BalanceTyped,
            each
                [Date] <> null
                and [Date] >= StartDate
                and [Date] <= EndDate
        ),


    // ============================================================
    // 5. CASH FLOW PREPARATION
    // ============================================================

    CashFlowColumns =
        Table.SelectColumns(
            CashFlowSource,
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
                "Debt",
                "OperatingCashFlowMargin",
                "FreeCashFlowMargin",
                "CapitalExpenditurePctRevenue",
                "CashCoverageRatio",
                "CashFlowRiskScore",
                "CashFlowRiskLevel",
                "CashFlowHealthScore",
                "CashFlowHealthStatus",
                "AlertFlag",
                "AlertType",
                "RecommendedAction",
                "DecisionPriority",
                "DataQualityStatus"
            },
            MissingField.UseNull
        ),

    CashFlowTyped =
        Table.TransformColumnTypes(
            CashFlowColumns,
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
                {"Debt", type number},
                {"OperatingCashFlowMargin", type number},
                {"FreeCashFlowMargin", type number},
                {"CapitalExpenditurePctRevenue", type number},
                {"CashCoverageRatio", type number},
                {"CashFlowRiskScore", type number},
                {"CashFlowRiskLevel", type text},
                {"CashFlowHealthScore", type number},
                {"CashFlowHealthStatus", type text},
                {"AlertFlag", type text},
                {"AlertType", type text},
                {"RecommendedAction", type text},
                {"DecisionPriority", type text},
                {"DataQualityStatus", type text}
            }
        ),

    CashFlowFiltered =
        Table.SelectRows(
            CashFlowTyped,
            each
                [Date] <> null
                and [Date] >= StartDate
                and [Date] <= EndDate
        ),


    // ============================================================
    // 6. FORECAST PREPARATION
    // ============================================================

    ForecastColumns =
        Table.SelectColumns(
            ForecastSource,
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
                "ModelVersion",
                "ForecastRiskScore",
                "ForecastRiskLevel",
                "ForecastStatus",
                "AlertFlag",
                "AlertType",
                "RecommendedAction"
            },
            MissingField.UseNull
        ),

    ForecastTyped =
        Table.TransformColumnTypes(
            ForecastColumns,
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
                {"ModelVersion", type text},
                {"ForecastRiskScore", type number},
                {"ForecastRiskLevel", type text},
                {"ForecastStatus", type text},
                {"AlertFlag", type text},
                {"AlertType", type text},
                {"RecommendedAction", type text}
            }
        ),

    ForecastFiltered =
        Table.SelectRows(
            ForecastTyped,
            each
                [ForecastDate] <> null
                and [ForecastDate] >= StartDate
                and [ForecastDate] <= EndDate
        ),


    // ============================================================
    // 7. CREATE COMMON BUSINESS KEY
    // ============================================================

    IncomeKey =
        Table.AddColumn(
            IncomeFiltered,
            "ExecutiveBusinessKey",
            each
                Date.ToText([Date], "yyyy-MM-dd")
                & "|"
                & [Company]
                & "|"
                & [BusinessUnit],
            type text
        ),

    BalanceKey =
        Table.AddColumn(
            BalanceFiltered,
            "ExecutiveBusinessKey",
            each
                Date.ToText([Date], "yyyy-MM-dd")
                & "|"
                & [Company]
                & "|"
                & [BusinessUnit],
            type text
        ),

    CashFlowKey =
        Table.AddColumn(
            CashFlowFiltered,
            "ExecutiveBusinessKey",
            each
                Date.ToText([Date], "yyyy-MM-dd")
                & "|"
                & [Company]
                & "|"
                & [BusinessUnit],
            type text
        ),


    // ============================================================
    // 8. MERGE INCOME + BALANCE SHEET
    // ============================================================

    MergeIncomeBalance =
        Table.NestedJoin(
            IncomeKey,
            {
                "ExecutiveBusinessKey"
            },
            BalanceKey,
            {
                "ExecutiveBusinessKey"
            },
            "BalanceData",
            JoinKind.LeftOuter
        ),

    ExpandBalance =
        Table.ExpandTableColumn(
            MergeIncomeBalance,
            "BalanceData",
            {
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
                "WorkingCapital",
                "CurrentRatio",
                "QuickRatio",
                "CashRatio",
                "DebtToAssetRatio",
                "DebtToEquity",
                "LiquidityRisk",
                "LeverageRisk",
                "BalanceSheetRiskScore",
                "BalanceSheetRiskLevel",
                "BalanceSheetHealthScore",
                "BalanceSheetHealthStatus",
                "BalanceSheetCheck"
            },
            {
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
                "WorkingCapital",
                "CurrentRatio",
                "QuickRatio",
                "CashRatio",
                "DebtToAssetRatio",
                "DebtToEquity",
                "LiquidityRisk",
                "LeverageRisk",
                "BalanceSheetRiskScore",
                "BalanceSheetRiskLevel",
                "BalanceSheetHealthScore",
                "BalanceSheetHealthStatus",
                "BalanceSheetCheck"
            }
        ),


    // ============================================================
    // 9. MERGE CASH FLOW
    // ============================================================

    MergeCashFlow =
        Table.NestedJoin(
            ExpandBalance,
            {
                "ExecutiveBusinessKey"
            },
            CashFlowKey,
            {
                "ExecutiveBusinessKey"
            },
            "CashFlowData",
            JoinKind.LeftOuter
        ),

    ExpandCashFlow =
        Table.ExpandTableColumn(
            MergeCashFlow,
            "CashFlowData",
            {
                "OperatingCashFlow",
                "CapitalExpenditure",
                "FreeCashFlow",
                "OperatingCashFlowMargin",
                "FreeCashFlowMargin",
                "CapitalExpenditurePctRevenue",
                "CashCoverageRatio",
                "CashFlowRiskScore",
                "CashFlowRiskLevel",
                "CashFlowHealthScore",
                "CashFlowHealthStatus"
            },
            {
                "OperatingCashFlow",
                "CapitalExpenditure",
                "FreeCashFlow",
                "OperatingCashFlowMargin",
                "FreeCashFlowMargin",
                "CapitalExpenditurePctRevenue",
                "CashCoverageRatio",
                "CashFlowRiskScore",
                "CashFlowRiskLevel",
                "CashFlowHealthScore",
                "CashFlowHealthStatus"
            }
        ),


    // ============================================================
    // 10. CALCULATE EXECUTIVE METRICS
    // ============================================================

    AddROA =
        Table.AddColumn(
            ExpandCashFlow,
            "ReturnOnAssets",
            each
                if [TotalAssets] = null
                   or [TotalAssets] = 0
                then null
                else [NetProfit] / [TotalAssets],
            type number
        ),

    AddROE =
        Table.AddColumn(
            AddROA,
            "ReturnOnEquity",
            each
                if [TotalEquity] = null
                   or [TotalEquity] = 0
                then null
                else [NetProfit] / [TotalEquity],
            type number
        ),

    AddDebtToCapital =
        Table.AddColumn(
            AddROE,
            "DebtToCapital",
            each
                if
                    [Debt] = null
                    or [TotalEquity] = null
                    or [Debt] + [TotalEquity] = 0
                then null
                else
                    [Debt]
                    / ([Debt] + [TotalEquity]),
            type number
        ),

    AddCashToAssets =
        Table.AddColumn(
            AddDebtToCapital,
            "CashToAssets",
            each
                if [TotalAssets] = null
                   or [TotalAssets] = 0
                then null
                else
                    [CashAndEquivalents]
                    / [TotalAssets],
            type number
        ),


    // ============================================================
    // 11. EXECUTIVE RISK ENGINE
    // ============================================================

    AddProfitRisk =
        Table.AddColumn(
            AddCashToAssets,
            "ExecutiveProfitRisk",
            each
                if [NetProfit] = null then 25
                else if [NetProfit] < 0 then 40
                else if [NetMargin] <> null and [NetMargin] < 0.05 then 20
                else 0,
            Int64.Type
        ),

    AddLiquidityRisk =
        Table.AddColumn(
            AddProfitRisk,
            "ExecutiveLiquidityRisk",
            each
                if [CurrentRatio] = null then 25
                else if [CurrentRatio] < 0.75 then 40
                else if [CurrentRatio] < 1 then 30
                else if [CurrentRatio] < 1.5 then 20
                else if [CurrentRatio] < 2 then 10
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
                else if [OperatingCashFlowMargin] <> null
                        and [OperatingCashFlowMargin] < 0.05
                then 30
                else if [FreeCashFlow] <> null
                        and [FreeCashFlow] < 0
                then 30
                else 0,
            Int64.Type
        ),

    AddLeverageRisk =
        Table.AddColumn(
            AddCashFlowRisk,
            "ExecutiveLeverageRisk",
            each
                if [DebtToAssetRatio] = null then 20
                else if [DebtToAssetRatio] >= 0.75 then 40
                else if [DebtToAssetRatio] >= 0.60 then 30
                else if [DebtToAssetRatio] >= 0.45 then 20
                else if [DebtToAssetRatio] >= 0.30 then 10
                else 0,
            Int64.Type
        ),

    AddIntegrityRisk =
        Table.AddColumn(
            AddLeverageRisk,
            "ExecutiveIntegrityRisk",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then 40
                else if [BalanceSheetCheck] = null
                then 15
                else 0,
            Int64.Type
        ),

    AddExecutiveRiskScore =
        Table.AddColumn(
            AddIntegrityRisk,
            "ExecutiveRiskScore",
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

    AddExecutiveRiskLevel =
        Table.AddColumn(
            AddExecutiveRiskScore,
            "ExecutiveRiskLevel",
            each
                if [ExecutiveRiskScore] >= 75 then "CRITICAL"
                else if [ExecutiveRiskScore] >= 50 then "HIGH"
                else if [ExecutiveRiskScore] >= 25 then "MEDIUM"
                else "LOW",
            type text
        ),

    AddExecutiveHealthScore =
        Table.AddColumn(
            AddExecutiveRiskLevel,
            "ExecutiveHealthScore",
            each
                List.Max(
                    {
                        0,
                        100 - [ExecutiveRiskScore]
                    }
                ),
            type number
        ),

    AddExecutiveHealthStatus =
        Table.AddColumn(
            AddExecutiveHealthScore,
            "ExecutiveHealthStatus",
            each
                if [ExecutiveHealthScore] >= 80 then "Excellent"
                else if [ExecutiveHealthScore] >= 65 then "Healthy"
                else if [ExecutiveHealthScore] >= 50 then "Watch"
                else if [ExecutiveHealthScore] >= 30 then "Weak"
                else "Critical",
            type text
        ),


    // ============================================================
    // 12. EXECUTIVE ALERT ENGINE
    // ============================================================

    AddAlertFlag =
        Table.AddColumn(
            AddExecutiveHealthStatus,
            "ExecutiveAlertFlag",
            each
                if
                    [ExecutiveRiskScore] >= 50
                    or [NetProfit] < 0
                    or [OperatingCashFlow] < 0
                    or [CurrentRatio] < 1
                    or [BalanceSheetCheck] = "Out of Balance"
                then "Yes"
                else "No",
            type text
        ),

    AddAlertType =
        Table.AddColumn(
            AddAlertFlag,
            "ExecutiveAlertType",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then "Balance Sheet Integrity"
                else if [NetProfit] < 0
                then "Profitability"
                else if [OperatingCashFlow] < 0
                then "Negative Cash Flow"
                else if [CurrentRatio] < 1
                then "Liquidity Stress"
                else if [ExecutiveRiskScore] >= 75
                then "Critical Financial Risk"
                else if [ExecutiveRiskScore] >= 50
                then "High Financial Risk"
                else if [ExecutiveRiskScore] >= 25
                then "Medium Financial Risk"
                else "None",
            type text
        ),

    AddRecommendedAction =
        Table.AddColumn(
            AddAlertType,
            "ExecutiveRecommendedAction",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then "Investigate balance sheet integrity"
                else if [NetProfit] < 0
                then "Review profitability and cost structure"
                else if [OperatingCashFlow] < 0
                then "Review operating cash generation"
                else if [CurrentRatio] < 1
                then "Strengthen short-term liquidity"
                else if [DebtToAssetRatio] >= 0.75
                then "Review leverage and debt capacity"
                else if [ExecutiveRiskScore] >= 50
                then "Escalate for management review"
                else "Continue normal monitoring",
            type text
        ),


    // ============================================================
    // 13. DECISION PRIORITY
    // ============================================================

    AddDecisionPriority =
        Table.AddColumn(
            AddRecommendedAction,
            "ExecutiveDecisionPriority",
            each
                if [ExecutiveRiskScore] >= 75
                then "P1 - Immediate"
                else if [ExecutiveRiskScore] >= 50
                then "P2 - High"
                else if [ExecutiveRiskScore] >= 25
                then "P3 - Monitor"
                else "P4 - Normal",
            type text
        ),

    AddManagementStatus =
        Table.AddColumn(
            AddDecisionPriority,
            "ExecutiveManagementStatus",
            each
                if [ExecutiveRiskScore] >= 75
                then "Executive Intervention Required"
                else if [ExecutiveRiskScore] >= 50
                then "Management Attention Required"
                else if [ExecutiveRiskScore] >= 25
                then "Management Monitoring"
                else "Business as Usual",
            type text
        ),


    // ============================================================
    // 14. EXECUTIVE PERFORMANCE CATEGORY
    // ============================================================

    AddPerformanceCategory =
        Table.AddColumn(
            AddManagementStatus,
            "ExecutivePerformanceCategory",
            each
                if [NetMargin] = null
                then "Unknown"
                else if [NetMargin] >= 0.15
                then "Excellent"
                else if [NetMargin] >= 0.10
                then "Strong"
                else if [NetMargin] >= 0.05
                then "Stable"
                else if [NetMargin] >= 0
                then "Weak"
                else "Loss Making",
            type text
        ),


    // ============================================================
    // 15. FORECAST SUMMARY
    // ============================================================

    ForecastGrouped =
        Table.Group(
            ForecastFiltered,
            {
                "ForecastDate",
                "Company",
                "BusinessUnit"
            },
            {
                {
                    "ForecastCount",
                    each Table.RowCount(_),
                    Int64.Type
                },
                {
                    "AverageForecastConfidence",
                    each
                        List.Average(
                            List.RemoveNulls(
                                [Confidence]
                            )
                        ),
                    type number
                },
                {
                    "AverageForecastRisk",
                    each
                        List.Average(
                            List.RemoveNulls(
                                [ForecastRiskScore]
                            )
                        ),
                    type number
                },
                {
                    "CriticalForecastCount",
                    each
                        List.Count(
                            List.Select(
                                [ForecastRiskLevel],
                                each _ = "CRITICAL"
                            )
                        ),
                    Int64.Type
                }
            }
        ),

    ForecastKey =
        Table.AddColumn(
            ForecastGrouped,
            "ExecutiveBusinessKey",
            each
                Date.ToText(
                    [ForecastDate],
                    "yyyy-MM-dd"
                )
                & "|"
                & [Company]
                & "|"
                & [BusinessUnit],
            type text
        ),

    ForecastRename =
        Table.RenameColumns(
            ForecastKey,
            {
                {
                    "ForecastDate",
                    "Date"
                }
            }
        ),


    // ============================================================
    // 16. FORECAST MERGE
    // ============================================================

    MergeForecast =
        Table.NestedJoin(
            AddPerformanceCategory,
            {
                "ExecutiveBusinessKey"
            },
            ForecastRename,
            {
                "ExecutiveBusinessKey"
            },
            "ForecastData",
            JoinKind.LeftOuter
        ),

    ExpandForecast =
        Table.ExpandTableColumn(
            MergeForecast,
            "ForecastData",
            {
                "ForecastCount",
                "AverageForecastConfidence",
                "AverageForecastRisk",
                "CriticalForecastCount"
            },
            {
                "ForecastCount",
                "AverageForecastConfidence",
                "AverageForecastRisk",
                "CriticalForecastCount"
            }
        ),


    // ============================================================
    // 17. DATA QUALITY
    // ============================================================

    AddDataQuality =
        Table.AddColumn(
            ExpandForecast,
            "ExecutiveDataQuality",
            each
                if
                    [Date] = null
                    or [Company] = null
                    or [Revenue] = null
                then "Incomplete"
                else if
                    [BalanceSheetCheck] = null
                    or [OperatingCashFlow] = null
                then "Partial"
                else "Complete",
            type text
        ),


    // ============================================================
    // 18. EXECUTIVE RECORD ID
    // ============================================================

    AddRecordID =
        Table.AddColumn(
            AddDataQuality,
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


    // ============================================================
    // 19. EXECUTIVE LABEL
    // ============================================================

    AddLabel =
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


    // ============================================================
    // 20. DEDUPLICATION
    // ============================================================

    Deduplicated =
        Table.Distinct(
            AddLabel,
            {
                "ExecutiveRecordID"
            }
        ),


    // ============================================================
    // 21. FINAL SORT
    // ============================================================

    Sorted =
        Table.Sort(
            Deduplicated,
            {
                {
                    "Date",
                    Order.Ascending
                },
                {
                    "Company",
                    Order.Ascending
                },
                {
                    "BusinessUnit",
                    Order.Ascending
                }
            }
        ),


    // ============================================================
    // 22. FINAL BUFFER
    // ============================================================

    ExecutiveDataModel =
        Table.Buffer(
            Sorted
        )

in
    ExecutiveDataModel