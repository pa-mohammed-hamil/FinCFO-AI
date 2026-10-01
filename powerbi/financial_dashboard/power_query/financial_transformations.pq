// ============================================================
// FINCO AI — FINANCIAL TRANSFORMATIONS
// File: financial_transformations.pq
//
// PURPOSE
//   Central reusable transformation layer for historical
//   financial data.
//
// INPUT
//   FinancialData
//
// OUTPUT
//   FinancialTransformations
//
// SUPPORTED AREAS
//   Revenue
//   Expenses
//   Profitability
//   Cash Flow
//   Balance Sheet
//   Liquidity
//   Leverage
//   Working Capital
//   Financial Risk
//   Data Quality
//   Alerts
//
// EXPECTED SOURCE COLUMNS
//   Date
//   Company
//   BusinessUnit
//   Revenue
//   CostOfRevenue
//   GrossProfit
//   OperatingExpenses
//   OperatingProfit
//   InterestExpense
//   TaxExpense
//   NetProfit
//   OperatingCashFlow
//   CapitalExpenditure
//   FreeCashFlow
//   TotalAssets
//   CurrentAssets
//   CurrentLiabilities
//   TotalLiabilities
//   TotalEquity
//   CashAndEquivalents
//   AccountsReceivable
//   Inventory
//   AccountsPayable
//   Debt
//   BudgetRevenue
//   BudgetExpenses
//
// ============================================================

let

    // ========================================================
    // 1. PARAMETERS
    // ========================================================

    FinancialStartDate =
        #date(2015, 1, 1),

    FinancialEndDate =
        #date(2030, 12, 31),


    // ========================================================
    // 2. SOURCE
    // ========================================================

    Source =
        try FinancialData
        otherwise
            #table(
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
                    "BudgetExpenses"
                },
                {}
            ),


    // ========================================================
    // 3. NORMALIZE COLUMN NAMES
    // ========================================================

    NormalizeColumnNames =
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
            "BudgetExpenses"
        },


    // ========================================================
    // 5. ADD MISSING COLUMNS
    // ========================================================

    AddMissingColumns =
        List.Accumulate(
            RequiredColumns,
            NormalizeColumnNames,
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
    // 6. DATA TYPES
    // ========================================================

    Typed =
        Table.TransformColumnTypes(
            AddMissingColumns,
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
                {"BudgetExpenses", type number}
            }
        ),


    // ========================================================
    // 7. CLEAN TEXT
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
    // 8. NUMERIC COLUMNS
    // ========================================================

    NumericColumns =
        {
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
            "BudgetExpenses"
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
    // 9. VALID DATE FILTER
    // ========================================================

    ValidDates =
        Table.SelectRows(
            CleanNumbers,
            each
                [Date] <> null
                and [Date] >= FinancialStartDate
                and [Date] <= FinancialEndDate
        ),


    // ========================================================
    // 10. SORT SOURCE
    // ========================================================

    SortedSource =
        Table.Sort(
            ValidDates,
            {
                {"Date", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending}
            }
        ),


    // ========================================================
    // 11. GROSS PROFIT
    //
    // Uses source GrossProfit when available.
    // Otherwise Revenue - CostOfRevenue.
    // ========================================================

    AddCalculatedGrossProfit =
        Table.AddColumn(
            SortedSource,
            "__CalculatedGrossProfit",
            each
                if [GrossProfit] <> null
                then [GrossProfit]
                else if
                    [Revenue] <> null
                    and [CostOfRevenue] <> null
                then
                    [Revenue] - [CostOfRevenue]
                else
                    null,
            type number
        ),


    RemoveGrossProfit =
        Table.RemoveColumns(
            AddCalculatedGrossProfit,
            {"GrossProfit"}
        ),


    RenameGrossProfit =
        Table.RenameColumns(
            RemoveGrossProfit,
            {
                {
                    "__CalculatedGrossProfit",
                    "GrossProfit"
                }
            }
        ),


    // ========================================================
    // 12. OPERATING PROFIT
    // ========================================================

    AddCalculatedOperatingProfit =
        Table.AddColumn(
            RenameGrossProfit,
            "__CalculatedOperatingProfit",
            each
                if [OperatingProfit] <> null
                then [OperatingProfit]
                else if
                    [GrossProfit] <> null
                    and [OperatingExpenses] <> null
                then
                    [GrossProfit]
                    - [OperatingExpenses]
                else
                    null,
            type number
        ),


    RemoveOperatingProfit =
        Table.RemoveColumns(
            AddCalculatedOperatingProfit,
            {"OperatingProfit"}
        ),


    RenameOperatingProfit =
        Table.RenameColumns(
            RemoveOperatingProfit,
            {
                {
                    "__CalculatedOperatingProfit",
                    "OperatingProfit"
                }
            }
        ),


    // ========================================================
    // 13. NET PROFIT
    // ========================================================

    AddCalculatedNetProfit =
        Table.AddColumn(
            RenameOperatingProfit,
            "__CalculatedNetProfit",
            each
                if [NetProfit] <> null
                then [NetProfit]
                else if [OperatingProfit] <> null
                then
                    [OperatingProfit]
                    - (
                        if [InterestExpense] = null
                        then 0
                        else [InterestExpense]
                    )
                    - (
                        if [TaxExpense] = null
                        then 0
                        else [TaxExpense]
                    )
                else
                    null,
            type number
        ),


    RemoveNetProfit =
        Table.RemoveColumns(
            AddCalculatedNetProfit,
            {"NetProfit"}
        ),


    RenameNetProfit =
        Table.RenameColumns(
            RemoveNetProfit,
            {
                {
                    "__CalculatedNetProfit",
                    "NetProfit"
                }
            }
        ),


    // ========================================================
    // 14. FREE CASH FLOW
    //
    // CapEx is normalized as a cash outflow.
    // ========================================================

    AddNormalizedCapEx =
        Table.AddColumn(
            RenameNetProfit,
            "__NormalizedCapEx",
            each
                if [CapitalExpenditure] = null
                then null
                else
                    -Number.Abs(
                        [CapitalExpenditure]
                    ),
            type number
        ),


    AddCalculatedFreeCashFlow =
        Table.AddColumn(
            AddNormalizedCapEx,
            "__CalculatedFreeCashFlow",
            each
                if [FreeCashFlow] <> null
                then [FreeCashFlow]
                else if
                    [OperatingCashFlow] <> null
                    and [__NormalizedCapEx] <> null
                then
                    [OperatingCashFlow]
                    + [__NormalizedCapEx]
                else
                    null,
            type number
        ),


    RemoveFreeCashFlow =
        Table.RemoveColumns(
            AddCalculatedFreeCashFlow,
            {"FreeCashFlow"}
        ),


    RenameFreeCashFlow =
        Table.RenameColumns(
            RemoveFreeCashFlow,
            {
                {
                    "__CalculatedFreeCashFlow",
                    "FreeCashFlow"
                }
            }
        ),


    RemoveTempCapEx =
        Table.RemoveColumns(
            RenameFreeCashFlow,
            {"__NormalizedCapEx"}
        ),


    // ========================================================
    // 15. DATE ATTRIBUTES
    // ========================================================

    AddYear =
        Table.AddColumn(
            RemoveTempCapEx,
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
    // 16. PROFITABILITY MARGINS
    // ========================================================

    AddGrossMargin =
        Table.AddColumn(
            AddYearMonthKey,
            "GrossMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [GrossProfit]
                    / [Revenue],
            type number
        ),


    AddOperatingMargin =
        Table.AddColumn(
            AddGrossMargin,
            "OperatingMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [OperatingProfit]
                    / [Revenue],
            type number
        ),


    AddNetMargin =
        Table.AddColumn(
            AddOperatingMargin,
            "NetMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [NetProfit]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 17. CASH FLOW MARGINS
    // ========================================================

    AddOperatingCashFlowMargin =
        Table.AddColumn(
            AddNetMargin,
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


    AddFreeCashFlowMargin =
        Table.AddColumn(
            AddOperatingCashFlowMargin,
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
    // 18. BALANCE SHEET RATIOS
    // ========================================================

    AddCurrentRatio =
        Table.AddColumn(
            AddFreeCashFlowMargin,
            "CurrentRatio",
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
            "QuickRatio",
            each
                if [CurrentLiabilities] = null
                    or [CurrentLiabilities] = 0
                then null
                else
                    (
                        (if [CurrentAssets] = null
                         then 0
                         else [CurrentAssets])
                        -
                        (if [Inventory] = null
                         then 0
                         else [Inventory])
                    )
                    / [CurrentLiabilities],
            type number
        ),


    AddCashRatio =
        Table.AddColumn(
            AddQuickRatio,
            "CashRatio",
            each
                if [CurrentLiabilities] = null
                    or [CurrentLiabilities] = 0
                then null
                else
                    [CashAndEquivalents]
                    / [CurrentLiabilities],
            type number
        ),


    AddDebtToAsset =
        Table.AddColumn(
            AddCashRatio,
            "DebtToAssetRatio",
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
            AddDebtToAsset,
            "DebtToEquity",
            each
                if [TotalEquity] = null
                    or [TotalEquity] = 0
                then null
                else
                    [Debt]
                    / [TotalEquity],
            type number
        ),


    AddLiabilityToAsset =
        Table.AddColumn(
            AddDebtToEquity,
            "LiabilityToAssetRatio",
            each
                if [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else
                    [TotalLiabilities]
                    / [TotalAssets],
            type number
        ),


    AddEquityToAsset =
        Table.AddColumn(
            AddLiabilityToAsset,
            "EquityToAssetRatio",
            each
                if [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else
                    [TotalEquity]
                    / [TotalAssets],
            type number
        ),


    // ========================================================
    // 19. WORKING CAPITAL
    // ========================================================

    AddWorkingCapital =
        Table.AddColumn(
            AddEquityToAsset,
            "WorkingCapital",
            each
                if [CurrentAssets] = null
                    or [CurrentLiabilities] = null
                then null
                else
                    [CurrentAssets]
                    - [CurrentLiabilities],
            type number
        ),


    AddCashCoverage =
        Table.AddColumn(
            AddWorkingCapital,
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
    // 20. BUDGET VARIANCES
    // ========================================================

    AddRevenueVariance =
        Table.AddColumn(
            AddCashCoverage,
            "RevenueVariance",
            each
                if [Revenue] = null
                    or [BudgetRevenue] = null
                then null
                else
                    [Revenue]
                    - [BudgetRevenue],
            type number
        ),


    AddRevenueVariancePct =
        Table.AddColumn(
            AddRevenueVariance,
            "RevenueVariancePct",
            each
                if [BudgetRevenue] = null
                    or [BudgetRevenue] = 0
                then null
                else
                    [RevenueVariance]
                    / Number.Abs([BudgetRevenue]),
            type number
        ),


    AddExpenseVariance =
        Table.AddColumn(
            AddRevenueVariancePct,
            "ExpenseVariance",
            each
                if [OperatingExpenses] = null
                    or [BudgetExpenses] = null
                then null
                else
                    [OperatingExpenses]
                    - [BudgetExpenses],
            type number
        ),


    AddExpenseVariancePct =
        Table.AddColumn(
            AddExpenseVariance,
            "ExpenseVariancePct",
            each
                if [BudgetExpenses] = null
                    or [BudgetExpenses] = 0
                then null
                else
                    [ExpenseVariance]
                    / Number.Abs([BudgetExpenses]),
            type number
        ),


    // ========================================================
    // 21. BALANCE SHEET EQUATION
    // ========================================================

    AddBalanceDifference =
        Table.AddColumn(
            AddExpenseVariancePct,
            "BalanceSheetDifference",
            each
                if [TotalAssets] = null
                    or [TotalLiabilities] = null
                    or [TotalEquity] = null
                then null
                else
                    [TotalAssets]
                    - (
                        [TotalLiabilities]
                        + [TotalEquity]
                    ),
            type number
        ),


    AddBalanceSheetCheck =
        Table.AddColumn(
            AddBalanceDifference,
            "BalanceSheetCheck",
            each
                if [BalanceSheetDifference] = null
                then "Unavailable"
                else if
                    Number.Abs(
                        [BalanceSheetDifference]
                    ) <= 0.01
                then "Balanced"
                else
                    "Out of Balance",
            type text
        ),


    // ========================================================
    // 22. PROFITABILITY STATUS
    // ========================================================

    AddProfitabilityStatus =
        Table.AddColumn(
            AddBalanceSheetCheck,
            "ProfitabilityStatus",
            each
                if [NetProfit] = null
                then "Unknown"
                else if [NetProfit] < 0
                then "Loss"
                else if [NetMargin] <> null
                    and [NetMargin] < 0.05
                then "Low Margin"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 23. LIQUIDITY STATUS
    // ========================================================

    AddLiquidityStatus =
        Table.AddColumn(
            AddProfitabilityStatus,
            "LiquidityStatus",
            each
                if [CurrentRatio] = null
                then "Unknown"
                else if [CurrentRatio] < 1
                then "Critical"
                else if [CurrentRatio] < 1.5
                then "Watch"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 24. CASH FLOW STATUS
    // ========================================================

    AddCashFlowStatus =
        Table.AddColumn(
            AddLiquidityStatus,
            "CashFlowStatus",
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
    // 25. LEVERAGE STATUS
    // ========================================================

    AddLeverageStatus =
        Table.AddColumn(
            AddCashFlowStatus,
            "LeverageStatus",
            each
                if [DebtToEquity] = null
                then "Unknown"
                else if [DebtToEquity] >= 2
                then "Critical"
                else if [DebtToEquity] >= 1
                then "Watch"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 26. PROFIT RISK
    // ========================================================

    AddProfitRisk =
        Table.AddColumn(
            AddLeverageStatus,
            "ProfitRisk",
            each
                if [NetProfit] = null
                then 20
                else if [NetProfit] < 0
                then 40
                else if [NetMargin] <> null
                    and [NetMargin] < 0.05
                then 20
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 27. LIQUIDITY RISK
    // ========================================================

    AddLiquidityRisk =
        Table.AddColumn(
            AddProfitRisk,
            "LiquidityRisk",
            each
                if [CurrentRatio] = null
                then 20
                else if [CurrentRatio] < 1
                then 40
                else if [CurrentRatio] < 1.5
                then 20
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 28. CASH FLOW RISK
    // ========================================================

    AddCashFlowRisk =
        Table.AddColumn(
            AddLiquidityRisk,
            "CashFlowRisk",
            each
                if [FreeCashFlow] = null
                then 20
                else if [FreeCashFlow] < 0
                then 30
                else if [FreeCashFlowMargin] <> null
                    and [FreeCashFlowMargin] < 0.05
                then 15
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 29. LEVERAGE RISK
    // ========================================================

    AddLeverageRisk =
        Table.AddColumn(
            AddCashFlowRisk,
            "LeverageRisk",
            each
                if [DebtToEquity] = null
                then 10
                else if [DebtToEquity] >= 2
                then 30
                else if [DebtToEquity] >= 1
                then 15
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 30. DATA INTEGRITY RISK
    // ========================================================

    AddIntegrityRisk =
        Table.AddColumn(
            AddLeverageRisk,
            "IntegrityRisk",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then 30
                else if [BalanceSheetCheck] = "Unavailable"
                then 10
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 31. FINANCIAL RISK SCORE
    // ========================================================

    AddFinancialRiskScore =
        Table.AddColumn(
            AddIntegrityRisk,
            "FinancialRiskScore",
            each
                Number.Min(
                    100,
                    [ProfitRisk]
                    + [LiquidityRisk]
                    + [CashFlowRisk]
                    + [LeverageRisk]
                    + [IntegrityRisk]
                ),
            type number
        ),


    // ========================================================
    // 32. FINANCIAL RISK LEVEL
    // ========================================================

    AddFinancialRiskLevel =
        Table.AddColumn(
            AddFinancialRiskScore,
            "FinancialRiskLevel",
            each
                if [FinancialRiskScore] >= 75
                then "CRITICAL"
                else if [FinancialRiskScore] >= 50
                then "HIGH"
                else if [FinancialRiskScore] >= 25
                then "MEDIUM"
                else
                    "LOW",
            type text
        ),


    // ========================================================
    // 33. FINANCIAL HEALTH SCORE
    // ========================================================

    AddFinancialHealthScore =
        Table.AddColumn(
            AddFinancialRiskLevel,
            "FinancialHealthScore",
            each
                Number.Max(
                    0,
                    100 - [FinancialRiskScore]
                ),
            type number
        ),


    // ========================================================
    // 34. FINANCIAL HEALTH STATUS
    // ========================================================

    AddFinancialHealthStatus =
        Table.AddColumn(
            AddFinancialHealthScore,
            "FinancialHealthStatus",
            each
                if [FinancialHealthScore] >= 80
                then "Excellent"
                else if [FinancialHealthScore] >= 65
                then "Healthy"
                else if [FinancialHealthScore] >= 50
                then "Watch"
                else if [FinancialHealthScore] >= 30
                then "Weak"
                else
                    "Critical",
            type text
        ),


    // ========================================================
    // 35. ALERT FLAG
    // ========================================================

    AddAlertFlag =
        Table.AddColumn(
            AddFinancialHealthStatus,
            "AlertFlag",
            each
                [FinancialRiskLevel] = "CRITICAL"
                or [FinancialRiskLevel] = "HIGH"
                or [BalanceSheetCheck] = "Out of Balance"
                or [RevenueVariancePct] <= -0.10
                or [ExpenseVariancePct] >= 0.10,
            type logical
        ),


    // ========================================================
    // 36. ALERT TYPE
    // ========================================================

    AddAlertType =
        Table.AddColumn(
            AddAlertFlag,
            "AlertType",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then "Balance Sheet Integrity Failure"
                else if [FinancialRiskLevel] = "CRITICAL"
                then "Critical Financial Risk"
                else if [FinancialRiskLevel] = "HIGH"
                then "High Financial Risk"
                else if [RevenueVariancePct] <= -0.20
                then "Severe Revenue Shortfall"
                else if [RevenueVariancePct] <= -0.10
                then "Revenue Shortfall"
                else if [ExpenseVariancePct] >= 0.20
                then "Severe Expense Overrun"
                else if [ExpenseVariancePct] >= 0.10
                then "Expense Overrun"
                else if [LiquidityStatus] = "Critical"
                then "Liquidity Risk"
                else if [CashFlowStatus] = "Negative"
                then "Negative Cash Flow"
                else if [LeverageStatus] = "Critical"
                then "Leverage Risk"
                else
                    null,
            type text
        ),


    // ========================================================
    // 37. RECOMMENDED ACTION
    // ========================================================

    AddRecommendedAction =
        Table.AddColumn(
            AddAlertType,
            "RecommendedAction",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then
                    "Investigate balance-sheet reconciliation and source-data integrity immediately"
                else if [FinancialRiskLevel] = "CRITICAL"
                then
                    "Immediate executive review of financial risk and mitigation options"
                else if [FinancialRiskLevel] = "HIGH"
                then
                    "Investigate financial risk drivers and execute mitigation plan"
                else if [RevenueVariancePct] <= -0.20
                then
                    "Perform revenue root-cause analysis and execute recovery plan"
                else if [ExpenseVariancePct] >= 0.20
                then
                    "Perform expense root-cause analysis and implement cost controls"
                else if [LiquidityStatus] = "Critical"
                then
                    "Improve liquidity and review short-term funding requirements"
                else if [CashFlowStatus] = "Negative"
                then
                    "Review operating cash generation, CapEx and working capital"
                else if [LeverageStatus] = "Critical"
                then
                    "Review debt structure, refinancing and capital allocation"
                else
                    "Continue normal financial monitoring",
            type text
        ),


    // ========================================================
    // 38. DATA QUALITY — DATE
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


    // ========================================================
    // 39. DATA QUALITY — INCOME STATEMENT
    // ========================================================

    AddIncomeStatementQuality =
        Table.AddColumn(
            AddDateQuality,
            "IncomeStatementQuality",
            each
                if [Revenue] = null
                    and [NetProfit] = null
                then "Missing"
                else if [Revenue] = null
                then "Incomplete"
                else
                    "Available",
            type text
        ),


    // ========================================================
    // 40. DATA QUALITY — BALANCE SHEET
    // ========================================================

    AddBalanceSheetQuality =
        Table.AddColumn(
            AddIncomeStatementQuality,
            "BalanceSheetQuality",
            each
                if [TotalAssets] = null
                    and [TotalLiabilities] = null
                    and [TotalEquity] = null
                then "Missing"
                else if [BalanceSheetCheck] = "Out of Balance"
                then "Needs Review"
                else
                    "Available",
            type text
        ),


    // ========================================================
    // 41. DATA QUALITY — CASH FLOW
    // ========================================================

    AddCashFlowQuality =
        Table.AddColumn(
            AddBalanceSheetQuality,
            "CashFlowQuality",
            each
                if [OperatingCashFlow] = null
                    and [FreeCashFlow] = null
                then "Missing"
                else if [OperatingCashFlow] = null
                then "Partial"
                else
                    "Available",
            type text
        ),


    // ========================================================
    // 42. OVERALL DATA QUALITY
    // ========================================================

    AddDataQualityStatus =
        Table.AddColumn(
            AddCashFlowQuality,
            "DataQualityStatus",
            each
                if [DateQuality] = "Invalid"
                then "Invalid"
                else if
                    [IncomeStatementQuality] = "Missing"
                    and [BalanceSheetQuality] = "Missing"
                    and [CashFlowQuality] = "Missing"
                then "Missing"
                else if
                    [IncomeStatementQuality] = "Incomplete"
                    or [BalanceSheetQuality] = "Needs Review"
                    or [CashFlowQuality] = "Partial"
                then "Needs Review"
                else
                    "Valid",
            type text
        ),


    // ========================================================
    // 43. DECISION PRIORITY
    // ========================================================

    AddDecisionPriority =
        Table.AddColumn(
            AddDataQualityStatus,
            "DecisionPriority",
            each
                if [FinancialRiskLevel] = "CRITICAL"
                    or [BalanceSheetCheck] = "Out of Balance"
                then "P0"
                else if [FinancialRiskLevel] = "HIGH"
                then "P1"
                else if [FinancialRiskLevel] = "MEDIUM"
                then "P2"
                else
                    "P3",
            type text
        ),


    // ========================================================
    // 44. FINANCIAL RECORD ID
    // ========================================================

    AddFinancialRecordID =
        Table.AddColumn(
            AddDecisionPriority,
            "FinancialRecordID",
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
    // 45. FINANCIAL LABEL
    // ========================================================

    AddFinancialLabel =
        Table.AddColumn(
            AddFinancialRecordID,
            "FinancialLabel",
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
    // 46. MANAGEMENT STATUS
    // ========================================================

    AddManagementStatus =
        Table.AddColumn(
            AddFinancialLabel,
            "ManagementStatus",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then "Immediate Action"
                else if [FinancialRiskLevel] = "CRITICAL"
                then "Immediate Action"
                else if [FinancialRiskLevel] = "HIGH"
                then "Management Review"
                else if [FinancialRiskLevel] = "MEDIUM"
                then "Monitor"
                else
                    "Normal",
            type text
        ),


    // ========================================================
    // 47. FINANCIAL CATEGORY
    // ========================================================

    AddFinancialCategory =
        Table.AddColumn(
            AddManagementStatus,
            "FinancialCategory",
            each
                if [FinancialRiskLevel] = "CRITICAL"
                    or [FinancialRiskLevel] = "HIGH"
                then "Risk"
                else if [ProfitabilityStatus] = "Loss"
                then "Profitability"
                else if [CashFlowStatus] = "Negative"
                then "Cash Flow"
                else if [LiquidityStatus] = "Critical"
                then "Liquidity"
                else if [LeverageStatus] = "Critical"
                then "Leverage"
                else
                    "Financial Performance",
            type text
        ),


    // ========================================================
    // 48. SORT FINAL DATA
    // ========================================================

    SortedFinal =
        Table.Sort(
            AddFinancialCategory,
            {
                {"Date", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending}
            }
        ),


    // ========================================================
    // 49. DEDUPLICATE
    // ========================================================

    Deduplicated =
        Table.Distinct(
            SortedFinal,
            {
                "FinancialRecordID"
            }
        ),


    // ========================================================
    // 50. FINAL BUFFER
    // ========================================================

    FinancialTransformations =
        Table.Buffer(
            Deduplicated
        )

in
    FinancialTransformations