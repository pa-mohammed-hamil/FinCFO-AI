let
    // ============================================================
    // FinCo AI — Historical Financials
    // File: historical_financials.pq
    //
    // Purpose:
    //   Clean, standardize, validate, and enrich historical
    //   financial statement data.
    //
    // Expected source query/table:
    //   HistoricalFinancials
    //
    // Supported financial fields:
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
    // ============================================================


    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------

    HistoricalStartDate =
        #date(2015, 1, 1),

    HistoricalEndDate =
        #date(2030, 12, 31),


    // ------------------------------------------------------------
    // Source
    // ------------------------------------------------------------

    Source =
        try HistoricalFinancials
        otherwise
            #table(
                type table[
                    Date = date,
                    Company = text,
                    BusinessUnit = text,
                    Revenue = number,
                    CostOfRevenue = number,
                    GrossProfit = number,
                    OperatingExpenses = number,
                    OperatingProfit = number,
                    InterestExpense = number,
                    TaxExpense = number,
                    NetProfit = number,
                    OperatingCashFlow = number,
                    CapitalExpenditure = number,
                    FreeCashFlow = number,
                    TotalAssets = number,
                    CurrentAssets = number,
                    CurrentLiabilities = number,
                    TotalLiabilities = number,
                    TotalEquity = number,
                    CashAndEquivalents = number,
                    AccountsReceivable = number,
                    Inventory = number,
                    AccountsPayable = number,
                    Debt = number,
                    BudgetRevenue = number,
                    BudgetExpenses = number
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

    SelectColumns =
        Table.SelectColumns(
            AddMissingColumns,
            RequiredColumns,
            MissingField.UseNull
        ),


    // ------------------------------------------------------------
    // Data types
    // ------------------------------------------------------------

    SetDataTypes =
        Table.TransformColumnTypes(
            SelectColumns,
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


    // ------------------------------------------------------------
    // Clean text
    // ------------------------------------------------------------

    CleanText =
        Table.TransformColumns(
            SetDataTypes,
            {
                {
                    "Company",
                    each
                        if _ = null or Text.Trim(Text.From(_)) = ""
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
                        if _ = null or Text.Trim(Text.From(_)) = ""
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
    // Clean numeric values
    // ------------------------------------------------------------

    NumericColumns = {
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


    CleanNumericColumns =
        List.Accumulate(
            NumericColumns,
            CleanText,
            (CurrentTable, ColumnName) =>
                Table.TransformColumns(
                    CurrentTable,
                    {
                        {
                            ColumnName,
                            each
                                if _ = null
                                then null
                                else
                                    try Number.From(_)
                                    otherwise null,
                            type number
                        }
                    }
                )
        ),


    // ------------------------------------------------------------
    // Date validation
    // ------------------------------------------------------------

    FilterValidDates =
        Table.SelectRows(
            CleanNumericColumns,
            each
                [Date] <> null
                and [Date] >= HistoricalStartDate
                and [Date] <= HistoricalEndDate
        ),


    // ------------------------------------------------------------
    // Sort source data
    // ------------------------------------------------------------

    SortBase =
        Table.Sort(
            FilterValidDates,
            {
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending},
                {"Date", Order.Ascending}
            }
        ),


    // ============================================================
    // DERIVED FINANCIAL VALUES
    // ============================================================


    // ------------------------------------------------------------
    // Gross Profit
    // ------------------------------------------------------------

    AddCalculatedGrossProfit =
        Table.AddColumn(
            SortBase,
            "CalculatedGrossProfit",
            each
                if [GrossProfit] <> null
                then [GrossProfit]
                else if
                    [Revenue] <> null
                    and [CostOfRevenue] <> null
                then
                    [Revenue] - [CostOfRevenue]
                else null,
            type number
        ),


    // ------------------------------------------------------------
    // Operating Profit
    // ------------------------------------------------------------

    AddCalculatedOperatingProfit =
        Table.AddColumn(
            AddCalculatedGrossProfit,
            "CalculatedOperatingProfit",
            each
                if [OperatingProfit] <> null
                then [OperatingProfit]
                else if
                    [CalculatedGrossProfit] <> null
                    and [OperatingExpenses] <> null
                then
                    [CalculatedGrossProfit]
                    - [OperatingExpenses]
                else null,
            type number
        ),


    // ------------------------------------------------------------
    // Net Profit
    // ------------------------------------------------------------

    AddCalculatedNetProfit =
        Table.AddColumn(
            AddCalculatedOperatingProfit,
            "CalculatedNetProfit",
            each
                if [NetProfit] <> null
                then [NetProfit]
                else if
                    [CalculatedOperatingProfit] <> null
                then
                    [CalculatedOperatingProfit]
                    -
                    (if [InterestExpense] = null
                     then 0
                     else [InterestExpense])
                    -
                    (if [TaxExpense] = null
                     then 0
                     else [TaxExpense])
                else null,
            type number
        ),


    // ------------------------------------------------------------
    // Free Cash Flow
    // ------------------------------------------------------------

    AddCalculatedFCF =
        Table.AddColumn(
            AddCalculatedNetProfit,
            "CalculatedFreeCashFlow",
            each
                if [FreeCashFlow] <> null
                then [FreeCashFlow]
                else if
                    [OperatingCashFlow] <> null
                    and [CapitalExpenditure] <> null
                then
                    [OperatingCashFlow]
                    - Number.Abs([CapitalExpenditure])
                else if
                    [OperatingCashFlow] <> null
                then
                    [OperatingCashFlow]
                else null,
            type number
        ),


    // ============================================================
    // DATE DIMENSIONS
    // ============================================================


    AddYear =
        Table.AddColumn(
            AddCalculatedFCF,
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
                "Q" & Text.From([QuarterNumber]),
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
            each
                Date.ToText(
                    [Date],
                    "yyyy-MM"
                ),
            type text
        ),


    AddMonthName =
        Table.AddColumn(
            AddMonth,
            "MonthName",
            each
                Date.MonthName([Date]),
            type text
        ),


    AddMonthStart =
        Table.AddColumn(
            AddMonthName,
            "MonthStart",
            each
                Date.StartOfMonth([Date]),
            type date
        ),


    AddMonthYear =
        Table.AddColumn(
            AddMonthStart,
            "MonthYear",
            each
                Date.ToText(
                    [Date],
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
                    [Date],
                    "yyyyMM"
                ),
            type text
        ),


    // ============================================================
    // PROFITABILITY METRICS
    // ============================================================


    AddGrossMargin =
        Table.AddColumn(
            AddPeriodKey,
            "GrossMargin",
            each
                if
                    [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [CalculatedGrossProfit]
                    / [Revenue],
            type number
        ),


    AddOperatingMargin =
        Table.AddColumn(
            AddGrossMargin,
            "OperatingMargin",
            each
                if
                    [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [CalculatedOperatingProfit]
                    / [Revenue],
            type number
        ),


    AddNetMargin =
        Table.AddColumn(
            AddOperatingMargin,
            "NetMargin",
            each
                if
                    [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [CalculatedNetProfit]
                    / [Revenue],
            type number
        ),


    // ============================================================
    // CASH FLOW METRICS
    // ============================================================


    AddFCFMargin =
        Table.AddColumn(
            AddNetMargin,
            "FCFMargin",
            each
                if
                    [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [CalculatedFreeCashFlow]
                    / [Revenue],
            type number
        ),


    AddOperatingCashFlowMargin =
        Table.AddColumn(
            AddFCFMargin,
            "OperatingCashFlowMargin",
            each
                if
                    [Revenue] = null
                    or [Revenue] = 0
                then null
                else
                    [OperatingCashFlow]
                    / [Revenue],
            type number
        ),


    // ============================================================
    // BALANCE SHEET RATIOS
    // ============================================================


    AddCurrentRatio =
        Table.AddColumn(
            AddOperatingCashFlowMargin,
            "CurrentRatio",
            each
                if
                    [CurrentLiabilities] = null
                    or [CurrentLiabilities] = 0
                then null
                else
                    [CurrentAssets]
                    / [CurrentLiabilities],
            type number
        ),


    AddDebtToAssetRatio =
        Table.AddColumn(
            AddCurrentRatio,
            "DebtToAssetRatio",
            each
                if
                    [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else
                    [Debt]
                    / [TotalAssets],
            type number
        ),


    AddDebtToEquity =
        Table.AddColumn(
            AddDebtToAssetRatio,
            "DebtToEquity",
            each
                if
                    [TotalEquity] = null
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
                if
                    [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else
                    [TotalLiabilities]
                    / [TotalAssets],
            type number
        ),


    // ============================================================
    // WORKING CAPITAL
    // ============================================================


    AddWorkingCapital =
        Table.AddColumn(
            AddLiabilityToAsset,
            "WorkingCapital",
            each
                if
                    [CurrentAssets] = null
                    or [CurrentLiabilities] = null
                then null
                else
                    [CurrentAssets]
                    - [CurrentLiabilities],
            type number
        ),


    AddNetWorkingCapital =
        Table.AddColumn(
            AddWorkingCapital,
            "NetWorkingCapital",
            each
                if
                    [CurrentAssets] = null
                    or [CurrentLiabilities] = null
                then null
                else
                    [CurrentAssets]
                    - [CurrentLiabilities],
            type number
        ),


    // ============================================================
    // BUDGET VARIANCES
    // ============================================================


    AddRevenueVariance =
        Table.AddColumn(
            AddNetWorkingCapital,
            "RevenueVariance",
            each
                if
                    [Revenue] = null
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
                if
                    [BudgetRevenue] = null
                    or [BudgetRevenue] = 0
                then null
                else
                    ([Revenue] - [BudgetRevenue])
                    /
                    Number.Abs(
                        [BudgetRevenue]
                    ),
            type number
        ),


    AddExpenseVariance =
        Table.AddColumn(
            AddRevenueVariancePct,
            "ExpenseVariance",
            each
                if
                    [OperatingExpenses] = null
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
                if
                    [BudgetExpenses] = null
                    or [BudgetExpenses] = 0
                then null
                else
                    ([OperatingExpenses] - [BudgetExpenses])
                    /
                    Number.Abs(
                        [BudgetExpenses]
                    ),
            type number
        ),


    // ============================================================
    // GROWTH METRICS
    // ============================================================


    AddRevenueGrowth =
        Table.AddColumn(
            AddExpenseVariancePct,
            "RevenueGrowthProxy",
            each null,
            type number
        ),


    // ============================================================
    // FINANCIAL HEALTH INDICATORS
    // ============================================================


    AddProfitabilityStatus =
        Table.AddColumn(
            AddRevenueGrowth,
            "ProfitabilityStatus",
            each
                if [CalculatedNetProfit] = null
                then "Unknown"

                else if [CalculatedNetProfit] < 0
                then "Loss"

                else if [NetMargin] <> null
                    and [NetMargin] < 0.05
                then "Low Margin"

                else
                    "Healthy",
            type text
        ),


    AddCashFlowStatus =
        Table.AddColumn(
            AddProfitabilityStatus,
            "CashFlowStatus",
            each
                if [CalculatedFreeCashFlow] = null
                then "Unknown"

                else if [CalculatedFreeCashFlow] < 0
                then "Negative FCF"

                else if
                    [FCFMargin] <> null
                    and [FCFMargin] < 0.05
                then "Weak FCF"

                else
                    "Healthy",
            type text
        ),


    AddLiquidityStatus =
        Table.AddColumn(
            AddCashFlowStatus,
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


    AddLeverageStatus =
        Table.AddColumn(
            AddLiquidityStatus,
            "LeverageStatus",
            each
                if [DebtToEquity] = null
                then "Unknown"

                else if [DebtToEquity] >= 2
                then "High Leverage"

                else if [DebtToEquity] >= 1
                then "Moderate Leverage"

                else
                    "Low Leverage",
            type text
        ),


    // ============================================================
    // FINANCIAL RISK SCORE
    // ============================================================


    AddProfitRisk =
        Table.AddColumn(
            AddLeverageStatus,
            "ProfitRisk",
            each
                if [CalculatedNetProfit] = null
                then 20

                else if [CalculatedNetProfit] < 0
                then 40

                else if
                    [NetMargin] <> null
                    and [NetMargin] < 0.05
                then 20

                else 0,
            Int64.Type
        ),


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

                else 0,
            Int64.Type
        ),


    AddCashFlowRisk =
        Table.AddColumn(
            AddLiquidityRisk,
            "CashFlowRisk",
            each
                if [CalculatedFreeCashFlow] = null
                then 20

                else if [CalculatedFreeCashFlow] < 0
                then 30

                else if
                    [FCFMargin] <> null
                    and [FCFMargin] < 0.05
                then 15

                else 0,
            Int64.Type
        ),


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

                else 0,
            Int64.Type
        ),


    AddFinancialRiskScore =
        Table.AddColumn(
            AddLeverageRisk,
            "FinancialRiskScore",
            each
                Number.Min(
                    100,
                    [ProfitRisk]
                    + [LiquidityRisk]
                    + [CashFlowRisk]
                    + [LeverageRisk]
                ),
            Int64.Type
        ),


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


    // ============================================================
    // FINANCIAL HEALTH SCORE
    // ============================================================


    AddFinancialHealthScore =
        Table.AddColumn(
            AddFinancialRiskLevel,
            "FinancialHealthScore",
            each
                Number.Max(
                    0,
                    100 - [FinancialRiskScore]
                ),
            Int64.Type
        ),


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


    // ============================================================
    // ALERT ENGINE INPUTS
    // ============================================================


    AddAlertFlag =
        Table.AddColumn(
            AddFinancialHealthStatus,
            "AlertFlag",
            each
                if
                    [FinancialRiskLevel]
                    = "CRITICAL"
                    or
                    [FinancialRiskLevel]
                    = "HIGH"
                then "YES"
                else "NO",
            type text
        ),


    AddAlertType =
        Table.AddColumn(
            AddAlertFlag,
            "AlertType",
            each
                if
                    [FinancialRiskLevel]
                    = "CRITICAL"
                then
                    "Critical Financial Risk"

                else if
                    [FinancialRiskLevel]
                    = "HIGH"
                then
                    "High Financial Risk"

                else if
                    [LiquidityStatus]
                    = "Critical"
                then
                    "Liquidity Risk"

                else if
                    [CashFlowStatus]
                    = "Negative FCF"
                then
                    "Negative Free Cash Flow"

                else if
                    [ProfitabilityStatus]
                    = "Loss"
                then
                    "Profitability Risk"

                else null,
            type text
        ),


    AddRecommendedAction =
        Table.AddColumn(
            AddAlertType,
            "RecommendedAction",
            each
                if
                    [FinancialRiskLevel]
                    = "CRITICAL"
                then
                    "Immediate management review and financial mitigation plan"

                else if
                    [LiquidityStatus]
                    = "Critical"
                then
                    "Review liquidity position, working capital, and funding requirements"

                else if
                    [CashFlowStatus]
                    = "Negative FCF"
                then
                    "Investigate cash conversion and reduce discretionary outflows"

                else if
                    [ProfitabilityStatus]
                    = "Loss"
                then
                    "Investigate revenue, cost structure, and margin deterioration"

                else if
                    [LeverageStatus]
                    = "High Leverage"
                then
                    "Review debt structure and refinancing options"

                else
                    "Continue normal financial monitoring",
            type text
        ),


    // ============================================================
    // DATA QUALITY
    // ============================================================


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


    AddRevenueQuality =
        Table.AddColumn(
            AddDateQuality,
            "RevenueQuality",
            each
                if [Revenue] = null
                then "Missing"

                else if [Revenue] = 0
                then "Zero"

                else "Valid",
            type text
        ),


    AddBalanceSheetQuality =
        Table.AddColumn(
            AddRevenueQuality,
            "BalanceSheetQuality",
            each
                if
                    [TotalAssets] = null
                    and [TotalLiabilities] = null
                    and [TotalEquity] = null
                then "Missing"

                else "Available",
            type text
        ),


    AddCashFlowQuality =
        Table.AddColumn(
            AddBalanceSheetQuality,
            "CashFlowQuality",
            each
                if [OperatingCashFlow] = null
                then "Missing"
                else "Available",
            type text
        ),


    AddDataQualityStatus =
        Table.AddColumn(
            AddCashFlowQuality,
            "DataQualityStatus",
            each
                if [DateQuality] = "Invalid"
                then "INVALID"

                else if [RevenueQuality] = "Missing"
                then "WARNING"

                else "VALID",
            type text
        ),


    // ============================================================
    // RECORD IDENTIFIER
    // ============================================================


    AddFinancialRecordID =
        Table.AddColumn(
            AddDataQualityStatus,
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


    // ============================================================
    // FINAL COLUMN ORDER
    // ============================================================


    ReorderedColumns =
        Table.ReorderColumns(
            AddFinancialLabel,
            {
                "FinancialRecordID",
                "Date",
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

                "Revenue",
                "CostOfRevenue",
                "CalculatedGrossProfit",
                "OperatingExpenses",
                "CalculatedOperatingProfit",
                "InterestExpense",
                "TaxExpense",
                "CalculatedNetProfit",

                "OperatingCashFlow",
                "CapitalExpenditure",
                "CalculatedFreeCashFlow",

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

                "GrossMargin",
                "OperatingMargin",
                "NetMargin",
                "FCFMargin",
                "OperatingCashFlowMargin",

                "CurrentRatio",
                "DebtToAssetRatio",
                "DebtToEquity",
                "LiabilityToAssetRatio",

                "WorkingCapital",
                "NetWorkingCapital",

                "RevenueVariance",
                "RevenueVariancePct",
                "ExpenseVariance",
                "ExpenseVariancePct",

                "ProfitabilityStatus",
                "CashFlowStatus",
                "LiquidityStatus",
                "LeverageStatus",

                "ProfitRisk",
                "LiquidityRisk",
                "CashFlowRisk",
                "LeverageRisk",
                "FinancialRiskScore",
                "FinancialRiskLevel",

                "FinancialHealthScore",
                "FinancialHealthStatus",

                "AlertFlag",
                "AlertType",
                "RecommendedAction",

                "DateQuality",
                "RevenueQuality",
                "BalanceSheetQuality",
                "CashFlowQuality",
                "DataQualityStatus",

                "FinancialLabel"
            },
            MissingField.UseNull
        ),


    // ============================================================
    // Remove duplicate financial records
    // ============================================================

    RemoveDuplicates =
        Table.Distinct(
            ReorderedColumns,
            {
                "FinancialRecordID"
            }
        ),


    // ============================================================
    // Final sort
    // ============================================================

    Sorted =
        Table.Sort(
            RemoveDuplicates,
            {
                {"Date", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending}
            }
        ),


    // ============================================================
    // Buffer final dataset
    // ============================================================

    Final =
        Table.Buffer(
            Sorted
        )

in
    Final