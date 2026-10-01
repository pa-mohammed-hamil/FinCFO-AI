// ============================================================
// FINCO AI — INCOME STATEMENT QUERY
// File: income_statement_query.pq
//
// PURPOSE
//   Production-ready Income Statement fact table for Power BI.
//
// INPUT
//   FinancialTransformations
//
// OUTPUT
//   IncomeStatement
//
// COVERS
//   Revenue
//   Cost of Revenue
//   Gross Profit
//   Operating Expenses
//   Operating Profit
//   Interest Expense
//   Tax Expense
//   Net Profit
//   Margins
//   Budget Variance
//   Growth
//   Profitability Risk
//   Financial Risk
//   Alerts
//   Data Quality
//
// RECOMMENDED MODEL
//   DimDate        1:* IncomeStatement
//   DimCompany     1:* IncomeStatement
//   DimBusinessUnit 1:* IncomeStatement
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
        try FinancialTransformations
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

    SetTypes =
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

                {"BudgetRevenue", type number},
                {"BudgetExpenses", type number}
            }
        ),


    // ========================================================
    // 7. CLEAN TEXT
    // ========================================================

    CleanText =
        Table.TransformColumns(
            SetTypes,
            {
                {
                    "Company",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(
                            Text.From(_)
                        ),
                    type text
                },
                {
                    "BusinessUnit",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(
                            Text.From(_)
                        ),
                    type text
                }
            }
        ),


    // ========================================================
    // 8. CLEAN NUMERIC VALUES
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
    // 9. DATE FILTER
    // ========================================================

    FilterDates =
        Table.SelectRows(
            CleanNumbers,
            each
                [Date] <> null
                and [Date] >= StartDate
                and [Date] <= EndDate
        ),


    // ========================================================
    // 10. SORT SOURCE
    // ========================================================

    SortedSource =
        Table.Sort(
            FilterDates,
            {
                {"Date", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending}
            }
        ),


    // ========================================================
    // 11. CALCULATE GROSS PROFIT
    // ========================================================

    AddGrossProfit =
        Table.AddColumn(
            SortedSource,
            "__GrossProfit",
            each
                if [GrossProfit] <> null
                then
                    [GrossProfit]
                else if
                    [Revenue] <> null
                    and [CostOfRevenue] <> null
                then
                    [Revenue]
                    - [CostOfRevenue]
                else
                    null,
            type number
        ),


    RemoveOriginalGrossProfit =
        Table.RemoveColumns(
            AddGrossProfit,
            {"GrossProfit"}
        ),


    RenameGrossProfit =
        Table.RenameColumns(
            RemoveOriginalGrossProfit,
            {
                {
                    "__GrossProfit",
                    "GrossProfit"
                }
            }
        ),


    // ========================================================
    // 12. CALCULATE OPERATING PROFIT
    // ========================================================

    AddOperatingProfit =
        Table.AddColumn(
            RenameGrossProfit,
            "__OperatingProfit",
            each
                if [OperatingProfit] <> null
                then
                    [OperatingProfit]
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


    RemoveOriginalOperatingProfit =
        Table.RemoveColumns(
            AddOperatingProfit,
            {"OperatingProfit"}
        ),


    RenameOperatingProfit =
        Table.RenameColumns(
            RemoveOriginalOperatingProfit,
            {
                {
                    "__OperatingProfit",
                    "OperatingProfit"
                }
            }
        ),


    // ========================================================
    // 13. CALCULATE NET PROFIT
    // ========================================================

    AddNetProfit =
        Table.AddColumn(
            RenameOperatingProfit,
            "__NetProfit",
            each
                if [NetProfit] <> null
                then
                    [NetProfit]
                else if [OperatingProfit] <> null
                then
                    [OperatingProfit]
                    -
                    (
                        if [InterestExpense] = null
                        then 0
                        else [InterestExpense]
                    )
                    -
                    (
                        if [TaxExpense] = null
                        then 0
                        else [TaxExpense]
                    )
                else
                    null,
            type number
        ),


    RemoveOriginalNetProfit =
        Table.RemoveColumns(
            AddNetProfit,
            {"NetProfit"}
        ),


    RenameNetProfit =
        Table.RenameColumns(
            RemoveOriginalNetProfit,
            {
                {
                    "__NetProfit",
                    "NetProfit"
                }
            }
        ),


    // ========================================================
    // 14. DATE DIMENSIONS
    // ========================================================

    AddYear =
        Table.AddColumn(
            RenameNetProfit,
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
            each Date.ToText(
                [Date],
                "MMM"
            ),
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
            each Date.ToText(
                [Date],
                "yyyy-MM"
            ),
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
    // 15. GROSS MARGIN
    // ========================================================

    AddGrossMargin =
        Table.AddColumn(
            AddYearMonthKey,
            "GrossMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then
                    null
                else
                    [GrossProfit]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 16. OPERATING MARGIN
    // ========================================================

    AddOperatingMargin =
        Table.AddColumn(
            AddGrossMargin,
            "OperatingMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then
                    null
                else
                    [OperatingProfit]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 17. NET MARGIN
    // ========================================================

    AddNetMargin =
        Table.AddColumn(
            AddOperatingMargin,
            "NetMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then
                    null
                else
                    [NetProfit]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 18. COST OF REVENUE %
    // ========================================================

    AddCostOfRevenuePct =
        Table.AddColumn(
            AddNetMargin,
            "CostOfRevenuePct",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then
                    null
                else
                    [CostOfRevenue]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 19. OPERATING EXPENSE %
    // ========================================================

    AddOperatingExpensePct =
        Table.AddColumn(
            AddCostOfRevenuePct,
            "OperatingExpensePct",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then
                    null
                else
                    [OperatingExpenses]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 20. INTEREST BURDEN
    // ========================================================

    AddInterestBurden =
        Table.AddColumn(
            AddOperatingExpensePct,
            "InterestBurden",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then
                    null
                else
                    [InterestExpense]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 21. TAX BURDEN
    // ========================================================

    AddTaxBurden =
        Table.AddColumn(
            AddInterestBurden,
            "TaxBurden",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then
                    null
                else
                    [TaxExpense]
                    / [Revenue],
            type number
        ),


    // ========================================================
    // 22. EBITDA-LIKE OPERATING PROFIT MARGIN
    // ========================================================

    AddProfitConversion =
        Table.AddColumn(
            AddTaxBurden,
            "ProfitConversionRate",
            each
                if [GrossProfit] = null
                    or [GrossProfit] = 0
                then
                    null
                else
                    [OperatingProfit]
                    / [GrossProfit],
            type number
        ),


    // ========================================================
    // 23. REVENUE BUDGET VARIANCE
    // ========================================================

    AddRevenueVariance =
        Table.AddColumn(
            AddProfitConversion,
            "RevenueVariance",
            each
                if [Revenue] = null
                    or [BudgetRevenue] = null
                then
                    null
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
                then
                    null
                else
                    [RevenueVariance]
                    / Number.Abs(
                        [BudgetRevenue]
                    ),
            type number
        ),


    // ========================================================
    // 24. EXPENSE BUDGET VARIANCE
    // ========================================================

    AddExpenseVariance =
        Table.AddColumn(
            AddRevenueVariancePct,
            "ExpenseVariance",
            each
                if [OperatingExpenses] = null
                    or [BudgetExpenses] = null
                then
                    null
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
                then
                    null
                else
                    [ExpenseVariance]
                    / Number.Abs(
                        [BudgetExpenses]
                    ),
            type number
        ),


    // ========================================================
    // 25. REVENUE STATUS
    // ========================================================

    AddRevenueStatus =
        Table.AddColumn(
            AddExpenseVariancePct,
            "RevenueStatus",
            each
                if [Revenue] = null
                then
                    "Unknown"
                else if [Revenue] < 0
                then
                    "Invalid"
                else if
                    [RevenueVariancePct] <> null
                    and [RevenueVariancePct] <= -0.20
                then
                    "Severe Shortfall"
                else if
                    [RevenueVariancePct] <> null
                    and [RevenueVariancePct] <= -0.10
                then
                    "Below Budget"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 26. GROSS PROFIT STATUS
    // ========================================================

    AddGrossProfitStatus =
        Table.AddColumn(
            AddRevenueStatus,
            "GrossProfitStatus",
            each
                if [GrossProfit] = null
                then
                    "Unknown"
                else if [GrossProfit] < 0
                then
                    "Loss"
                else if
                    [GrossMargin] <> null
                    and [GrossMargin] < 0.20
                then
                    "Low Margin"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 27. OPERATING PROFIT STATUS
    // ========================================================

    AddOperatingProfitStatus =
        Table.AddColumn(
            AddGrossProfitStatus,
            "OperatingProfitStatus",
            each
                if [OperatingProfit] = null
                then
                    "Unknown"
                else if [OperatingProfit] < 0
                then
                    "Loss"
                else if
                    [OperatingMargin] <> null
                    and [OperatingMargin] < 0.05
                then
                    "Low Margin"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 28. NET PROFIT STATUS
    // ========================================================

    AddNetProfitStatus =
        Table.AddColumn(
            AddOperatingProfitStatus,
            "NetProfitStatus",
            each
                if [NetProfit] = null
                then
                    "Unknown"
                else if [NetProfit] < 0
                then
                    "Loss"
                else if
                    [NetMargin] <> null
                    and [NetMargin] < 0.05
                then
                    "Low Margin"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 29. PROFITABILITY RISK
    // ========================================================

    AddProfitabilityRisk =
        Table.AddColumn(
            AddNetProfitStatus,
            "ProfitabilityRisk",
            each
                if [NetProfit] = null
                then
                    20
                else if [NetProfit] < 0
                then
                    40
                else if
                    [NetMargin] <> null
                    and [NetMargin] < 0.05
                then
                    20
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 30. GROSS MARGIN RISK
    // ========================================================

    AddGrossMarginRisk =
        Table.AddColumn(
            AddProfitabilityRisk,
            "GrossMarginRisk",
            each
                if [GrossMargin] = null
                then
                    15
                else if [GrossMargin] < 0
                then
                    30
                else if [GrossMargin] < 0.20
                then
                    20
                else if [GrossMargin] < 0.30
                then
                    10
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 31. OPERATING MARGIN RISK
    // ========================================================

    AddOperatingMarginRisk =
        Table.AddColumn(
            AddGrossMarginRisk,
            "OperatingMarginRisk",
            each
                if [OperatingMargin] = null
                then
                    15
                else if [OperatingMargin] < 0
                then
                    30
                else if [OperatingMargin] < 0.05
                then
                    20
                else if [OperatingMargin] < 0.10
                then
                    10
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 32. REVENUE RISK
    // ========================================================

    AddRevenueRisk =
        Table.AddColumn(
            AddOperatingMarginRisk,
            "RevenueRisk",
            each
                if [Revenue] = null
                then
                    20
                else if [RevenueVariancePct] <> null
                    and [RevenueVariancePct] <= -0.30
                then
                    40
                else if [RevenueVariancePct] <> null
                    and [RevenueVariancePct] <= -0.20
                then
                    30
                else if [RevenueVariancePct] <> null
                    and [RevenueVariancePct] <= -0.10
                then
                    20
                else if [RevenueVariancePct] <> null
                    and [RevenueVariancePct] < 0
                then
                    10
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 33. EXPENSE RISK
    // ========================================================

    AddExpenseRisk =
        Table.AddColumn(
            AddRevenueRisk,
            "ExpenseRisk",
            each
                if [OperatingExpenses] = null
                then
                    20
                else if [ExpenseVariancePct] <> null
                    and [ExpenseVariancePct] >= 0.30
                then
                    40
                else if [ExpenseVariancePct] <> null
                    and [ExpenseVariancePct] >= 0.20
                then
                    30
                else if [ExpenseVariancePct] <> null
                    and [ExpenseVariancePct] >= 0.10
                then
                    20
                else if [ExpenseVariancePct] <> null
                    and [ExpenseVariancePct] > 0
                then
                    10
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 34. FINANCIAL RISK SCORE
    // ========================================================

    AddFinancialRiskScore =
        Table.AddColumn(
            AddExpenseRisk,
            "FinancialRiskScore",
            each
                List.Min(
                    {
                        100,
                        [ProfitabilityRisk]
                        + [GrossMarginRisk]
                        + [OperatingMarginRisk]
                        + [RevenueRisk]
                        + [ExpenseRisk]
                    }
                ),
            type number
        ),


    // ========================================================
    // 35. FINANCIAL RISK LEVEL
    // ========================================================

    AddFinancialRiskLevel =
        Table.AddColumn(
            AddFinancialRiskScore,
            "FinancialRiskLevel",
            each
                if [FinancialRiskScore] >= 75
                then
                    "CRITICAL"
                else if [FinancialRiskScore] >= 50
                then
                    "HIGH"
                else if [FinancialRiskScore] >= 25
                then
                    "MEDIUM"
                else
                    "LOW",
            type text
        ),


    // ========================================================
    // 36. FINANCIAL HEALTH SCORE
    // ========================================================

    AddFinancialHealthScore =
        Table.AddColumn(
            AddFinancialRiskLevel,
            "FinancialHealthScore",
            each
                List.Max(
                    {
                        0,
                        100
                        - [FinancialRiskScore]
                    }
                ),
            type number
        ),


    // ========================================================
    // 37. FINANCIAL HEALTH STATUS
    // ========================================================

    AddFinancialHealthStatus =
        Table.AddColumn(
            AddFinancialHealthScore,
            "FinancialHealthStatus",
            each
                if [FinancialHealthScore] >= 80
                then
                    "Excellent"
                else if [FinancialHealthScore] >= 65
                then
                    "Healthy"
                else if [FinancialHealthScore] >= 50
                then
                    "Watch"
                else if [FinancialHealthScore] >= 30
                then
                    "Weak"
                else
                    "Critical",
            type text
        ),


    // ========================================================
    // 38. ALERT FLAG
    // ========================================================

    AddAlertFlag =
        Table.AddColumn(
            AddFinancialHealthStatus,
            "AlertFlag",
            each
                [FinancialRiskLevel] = "CRITICAL"
                or [FinancialRiskLevel] = "HIGH"
                or [RevenueVariancePct] <= -0.10
                or [ExpenseVariancePct] >= 0.10
                or [NetProfitStatus] = "Loss",
            type logical
        ),


    // ========================================================
    // 39. ALERT TYPE
    // ========================================================

    AddAlertType =
        Table.AddColumn(
            AddAlertFlag,
            "AlertType",
            each
                if [FinancialRiskLevel] = "CRITICAL"
                then
                    "Critical Income Statement Risk"
                else if [FinancialRiskLevel] = "HIGH"
                then
                    "High Income Statement Risk"
                else if [NetProfitStatus] = "Loss"
                then
                    "Net Loss"
                else if [RevenueVariancePct] <= -0.20
                then
                    "Severe Revenue Shortfall"
                else if [RevenueVariancePct] <= -0.10
                then
                    "Revenue Shortfall"
                else if [ExpenseVariancePct] >= 0.20
                then
                    "Severe Expense Overrun"
                else if [ExpenseVariancePct] >= 0.10
                then
                    "Expense Overrun"
                else if [GrossMarginRisk] >= 20
                then
                    "Gross Margin Pressure"
                else if [OperatingMarginRisk] >= 20
                then
                    "Operating Margin Pressure"
                else
                    null,
            type text
        ),


    // ========================================================
    // 40. RECOMMENDED ACTION
    // ========================================================

    AddRecommendedAction =
        Table.AddColumn(
            AddAlertType,
            "RecommendedAction",
            each
                if [FinancialRiskLevel] = "CRITICAL"
                then
                    "Immediate executive review and profitability recovery plan"
                else if [FinancialRiskLevel] = "HIGH"
                then
                    "Investigate profitability drivers and execute mitigation plan"
                else if [NetProfitStatus] = "Loss"
                then
                    "Perform root-cause analysis of net loss and establish recovery actions"
                else if [RevenueVariancePct] <= -0.20
                then
                    "Review sales pipeline, pricing, demand and revenue recovery actions"
                else if [ExpenseVariancePct] >= 0.20
                then
                    "Perform expense root-cause analysis and implement cost controls"
                else if [GrossMarginRisk] >= 20
                then
                    "Review pricing, product mix and cost-of-revenue drivers"
                else if [OperatingMarginRisk] >= 20
                then
                    "Review operating expenses and improve operating leverage"
                else
                    "Continue normal profitability monitoring",
            type text
        ),


    // ========================================================
    // 41. DATA QUALITY
    // ========================================================

    AddDateQuality =
        Table.AddColumn(
            AddRecommendedAction,
            "DateQuality",
            each
                if [Date] = null
                then
                    "Invalid"
                else
                    "Valid",
            type text
        ),


    AddIncomeStatementDataQuality =
        Table.AddColumn(
            AddDateQuality,
            "IncomeStatementDataQuality",
            each
                if [Revenue] = null
                    and [NetProfit] = null
                then
                    "Missing"
                else if [Revenue] = null
                    or [GrossProfit] = null
                    or [OperatingProfit] = null
                    or [NetProfit] = null
                then
                    "Incomplete"
                else
                    "Available",
            type text
        ),


    AddMarginQuality =
        Table.AddColumn(
            AddIncomeStatementDataQuality,
            "MarginQuality",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then
                    "Unavailable"
                else
                    "Available",
            type text
        ),


    AddDataQualityStatus =
        Table.AddColumn(
            AddMarginQuality,
            "DataQualityStatus",
            each
                if [DateQuality] = "Invalid"
                then
                    "Invalid"
                else if [IncomeStatementDataQuality] = "Missing"
                then
                    "Missing"
                else if [IncomeStatementDataQuality] = "Incomplete"
                then
                    "Needs Review"
                else
                    "Valid",
            type text
        ),


    // ========================================================
    // 42. DECISION PRIORITY
    // ========================================================

    AddDecisionPriority =
        Table.AddColumn(
            AddDataQualityStatus,
            "DecisionPriority",
            each
                if [FinancialRiskLevel] = "CRITICAL"
                    or [NetProfitStatus] = "Loss"
                then
                    "P0"
                else if [FinancialRiskLevel] = "HIGH"
                then
                    "P1"
                else if [FinancialRiskLevel] = "MEDIUM"
                then
                    "P2"
                else
                    "P3",
            type text
        ),


    // ========================================================
    // 43. FINANCIAL RECORD ID
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
    // 44. INCOME STATEMENT LABEL
    // ========================================================

    AddIncomeStatementLabel =
        Table.AddColumn(
            AddFinancialRecordID,
            "IncomeStatementLabel",
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
    // 45. MANAGEMENT STATUS
    // ========================================================

    AddManagementStatus =
        Table.AddColumn(
            AddIncomeStatementLabel,
            "ManagementStatus",
            each
                if [FinancialRiskLevel] = "CRITICAL"
                then
                    "Immediate Action"
                else if [FinancialRiskLevel] = "HIGH"
                then
                    "Management Review"
                else if [FinancialRiskLevel] = "MEDIUM"
                then
                    "Monitor"
                else
                    "Normal",
            type text
        ),


    // ========================================================
    // 46. PERFORMANCE CATEGORY
    // ========================================================

    AddPerformanceCategory =
        Table.AddColumn(
            AddManagementStatus,
            "PerformanceCategory",
            each
                if [NetProfitStatus] = "Loss"
                then
                    "Loss"
                else if [RevenueStatus] = "Severe Shortfall"
                then
                    "Revenue Risk"
                else if [ExpenseVariancePct] <> null
                    and [ExpenseVariancePct] >= 0.20
                then
                    "Cost Pressure"
                else if [GrossProfitStatus] = "Low Margin"
                then
                    "Gross Margin Pressure"
                else if [OperatingProfitStatus] = "Low Margin"
                then
                    "Operating Margin Pressure"
                else
                    "Healthy",
            type text
        ),


    // ========================================================
    // 47. FINAL SORT
    // ========================================================

    SortedFinal =
        Table.Sort(
            AddPerformanceCategory,
            {
                {"Date", Order.Ascending},
                {"Company", Order.Ascending},
                {"BusinessUnit", Order.Ascending}
            }
        ),


    // ========================================================
    // 48. REMOVE DUPLICATES
    // ========================================================

    Deduplicated =
        Table.Distinct(
            SortedFinal,
            {
                "FinancialRecordID"
            }
        ),


    // ========================================================
    // 49. FINAL BUFFER
    // ========================================================

    IncomeStatement =
        Table.Buffer(
            Deduplicated
        )

in
    IncomeStatement