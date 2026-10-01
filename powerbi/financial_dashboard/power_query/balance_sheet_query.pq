// ============================================================
// FINCO AI — BALANCE SHEET QUERY
// File: balance_sheet_query.pq
//
// PURPOSE
//   Balance-sheet transformation layer for Financial Dashboard.
//
// SOURCE
//   HistoricalFinancials
//
// OUTPUT
//   BalanceSheet
//
// EXPECTED SOURCE COLUMNS
//   Date
//   Company
//   BusinessUnit
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
                    "TotalAssets",
                    "CurrentAssets",
                    "CurrentLiabilities",
                    "TotalLiabilities",
                    "TotalEquity",
                    "CashAndEquivalents",
                    "AccountsReceivable",
                    "Inventory",
                    "AccountsPayable",
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
            "TotalAssets",
            "CurrentAssets",
            "CurrentLiabilities",
            "TotalLiabilities",
            "TotalEquity",
            "CashAndEquivalents",
            "AccountsReceivable",
            "Inventory",
            "AccountsPayable",
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

                {"TotalAssets", type number},
                {"CurrentAssets", type number},
                {"CurrentLiabilities", type number},
                {"TotalLiabilities", type number},
                {"TotalEquity", type number},

                {"CashAndEquivalents", type number},
                {"AccountsReceivable", type number},
                {"Inventory", type number},
                {"AccountsPayable", type number},
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
    // 7. CLEAN NUMERIC VALUES
    // ========================================================

    NumericColumns =
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
    // 8. DATE VALIDATION
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
    // 10. WORKING CAPITAL
    // ========================================================

    AddWorkingCapital =
        Table.AddColumn(
            AddYearMonthKey,
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


    // ========================================================
    // 11. NET WORKING CAPITAL
    // ========================================================

    AddNetWorkingCapital =
        Table.AddColumn(
            AddWorkingCapital,
            "NetWorkingCapital",
            each
                if [CurrentAssets] = null
                    or [CurrentLiabilities] = null
                then null
                else
                    [CurrentAssets]
                        - [CurrentLiabilities],
            type number
        ),


    // ========================================================
    // 12. CURRENT RATIO
    // ========================================================

    AddCurrentRatio =
        Table.AddColumn(
            AddNetWorkingCapital,
            "CurrentRatio",
            each
                if [CurrentLiabilities] = null
                    or [CurrentLiabilities] = 0
                    or [CurrentAssets] = null
                then null
                else
                    [CurrentAssets]
                        / [CurrentLiabilities],
            type number
        ),


    // ========================================================
    // 13. QUICK RATIO
    // ========================================================

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


    // ========================================================
    // 14. CASH RATIO
    // ========================================================

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


    // ========================================================
    // 15. DEBT TO ASSET
    // ========================================================

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


    // ========================================================
    // 16. DEBT TO EQUITY
    // ========================================================

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


    // ========================================================
    // 17. LIABILITIES TO ASSETS
    // ========================================================

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


    // ========================================================
    // 18. EQUITY TO ASSETS
    // ========================================================

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
    // 19. CASH TO ASSETS
    // ========================================================

    AddCashToAsset =
        Table.AddColumn(
            AddEquityToAsset,
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
    // 20. RECEIVABLES TO ASSETS
    // ========================================================

    AddReceivablesToAsset =
        Table.AddColumn(
            AddCashToAsset,
            "ReceivablesToAssetRatio",
            each
                if [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else
                    [AccountsReceivable]
                        / [TotalAssets],
            type number
        ),


    // ========================================================
    // 21. INVENTORY TO ASSETS
    // ========================================================

    AddInventoryToAsset =
        Table.AddColumn(
            AddReceivablesToAsset,
            "InventoryToAssetRatio",
            each
                if [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else
                    [Inventory]
                        / [TotalAssets],
            type number
        ),


    // ========================================================
    // 22. WORKING CAPITAL TO ASSETS
    // ========================================================

    AddWorkingCapitalToAsset =
        Table.AddColumn(
            AddInventoryToAsset,
            "WorkingCapitalToAssetRatio",
            each
                if [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else
                    [WorkingCapital]
                        / [TotalAssets],
            type number
        ),


    // ========================================================
    // 23. BALANCE SHEET EQUATION
    // Assets = Liabilities + Equity
    // ========================================================

    AddBalanceDifference =
        Table.AddColumn(
            AddWorkingCapitalToAsset,
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


    // ========================================================
    // 24. BALANCE SHEET CHECK
    // ========================================================

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
    // 25. LIQUIDITY STATUS
    // ========================================================

    AddLiquidityStatus =
        Table.AddColumn(
            AddBalanceSheetCheck,
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
    // 26. LEVERAGE STATUS
    // ========================================================

    AddLeverageStatus =
        Table.AddColumn(
            AddLiquidityStatus,
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
    // 27. WORKING CAPITAL STATUS
    // ========================================================

    AddWorkingCapitalStatus =
        Table.AddColumn(
            AddLeverageStatus,
            "WorkingCapitalStatus",
            each
                if [WorkingCapital] = null
                then "Unknown"
                else if [WorkingCapital] < 0
                then "Negative"
                else
                    "Positive",
            type text
        ),


    // ========================================================
    // 28. ASSET QUALITY STATUS
    // ========================================================

    AddAssetQualityStatus =
        Table.AddColumn(
            AddWorkingCapitalStatus,
            "AssetQualityStatus",
            each
                if [TotalAssets] = null
                then "Missing"
                else if [TotalAssets] < 0
                then "Invalid"
                else
                    "Valid",
            type text
        ),


    // ========================================================
    // 29. LIQUIDITY RISK
    // ========================================================

    AddLiquidityRisk =
        Table.AddColumn(
            AddAssetQualityStatus,
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
    // 30. LEVERAGE RISK
    // ========================================================

    AddLeverageRisk =
        Table.AddColumn(
            AddLiquidityRisk,
            "LeverageRisk",
            each
                if [DebtToEquity] = null
                then 10
                else if [DebtToEquity] >= 2
                then 40
                else if [DebtToEquity] >= 1
                then 20
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 31. WORKING CAPITAL RISK
    // ========================================================

    AddWorkingCapitalRisk =
        Table.AddColumn(
            AddLeverageRisk,
            "WorkingCapitalRisk",
            each
                if [WorkingCapital] = null
                then 15
                else if [WorkingCapital] < 0
                then 30
                else
                    0,
            Int64.Type
        ),


    // ========================================================
    // 32. BALANCE SHEET INTEGRITY RISK
    // ========================================================

    AddBalanceIntegrityRisk =
        Table.AddColumn(
            AddWorkingCapitalRisk,
            "BalanceIntegrityRisk",
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
    // 33. OVERALL BALANCE SHEET RISK
    // ========================================================

    AddBalanceSheetRiskScore =
        Table.AddColumn(
            AddBalanceIntegrityRisk,
            "BalanceSheetRiskScore",
            each
                Number.Min(
                    100,
                    [LiquidityRisk]
                    + [LeverageRisk]
                    + [WorkingCapitalRisk]
                    + [BalanceIntegrityRisk]
                ),
            type number
        ),


    // ========================================================
    // 34. RISK LEVEL
    // ========================================================

    AddBalanceSheetRiskLevel =
        Table.AddColumn(
            AddBalanceSheetRiskScore,
            "BalanceSheetRiskLevel",
            each
                if [BalanceSheetRiskScore] >= 75
                then "CRITICAL"
                else if [BalanceSheetRiskScore] >= 50
                then "HIGH"
                else if [BalanceSheetRiskScore] >= 25
                then "MEDIUM"
                else
                    "LOW",
            type text
        ),


    // ========================================================
    // 35. HEALTH SCORE
    // ========================================================

    AddBalanceSheetHealthScore =
        Table.AddColumn(
            AddBalanceSheetRiskLevel,
            "BalanceSheetHealthScore",
            each
                Number.Max(
                    0,
                    100 - [BalanceSheetRiskScore]
                ),
            type number
        ),


    // ========================================================
    // 36. HEALTH STATUS
    // ========================================================

    AddBalanceSheetHealthStatus =
        Table.AddColumn(
            AddBalanceSheetHealthScore,
            "BalanceSheetHealthStatus",
            each
                if [BalanceSheetHealthScore] >= 80
                then "Excellent"
                else if [BalanceSheetHealthScore] >= 65
                then "Healthy"
                else if [BalanceSheetHealthScore] >= 50
                then "Watch"
                else if [BalanceSheetHealthScore] >= 30
                then "Weak"
                else
                    "Critical",
            type text
        ),


    // ========================================================
    // 37. ALERT FLAG
    // ========================================================

    AddAlertFlag =
        Table.AddColumn(
            AddBalanceSheetHealthStatus,
            "AlertFlag",
            each
                [BalanceSheetRiskLevel] = "CRITICAL"
                or [BalanceSheetRiskLevel] = "HIGH"
                or [BalanceSheetCheck] = "Out of Balance",
            type logical
        ),


    // ========================================================
    // 38. ALERT TYPE
    // ========================================================

    AddAlertType =
        Table.AddColumn(
            AddAlertFlag,
            "AlertType",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then
                    "Balance Sheet Integrity Failure"
                else if [BalanceSheetRiskLevel] = "CRITICAL"
                then
                    "Critical Balance Sheet Risk"
                else if [BalanceSheetRiskLevel] = "HIGH"
                then
                    "High Balance Sheet Risk"
                else if [LiquidityStatus] = "Critical"
                then
                    "Liquidity Risk"
                else if [LeverageStatus] = "Critical"
                then
                    "Leverage Risk"
                else if [WorkingCapitalStatus] = "Negative"
                then
                    "Negative Working Capital"
                else
                    null,
            type text
        ),


    // ========================================================
    // 39. RECOMMENDED ACTION
    // ========================================================

    AddRecommendedAction =
        Table.AddColumn(
            AddAlertType,
            "RecommendedAction",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then
                    "Investigate balance-sheet reconciliation and source-data integrity immediately"
                else if [BalanceSheetRiskLevel] = "CRITICAL"
                then
                    "Immediate management review of liquidity, leverage and capital structure"
                else if [BalanceSheetRiskLevel] = "HIGH"
                then
                    "Investigate balance-sheet risk drivers and execute mitigation plan"
                else if [LiquidityStatus] = "Critical"
                then
                    "Improve liquidity and review short-term funding requirements"
                else if [LeverageStatus] = "Critical"
                then
                    "Review debt structure, refinancing options and capital allocation"
                else if [WorkingCapitalStatus] = "Negative"
                then
                    "Review receivables, inventory, payables and working-capital cycle"
                else
                    "Continue normal balance-sheet monitoring",
            type text
        ),


    // ========================================================
    // 40. DATA QUALITY
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


    AddBalanceSheetDataQuality =
        Table.AddColumn(
            AddDateQuality,
            "BalanceSheetDataQuality",
            each
                if [TotalAssets] = null
                    and [TotalLiabilities] = null
                    and [TotalEquity] = null
                then "Missing"
                else if
                    [TotalAssets] < 0
                    or [TotalLiabilities] < 0
                then
                    "Needs Review"
                else
                    "Available",
            type text
        ),


    AddRatioQuality =
        Table.AddColumn(
            AddBalanceSheetDataQuality,
            "RatioQuality",
            each
                if [CurrentRatio] = null
                    and [DebtToEquity] = null
                then "Incomplete"
                else
                    "Available",
            type text
        ),


    AddDataQualityStatus =
        Table.AddColumn(
            AddRatioQuality,
            "DataQualityStatus",
            each
                if [DateQuality] = "Invalid"
                then "Invalid"
                else if [BalanceSheetDataQuality] = "Missing"
                then "Incomplete"
                else if [BalanceSheetDataQuality] = "Needs Review"
                then "Needs Review"
                else
                    "Valid",
            type text
        ),


    // ========================================================
    // 41. DECISION PRIORITY
    // ========================================================

    AddDecisionPriority =
        Table.AddColumn(
            AddDataQualityStatus,
            "DecisionPriority",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                    or [BalanceSheetRiskLevel] = "CRITICAL"
                then "P0"
                else if [BalanceSheetRiskLevel] = "HIGH"
                then "P1"
                else if [BalanceSheetRiskLevel] = "MEDIUM"
                then "P2"
                else
                    "P3",
            type text
        ),


    // ========================================================
    // 42. BALANCE SHEET RECORD ID
    // ========================================================

    AddRecordID =
        Table.AddColumn(
            AddDecisionPriority,
            "BalanceSheetRecordID",
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
    // 43. DISPLAY LABEL
    // ========================================================

    AddDisplayLabel =
        Table.AddColumn(
            AddRecordID,
            "BalanceSheetLabel",
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
    // 44. MANAGEMENT STATUS
    // ========================================================

    AddManagementStatus =
        Table.AddColumn(
            AddDisplayLabel,
            "ManagementStatus",
            each
                if [BalanceSheetCheck] = "Out of Balance"
                then "Immediate Action"
                else if [BalanceSheetRiskLevel] = "CRITICAL"
                then "Immediate Action"
                else if [BalanceSheetRiskLevel] = "HIGH"
                then "Management Review"
                else if [BalanceSheetRiskLevel] = "MEDIUM"
                then "Monitor"
                else
                    "Normal",
            type text
        ),


    // ========================================================
    // 45. SORT
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
    // 46. REMOVE DUPLICATES
    // ========================================================

    Deduplicated =
        Table.Distinct(
            Sorted,
            {
                "BalanceSheetRecordID"
            }
        ),


    // ========================================================
    // 47. BUFFER
    // ========================================================

    Final =
        Table.Buffer(
            Deduplicated
        )

in
    Final