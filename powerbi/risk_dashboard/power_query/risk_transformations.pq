let
    // ============================================================
    // FinCo AI - Risk Transformations Query
    // File:
    //   powerbi/risk_dashboard/risk_transformations.pq
    //
    // Purpose:
    //   Normalize financial data and create standardized risk
    //   indicators used by:
    //
    //       risk_factors_query.pq
    //              ↓
    //       risk_scores_query.pq
    //
    // Expected source:
    //   FinancialMetrics
    //
    // Main responsibilities:
    //   - Data type normalization
    //   - Null / invalid value handling
    //   - Period normalization
    //   - Financial metric derivation
    //   - Growth calculations
    //   - Margin calculations
    //   - Liquidity calculations
    //   - Leverage calculations
    //   - Cash-flow calculations
    //   - Budget variance calculations
    //   - Financial health indicators
    // ============================================================


    // ------------------------------------------------------------
    // 1. SOURCE
    // ------------------------------------------------------------
    Source =
        try FinancialMetrics
        otherwise #table(
            {
                "CompanyId",
                "CompanyName",
                "Period",
                "Revenue",
                "PreviousRevenue",
                "NetProfit",
                "PreviousNetProfit",
                "GrossProfit",
                "PreviousGrossProfit",
                "Cash",
                "CurrentAssets",
                "CurrentLiabilities",
                "TotalDebt",
                "TotalAssets",
                "TotalEquity",
                "OperatingCashFlow",
                "BudgetRevenue",
                "BudgetProfit"
            },
            {}
        ),


    // ------------------------------------------------------------
    // 2. STANDARDIZE COLUMN NAMES
    // ------------------------------------------------------------
    StandardNames =
        Table.TransformColumnNames(
            Source,
            each Text.Trim(_)
        ),


    // ------------------------------------------------------------
    // 3. TYPE CONVERSION
    // ------------------------------------------------------------
    Typed =
        Table.TransformColumnTypes(
            StandardNames,
            {
                {"CompanyId", type text},
                {"CompanyName", type text},
                {"Period", type date},

                {"Revenue", type number},
                {"PreviousRevenue", type number},

                {"NetProfit", type number},
                {"PreviousNetProfit", type number},

                {"GrossProfit", type number},
                {"PreviousGrossProfit", type number},

                {"Cash", type number},
                {"CurrentAssets", type number},
                {"CurrentLiabilities", type number},

                {"TotalDebt", type number},
                {"TotalAssets", type number},
                {"TotalEquity", type number},

                {"OperatingCashFlow", type number},

                {"BudgetRevenue", type number},
                {"BudgetProfit", type number}
            },
            "en-US"
        ),


    // ------------------------------------------------------------
    // 4. TEXT CLEANUP
    // ------------------------------------------------------------
    CleanCompanyId =
        Table.TransformColumns(
            Typed,
            {
                {
                    "CompanyId",
                    each if _ = null then null else Text.Trim(_),
                    type text
                },
                {
                    "CompanyName",
                    each if _ = null then null else Text.Trim(_),
                    type text
                }
            }
        ),


    // ------------------------------------------------------------
    // 5. REPLACE INVALID NUMERIC VALUES
    // ------------------------------------------------------------
    ReplaceErrors =
        Table.ReplaceErrorValues(
            CleanCompanyId,
            {
                {"Revenue", null},
                {"PreviousRevenue", null},
                {"NetProfit", null},
                {"PreviousNetProfit", null},
                {"GrossProfit", null},
                {"PreviousGrossProfit", null},
                {"Cash", null},
                {"CurrentAssets", null},
                {"CurrentLiabilities", null},
                {"TotalDebt", null},
                {"TotalAssets", null},
                {"TotalEquity", null},
                {"OperatingCashFlow", null},
                {"BudgetRevenue", null},
                {"BudgetProfit", null}
            }
        ),


    // ------------------------------------------------------------
    // 6. SAFE DIVISION FUNCTION
    // ------------------------------------------------------------
    SafeDivide =
        (numerator as nullable number, denominator as nullable number)
        as nullable number =>
            if numerator = null
                or denominator = null
                or denominator = 0
            then
                null
            else
                numerator / denominator,


    // ------------------------------------------------------------
    // 7. SAFE GROWTH FUNCTION
    // ------------------------------------------------------------
    GrowthRate =
        (current as nullable number, previous as nullable number)
        as nullable number =>
            if current = null
                or previous = null
                or previous = 0
            then
                null
            else
                (current - previous) / Number.Abs(previous),


    // ------------------------------------------------------------
    // 8. PERIOD ATTRIBUTES
    // ------------------------------------------------------------
    AddYear =
        Table.AddColumn(
            ReplaceErrors,
            "RiskYear",
            each
                if [Period] = null
                then null
                else Date.Year([Period]),
            Int64.Type
        ),

    AddMonth =
        Table.AddColumn(
            AddYear,
            "RiskMonth",
            each
                if [Period] = null
                then null
                else Date.Month([Period]),
            Int64.Type
        ),

    AddQuarter =
        Table.AddColumn(
            AddMonth,
            "RiskQuarter",
            each
                if [Period] = null
                then null
                else "Q" & Text.From(Date.QuarterOfYear([Period])),
            type text
        ),

    AddPeriodKey =
        Table.AddColumn(
            AddQuarter,
            "RiskPeriodKey",
            each
                if [Period] = null
                then null
                else Date.ToText([Period], "yyyy-MM"),
            type text
        ),


    // ------------------------------------------------------------
    // 9. REVENUE TRANSFORMATIONS
    // ------------------------------------------------------------
    AddRevenueGrowth =
        Table.AddColumn(
            AddPeriodKey,
            "RevenueGrowthPct",
            each GrowthRate(
                [Revenue],
                [PreviousRevenue]
            ),
            type number
        ),

    AddRevenueDecline =
        Table.AddColumn(
            AddRevenueGrowth,
            "RevenueDeclinePct",
            each
                if [RevenueGrowthPct] = null
                then null
                else if [RevenueGrowthPct] < 0
                then Number.Abs([RevenueGrowthPct])
                else 0,
            type number
        ),


    // ------------------------------------------------------------
    // 10. PROFIT TRANSFORMATIONS
    // ------------------------------------------------------------
    AddProfitGrowth =
        Table.AddColumn(
            AddRevenueDecline,
            "ProfitGrowthPct",
            each GrowthRate(
                [NetProfit],
                [PreviousNetProfit]
            ),
            type number
        ),

    AddProfitDecline =
        Table.AddColumn(
            AddProfitGrowth,
            "ProfitDeclinePct",
            each
                if [ProfitGrowthPct] = null
                then null
                else if [ProfitGrowthPct] < 0
                then Number.Abs([ProfitGrowthPct])
                else 0,
            type number
        ),


    // ------------------------------------------------------------
    // 11. GROSS PROFIT TRANSFORMATIONS
    // ------------------------------------------------------------
    AddGrossProfitGrowth =
        Table.AddColumn(
            AddProfitDecline,
            "GrossProfitGrowthPct",
            each GrowthRate(
                [GrossProfit],
                [PreviousGrossProfit]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 12. NET PROFIT MARGIN
    // ------------------------------------------------------------
    AddNetMargin =
        Table.AddColumn(
            AddGrossProfitGrowth,
            "NetProfitMargin",
            each SafeDivide(
                [NetProfit],
                [Revenue]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 13. GROSS MARGIN
    // ------------------------------------------------------------
    AddGrossMargin =
        Table.AddColumn(
            AddNetMargin,
            "GrossMargin",
            each SafeDivide(
                [GrossProfit],
                [Revenue]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 14. OPERATING CASH FLOW MARGIN
    // ------------------------------------------------------------
    AddOperatingCashFlowMargin =
        Table.AddColumn(
            AddGrossMargin,
            "OperatingCashFlowMargin",
            each SafeDivide(
                [OperatingCashFlow],
                [Revenue]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 15. CURRENT RATIO
    // ------------------------------------------------------------
    AddCurrentRatio =
        Table.AddColumn(
            AddOperatingCashFlowMargin,
            "CurrentRatio",
            each SafeDivide(
                [CurrentAssets],
                [CurrentLiabilities]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 16. CASH RATIO
    // ------------------------------------------------------------
    AddCashRatio =
        Table.AddColumn(
            AddCurrentRatio,
            "CashRatio",
            each SafeDivide(
                [Cash],
                [CurrentLiabilities]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 17. DEBT TO EQUITY
    // ------------------------------------------------------------
    AddDebtToEquity =
        Table.AddColumn(
            AddCashRatio,
            "DebtToEquity",
            each SafeDivide(
                [TotalDebt],
                [TotalEquity]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 18. DEBT TO ASSETS
    // ------------------------------------------------------------
    AddDebtToAssets =
        Table.AddColumn(
            AddDebtToEquity,
            "DebtToAssets",
            each SafeDivide(
                [TotalDebt],
                [TotalAssets]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 19. EQUITY RATIO
    // ------------------------------------------------------------
    AddEquityRatio =
        Table.AddColumn(
            AddDebtToAssets,
            "EquityRatio",
            each SafeDivide(
                [TotalEquity],
                [TotalAssets]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 20. CASH / DEBT
    // ------------------------------------------------------------
    AddCashToDebt =
        Table.AddColumn(
            AddEquityRatio,
            "CashToDebt",
            each SafeDivide(
                [Cash],
                [TotalDebt]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 21. CASH / CURRENT LIABILITIES
    // ------------------------------------------------------------
    AddCashToCurrentLiabilities =
        Table.AddColumn(
            AddCashToDebt,
            "CashToCurrentLiabilities",
            each SafeDivide(
                [Cash],
                [CurrentLiabilities]
            ),
            type number
        ),


    // ------------------------------------------------------------
    // 22. REVENUE BUDGET VARIANCE
    // ------------------------------------------------------------
    AddRevenueBudgetVariance =
        Table.AddColumn(
            AddCashToCurrentLiabilities,
            "RevenueBudgetVariancePct",
            each
                if [BudgetRevenue] = null
                    or [BudgetRevenue] = 0
                then null
                else
                    ([Revenue] - [BudgetRevenue])
                    / Number.Abs([BudgetRevenue]),
            type number
        ),


    // ------------------------------------------------------------
    // 23. PROFIT BUDGET VARIANCE
    // ------------------------------------------------------------
    AddProfitBudgetVariance =
        Table.AddColumn(
            AddRevenueBudgetVariance,
            "ProfitBudgetVariancePct",
            each
                if [BudgetProfit] = null
                    or [BudgetProfit] = 0
                then null
                else
                    ([NetProfit] - [BudgetProfit])
                    / Number.Abs([BudgetProfit]),
            type number
        ),


    // ------------------------------------------------------------
    // 24. LOSS FLAG
    // ------------------------------------------------------------
    AddLossFlag =
        Table.AddColumn(
            AddProfitBudgetVariance,
            "LossFlag",
            each
                if [NetProfit] = null
                then false
                else [NetProfit] < 0,
            type logical
        ),


    // ------------------------------------------------------------
    // 25. REVENUE DECLINE FLAG
    // ------------------------------------------------------------
    AddRevenueDeclineFlag =
        Table.AddColumn(
            AddLossFlag,
            "RevenueDeclineFlag",
            each
                if [RevenueGrowthPct] = null
                then false
                else [RevenueGrowthPct] <= -0.10,
            type logical
        ),


    // ------------------------------------------------------------
    // 26. PROFIT DECLINE FLAG
    // ------------------------------------------------------------
    AddProfitDeclineFlag =
        Table.AddColumn(
            AddRevenueDeclineFlag,
            "ProfitDeclineFlag",
            each
                if [ProfitGrowthPct] = null
                then false
                else [ProfitGrowthPct] <= -0.15,
            type logical
        ),


    // ------------------------------------------------------------
    // 27. LIQUIDITY PRESSURE FLAG
    // ------------------------------------------------------------
    AddLiquidityPressure =
        Table.AddColumn(
            AddProfitDeclineFlag,
            "LiquidityPressureFlag",
            each
                if [CurrentRatio] = null
                then false
                else [CurrentRatio] < 1.00,
            type logical
        ),


    // ------------------------------------------------------------
    // 28. HIGH LEVERAGE FLAG
    // ------------------------------------------------------------
    AddHighLeverage =
        Table.AddColumn(
            AddLiquidityPressure,
            "HighLeverageFlag",
            each
                if [DebtToEquity] = null
                then false
                else [DebtToEquity] >= 2.00,
            type logical
        ),


    // ------------------------------------------------------------
    // 29. NEGATIVE CASH FLOW FLAG
    // ------------------------------------------------------------
    AddNegativeCashFlow =
        Table.AddColumn(
            AddHighLeverage,
            "NegativeCashFlowFlag",
            each
                if [OperatingCashFlow] = null
                then false
                else [OperatingCashFlow] < 0,
            type logical
        ),


    // ------------------------------------------------------------
    // 30. NEGATIVE MARGIN FLAG
    // ------------------------------------------------------------
    AddNegativeMargin =
        Table.AddColumn(
            AddNegativeCashFlow,
            "NegativeMarginFlag",
            each
                if [NetProfitMargin] = null
                then false
                else [NetProfitMargin] < 0,
            type logical
        ),


    // ------------------------------------------------------------
    // 31. SEVERE LOSS FLAG
    // ------------------------------------------------------------
    AddSevereLossFlag =
        Table.AddColumn(
            AddNegativeMargin,
            "SevereLossRiskFlag",
            each
                (
                    [LossFlag] = true
                    and
                    (
                        [ProfitDeclineFlag] = true
                        or [LiquidityPressureFlag] = true
                        or [NegativeCashFlowFlag] = true
                    )
                ),
            type logical
        ),


    // ------------------------------------------------------------
    // 32. FINANCIAL PROFILE FLAG
    // ------------------------------------------------------------
    AddFinancialProfileFlag =
        Table.AddColumn(
            AddSevereLossFlag,
            "FinancialProfileDownFlag",
            each
                (
                    [RevenueDeclineFlag] = true
                    or [ProfitDeclineFlag] = true
                    or [LiquidityPressureFlag] = true
                    or [HighLeverageFlag] = true
                    or [NegativeCashFlowFlag] = true
                    or [NegativeMarginFlag] = true
                ),
            type logical
        ),


    // ------------------------------------------------------------
    // 33. RISK FACTOR COUNT
    // ------------------------------------------------------------
    AddRiskFactorCount =
        Table.AddColumn(
            AddFinancialProfileFlag,
            "RiskFactorCount",
            each
                List.Count(
                    List.Select(
                        {
                            [RevenueDeclineFlag],
                            [ProfitDeclineFlag],
                            [LiquidityPressureFlag],
                            [HighLeverageFlag],
                            [NegativeCashFlowFlag],
                            [NegativeMarginFlag]
                        },
                        each _ = true
                    )
                ),
            Int64.Type
        ),


    // ------------------------------------------------------------
    // 34. DATA COMPLETENESS
    // ------------------------------------------------------------
    AddDataCompleteness =
        Table.AddColumn(
            AddRiskFactorCount,
            "RiskDataCompleteness",
            each
                let
                    Fields = {
                        [Revenue],
                        [PreviousRevenue],
                        [NetProfit],
                        [PreviousNetProfit],
                        [GrossProfit],
                        [Cash],
                        [CurrentAssets],
                        [CurrentLiabilities],
                        [TotalDebt],
                        [TotalAssets],
                        [TotalEquity],
                        [OperatingCashFlow]
                    },

                    Available =
                        List.Count(
                            List.RemoveNulls(Fields)
                        ),

                    Total =
                        List.Count(Fields)
                in
                    Number.Round(
                        Available / Total,
                        4
                    ),
            type number
        ),


    // ------------------------------------------------------------
    // 35. DATA QUALITY BAND
    // ------------------------------------------------------------
    AddDataQualityBand =
        Table.AddColumn(
            AddDataCompleteness,
            "RiskDataQuality",
            each
                if [RiskDataCompleteness] >= 0.90 then
                    "HIGH"
                else if [RiskDataCompleteness] >= 0.70 then
                    "MEDIUM"
                else if [RiskDataCompleteness] >= 0.50 then
                    "LOW"
                else
                    "INSUFFICIENT",
            type text
        ),


    // ------------------------------------------------------------
    // 36. LIQUIDITY STATUS
    // ------------------------------------------------------------
    AddLiquidityStatus =
        Table.AddColumn(
            AddDataQualityBand,
            "LiquidityStatus",
            each
                if [CurrentRatio] = null then
                    "UNKNOWN"
                else if [CurrentRatio] < 0.75 then
                    "CRITICAL"
                else if [CurrentRatio] < 1.00 then
                    "HIGH_RISK"
                else if [CurrentRatio] < 1.25 then
                    "WATCH"
                else
                    "HEALTHY",
            type text
        ),


    // ------------------------------------------------------------
    // 37. LEVERAGE STATUS
    // ------------------------------------------------------------
    AddLeverageStatus =
        Table.AddColumn(
            AddLiquidityStatus,
            "LeverageStatus",
            each
                if [DebtToEquity] = null then
                    "UNKNOWN"
                else if [DebtToEquity] >= 3.00 then
                    "CRITICAL"
                else if [DebtToEquity] >= 2.00 then
                    "HIGH_RISK"
                else if [DebtToEquity] >= 1.00 then
                    "WATCH"
                else
                    "HEALTHY",
            type text
        ),


    // ------------------------------------------------------------
    // 38. PROFITABILITY STATUS
    // ------------------------------------------------------------
    AddProfitabilityStatus =
        Table.AddColumn(
            AddLeverageStatus,
            "ProfitabilityStatus",
            each
                if [NetProfitMargin] = null then
                    "UNKNOWN"
                else if [NetProfitMargin] < -0.10 then
                    "CRITICAL"
                else if [NetProfitMargin] < 0 then
                    "LOSS"
                else if [NetProfitMargin] < 0.05 then
                    "LOW_MARGIN"
                else if [NetProfitMargin] < 0.10 then
                    "WATCH"
                else
                    "HEALTHY",
            type text
        ),


    // ------------------------------------------------------------
    // 39. CASH FLOW STATUS
    // ------------------------------------------------------------
    AddCashFlowStatus =
        Table.AddColumn(
            AddProfitabilityStatus,
            "CashFlowStatus",
            each
                if [OperatingCashFlow] = null then
                    "UNKNOWN"
                else if [OperatingCashFlow] < 0
                    and [CashToCurrentLiabilities] <> null
                    and [CashToCurrentLiabilities] < 0.50 then
                    "CRITICAL"
                else if [OperatingCashFlow] < 0 then
                    "HIGH_RISK"
                else if [CashToCurrentLiabilities] <> null
                    and [CashToCurrentLiabilities] < 1.00 then
                    "WATCH"
                else
                    "HEALTHY",
            type text
        ),


    // ------------------------------------------------------------
    // 40. OVERALL TRANSFORMATION STATUS
    // ------------------------------------------------------------
    AddTransformationStatus =
        Table.AddColumn(
            AddCashFlowStatus,
            "TransformationStatus",
            each
                if [RiskDataQuality] = "INSUFFICIENT" then
                    "DATA_QUALITY_REVIEW"
                else if [SevereLossRiskFlag] = true then
                    "SEVERE_RISK"
                else if [FinancialProfileDownFlag] = true then
                    "RISK_DETECTED"
                else
                    "NORMAL",
            type text
        ),


    // ------------------------------------------------------------
    // 41. ROUND NUMERIC DERIVED VALUES
    // ------------------------------------------------------------
    Rounded =
        Table.TransformColumns(
            AddTransformationStatus,
            {
                {
                    "RevenueGrowthPct",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "RevenueDeclinePct",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "ProfitGrowthPct",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "ProfitDeclinePct",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "GrossProfitGrowthPct",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "NetProfitMargin",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "GrossMargin",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "OperatingCashFlowMargin",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "CurrentRatio",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "CashRatio",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "DebtToEquity",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "DebtToAssets",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "EquityRatio",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "CashToDebt",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "CashToCurrentLiabilities",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "RevenueBudgetVariancePct",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "ProfitBudgetVariancePct",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                },
                {
                    "RiskDataCompleteness",
                    each if _ = null then null else Number.Round(_, 4),
                    type number
                }
            }
        ),


    // ------------------------------------------------------------
    // 42. FINAL OUTPUT
    // ------------------------------------------------------------
    Final =
        Table.SelectColumns(
            Rounded,
            {
                "CompanyId",
                "CompanyName",
                "Period",

                "RiskYear",
                "RiskMonth",
                "RiskQuarter",
                "RiskPeriodKey",

                "Revenue",
                "PreviousRevenue",
                "RevenueGrowthPct",
                "RevenueDeclinePct",

                "NetProfit",
                "PreviousNetProfit",
                "ProfitGrowthPct",
                "ProfitDeclinePct",

                "GrossProfit",
                "PreviousGrossProfit",
                "GrossProfitGrowthPct",

                "NetProfitMargin",
                "GrossMargin",
                "OperatingCashFlowMargin",

                "Cash",
                "CurrentAssets",
                "CurrentLiabilities",
                "CurrentRatio",
                "CashRatio",
                "CashToCurrentLiabilities",

                "TotalDebt",
                "TotalAssets",
                "TotalEquity",
                "DebtToEquity",
                "DebtToAssets",
                "EquityRatio",
                "CashToDebt",

                "OperatingCashFlow",

                "BudgetRevenue",
                "BudgetProfit",
                "RevenueBudgetVariancePct",
                "ProfitBudgetVariancePct",

                "LossFlag",
                "RevenueDeclineFlag",
                "ProfitDeclineFlag",
                "LiquidityPressureFlag",
                "HighLeverageFlag",
                "NegativeCashFlowFlag",
                "NegativeMarginFlag",
                "SevereLossRiskFlag",
                "FinancialProfileDownFlag",

                "RiskFactorCount",

                "RiskDataCompleteness",
                "RiskDataQuality",

                "LiquidityStatus",
                "LeverageStatus",
                "ProfitabilityStatus",
                "CashFlowStatus",

                "TransformationStatus"
            }
        )

in
    Final