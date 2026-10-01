let
    // ============================================================
    // FinCo AI - Risk Factors Query
    // File: powerbi/risk_dashboard/risk_factors_query.pq
    //
    // Purpose:
    //   Produces normalized risk factors for the Risk Dashboard.
    //
    // Expected source:
    //   FinancialMetrics table/query
    //
    // Expected columns:
    //   CompanyId
    //   CompanyName
    //   Period
    //   Revenue
    //   PreviousRevenue
    //   NetProfit
    //   PreviousNetProfit
    //   GrossProfit
    //   PreviousGrossProfit
    //   Cash
    //   CurrentAssets
    //   CurrentLiabilities
    //   TotalDebt
    //   TotalAssets
    //   TotalEquity
    //   OperatingCashFlow
    //   BudgetRevenue
    //   BudgetProfit
    //
    // Output:
    //   One row per company / period / risk factor.
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
    // 2. TYPE NORMALIZATION
    // ------------------------------------------------------------
    TypedSource =
        Table.TransformColumnTypes(
            Source,
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
    // 3. SAFE PERCENTAGE FUNCTION
    // ------------------------------------------------------------
    SafePct =
        (current as nullable number, previous as nullable number) as nullable number =>
            if current = null or previous = null or previous = 0
            then null
            else (current - previous) / Number.Abs(previous),


    // ------------------------------------------------------------
    // 4. DERIVED FINANCIAL METRICS
    // ------------------------------------------------------------
    AddRevenueChange =
        Table.AddColumn(
            TypedSource,
            "RevenueChangePct",
            each SafePct([Revenue], [PreviousRevenue]),
            type number
        ),

    AddProfitChange =
        Table.AddColumn(
            AddRevenueChange,
            "ProfitChangePct",
            each SafePct([NetProfit], [PreviousNetProfit]),
            type number
        ),

    AddGrossProfitChange =
        Table.AddColumn(
            AddProfitChange,
            "GrossProfitChangePct",
            each SafePct([GrossProfit], [PreviousGrossProfit]),
            type number
        ),

    AddCurrentRatio =
        Table.AddColumn(
            AddGrossProfitChange,
            "CurrentRatio",
            each
                if [CurrentLiabilities] = null
                    or [CurrentLiabilities] = 0
                then null
                else [CurrentAssets] / [CurrentLiabilities],
            type number
        ),

    AddDebtToEquity =
        Table.AddColumn(
            AddCurrentRatio,
            "DebtToEquity",
            each
                if [TotalEquity] = null
                    or [TotalEquity] = 0
                then null
                else [TotalDebt] / [TotalEquity],
            type number
        ),

    AddDebtToAssets =
        Table.AddColumn(
            AddDebtToEquity,
            "DebtToAssets",
            each
                if [TotalAssets] = null
                    or [TotalAssets] = 0
                then null
                else [TotalDebt] / [TotalAssets],
            type number
        ),

    AddProfitMargin =
        Table.AddColumn(
            AddDebtToAssets,
            "NetProfitMargin",
            each
                if [Revenue] = null
                    or [Revenue] = 0
                then null
                else [NetProfit] / [Revenue],
            type number
        ),

    AddCashToLiabilities =
        Table.AddColumn(
            AddProfitMargin,
            "CashToCurrentLiabilities",
            each
                if [CurrentLiabilities] = null
                    or [CurrentLiabilities] = 0
                then null
                else [Cash] / [CurrentLiabilities],
            type number
        ),


    // ------------------------------------------------------------
    // 5. BUDGET VARIANCE
    // ------------------------------------------------------------
    AddRevenueBudgetVariance =
        Table.AddColumn(
            AddCashToLiabilities,
            "RevenueBudgetVariancePct",
            each SafePct([Revenue], [BudgetRevenue]),
            type number
        ),

    AddProfitBudgetVariance =
        Table.AddColumn(
            AddRevenueBudgetVariance,
            "ProfitBudgetVariancePct",
            each SafePct([NetProfit], [BudgetProfit]),
            type number
        ),


    // ------------------------------------------------------------
    // 6. RISK FACTOR TABLE
    // ------------------------------------------------------------

    // Revenue decline
    RevenueRisk =
        Table.AddColumn(
            AddProfitBudgetVariance,
            "RevenueRisk",
            each
                if [RevenueChangePct] = null then "UNKNOWN"
                else if [RevenueChangePct] <= -0.20 then "CRITICAL"
                else if [RevenueChangePct] <= -0.10 then "HIGH"
                else if [RevenueChangePct] <= -0.05 then "MEDIUM"
                else "LOW",
            type text
        ),

    // Profit decline
    ProfitRisk =
        Table.AddColumn(
            RevenueRisk,
            "ProfitRisk",
            each
                if [ProfitChangePct] = null then "UNKNOWN"
                else if [ProfitChangePct] <= -0.30 then "CRITICAL"
                else if [ProfitChangePct] <= -0.15 then "HIGH"
                else if [ProfitChangePct] <= -0.05 then "MEDIUM"
                else "LOW",
            type text
        ),

    // Liquidity risk
    LiquidityRisk =
        Table.AddColumn(
            ProfitRisk,
            "LiquidityRisk",
            each
                if [CurrentRatio] = null then "UNKNOWN"
                else if [CurrentRatio] < 0.75 then "CRITICAL"
                else if [CurrentRatio] < 1.00 then "HIGH"
                else if [CurrentRatio] < 1.25 then "MEDIUM"
                else "LOW",
            type text
        ),

    // Debt risk
    DebtRisk =
        Table.AddColumn(
            LiquidityRisk,
            "DebtRisk",
            each
                if [DebtToEquity] = null then "UNKNOWN"
                else if [DebtToEquity] >= 3.00 then "CRITICAL"
                else if [DebtToEquity] >= 2.00 then "HIGH"
                else if [DebtToEquity] >= 1.00 then "MEDIUM"
                else "LOW",
            type text
        ),

    // Profitability risk
    MarginRisk =
        Table.AddColumn(
            DebtRisk,
            "MarginRisk",
            each
                if [NetProfitMargin] = null then "UNKNOWN"
                else if [NetProfitMargin] < -0.10 then "CRITICAL"
                else if [NetProfitMargin] < 0 then "HIGH"
                else if [NetProfitMargin] < 0.05 then "MEDIUM"
                else "LOW",
            type text
        ),

    // Cash-flow risk
    CashFlowRisk =
        Table.AddColumn(
            MarginRisk,
            "CashFlowRisk",
            each
                if [OperatingCashFlow] = null then "UNKNOWN"
                else if [OperatingCashFlow] < 0 and [Cash] <= 0 then "CRITICAL"
                else if [OperatingCashFlow] < 0 then "HIGH"
                else if [CashToCurrentLiabilities] < 0.50 then "HIGH"
                else if [CashToCurrentLiabilities] < 1.00 then "MEDIUM"
                else "LOW",
            type text
        ),


    // ------------------------------------------------------------
    // 7. OVERALL RISK SCORE
    // ------------------------------------------------------------
    AddRevenueScore =
        Table.AddColumn(
            CashFlowRisk,
            "RevenueRiskScore",
            each
                if [RevenueRisk] = "CRITICAL" then 100
                else if [RevenueRisk] = "HIGH" then 75
                else if [RevenueRisk] = "MEDIUM" then 50
                else if [RevenueRisk] = "LOW" then 10
                else 0,
            Int64.Type
        ),

    AddProfitScore =
        Table.AddColumn(
            AddRevenueScore,
            "ProfitRiskScore",
            each
                if [ProfitRisk] = "CRITICAL" then 100
                else if [ProfitRisk] = "HIGH" then 75
                else if [ProfitRisk] = "MEDIUM" then 50
                else if [ProfitRisk] = "LOW" then 10
                else 0,
            Int64.Type
        ),

    AddLiquidityScore =
        Table.AddColumn(
            AddProfitScore,
            "LiquidityRiskScore",
            each
                if [LiquidityRisk] = "CRITICAL" then 100
                else if [LiquidityRisk] = "HIGH" then 75
                else if [LiquidityRisk] = "MEDIUM" then 50
                else if [LiquidityRisk] = "LOW" then 10
                else 0,
            Int64.Type
        ),

    AddDebtScore =
        Table.AddColumn(
            AddLiquidityScore,
            "DebtRiskScore",
            each
                if [DebtRisk] = "CRITICAL" then 100
                else if [DebtRisk] = "HIGH" then 75
                else if [DebtRisk] = "MEDIUM" then 50
                else if [DebtRisk] = "LOW" then 10
                else 0,
            Int64.Type
        ),

    AddMarginScore =
        Table.AddColumn(
            AddDebtScore,
            "MarginRiskScore",
            each
                if [MarginRisk] = "CRITICAL" then 100
                else if [MarginRisk] = "HIGH" then 75
                else if [MarginRisk] = "MEDIUM" then 50
                else if [MarginRisk] = "LOW" then 10
                else 0,
            Int64.Type
        ),

    AddCashFlowScore =
        Table.AddColumn(
            AddMarginScore,
            "CashFlowRiskScore",
            each
                if [CashFlowRisk] = "CRITICAL" then 100
                else if [CashFlowRisk] = "HIGH" then 75
                else if [CashFlowRisk] = "MEDIUM" then 50
                else if [CashFlowRisk] = "LOW" then 10
                else 0,
            Int64.Type
        ),


    // ------------------------------------------------------------
    // 8. WEIGHTED RISK SCORE
    // ------------------------------------------------------------
    AddOverallRiskScore =
        Table.AddColumn(
            AddCashFlowScore,
            "OverallRiskScore",
            each
                Number.Round(
                    (
                        [RevenueRiskScore] * 0.20 +
                        [ProfitRiskScore] * 0.20 +
                        [LiquidityRiskScore] * 0.20 +
                        [DebtRiskScore] * 0.15 +
                        [MarginRiskScore] * 0.10 +
                        [CashFlowRiskScore] * 0.15
                    ),
                    2
                ),
            type number
        ),


    // ------------------------------------------------------------
    // 9. OVERALL SEVERITY
    // ------------------------------------------------------------
    AddOverallSeverity =
        Table.AddColumn(
            AddOverallRiskScore,
            "OverallSeverity",
            each
                if [OverallRiskScore] >= 80 then "CRITICAL"
                else if [OverallRiskScore] >= 60 then "HIGH"
                else if [OverallRiskScore] >= 35 then "MEDIUM"
                else if [OverallRiskScore] > 0 then "LOW"
                else "INFO",
            type text
        ),


    // ------------------------------------------------------------
    // 10. PRIMARY RISK FACTOR
    // ------------------------------------------------------------
    AddPrimaryRiskFactor =
        Table.AddColumn(
            AddOverallSeverity,
            "PrimaryRiskFactor",
            each
                let
                    Scores = {
                        [Revenue = [RevenueRiskScore]],
                        [Profitability = [ProfitRiskScore]],
                        [Liquidity = [LiquidityRiskScore]],
                        [Debt = [DebtRiskScore]],
                        [Margin = [MarginRiskScore]],
                        [CashFlow = [CashFlowRiskScore]]
                    },

                    ScoreTable =
                        Table.FromRecords(Scores),

                    Unpivoted =
                        Table.UnpivotOtherColumns(
                            ScoreTable,
                            {},
                            "RiskFactor",
                            "Score"
                        ),

                    Sorted =
                        Table.Sort(
                            Unpivoted,
                            {{"Score", Order.Descending}}
                        )
                in
                    if Table.RowCount(Sorted) = 0
                    then null
                    else Sorted{0}[RiskFactor],
            type text
        ),


    // ------------------------------------------------------------
    // 11. RISK FACTOR COUNT
    // ------------------------------------------------------------
    AddHighRiskCount =
        Table.AddColumn(
            AddPrimaryRiskFactor,
            "HighRiskFactorCount",
            each
                List.Count(
                    List.Select(
                        {
                            [RevenueRisk],
                            [ProfitRisk],
                            [LiquidityRisk],
                            [DebtRisk],
                            [MarginRisk],
                            [CashFlowRisk]
                        },
                        each _ = "HIGH" or _ = "CRITICAL"
                    )
                ),
            Int64.Type
        ),


    // ------------------------------------------------------------
    // 12. ALERT FLAGS
    // ------------------------------------------------------------
    AddFinancialProfileDown =
        Table.AddColumn(
            AddHighRiskCount,
            "FinancialProfileDown",
            each
                [OverallRiskScore] >= 60,
            type logical
        ),

    AddPreviousPeriodDecline =
        Table.AddColumn(
            AddFinancialProfileDown,
            "PreviousPeriodDecline",
            each
                (
                    ([RevenueChangePct] <> null and [RevenueChangePct] <= -0.10)
                    or
                    ([ProfitChangePct] <> null and [ProfitChangePct] <= -0.15)
                ),
            type logical
        ),

    AddSevereLossRisk =
        Table.AddColumn(
            AddPreviousPeriodDecline,
            "SevereLossRisk",
            each
                (
                    ([NetProfit] <> null and [NetProfit] < 0)
                    and
                    (
                        [ProfitRiskScore] >= 75
                        or [LiquidityRiskScore] >= 75
                        or [CashFlowRiskScore] >= 75
                    )
                ),
            type logical
        ),


    // ------------------------------------------------------------
    // 13. RISK EXPLANATION
    // ------------------------------------------------------------
    AddRiskExplanation =
        Table.AddColumn(
            AddSevereLossRisk,
            "RiskExplanation",
            each
                if [OverallSeverity] = "CRITICAL" then
                    "Multiple critical financial risk indicators require immediate review."
                else if [PrimaryRiskFactor] = "Liquidity" then
                    "Liquidity indicators show elevated pressure on short-term obligations."
                else if [PrimaryRiskFactor] = "Profitability" then
                    "Profitability has deteriorated materially compared with the previous period."
                else if [PrimaryRiskFactor] = "Revenue" then
                    "Revenue has declined materially compared with the previous period."
                else if [PrimaryRiskFactor] = "Debt" then
                    "Leverage indicators show elevated financial risk."
                else if [PrimaryRiskFactor] = "CashFlow" then
                    "Operating cash-flow indicators suggest potential liquidity pressure."
                else if [PrimaryRiskFactor] = "Margin" then
                    "Net margin performance indicates profitability pressure."
                else
                    "No significant financial risk factor detected.",
            type text
        ),


    // ------------------------------------------------------------
    // 14. RECOMMENDED ACTION
    // ------------------------------------------------------------
    AddRecommendedAction =
        Table.AddColumn(
            AddRiskExplanation,
            "RecommendedAction",
            each
                if [OverallSeverity] = "CRITICAL" then
                    "Escalate for immediate management review and perform root-cause and liquidity analysis."
                else if [PrimaryRiskFactor] = "Liquidity" then
                    "Review cash position, working capital, receivables, payables and near-term obligations."
                else if [PrimaryRiskFactor] = "Profitability" then
                    "Investigate margin compression, cost increases and revenue deterioration."
                else if [PrimaryRiskFactor] = "Revenue" then
                    "Investigate revenue drivers, customer concentration and period-over-period declines."
                else if [PrimaryRiskFactor] = "Debt" then
                    "Review debt maturity, leverage, interest burden and refinancing exposure."
                else if [PrimaryRiskFactor] = "CashFlow" then
                    "Review operating cash flow and near-term cash requirements."
                else if [PrimaryRiskFactor] = "Margin" then
                    "Analyze gross and net margin drivers and identify cost-reduction opportunities."
                else
                    "Continue monitoring financial performance.",
            type text
        ),


    // ------------------------------------------------------------
    // 15. DISPLAY FORMATTING
    // ------------------------------------------------------------
    AddRiskScoreRounded =
        Table.TransformColumns(
            AddRecommendedAction,
            {
                {
                    "OverallRiskScore",
                    each Number.Round(_, 1),
                    type number
                }
            }
        ),


    // ------------------------------------------------------------
    // 16. FINAL COLUMN SELECTION
    // ------------------------------------------------------------
    Final =
        Table.SelectColumns(
            AddRiskScoreRounded,
            {
                "CompanyId",
                "CompanyName",
                "Period",

                "Revenue",
                "PreviousRevenue",
                "RevenueChangePct",

                "NetProfit",
                "PreviousNetProfit",
                "ProfitChangePct",

                "GrossProfit",
                "PreviousGrossProfit",
                "GrossProfitChangePct",

                "Cash",
                "CurrentAssets",
                "CurrentLiabilities",
                "CurrentRatio",

                "TotalDebt",
                "TotalAssets",
                "TotalEquity",
                "DebtToEquity",
                "DebtToAssets",

                "OperatingCashFlow",
                "CashToCurrentLiabilities",

                "BudgetRevenue",
                "BudgetProfit",
                "RevenueBudgetVariancePct",
                "ProfitBudgetVariancePct",

                "RevenueRisk",
                "ProfitRisk",
                "LiquidityRisk",
                "DebtRisk",
                "MarginRisk",
                "CashFlowRisk",

                "RevenueRiskScore",
                "ProfitRiskScore",
                "LiquidityRiskScore",
                "DebtRiskScore",
                "MarginRiskScore",
                "CashFlowRiskScore",

                "OverallRiskScore",
                "OverallSeverity",
                "PrimaryRiskFactor",
                "HighRiskFactorCount",

                "FinancialProfileDown",
                "PreviousPeriodDecline",
                "SevereLossRisk",

                "RiskExplanation",
                "RecommendedAction"
            }
        )

in
    Final