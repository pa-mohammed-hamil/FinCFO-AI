let
    // ============================================================
    // FinCo AI - Risk Scores Query
    // File:
    //   powerbi/risk_dashboard/risk_scores_query.pq
    //
    // Purpose:
    //   Converts risk factors into normalized risk scores.
    //
    // Expected source:
    //   RiskFactors
    //
    // Main outputs:
    //   RevenueRiskScore
    //   ProfitRiskScore
    //   LiquidityRiskScore
    //   DebtRiskScore
    //   MarginRiskScore
    //   CashFlowRiskScore
    //   OverallRiskScore
    //   BusinessContinuityRiskScore
    //   FinancialRiskScore
    //   RiskSeverity
    //   RiskCategory
    //   RiskTrend
    //   AlertPriority
    // ============================================================


    // ------------------------------------------------------------
    // 1. SOURCE
    // ------------------------------------------------------------
    Source =
        try RiskFactors
        otherwise #table(
            {
                "CompanyId",
                "CompanyName",
                "Period",
                "RevenueChangePct",
                "ProfitChangePct",
                "CurrentRatio",
                "DebtToEquity",
                "NetProfitMargin",
                "CashToCurrentLiabilities",
                "OperatingCashFlow",
                "OverallRiskScore",
                "FinancialProfileDown",
                "PreviousPeriodDecline",
                "SevereLossRisk"
            },
            {}
        ),


    // ------------------------------------------------------------
    // 2. TYPE NORMALIZATION
    // ------------------------------------------------------------
    Typed =
        Table.TransformColumnTypes(
            Source,
            {
                {"CompanyId", type text},
                {"CompanyName", type text},
                {"Period", type date},

                {"RevenueChangePct", type number},
                {"ProfitChangePct", type number},
                {"CurrentRatio", type number},
                {"DebtToEquity", type number},
                {"NetProfitMargin", type number},
                {"CashToCurrentLiabilities", type number},
                {"OperatingCashFlow", type number},

                {"OverallRiskScore", type number},

                {"FinancialProfileDown", type logical},
                {"PreviousPeriodDecline", type logical},
                {"SevereLossRisk", type logical}
            },
            "en-US"
        ),


    // ------------------------------------------------------------
    // 3. CLAMP FUNCTION
    // ------------------------------------------------------------
    Clamp =
        (value as nullable number, minimum as number, maximum as number)
        as nullable number =>
            if value = null then
                null
            else if value < minimum then
                minimum
            else if value > maximum then
                maximum
            else
                value,


    // ------------------------------------------------------------
    // 4. REVENUE RISK SCORE
    // ------------------------------------------------------------
    AddRevenueScore =
        Table.AddColumn(
            Typed,
            "RevenueRiskScore",
            each
                if [RevenueChangePct] = null then
                    null
                else if [RevenueChangePct] <= -0.30 then
                    100
                else if [RevenueChangePct] <= -0.20 then
                    90
                else if [RevenueChangePct] <= -0.10 then
                    75
                else if [RevenueChangePct] <= -0.05 then
                    50
                else if [RevenueChangePct] < 0 then
                    25
                else
                    0,
            type number
        ),


    // ------------------------------------------------------------
    // 5. PROFITABILITY RISK SCORE
    // ------------------------------------------------------------
    AddProfitScore =
        Table.AddColumn(
            AddRevenueScore,
            "ProfitRiskScore",
            each
                if [ProfitChangePct] = null then
                    null
                else if [ProfitChangePct] <= -0.50 then
                    100
                else if [ProfitChangePct] <= -0.30 then
                    90
                else if [ProfitChangePct] <= -0.15 then
                    75
                else if [ProfitChangePct] <= -0.05 then
                    50
                else if [ProfitChangePct] < 0 then
                    25
                else
                    0,
            type number
        ),


    // ------------------------------------------------------------
    // 6. LIQUIDITY RISK SCORE
    // ------------------------------------------------------------
    AddLiquidityScore =
        Table.AddColumn(
            AddProfitScore,
            "LiquidityRiskScore",
            each
                if [CurrentRatio] = null then
                    null
                else if [CurrentRatio] < 0.50 then
                    100
                else if [CurrentRatio] < 0.75 then
                    90
                else if [CurrentRatio] < 1.00 then
                    75
                else if [CurrentRatio] < 1.25 then
                    50
                else if [CurrentRatio] < 1.50 then
                    25
                else
                    0,
            type number
        ),


    // ------------------------------------------------------------
    // 7. DEBT RISK SCORE
    // ------------------------------------------------------------
    AddDebtScore =
        Table.AddColumn(
            AddLiquidityScore,
            "DebtRiskScore",
            each
                if [DebtToEquity] = null then
                    null
                else if [DebtToEquity] >= 4.00 then
                    100
                else if [DebtToEquity] >= 3.00 then
                    90
                else if [DebtToEquity] >= 2.00 then
                    75
                else if [DebtToEquity] >= 1.00 then
                    50
                else if [DebtToEquity] >= 0.50 then
                    25
                else
                    0,
            type number
        ),


    // ------------------------------------------------------------
    // 8. MARGIN RISK SCORE
    // ------------------------------------------------------------
    AddMarginScore =
        Table.AddColumn(
            AddDebtScore,
            "MarginRiskScore",
            each
                if [NetProfitMargin] = null then
                    null
                else if [NetProfitMargin] <= -0.20 then
                    100
                else if [NetProfitMargin] < -0.10 then
                    90
                else if [NetProfitMargin] < 0 then
                    75
                else if [NetProfitMargin] < 0.05 then
                    50
                else if [NetProfitMargin] < 0.10 then
                    25
                else
                    0,
            type number
        ),


    // ------------------------------------------------------------
    // 9. CASH-FLOW RISK SCORE
    // ------------------------------------------------------------
    AddCashFlowScore =
        Table.AddColumn(
            AddMarginScore,
            "CashFlowRiskScore",
            each
                if [OperatingCashFlow] = null then
                    null
                else if [OperatingCashFlow] < 0
                    and [CashToCurrentLiabilities] <> null
                    and [CashToCurrentLiabilities] < 0.50 then
                    100
                else if [OperatingCashFlow] < 0 then
                    90
                else if [CashToCurrentLiabilities] <> null
                    and [CashToCurrentLiabilities] < 0.50 then
                    75
                else if [CashToCurrentLiabilities] <> null
                    and [CashToCurrentLiabilities] < 1.00 then
                    50
                else if [CashToCurrentLiabilities] <> null
                    and [CashToCurrentLiabilities] < 1.50 then
                    25
                else
                    0,
            type number
        ),


    // ------------------------------------------------------------
    // 10. COMPONENT VALIDITY COUNT
    // ------------------------------------------------------------
    AddAvailableFactors =
        Table.AddColumn(
            AddCashFlowScore,
            "AvailableRiskFactors",
            each
                List.Count(
                    List.RemoveNulls(
                        {
                            [RevenueRiskScore],
                            [ProfitRiskScore],
                            [LiquidityRiskScore],
                            [DebtRiskScore],
                            [MarginRiskScore],
                            [CashFlowRiskScore]
                        }
                    )
                ),
            Int64.Type
        ),


    // ------------------------------------------------------------
    // 11. WEIGHTED FINANCIAL RISK SCORE
    //
    // Weights:
    // Revenue       20%
    // Profit        20%
    // Liquidity     20%
    // Debt          15%
    // Margin        10%
    // Cash Flow     15%
    // ------------------------------------------------------------
    AddFinancialRiskScore =
        Table.AddColumn(
            AddAvailableFactors,
            "FinancialRiskScore",
            each
                let
                    Revenue = if [RevenueRiskScore] = null then 0 else [RevenueRiskScore],
                    Profit = if [ProfitRiskScore] = null then 0 else [ProfitRiskScore],
                    Liquidity = if [LiquidityRiskScore] = null then 0 else [LiquidityRiskScore],
                    Debt = if [DebtRiskScore] = null then 0 else [DebtRiskScore],
                    Margin = if [MarginRiskScore] = null then 0 else [MarginRiskScore],
                    CashFlow = if [CashFlowRiskScore] = null then 0 else [CashFlowRiskScore],

                    WeightedScore =
                        Revenue * 0.20 +
                        Profit * 0.20 +
                        Liquidity * 0.20 +
                        Debt * 0.15 +
                        Margin * 0.10 +
                        CashFlow * 0.15,

                    WeightAvailable =
                        (if [RevenueRiskScore] = null then 0 else 0.20) +
                        (if [ProfitRiskScore] = null then 0 else 0.20) +
                        (if [LiquidityRiskScore] = null then 0 else 0.20) +
                        (if [DebtRiskScore] = null then 0 else 0.15) +
                        (if [MarginRiskScore] = null then 0 else 0.10) +
                        (if [CashFlowRiskScore] = null then 0 else 0.15)
                in
                    if WeightAvailable = 0 then
                        null
                    else
                        Number.Round(
                            WeightedScore / WeightAvailable,
                            2
                        ),
            type number
        ),


    // ------------------------------------------------------------
    // 12. BUSINESS CONTINUITY RISK
    //
    // Focuses on:
    //   Liquidity
    //   Cash flow
    //   Debt
    //   Profitability
    // ------------------------------------------------------------
    AddBusinessContinuityRisk =
        Table.AddColumn(
            AddFinancialRiskScore,
            "BusinessContinuityRiskScore",
            each
                let
                    Liquidity =
                        if [LiquidityRiskScore] = null
                        then 0
                        else [LiquidityRiskScore],

                    CashFlow =
                        if [CashFlowRiskScore] = null
                        then 0
                        else [CashFlowRiskScore],

                    Debt =
                        if [DebtRiskScore] = null
                        then 0
                        else [DebtRiskScore],

                    Profit =
                        if [ProfitRiskScore] = null
                        then 0
                        else [ProfitRiskScore],

                    Score =
                        Liquidity * 0.35 +
                        CashFlow * 0.30 +
                        Debt * 0.20 +
                        Profit * 0.15
                in
                    Number.Round(Score, 2),
            type number
        ),


    // ------------------------------------------------------------
    // 13. LOSS RISK SCORE
    // ------------------------------------------------------------
    AddLossRiskScore =
        Table.AddColumn(
            AddBusinessContinuityRisk,
            "LossRiskScore",
            each
                let
                    Profit =
                        if [ProfitRiskScore] = null
                        then 0
                        else [ProfitRiskScore],

                    Margin =
                        if [MarginRiskScore] = null
                        then 0
                        else [MarginRiskScore],

                    Revenue =
                        if [RevenueRiskScore] = null
                        then 0
                        else [RevenueRiskScore],

                    CashFlow =
                        if [CashFlowRiskScore] = null
                        then 0
                        else [CashFlowRiskScore],

                    Score =
                        Profit * 0.40 +
                        Margin * 0.25 +
                        Revenue * 0.20 +
                        CashFlow * 0.15
                in
                    Number.Round(Score, 2),
            type number
        ),


    // ------------------------------------------------------------
    // 14. FINAL OVERALL RISK SCORE
    // ------------------------------------------------------------
    AddOverallScore =
        Table.AddColumn(
            AddLossRiskScore,
            "OverallRiskScoreCalculated",
            each
                let
                    Financial =
                        if [FinancialRiskScore] = null
                        then 0
                        else [FinancialRiskScore],

                    Continuity =
                        if [BusinessContinuityRiskScore] = null
                        then 0
                        else [BusinessContinuityRiskScore],

                    Loss =
                        if [LossRiskScore] = null
                        then 0
                        else [LossRiskScore],

                    Score =
                        Financial * 0.50 +
                        Continuity * 0.30 +
                        Loss * 0.20
                in
                    Number.Round(
                        Clamp(Score, 0, 100),
                        2
                    ),
            type number
        ),


    // ------------------------------------------------------------
    // 15. RISK SEVERITY
    // ------------------------------------------------------------
    AddRiskSeverity =
        Table.AddColumn(
            AddOverallScore,
            "RiskSeverity",
            each
                if [OverallRiskScoreCalculated] = null then
                    "UNKNOWN"
                else if [OverallRiskScoreCalculated] >= 80 then
                    "CRITICAL"
                else if [OverallRiskScoreCalculated] >= 60 then
                    "HIGH"
                else if [OverallRiskScoreCalculated] >= 35 then
                    "MEDIUM"
                else if [OverallRiskScoreCalculated] >= 15 then
                    "LOW"
                else
                    "INFO",
            type text
        ),


    // ------------------------------------------------------------
    // 16. RISK CATEGORY
    // ------------------------------------------------------------
    AddRiskCategory =
        Table.AddColumn(
            AddRiskSeverity,
            "RiskCategory",
            each
                let
                    Scores = {
                        [Category = "Revenue", Score = [RevenueRiskScore]],
                        [Category = "Profitability", Score = [ProfitRiskScore]],
                        [Category = "Liquidity", Score = [LiquidityRiskScore]],
                        [Category = "Debt", Score = [DebtRiskScore]],
                        [Category = "Margin", Score = [MarginRiskScore]],
                        [Category = "Cash Flow", Score = [CashFlowRiskScore]]
                    },

                    ValidScores =
                        List.Select(
                            Scores,
                            each [Score] <> null
                        ),

                    Sorted =
                        List.Sort(
                            ValidScores,
                            (x, y) =>
                                if x[Score] > y[Score]
                                then -1
                                else if x[Score] < y[Score]
                                then 1
                                else 0
                        )
                in
                    if List.Count(Sorted) = 0
                    then "Unknown"
                    else Sorted{0}[Category],
            type text
        ),


    // ------------------------------------------------------------
    // 17. RISK TREND
    // ------------------------------------------------------------
    AddRiskTrend =
        Table.AddColumn(
            AddRiskCategory,
            "RiskTrend",
            each
                if [PreviousPeriodDecline] = true
                    and [SevereLossRisk] = true then
                    "DETERIORATING_FAST"
                else if [PreviousPeriodDecline] = true then
                    "DETERIORATING"
                else if [FinancialProfileDown] = true then
                    "ELEVATED"
                else
                    "STABLE",
            type text
        ),


    // ------------------------------------------------------------
    // 18. ALERT PRIORITY
    // ------------------------------------------------------------
    AddAlertPriority =
        Table.AddColumn(
            AddRiskTrend,
            "AlertPriority",
            each
                if [SevereLossRisk] = true then
                    1
                else if [OverallRiskScoreCalculated] >= 80 then
                    1
                else if [FinancialProfileDown] = true then
                    2
                else if [PreviousPeriodDecline] = true then
                    3
                else if [OverallRiskScoreCalculated] >= 35 then
                    4
                else
                    5,
            Int64.Type
        ),


    // ------------------------------------------------------------
    // 19. ALERT STATUS
    // ------------------------------------------------------------
    AddAlertStatus =
        Table.AddColumn(
            AddAlertPriority,
            "AlertStatus",
            each
                if [SevereLossRisk] = true then
                    "ACTION_REQUIRED"
                else if [OverallRiskScoreCalculated] >= 80 then
                    "ESCALATE"
                else if [OverallRiskScoreCalculated] >= 60 then
                    "REVIEW"
                else if [OverallRiskScoreCalculated] >= 35 then
                    "MONITOR"
                else
                    "NORMAL",
            type text
        ),


    // ------------------------------------------------------------
    // 20. DECISION SIGNAL
    // ------------------------------------------------------------
    AddDecisionSignal =
        Table.AddColumn(
            AddAlertStatus,
            "DecisionSignal",
            each
                if [SevereLossRisk] = true then
                    "SEVERE_LOSS_RISK"
                else if [BusinessContinuityRiskScore] >= 80 then
                    "BUSINESS_CONTINUITY_RISK"
                else if [LossRiskScore] >= 80 then
                    "HIGH_LOSS_RISK"
                else if [LiquidityRiskScore] <> null
                    and [LiquidityRiskScore] >= 75 then
                    "LIQUIDITY_PRESSURE"
                else if [DebtRiskScore] <> null
                    and [DebtRiskScore] >= 75 then
                    "HIGH_LEVERAGE"
                else if [ProfitRiskScore] <> null
                    and [ProfitRiskScore] >= 75 then
                    "PROFITABILITY_DECLINE"
                else if [RevenueRiskScore] <> null
                    and [RevenueRiskScore] >= 75 then
                    "REVENUE_DECLINE"
                else
                    "NO_MAJOR_RISK",
            type text
        ),


    // ------------------------------------------------------------
    // 21. MANAGEMENT ACTION
    // ------------------------------------------------------------
    AddManagementAction =
        Table.AddColumn(
            AddDecisionSignal,
            "ManagementAction",
            each
                if [AlertStatus] = "ESCALATE" then
                    "Immediate management review, root-cause analysis and scenario assessment required."
                else if [AlertStatus] = "REVIEW" then
                    "Management review recommended with supporting financial and RAG evidence."
                else if [AlertStatus] = "MONITOR" then
                    "Continue monitoring the affected financial indicators."
                else
                    "No immediate management intervention required.",
            type text
        ),


    // ------------------------------------------------------------
    // 22. DATA QUALITY STATUS
    // ------------------------------------------------------------
    AddDataQuality =
        Table.AddColumn(
            AddManagementAction,
            "RiskDataQuality",
            each
                if [AvailableRiskFactors] >= 5 then
                    "HIGH"
                else if [AvailableRiskFactors] >= 3 then
                    "MEDIUM"
                else if [AvailableRiskFactors] >= 1 then
                    "LOW"
                else
                    "INSUFFICIENT",
            type text
        ),


    // ------------------------------------------------------------
    // 23. FINAL ROUNDING
    // ------------------------------------------------------------
    Rounded =
        Table.TransformColumns(
            AddDataQuality,
            {
                {
                    "FinancialRiskScore",
                    each if _ = null then null else Number.Round(_, 2),
                    type number
                },
                {
                    "BusinessContinuityRiskScore",
                    each if _ = null then null else Number.Round(_, 2),
                    type number
                },
                {
                    "LossRiskScore",
                    each if _ = null then null else Number.Round(_, 2),
                    type number
                },
                {
                    "OverallRiskScoreCalculated",
                    each if _ = null then null else Number.Round(_, 2),
                    type number
                }
            }
        ),


    // ------------------------------------------------------------
    // 24. FINAL OUTPUT
    // ------------------------------------------------------------
    Final =
        Table.SelectColumns(
            Rounded,
            {
                "CompanyId",
                "CompanyName",
                "Period",

                "RevenueRiskScore",
                "ProfitRiskScore",
                "LiquidityRiskScore",
                "DebtRiskScore",
                "MarginRiskScore",
                "CashFlowRiskScore",

                "FinancialRiskScore",
                "BusinessContinuityRiskScore",
                "LossRiskScore",
                "OverallRiskScoreCalculated",

                "RiskSeverity",
                "RiskCategory",
                "RiskTrend",

                "AvailableRiskFactors",
                "RiskDataQuality",

                "AlertPriority",
                "AlertStatus",
                "DecisionSignal",
                "ManagementAction",

                "FinancialProfileDown",
                "PreviousPeriodDecline",
                "SevereLossRisk"
            }
        ),

    // Rename calculated score to the dashboard-friendly name.
    Renamed =
        Table.RenameColumns(
            Final,
            {
                {
                    "OverallRiskScoreCalculated",
                    "OverallRiskScore"
                }
            }
        )

in
    Renamed