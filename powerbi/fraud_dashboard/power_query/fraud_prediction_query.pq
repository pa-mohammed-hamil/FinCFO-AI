// ============================================================================
// FinCo AI — Fraud Prediction Query
// File: fraud_prediction_query.pq
// ============================================================================
//
// PURPOSE
// -------
// Power BI preparation layer for fraud prediction analytics.
//
// ARCHITECTURE
// ------------
//
// Transactions
//      ↓
// common_transformations
//      ↓
// anomaly_query
//      ↓
// Fraud ML Prediction
//      ↓
// fraud_prediction_query.pq
//      ↓
// Fraud Dashboard
//      ↓
// Alert Engine / Supervisor Agent / Human Review
//
// EXPECTED UPSTREAM QUERIES
// -------------------------
//   anomaly_query
//   FraudPredictions       (optional ML output)
//
// EXPECTED ML OUTPUT
// ------------------
// FraudPredictions may contain:
//
//   TransactionId
//   CompanyId
//   PredictionDate
//   FraudProbability
//   FraudScore
//   Prediction
//   ModelName
//   ModelVersion
//   Threshold
//
// IMPORTANT
// ---------
// This query does NOT replace the production ML model.
//
// Authoritative model execution remains in:
//
//   backend/app/fraud/preprocessing.py
//   backend/app/fraud/feature_engineering.py
//   backend/app/fraud/supervised_model.py
//   backend/app/fraud/anomaly_model.py
//   backend/app/fraud/scoring.py
//   backend/app/fraud/explainability.py
//
// Power BI is the analytical / monitoring layer.
// ============================================================================

let

    // =========================================================================
    // 1. HELPER FUNCTIONS
    // =========================================================================

    SafeText = (value as any) as nullable text =>
        try
            if value = null
            then null
            else Text.Trim(Text.From(value))
        otherwise null,


    SafeNumber = (value as any) as nullable number =>
        try Number.From(value)
        otherwise null,


    SafeDate = (value as any) as nullable date =>
        try Date.From(value)
        otherwise null,


    SafeDivide = (
        numerator as any,
        denominator as any
    ) as nullable number =>
        let
            N = SafeNumber(numerator),
            D = SafeNumber(denominator)
        in
            if N = null or D = null or D = 0
            then null
            else N / D,


    Clamp = (
        value as any,
        minimum as number,
        maximum as number
    ) as number =>
        let
            V = SafeNumber(value)
        in
            if V = null
            then 0
            else if V < minimum
            then minimum
            else if V > maximum
            then maximum
            else V,


    HasColumn = (
        table as table,
        columnName as text
    ) as logical =>
        List.Contains(
            Table.ColumnNames(table),
            columnName
        ),


    // =========================================================================
    // 2. ANOMALY SOURCE
    // =========================================================================

    AnomalySource =
        try anomaly_query
        otherwise #table(
            {
                "AnomalyRecordKey",
                "TransactionId",
                "CompanyId",
                "CompanyName",
                "TransactionDate",
                "CustomerId",
                "SupplierId",
                "AccountId",
                "TransactionType",
                "TransactionCategory",
                "PaymentMethod",
                "Currency",
                "Amount",
                "AnomalyAbsoluteAmount",
                "AnomalyRuleScore",
                "AnomalyRiskBand",
                "AnomalySignalCount",
                "AnomalySignalStrength",
                "AnomalyDuplicateFlag",
                "AnomalyHighValueFlag",
                "AnomalyVeryHighValueFlag",
                "AnomalyRoundValueFlag",
                "AnomalyWeekendFlag",
                "AnomalyMonthEndFlag",
                "AnomalyYearEndFlag",
                "AnomalyMissingPartyFlag",
                "AnomalyPaymentMethodFlag",
                "AnomalyCategoryRiskFlag",
                "AnomalyStatusRiskFlag",
                "AnomalyConcentrationFlag",
                "AnomalyCustomerAmountShare",
                "AnomalySupplierAmountShare",
                "AnomalyExplanation",
                "AnomalyMLReady",
                "AnomalySupervisorAgentSignal",
                "AnomalyHumanReviewRequired",
                "AnomalyDecision",
                "AnomalyStatus"
            },
            {}
        ),


    // =========================================================================
    // 3. STANDARDIZE SOURCE
    // =========================================================================

    Standardized =
        Table.TransformColumnNames(
            AnomalySource,
            each Text.Trim(_)
        ),


    // =========================================================================
    // 4. REQUIRED COLUMNS
    // =========================================================================

    RequiredColumns = {
        "AnomalyRecordKey",
        "TransactionId",
        "CompanyId",
        "CompanyName",
        "TransactionDate",
        "CustomerId",
        "SupplierId",
        "AccountId",
        "TransactionType",
        "TransactionCategory",
        "PaymentMethod",
        "Currency",
        "Amount",
        "AnomalyAbsoluteAmount",
        "AnomalyRuleScore",
        "AnomalyRiskBand",
        "AnomalySignalCount",
        "AnomalySignalStrength",
        "AnomalyDuplicateFlag",
        "AnomalyHighValueFlag",
        "AnomalyVeryHighValueFlag",
        "AnomalyRoundValueFlag",
        "AnomalyWeekendFlag",
        "AnomalyMonthEndFlag",
        "AnomalyYearEndFlag",
        "AnomalyMissingPartyFlag",
        "AnomalyPaymentMethodFlag",
        "AnomalyCategoryRiskFlag",
        "AnomalyStatusRiskFlag",
        "AnomalyConcentrationFlag",
        "AnomalyCustomerAmountShare",
        "AnomalySupplierAmountShare",
        "AnomalyExplanation",
        "AnomalyMLReady",
        "AnomalySupervisorAgentSignal",
        "AnomalyHumanReviewRequired",
        "AnomalyDecision",
        "AnomalyStatus"
    },


    EnsureColumns =
        List.Accumulate(
            RequiredColumns,
            Standardized,
            (state, columnName) =>
                if HasColumn(state, columnName)
                then state
                else
                    Table.AddColumn(
                        state,
                        columnName,
                        each null
                    )
        ),


    // =========================================================================
    // 5. TYPE NORMALIZATION
    // =========================================================================

    Typed =
        Table.TransformColumns(
            EnsureColumns,
            {
                {
                    "AnomalyRecordKey",
                    each SafeText(_),
                    type text
                },
                {
                    "TransactionId",
                    each SafeText(_),
                    type text
                },
                {
                    "CompanyId",
                    each SafeText(_),
                    type text
                },
                {
                    "CompanyName",
                    each SafeText(_),
                    type text
                },
                {
                    "TransactionDate",
                    each SafeDate(_),
                    type date
                },
                {
                    "CustomerId",
                    each SafeText(_),
                    type text
                },
                {
                    "SupplierId",
                    each SafeText(_),
                    type text
                },
                {
                    "AccountId",
                    each SafeText(_),
                    type text
                },
                {
                    "TransactionType",
                    each SafeText(_),
                    type text
                },
                {
                    "TransactionCategory",
                    each SafeText(_),
                    type text
                },
                {
                    "PaymentMethod",
                    each SafeText(_),
                    type text
                },
                {
                    "Currency",
                    each SafeText(_),
                    type text
                },
                {
                    "Amount",
                    each SafeNumber(_),
                    type number
                },
                {
                    "AnomalyAbsoluteAmount",
                    each SafeNumber(_),
                    type number
                },
                {
                    "AnomalyRuleScore",
                    each Clamp(_, 0, 100),
                    type number
                },
                {
                    "AnomalyRiskBand",
                    each SafeText(_),
                    type text
                },
                {
                    "AnomalySignalCount",
                    each SafeNumber(_),
                    Int64.Type
                },
                {
                    "AnomalySignalStrength",
                    each SafeText(_),
                    type text
                }
            }
        ),


    // =========================================================================
    // 6. OPTIONAL ML PREDICTION SOURCE
    // =========================================================================

    MLPredictionSource =
        try FraudPredictions
        otherwise #table(
            {
                "TransactionId",
                "CompanyId",
                "PredictionDate",
                "FraudProbability",
                "FraudScore",
                "Prediction",
                "ModelName",
                "ModelVersion",
                "Threshold"
            },
            {}
        ),


    // =========================================================================
    // 7. STANDARDIZE ML PREDICTIONS
    // =========================================================================

    MLStandardized =
        Table.TransformColumnNames(
            MLPredictionSource,
            each Text.Trim(_)
        ),


    // =========================================================================
    // 8. ENSURE ML COLUMNS
    // =========================================================================

    MLRequiredColumns = {
        "TransactionId",
        "CompanyId",
        "PredictionDate",
        "FraudProbability",
        "FraudScore",
        "Prediction",
        "ModelName",
        "ModelVersion",
        "Threshold"
    },


    EnsureMLColumns =
        List.Accumulate(
            MLRequiredColumns,
            MLStandardized,
            (state, columnName) =>
                if HasColumn(state, columnName)
                then state
                else
                    Table.AddColumn(
                        state,
                        columnName,
                        each null
                    )
        ),


    // =========================================================================
    // 9. TYPE ML PREDICTION DATA
    // =========================================================================

    TypedML =
        Table.TransformColumns(
            EnsureMLColumns,
            {
                {
                    "TransactionId",
                    each SafeText(_),
                    type text
                },
                {
                    "CompanyId",
                    each SafeText(_),
                    type text
                },
                {
                    "PredictionDate",
                    each SafeDate(_),
                    type date
                },
                {
                    "FraudProbability",
                    each Clamp(_, 0, 1),
                    type number
                },
                {
                    "FraudScore",
                    each Clamp(_, 0, 100),
                    type number
                },
                {
                    "Prediction",
                    each SafeText(_),
                    type text
                },
                {
                    "ModelName",
                    each SafeText(_),
                    type text
                },
                {
                    "ModelVersion",
                    each SafeText(_),
                    type text
                },
                {
                    "Threshold",
                    each
                        if _ = null
                        then null
                        else Clamp(_, 0, 1),
                    type number
                }
            }
        ),


    // =========================================================================
    // 10. JOIN ML PREDICTIONS
    // =========================================================================

    JoinML =
        Table.NestedJoin(
            Typed,
            {
                "TransactionId",
                "CompanyId"
            },
            TypedML,
            {
                "TransactionId",
                "CompanyId"
            },
            "MLPredictionData",
            JoinKind.LeftOuter
        ),


    // =========================================================================
    // 11. EXPAND ML OUTPUT
    // =========================================================================

    ExpandML =
        Table.ExpandTableColumn(
            JoinML,
            "MLPredictionData",
            {
                "PredictionDate",
                "FraudProbability",
                "FraudScore",
                "Prediction",
                "ModelName",
                "ModelVersion",
                "Threshold"
            },
            {
                "FraudPredictionDate",
                "MLFraudProbability",
                "MLFraudScore",
                "MLFraudPrediction",
                "FraudModelName",
                "FraudModelVersion",
                "FraudModelThreshold"
            }
        ),


    // =========================================================================
    // 12. ML AVAILABILITY
    // =========================================================================

    AddMLAvailability =
        Table.AddColumn(
            ExpandML,
            "FraudMLPredictionAvailable",
            each
                if [MLFraudScore] <> null
                    or [MLFraudProbability] <> null
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 13. NORMALIZE ML SCORE
    // =========================================================================

    AddNormalizedMLScore =
        Table.AddColumn(
            AddMLAvailability,
            "NormalizedMLFraudScore",
            each
                if [MLFraudScore] <> null
                then Clamp(
                    [MLFraudScore],
                    0,
                    100
                )

                else if [MLFraudProbability] <> null
                then
                    Clamp(
                        [MLFraudProbability] * 100,
                        0,
                        100
                    )

                else null,
            type number
        ),


    // =========================================================================
    // 14. NORMALIZE ML PROBABILITY
    // =========================================================================

    AddNormalizedProbability =
        Table.AddColumn(
            AddNormalizedMLScore,
            "NormalizedFraudProbability",
            each
                if [MLFraudProbability] <> null
                then Clamp(
                    [MLFraudProbability],
                    0,
                    1
                )

                else if [MLFraudScore] <> null
                then
                    Clamp(
                        [MLFraudScore] / 100,
                        0,
                        1
                    )

                else null,
            type number
        ),


    // =========================================================================
    // 15. ML RISK BAND
    // =========================================================================

    AddMLRiskBand =
        Table.AddColumn(
            AddNormalizedProbability,
            "MLFraudRiskBand",
            each
                if [NormalizedMLFraudScore] = null
                then "NOT_AVAILABLE"

                else if [NormalizedMLFraudScore] >= 80
                then "CRITICAL"

                else if [NormalizedMLFraudScore] >= 60
                then "HIGH"

                else if [NormalizedMLFraudScore] >= 35
                then "MEDIUM"

                else if [NormalizedMLFraudScore] >= 15
                then "LOW"

                else "INFO",
            type text
        ),


    // =========================================================================
    // 16. PREDICTION CLASS
    // =========================================================================

    AddPredictionClass =
        Table.AddColumn(
            AddMLRiskBand,
            "FraudPredictionClass",
            each
                if [NormalizedMLFraudScore] = null
                then "UNKNOWN"

                else if [NormalizedMLFraudScore] >= 80
                then "CRITICAL_FRAUD"

                else if [NormalizedMLFraudScore] >= 60
                then "HIGH_FRAUD_RISK"

                else if [NormalizedMLFraudScore] >= 35
                then "SUSPICIOUS"

                else "NORMAL",
            type text
        ),


    // =========================================================================
    // 17. PREDICTION CONFIDENCE
    // =========================================================================

    AddPredictionConfidence =
        Table.AddColumn(
            AddPredictionClass,
            "FraudPredictionConfidence",
            each
                if [NormalizedFraudProbability] = null
                then "UNKNOWN"

                else if [NormalizedFraudProbability] >= 0.90
                then "VERY_HIGH"

                else if [NormalizedFraudProbability] >= 0.75
                then "HIGH"

                else if [NormalizedFraudProbability] >= 0.50
                then "MEDIUM"

                else if [NormalizedFraudProbability] >= 0.25
                then "LOW"

                else "VERY_LOW",
            type text
        ),


    // =========================================================================
    // 18. COMBINE ML + RULE SCORE
    // =========================================================================

    AddCombinedFraudScore =
        Table.AddColumn(
            AddPredictionConfidence,
            "CombinedFraudScore",
            each
                let
                    MLScore =
                        [NormalizedMLFraudScore],

                    RuleScore =
                        [AnomalyRuleScore]
                in
                    if MLScore <> null
                    then
                        Clamp(
                            MLScore * 0.70
                            + RuleScore * 0.30,
                            0,
                            100
                        )
                    else
                        Clamp(
                            RuleScore,
                            0,
                            100
                        ),
            type number
        ),


    // =========================================================================
    // 19. COMBINED RISK BAND
    // =========================================================================

    AddCombinedRiskBand =
        Table.AddColumn(
            AddCombinedFraudScore,
            "CombinedFraudRiskBand",
            each
                if [CombinedFraudScore] >= 80
                then "CRITICAL"

                else if [CombinedFraudScore] >= 60
                then "HIGH"

                else if [CombinedFraudScore] >= 35
                then "MEDIUM"

                else if [CombinedFraudScore] >= 15
                then "LOW"

                else "INFO",
            type text
        ),


    // =========================================================================
    // 20. ML / RULE AGREEMENT
    // =========================================================================

    AddModelRuleAgreement =
        Table.AddColumn(
            AddCombinedRiskBand,
            "FraudModelRuleAgreement",
            each
                if [NormalizedMLFraudScore] = null
                then "NO_ML_MODEL"

                else if
                    [NormalizedMLFraudScore] >= 60
                    and [AnomalyRuleScore] >= 60
                then "HIGH_AGREEMENT"

                else if
                    [NormalizedMLFraudScore] < 35
                    and [AnomalyRuleScore] < 35
                then "NORMAL_AGREEMENT"

                else "MODEL_RULE_DISAGREEMENT",
            type text
        ),


    // =========================================================================
    // 21. MODEL DISAGREEMENT FLAG
    // =========================================================================

    AddModelDisagreementFlag =
        Table.AddColumn(
            AddModelRuleAgreement,
            "FraudModelDisagreementFlag",
            each
                if [FraudModelRuleAgreement] = "MODEL_RULE_DISAGREEMENT"
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 22. HIGH CONFIDENCE FRAUD
    // =========================================================================

    AddHighConfidenceFraud =
        Table.AddColumn(
            AddModelDisagreementFlag,
            "HighConfidenceFraudFlag",
            each
                if
                    [NormalizedFraudProbability] <> null
                    and [NormalizedFraudProbability] >= 0.80
                    and [CombinedFraudScore] >= 60
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 23. FALSE POSITIVE REVIEW SIGNAL
    // =========================================================================

    AddReviewSignal =
        Table.AddColumn(
            AddHighConfidenceFraud,
            "FraudPredictionReviewSignal",
            each
                if
                    [FraudModelDisagreementFlag] = 1
                    and [CombinedFraudScore] >= 35
                then "REVIEW_MODEL_RULE_CONFLICT"

                else if
                    [CombinedFraudScore] >= 80
                then "CRITICAL_REVIEW"

                else if
                    [CombinedFraudScore] >= 60
                then "HIGH_PRIORITY_REVIEW"

                else if
                    [CombinedFraudScore] >= 35
                then "STANDARD_REVIEW"

                else "NO_REVIEW",
            type text
        ),


    // =========================================================================
    // 24. INVESTIGATION PRIORITY
    // =========================================================================

    AddInvestigationPriority =
        Table.AddColumn(
            AddReviewSignal,
            "FraudPredictionInvestigationPriority",
            each
                if [CombinedFraudScore] >= 80
                then "P1 — CRITICAL"

                else if [CombinedFraudScore] >= 60
                then "P2 — HIGH"

                else if [CombinedFraudScore] >= 35
                then "P3 — MEDIUM"

                else "P4 — LOW",
            type text
        ),


    // =========================================================================
    // 25. INVESTIGATION REQUIRED
    // =========================================================================

    AddInvestigationRequired =
        Table.AddColumn(
            AddInvestigationPriority,
            "FraudPredictionInvestigationRequired",
            each
                if [CombinedFraudScore] >= 60
                    or [HighConfidenceFraudFlag] = 1
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 26. HUMAN REVIEW
    // =========================================================================

    AddHumanReview =
        Table.AddColumn(
            AddInvestigationRequired,
            "FraudPredictionHumanReviewRequired",
            each
                if [CombinedFraudScore] >= 60
                    or [FraudModelDisagreementFlag] = 1
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 27. SUPERVISOR AGENT SIGNAL
    // =========================================================================

    AddSupervisorSignal =
        Table.AddColumn(
            AddHumanReview,
            "FraudPredictionSupervisorSignal",
            each
                if [CombinedFraudScore] >= 80
                then "ESCALATE_TO_SUPERVISOR"

                else if [CombinedFraudScore] >= 60
                then "START_FRAUD_INVESTIGATION"

                else if [CombinedFraudScore] >= 35
                then "REVIEW_TRANSACTION"

                else "NO_IMMEDIATE_ACTION",
            type text
        ),


    // =========================================================================
    // 28. RAG EVIDENCE TRIGGER
    // =========================================================================

    AddRAGTrigger =
        Table.AddColumn(
            AddSupervisorSignal,
            "FraudPredictionRAGEvidenceRequired",
            each
                if [CombinedFraudScore] >= 35
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 29. FORECAST TRIGGER
    // =========================================================================

    AddForecastTrigger =
        Table.AddColumn(
            AddRAGTrigger,
            "FraudPredictionForecastRequired",
            each
                if [CombinedFraudScore] >= 60
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 30. WHAT-IF TRIGGER
    // =========================================================================

    AddWhatIfTrigger =
        Table.AddColumn(
            AddForecastTrigger,
            "FraudPredictionWhatIfRequired",
            each
                if [CombinedFraudScore] >= 80
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 31. FINAL DECISION
    // =========================================================================

    AddDecision =
        Table.AddColumn(
            AddWhatIfTrigger,
            "FraudPredictionDecision",
            each
                if [CombinedFraudScore] >= 80
                then "BLOCK_AND_ESCALATE"

                else if [CombinedFraudScore] >= 60
                then "HOLD_AND_INVESTIGATE"

                else if [CombinedFraudScore] >= 35
                then "MONITOR"

                else "NORMAL",
            type text
        ),


    // =========================================================================
    // 32. MODEL STATUS
    // =========================================================================

    AddModelStatus =
        Table.AddColumn(
            AddDecision,
            "FraudPredictionModelStatus",
            each
                if [FraudMLPredictionAvailable] = "YES"
                then
                    if [FraudModelName] <> null
                    then "ML_MODEL_AVAILABLE"
                    else "ML_SCORE_AVAILABLE"
                else "RULE_BASED_ONLY",
            type text
        ),


    // =========================================================================
    // 33. PREDICTION STATUS
    // =========================================================================

    AddPredictionStatus =
        Table.AddColumn(
            AddModelStatus,
            "FraudPredictionStatus",
            each
                if [CombinedFraudScore] >= 80
                then "CRITICAL_FRAUD_PREDICTION"

                else if [CombinedFraudScore] >= 60
                then "HIGH_FRAUD_PREDICTION"

                else if [CombinedFraudScore] >= 35
                then "FRAUD_WATCH"

                else "NORMAL",
            type text
        ),


    // =========================================================================
    // 34. EXPLANATION
    // =========================================================================

    AddExplanation =
        Table.AddColumn(
            AddPredictionStatus,
            "FraudPredictionExplanation",
            each
                if [FraudModelDisagreementFlag] = 1
                then
                    "ML model and deterministic anomaly rules disagree"

                else if [HighConfidenceFraudFlag] = 1
                then
                    "High-confidence ML fraud prediction supported by anomaly signals"

                else if [CombinedFraudScore] >= 80
                then
                    "Critical fraud prediction requiring immediate escalation"

                else if [CombinedFraudScore] >= 60
                then
                    "High fraud probability requiring investigation"

                else if [CombinedFraudScore] >= 35
                then
                    "Suspicious transaction requiring monitoring"

                else
                    [AnomalyExplanation],
            type text
        ),


    // =========================================================================
    // 35. PREDICTION RECORD KEY
    // =========================================================================

    AddPredictionKey =
        Table.AddColumn(
            AddExplanation,
            "FraudPredictionRecordKey",
            each
                Text.Combine(
                    {
                        if [CompanyId] = null
                        then ""
                        else [CompanyId],

                        if [TransactionId] = null
                        then ""
                        else [TransactionId],

                        if [TransactionDate] = null
                        then ""
                        else Date.ToText(
                            [TransactionDate],
                            "yyyyMMdd"
                        )
                    },
                    "|"
                ),
            type text
        ),


    // =========================================================================
    // 36. DATA QUALITY
    // =========================================================================

    AddDataQuality =
        Table.AddColumn(
            AddPredictionKey,
            "FraudPredictionDataQuality",
            each
                if
                    [TransactionId] <> null
                    and [CompanyId] <> null
                    and [TransactionDate] <> null
                    and [Amount] <> null
                then "HIGH"

                else if
                    [TransactionId] <> null
                    and [TransactionDate] <> null
                    and [Amount] <> null
                then "MEDIUM"

                else "LOW",
            type text
        ),


    // =========================================================================
    // 37. ROUND SCORES
    // =========================================================================

    Rounded =
        Table.TransformColumns(
            AddDataQuality,
            {
                {
                    "AnomalyRuleScore",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                },
                {
                    "NormalizedMLFraudScore",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                },
                {
                    "NormalizedFraudProbability",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 4),
                    type number
                },
                {
                    "CombinedFraudScore",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                }
            }
        ),


    // =========================================================================
    // 38. SORT
    // =========================================================================

    Sorted =
        Table.Sort(
            Rounded,
            {
                {
                    "TransactionDate",
                    Order.Descending
                },
                {
                    "CombinedFraudScore",
                    Order.Descending
                },
                {
                    "AnomalyAbsoluteAmount",
                    Order.Descending
                }
            }
        ),


    // =========================================================================
    // 39. PREFERRED COLUMN ORDER
    // =========================================================================

    PreferredColumns = {
        "FraudPredictionRecordKey",

        "TransactionId",
        "CompanyId",
        "CompanyName",
        "TransactionDate",

        "CustomerId",
        "SupplierId",
        "AccountId",

        "TransactionType",
        "TransactionCategory",
        "PaymentMethod",
        "Currency",

        "Amount",
        "AnomalyAbsoluteAmount",

        "AnomalyRuleScore",
        "AnomalyRiskBand",
        "AnomalySignalCount",
        "AnomalySignalStrength",

        "AnomalyDuplicateFlag",
        "AnomalyHighValueFlag",
        "AnomalyVeryHighValueFlag",
        "AnomalyRoundValueFlag",
        "AnomalyWeekendFlag",
        "AnomalyMonthEndFlag",
        "AnomalyYearEndFlag",
        "AnomalyMissingPartyFlag",
        "AnomalyPaymentMethodFlag",
        "AnomalyCategoryRiskFlag",
        "AnomalyStatusRiskFlag",
        "AnomalyConcentrationFlag",

        "AnomalyCustomerAmountShare",
        "AnomalySupplierAmountShare",

        "FraudPredictionDate",
        "FraudMLPredictionAvailable",
        "MLFraudProbability",
        "MLFraudScore",
        "MLFraudPrediction",

        "FraudModelName",
        "FraudModelVersion",
        "FraudModelThreshold",

        "NormalizedMLFraudScore",
        "NormalizedFraudProbability",

        "MLFraudRiskBand",
        "FraudPredictionClass",
        "FraudPredictionConfidence",

        "CombinedFraudScore",
        "CombinedFraudRiskBand",

        "FraudModelRuleAgreement",
        "FraudModelDisagreementFlag",
        "HighConfidenceFraudFlag",

        "FraudPredictionReviewSignal",
        "FraudPredictionInvestigationPriority",
        "FraudPredictionInvestigationRequired",

        "FraudPredictionSupervisorSignal",
        "FraudPredictionRAGEvidenceRequired",
        "FraudPredictionForecastRequired",
        "FraudPredictionWhatIfRequired",

        "FraudPredictionHumanReviewRequired",
        "FraudPredictionDecision",
        "FraudPredictionModelStatus",
        "FraudPredictionStatus",

        "FraudPredictionExplanation",
        "AnomalyExplanation",

        "FraudPredictionDataQuality"
    },


    ExistingPreferredColumns =
        List.Intersect(
            {
                PreferredColumns,
                Table.ColumnNames(Sorted)
            }
        ),


    RemainingColumns =
        List.Difference(
            Table.ColumnNames(Sorted),
            ExistingPreferredColumns
        ),


    FinalColumnOrder =
        List.Combine(
            {
                ExistingPreferredColumns,
                RemainingColumns
            }
        ),


    Final =
        Table.ReorderColumns(
            Sorted,
            FinalColumnOrder,
            MissingField.Ignore
        )

in
    Final