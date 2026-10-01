
// ============================================================================
// FinCo AI — Anomaly Detection Query
// File: anomaly_query.pq
// ============================================================================
//
// PURPOSE
// -------
// Transaction anomaly detection preparation layer for:
//
//   Fraud Detection
//   Financial Monitoring
//   Fraud Dashboard
//   Alert Engine
//   ML Anomaly Detection
//   Supervisor Agent
//
// ARCHITECTURE
// ------------
//
//   Transactions
//        ↓
//   Common Transformations
//        ↓
//   anomaly_query.pq
//        ↓
//   Fraud / Anomaly Features
//        ↓
//   ML Anomaly Model
//        ↓
//   Fraud Scoring
//        ↓
//   Alert Engine
//        ↓
//   Fraud Dashboard
//
// EXPECTED UPSTREAM QUERY
// -----------------------
//   Transactions
//
// OPTIONAL
// --------
//   common_transformations
//
// IMPORTANT
// ---------
// Production ML anomaly scoring should remain in:
//
//   backend/app/fraud/anomaly_model.py
//   backend/app/fraud/feature_engineering.py
//   backend/app/fraud/scoring.py
//
// Power Query provides deterministic preparation and explainable
// dashboard-layer anomaly signals.
// ============================================================================

let

    // =========================================================================
    // 1. SOURCE
    // =========================================================================

    Source =
        try Transactions
        otherwise #table(
            {
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
                "Description",
                "Status"
            },
            {}
        ),


    // =========================================================================
    // 2. HELPER FUNCTIONS
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


    SafeDivide = (numerator as any, denominator as any) as nullable number =>
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
    // 3. STANDARDIZE COLUMN NAMES
    // =========================================================================

    Standardized =
        Table.TransformColumnNames(
            Source,
            each Text.Trim(_)
        ),


    // =========================================================================
    // 4. REQUIRED COLUMNS
    // =========================================================================

    RequiredColumns = {
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
        "Description",
        "Status"
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
                    "Description",
                    each SafeText(_),
                    type text
                },
                {
                    "Status",
                    each SafeText(_),
                    type text
                }
            }
        ),


    // =========================================================================
    // 6. CLEAN TEXT VALUES
    // =========================================================================

    CleanText =
        Table.TransformColumns(
            Typed,
            {
                {
                    "TransactionType",
                    each
                        if _ = null
                        then null
                        else Text.Upper(Text.Trim(_)),
                    type text
                },
                {
                    "TransactionCategory",
                    each
                        if _ = null
                        then null
                        else Text.Upper(Text.Trim(_)),
                    type text
                },
                {
                    "PaymentMethod",
                    each
                        if _ = null
                        then null
                        else Text.Upper(Text.Trim(_)),
                    type text
                },
                {
                    "Currency",
                    each
                        if _ = null
                        then null
                        else Text.Upper(Text.Trim(_)),
                    type text
                },
                {
                    "Status",
                    each
                        if _ = null
                        then null
                        else Text.Upper(Text.Trim(_)),
                    type text
                }
            }
        ),


    // =========================================================================
    // 7. VALID TRANSACTIONS
    // =========================================================================

    ValidTransactions =
        Table.SelectRows(
            CleanText,
            each
                [TransactionId] <> null
                and [TransactionDate] <> null
                and [Amount] <> null
        ),


    // =========================================================================
    // 8. ABSOLUTE AMOUNT
    // =========================================================================

    AddAbsoluteAmount =
        Table.AddColumn(
            ValidTransactions,
            "AnomalyAbsoluteAmount",
            each Number.Abs([Amount]),
            type number
        ),


    // =========================================================================
    // 9. DATE FEATURES
    // =========================================================================

    AddYear =
        Table.AddColumn(
            AddAbsoluteAmount,
            "AnomalyYear",
            each Date.Year([TransactionDate]),
            Int64.Type
        ),


    AddMonth =
        Table.AddColumn(
            AddYear,
            "AnomalyMonth",
            each Date.Month([TransactionDate]),
            Int64.Type
        ),


    AddMonthName =
        Table.AddColumn(
            AddMonth,
            "AnomalyMonthName",
            each Date.MonthName([TransactionDate]),
            type text
        ),


    AddQuarter =
        Table.AddColumn(
            AddMonthName,
            "AnomalyQuarter",
            each
                "Q" &
                Text.From(
                    Date.QuarterOfYear([TransactionDate])
                ),
            type text
        ),


    AddPeriodKey =
        Table.AddColumn(
            AddQuarter,
            "AnomalyPeriodKey",
            each
                Date.Year([TransactionDate]) * 100
                + Date.Month([TransactionDate]),
            Int64.Type
        ),


    // =========================================================================
    // 10. DAY FEATURES
    // =========================================================================

    AddDayOfWeek =
        Table.AddColumn(
            AddPeriodKey,
            "AnomalyDayOfWeek",
            each
                Date.DayOfWeek(
                    [TransactionDate],
                    Day.Monday
                ),
            Int64.Type
        ),


    AddDayName =
        Table.AddColumn(
            AddDayOfWeek,
            "AnomalyDayName",
            each Date.DayOfWeekName([TransactionDate]),
            type text
        ),


    AddWeekendFlag =
        Table.AddColumn(
            AddDayName,
            "AnomalyWeekendFlag",
            each
                if [AnomalyDayOfWeek] >= 5
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 11. MONTH-END FEATURE
    // =========================================================================

    AddMonthEndFlag =
        Table.AddColumn(
            AddWeekendFlag,
            "AnomalyMonthEndFlag",
            each
                if Date.Day([TransactionDate])
                    >= Date.DaysInMonth(
                        [TransactionDate]
                    ) - 2
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 12. YEAR-END FEATURE
    // =========================================================================

    AddYearEndFlag =
        Table.AddColumn(
            AddMonthEndFlag,
            "AnomalyYearEndFlag",
            each
                if
                    Date.Month([TransactionDate]) = 12
                    and Date.Day([TransactionDate]) >= 29
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 13. TRANSACTION DIRECTION
    // =========================================================================

    AddDirection =
        Table.AddColumn(
            AddYearEndFlag,
            "AnomalyTransactionDirection",
            each
                if List.Contains(
                    {
                        "PAYMENT",
                        "EXPENSE",
                        "WITHDRAWAL",
                        "REFUND",
                        "DEBIT"
                    },
                    [TransactionType]
                )
                then "OUTFLOW"

                else if List.Contains(
                    {
                        "RECEIPT",
                        "REVENUE",
                        "DEPOSIT",
                        "CREDIT",
                        "COLLECTION"
                    },
                    [TransactionType]
                )
                then "INFLOW"

                else "OTHER",
            type text
        ),


    // =========================================================================
    // 14. SIGN FEATURE
    // =========================================================================

    AddAmountSign =
        Table.AddColumn(
            AddDirection,
            "AnomalyAmountSign",
            each
                if [Amount] > 0
                then "POSITIVE"
                else if [Amount] < 0
                then "NEGATIVE"
                else "ZERO",
            type text
        ),


    // =========================================================================
    // 15. ZERO AMOUNT FLAG
    // =========================================================================

    AddZeroAmountFlag =
        Table.AddColumn(
            AddAmountSign,
            "AnomalyZeroAmountFlag",
            each
                if [Amount] = 0
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 16. HIGH VALUE FLAG
    // =========================================================================

    AddHighValueFlag =
        Table.AddColumn(
            AddZeroAmountFlag,
            "AnomalyHighValueFlag",
            each
                if [AnomalyAbsoluteAmount] >= 500000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 17. VERY HIGH VALUE FLAG
    // =========================================================================

    AddVeryHighValueFlag =
        Table.AddColumn(
            AddHighValueFlag,
            "AnomalyVeryHighValueFlag",
            each
                if [AnomalyAbsoluteAmount] >= 5000000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 18. ROUND VALUE FLAG
    // =========================================================================

    AddRoundValueFlag =
        Table.AddColumn(
            AddVeryHighValueFlag,
            "AnomalyRoundValueFlag",
            each
                if
                    [AnomalyAbsoluteAmount] <> null
                    and [AnomalyAbsoluteAmount] > 0
                    and Number.Mod(
                        [AnomalyAbsoluteAmount],
                        10000
                    ) = 0
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 19. MISSING PARTY FLAG
    // =========================================================================

    AddMissingPartyFlag =
        Table.AddColumn(
            AddRoundValueFlag,
            "AnomalyMissingPartyFlag",
            each
                if
                    [CustomerId] = null
                    and [SupplierId] = null
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 20. PAYMENT METHOD RISK
    // =========================================================================

    AddPaymentMethodFlag =
        Table.AddColumn(
            AddMissingPartyFlag,
            "AnomalyPaymentMethodFlag",
            each
                if List.Contains(
                    {
                        "CASH",
                        "MANUAL",
                        "UNKNOWN",
                        "OTHER"
                    },
                    [PaymentMethod]
                )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 21. UNUSUAL TRANSACTION CATEGORY
    // =========================================================================

    AddCategoryRiskFlag =
        Table.AddColumn(
            AddPaymentMethodFlag,
            "AnomalyCategoryRiskFlag",
            each
                if List.Contains(
                    {
                        "MISCELLANEOUS",
                        "OTHER",
                        "UNKNOWN",
                        "ADJUSTMENT",
                        "MANUAL"
                    },
                    [TransactionCategory]
                )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 22. FAILED / REVERSED TRANSACTION FLAG
    // =========================================================================

    AddStatusRiskFlag =
        Table.AddColumn(
            AddCategoryRiskFlag,
            "AnomalyStatusRiskFlag",
            each
                if List.Contains(
                    {
                        "FAILED",
                        "REVERSED",
                        "CANCELLED",
                        "VOID",
                        "REJECTED"
                    },
                    [Status]
                )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 23. DUPLICATE TRANSACTION ANALYSIS
    // =========================================================================

    DuplicateGroups =
        Table.Group(
            AddStatusRiskFlag,
            {
                "CompanyId",
                "TransactionDate",
                "Amount",
                "CustomerId",
                "SupplierId"
            },
            {
                {
                    "AnomalyDuplicateCount",
                    each Table.RowCount(_),
                    Int64.Type
                }
            }
        ),


    JoinDuplicateData =
        Table.NestedJoin(
            AddStatusRiskFlag,
            {
                "CompanyId",
                "TransactionDate",
                "Amount",
                "CustomerId",
                "SupplierId"
            },
            DuplicateGroups,
            {
                "CompanyId",
                "TransactionDate",
                "Amount",
                "CustomerId",
                "SupplierId"
            },
            "DuplicateData",
            JoinKind.LeftOuter
        ),


    ExpandDuplicateData =
        Table.ExpandTableColumn(
            JoinDuplicateData,
            "DuplicateData",
            {
                "AnomalyDuplicateCount"
            },
            {
                "AnomalyDuplicateCount"
            }
        ),


    AddDuplicateFlag =
        Table.AddColumn(
            ExpandDuplicateData,
            "AnomalyDuplicateFlag",
            each
                if [AnomalyDuplicateCount] > 1
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 24. CUSTOMER TRANSACTION FREQUENCY
    // =========================================================================

    CustomerFrequency =
        Table.Group(
            AddDuplicateFlag,
            {
                "CompanyId",
                "CustomerId"
            },
            {
                {
                    "CustomerTransactionCount",
                    each Table.RowCount(_),
                    Int64.Type
                },
                {
                    "CustomerTotalAmount",
                    each
                        List.Sum(
                            List.Transform(
                                [AnomalyAbsoluteAmount],
                                each
                                    if _ = null
                                    then 0
                                    else _
                            )
                        ),
                    type number
                }
            }
        ),


    JoinCustomerFrequency =
        Table.NestedJoin(
            AddDuplicateFlag,
            {
                "CompanyId",
                "CustomerId"
            },
            CustomerFrequency,
            {
                "CompanyId",
                "CustomerId"
            },
            "CustomerFrequency",
            JoinKind.LeftOuter
        ),


    ExpandCustomerFrequency =
        Table.ExpandTableColumn(
            JoinCustomerFrequency,
            "CustomerFrequency",
            {
                "CustomerTransactionCount",
                "CustomerTotalAmount"
            },
            {
                "CustomerTransactionCount",
                "CustomerTotalAmount"
            }
        ),


    // =========================================================================
    // 25. SUPPLIER TRANSACTION FREQUENCY
    // =========================================================================

    SupplierFrequency =
        Table.Group(
            ExpandCustomerFrequency,
            {
                "CompanyId",
                "SupplierId"
            },
            {
                {
                    "SupplierTransactionCount",
                    each Table.RowCount(_),
                    Int64.Type
                },
                {
                    "SupplierTotalAmount",
                    each
                        List.Sum(
                            List.Transform(
                                [AnomalyAbsoluteAmount],
                                each
                                    if _ = null
                                    then 0
                                    else _
                            )
                        ),
                    type number
                }
            }
        ),


    JoinSupplierFrequency =
        Table.NestedJoin(
            ExpandCustomerFrequency,
            {
                "CompanyId",
                "SupplierId"
            },
            SupplierFrequency,
            {
                "CompanyId",
                "SupplierId"
            },
            "SupplierFrequency",
            JoinKind.LeftOuter
        ),


    ExpandSupplierFrequency =
        Table.ExpandTableColumn(
            JoinSupplierFrequency,
            "SupplierFrequency",
            {
                "SupplierTransactionCount",
                "SupplierTotalAmount"
            },
            {
                "SupplierTransactionCount",
                "SupplierTotalAmount"
            }
        ),


    // =========================================================================
    // 26. CUSTOMER CONCENTRATION
    // =========================================================================

    AddCustomerConcentration =
        Table.AddColumn(
            ExpandSupplierFrequency,
            "AnomalyCustomerAmountShare",
            each
                SafeDivide(
                    [AnomalyAbsoluteAmount],
                    [CustomerTotalAmount]
                ),
            type number
        ),


    // =========================================================================
    // 27. SUPPLIER CONCENTRATION
    // =========================================================================

    AddSupplierConcentration =
        Table.AddColumn(
            AddCustomerConcentration,
            "AnomalySupplierAmountShare",
            each
                SafeDivide(
                    [AnomalyAbsoluteAmount],
                    [SupplierTotalAmount]
                ),
            type number
        ),


    // =========================================================================
    // 28. HIGH CUSTOMER FREQUENCY
    // =========================================================================

    AddHighCustomerFrequency =
        Table.AddColumn(
            AddSupplierConcentration,
            "AnomalyHighCustomerFrequencyFlag",
            each
                if
                    [CustomerTransactionCount] <> null
                    and [CustomerTransactionCount] >= 100
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 29. HIGH SUPPLIER FREQUENCY
    // =========================================================================

    AddHighSupplierFrequency =
        Table.AddColumn(
            AddHighCustomerFrequency,
            "AnomalyHighSupplierFrequencyFlag",
            each
                if
                    [SupplierTransactionCount] <> null
                    and [SupplierTransactionCount] >= 100
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 30. EXTREME CONCENTRATION
    // =========================================================================

    AddConcentrationFlag =
        Table.AddColumn(
            AddHighSupplierFrequency,
            "AnomalyConcentrationFlag",
            each
                if
                    (
                        [AnomalyCustomerAmountShare] <> null
                        and [AnomalyCustomerAmountShare] >= 0.50
                    )
                    or
                    (
                        [AnomalySupplierAmountShare] <> null
                        and [AnomalySupplierAmountShare] >= 0.50
                    )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 31. RULE-BASED ANOMALY SCORE
    // =========================================================================

    AddAnomalyRuleScore =
        Table.AddColumn(
            AddConcentrationFlag,
            "AnomalyRuleScore",
            each
                Clamp(
                    [AnomalyDuplicateFlag] * 25
                    + [AnomalyVeryHighValueFlag] * 20
                    + [AnomalyHighValueFlag] * 10
                    + [AnomalyRoundValueFlag] * 10
                    + [AnomalyWeekendFlag] * 10
                    + [AnomalyMonthEndFlag] * 8
                    + [AnomalyYearEndFlag] * 5
                    + [AnomalyPaymentMethodFlag] * 5
                    + [AnomalyMissingPartyFlag] * 5
                    + [AnomalyStatusRiskFlag] * 5
                    + [AnomalyCategoryRiskFlag] * 3
                    + [AnomalyConcentrationFlag] * 10,
                    0,
                    100
                ),
            type number
        ),


    // =========================================================================
    // 32. ANOMALY BAND
    // =========================================================================

    AddAnomalyBand =
        Table.AddColumn(
            AddAnomalyRuleScore,
            "AnomalyRiskBand",
            each
                if [AnomalyRuleScore] >= 80
                then "CRITICAL"
                else if [AnomalyRuleScore] >= 60
                then "HIGH"
                else if [AnomalyRuleScore] >= 35
                then "MEDIUM"
                else if [AnomalyRuleScore] >= 15
                then "LOW"
                else "INFO",
            type text
        ),


    // =========================================================================
    // 33. ANOMALY SIGNAL COUNT
    // =========================================================================

    AddSignalCount =
        Table.AddColumn(
            AddAnomalyBand,
            "AnomalySignalCount",
            each
                [AnomalyDuplicateFlag]
                + [AnomalyVeryHighValueFlag]
                + [AnomalyHighValueFlag]
                + [AnomalyRoundValueFlag]
                + [AnomalyWeekendFlag]
                + [AnomalyMonthEndFlag]
                + [AnomalyYearEndFlag]
                + [AnomalyPaymentMethodFlag]
                + [AnomalyMissingPartyFlag]
                + [AnomalyStatusRiskFlag]
                + [AnomalyCategoryRiskFlag]
                + [AnomalyConcentrationFlag],
            Int64.Type
        ),


    // =========================================================================
    // 34. SIGNAL STRENGTH
    // =========================================================================

    AddSignalStrength =
        Table.AddColumn(
            AddSignalCount,
            "AnomalySignalStrength",
            each
                if [AnomalySignalCount] >= 6
                then "VERY_HIGH"
                else if [AnomalySignalCount] >= 4
                then "HIGH"
                else if [AnomalySignalCount] >= 2
                then "MEDIUM"
                else if [AnomalySignalCount] = 1
                then "LOW"
                else "NONE",
            type text
        ),


    // =========================================================================
    // 35. INVESTIGATION PRIORITY
    // =========================================================================

    AddInvestigationPriority =
        Table.AddColumn(
            AddSignalStrength,
            "AnomalyInvestigationPriority",
            each
                if [AnomalyRuleScore] >= 80
                then "P1 — CRITICAL"
                else if [AnomalyRuleScore] >= 60
                then "P2 — HIGH"
                else if [AnomalyRuleScore] >= 35
                then "P3 — MEDIUM"
                else "P4 — LOW",
            type text
        ),


    // =========================================================================
    // 36. INVESTIGATION FLAG
    // =========================================================================

    AddInvestigationFlag =
        Table.AddColumn(
            AddInvestigationPriority,
            "AnomalyInvestigationRequired",
            each
                if [AnomalyRuleScore] >= 60
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 37. ANOMALY EXPLANATION
    // =========================================================================

    AddExplanation =
        Table.AddColumn(
            AddInvestigationFlag,
            "AnomalyExplanation",
            each
                if [AnomalyDuplicateFlag] = 1
                then "Potential duplicate transaction"

                else if [AnomalyVeryHighValueFlag] = 1
                then "Very high-value transaction"

                else if [AnomalyHighValueFlag] = 1
                then "High-value transaction"

                else if [AnomalyConcentrationFlag] = 1
                then "High customer or supplier concentration"

                else if [AnomalyWeekendFlag] = 1
                then "Transaction occurred during weekend"

                else if [AnomalyMonthEndFlag] = 1
                then "Transaction occurred near month-end"

                else if [AnomalyYearEndFlag] = 1
                then "Transaction occurred near year-end"

                else if [AnomalyRoundValueFlag] = 1
                then "Unusual round-value transaction"

                else if [AnomalyStatusRiskFlag] = 1
                then "Transaction has failed/reversed/cancelled status"

                else if [AnomalyPaymentMethodFlag] = 1
                then "Higher-risk payment method"

                else if [AnomalyMissingPartyFlag] = 1
                then "Missing transaction counterparty"

                else
                    "No significant deterministic anomaly detected",
            type text
        ),


    // =========================================================================
    // 38. ML FEATURE — LOG AMOUNT
    // =========================================================================

    AddLogAmount =
        Table.AddColumn(
            AddExplanation,
            "AnomalyLogAmount",
            each
                if [AnomalyAbsoluteAmount] > 0
                then Number.Log(1 + [AnomalyAbsoluteAmount])
                else 0,
            type number
        ),


    // =========================================================================
    // 39. ML FEATURE — AMOUNT BUCKET
    // =========================================================================

    AddAmountBucket =
        Table.AddColumn(
            AddLogAmount,
            "AnomalyAmountBucket",
            each
                if [AnomalyAbsoluteAmount] >= 5000000
                then "EXTREME"

                else if [AnomalyAbsoluteAmount] >= 1000000
                then "VERY_HIGH"

                else if [AnomalyAbsoluteAmount] >= 500000
                then "HIGH"

                else if [AnomalyAbsoluteAmount] >= 100000
                then "MEDIUM"

                else if [AnomalyAbsoluteAmount] >= 10000
                then "LOW"

                else "VERY_LOW",
            type text
        ),


    // =========================================================================
    // 40. ML FEATURE — PARTY COMPLETENESS
    // =========================================================================

    AddPartyCompleteness =
        Table.AddColumn(
            AddAmountBucket,
            "AnomalyPartyCompleteness",
            each
                (
                    (if [CustomerId] <> null then 1 else 0)
                    +
                    (if [SupplierId] <> null then 1 else 0)
                    +
                    (if [AccountId] <> null then 1 else 0)
                )
                / 3,
            type number
        ),


    // =========================================================================
    // 41. ML FEATURE — TRANSACTION COMPLETENESS
    // =========================================================================

    AddTransactionCompleteness =
        Table.AddColumn(
            AddPartyCompleteness,
            "AnomalyTransactionCompleteness",
            each
                let
                    Fields = {
                        [TransactionId],
                        [CompanyId],
                        [TransactionDate],
                        [Amount],
                        [TransactionType],
                        [TransactionCategory],
                        [PaymentMethod],
                        [Currency]
                    },

                    Available =
                        List.Count(
                            List.Select(
                                Fields,
                                each _ <> null
                            )
                        )
                in
                    Available / List.Count(Fields),
            type number
        ),


    // =========================================================================
    // 42. DATA QUALITY
    // =========================================================================

    AddDataQuality =
        Table.AddColumn(
            AddTransactionCompleteness,
            "AnomalyDataQuality",
            each
                if [AnomalyTransactionCompleteness] >= 0.95
                then "HIGH"
                else if [AnomalyTransactionCompleteness] >= 0.80
                then "MEDIUM"
                else if [AnomalyTransactionCompleteness] >= 0.60
                then "LOW"
                else "CRITICAL",
            type text
        ),


    // =========================================================================
    // 43. ML READINESS
    // =========================================================================

    AddMLReady =
        Table.AddColumn(
            AddDataQuality,
            "AnomalyMLReady",
            each
                if
                    [AnomalyTransactionCompleteness] >= 0.80
                    and [Amount] <> null
                    and [TransactionDate] <> null
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 44. ML FEATURE VECTOR STATUS
    // =========================================================================

    AddFeatureStatus =
        Table.AddColumn(
            AddMLReady,
            "AnomalyFeatureStatus",
            each
                if [AnomalyMLReady] = "YES"
                then "FEATURES_READY"
                else "FEATURES_INCOMPLETE",
            type text
        ),


    // =========================================================================
    // 45. SUPERVISOR AGENT SIGNAL
    // =========================================================================

    AddSupervisorSignal =
        Table.AddColumn(
            AddFeatureStatus,
            "AnomalySupervisorAgentSignal",
            each
                if [AnomalyRuleScore] >= 80
                then "ESCALATE_TO_SUPERVISOR"

                else if [AnomalyRuleScore] >= 60
                then "START_FRAUD_INVESTIGATION"

                else if [AnomalyRuleScore] >= 35
                then "REVIEW_ANOMALY"

                else "NO_IMMEDIATE_ACTION",
            type text
        ),


    // =========================================================================
    // 46. RAG EVIDENCE TRIGGER
    // =========================================================================

    AddRAGTrigger =
        Table.AddColumn(
            AddSupervisorSignal,
            "AnomalyRAGEvidenceRequired",
            each
                if [AnomalyRuleScore] >= 35
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 47. HUMAN REVIEW
    // =========================================================================

    AddHumanReview =
        Table.AddColumn(
            AddRAGTrigger,
            "AnomalyHumanReviewRequired",
            each
                if [AnomalyRuleScore] >= 60
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 48. ANOMALY DECISION
    // =========================================================================

    AddDecision =
        Table.AddColumn(
            AddHumanReview,
            "AnomalyDecision",
            each
                if [AnomalyRuleScore] >= 80
                then "BLOCK_AND_ESCALATE"

                else if [AnomalyRuleScore] >= 60
                then "HOLD_AND_INVESTIGATE"

                else if [AnomalyRuleScore] >= 35
                then "MONITOR"

                else "NORMAL",
            type text
        ),


    // =========================================================================
    // 49. ANOMALY STATUS
    // =========================================================================

    AddStatus =
        Table.AddColumn(
            AddDecision,
            "AnomalyStatus",
            each
                if [AnomalyRuleScore] >= 80
                then "CRITICAL_ANOMALY"

                else if [AnomalyRuleScore] >= 60
                then "HIGH_ANOMALY"

                else if [AnomalyRuleScore] >= 35
                then "ANOMALY_WATCH"

                else "NORMAL",
            type text
        ),


    // =========================================================================
    // 50. ANOMALY RECORD KEY
    // =========================================================================

    AddRecordKey =
        Table.AddColumn(
            AddStatus,
            "AnomalyRecordKey",
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
    // 51. ROUND NUMERIC FEATURES
    // =========================================================================

    Rounded =
        Table.TransformColumns(
            AddRecordKey,
            {
                {
                    "AnomalyAbsoluteAmount",
                    each Number.Round(_, 2),
                    type number
                },
                {
                    "AnomalyRuleScore",
                    each Number.Round(_, 2),
                    type number
                },
                {
                    "AnomalyCustomerAmountShare",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 4),
                    type number
                },
                {
                    "AnomalySupplierAmountShare",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 4),
                    type number
                },
                {
                    "AnomalyLogAmount",
                    each Number.Round(_, 4),
                    type number
                },
                {
                    "AnomalyPartyCompleteness",
                    each Number.Round(_, 4),
                    type number
                },
                {
                    "AnomalyTransactionCompleteness",
                    each Number.Round(_, 4),
                    type number
                }
            }
        ),


    // =========================================================================
    // 52. SORT
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
                    "AnomalyRuleScore",
                    Order.Descending
                },
                {
                    "AnomalyAbsoluteAmount",
                    Order.Descending
                }
            }
        ),


    // =========================================================================
    // 53. PREFERRED COLUMN ORDER
    // =========================================================================

    PreferredColumns = {
        "AnomalyRecordKey",

        "TransactionId",
        "CompanyId",
        "CompanyName",
        "TransactionDate",

        "AnomalyYear",
        "AnomalyMonth",
        "AnomalyMonthName",
        "AnomalyQuarter",
        "AnomalyPeriodKey",
        "AnomalyDayOfWeek",
        "AnomalyDayName",

        "CustomerId",
        "SupplierId",
        "AccountId",

        "TransactionType",
        "TransactionCategory",
        "PaymentMethod",
        "Currency",
        "Status",

        "Amount",
        "AnomalyAbsoluteAmount",
        "AnomalyAmountSign",
        "AnomalyTransactionDirection",

        "AnomalyZeroAmountFlag",
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

        "AnomalyDuplicateCount",
        "AnomalyDuplicateFlag",

        "CustomerTransactionCount",
        "CustomerTotalAmount",
        "SupplierTransactionCount",
        "SupplierTotalAmount",

        "AnomalyCustomerAmountShare",
        "AnomalySupplierAmountShare",

        "AnomalyHighCustomerFrequencyFlag",
        "AnomalyHighSupplierFrequencyFlag",
        "AnomalyConcentrationFlag",

        "AnomalyRuleScore",
        "AnomalyRiskBand",
        "AnomalySignalCount",
        "AnomalySignalStrength",

        "AnomalyInvestigationPriority",
        "AnomalyInvestigationRequired",
        "AnomalyExplanation",

        "AnomalyLogAmount",
        "AnomalyAmountBucket",
        "AnomalyPartyCompleteness",
        "AnomalyTransactionCompleteness",

        "AnomalyDataQuality",
        "AnomalyMLReady",
        "AnomalyFeatureStatus",

        "AnomalySupervisorAgentSignal",
        "AnomalyRAGEvidenceRequired",
        "AnomalyHumanReviewRequired",

        "AnomalyDecision",
        "AnomalyStatus",

        "Description"
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