// ============================================================================
// FinCo AI — Fraud Transformations
// File: fraud_transformations.pq
// ============================================================================
//
// PURPOSE
// -------
// Enterprise fraud-specific transformation and feature-engineering layer.
//
// ARCHITECTURE
// ------------
//
//   Transactions
//        ↓
//   common_transformations.pq
//        ↓
//   fraud_transformations.pq
//        ↓
//   anomaly_query.pq
//        ↓
//   fraud_prediction_query.pq
//        ↓
//   Fraud Dashboard
//        ↓
//   Alert Engine / Supervisor Agent
//
// EXPECTED UPSTREAM QUERY
// -----------------------
//   common_transformations
//
// FALLBACK
// --------
// If common_transformations is unavailable, the query attempts to use
// Transactions directly.
//
// IMPORTANT
// ---------
// This query creates deterministic, explainable fraud features.
// Production ML scoring remains in:
//
//   backend/app/fraud/feature_engineering.py
//   backend/app/fraud/anomaly_model.py
//   backend/app/fraud/supervised_model.py
//   backend/app/fraud/scoring.py
//   backend/app/fraud/explainability.py
//
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
    // 2. SOURCE
    // =========================================================================

    Source =
        try common_transformations
        otherwise
            try Transactions
            otherwise
                #table(
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
    // 3. STANDARDIZE COLUMN NAMES
    // =========================================================================

    Standardized =
        Table.TransformColumnNames(
            Source,
            each Text.Trim(_)
        ),


    // =========================================================================
    // 4. REQUIRED BASE COLUMNS
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
    // 6. TEXT NORMALIZATION
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
    // 7. REMOVE INVALID TRANSACTIONS
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
    // 8. TRANSACTION KEY
    // =========================================================================

    AddFraudTransactionKey =
        Table.AddColumn(
            ValidTransactions,
            "FraudTransactionKey",
            each
                Text.Combine(
                    {
                        if [CompanyId] = null
                        then ""
                        else [CompanyId],

                        if [TransactionId] = null
                        then ""
                        else [TransactionId]
                    },
                    "|"
                ),
            type text
        ),


    // =========================================================================
    // 9. ABSOLUTE AMOUNT
    // =========================================================================

    AddAbsoluteAmount =
        Table.AddColumn(
            AddFraudTransactionKey,
            "FraudAbsoluteAmount",
            each
                Number.Abs(
                    [Amount]
                ),
            type number
        ),


    // =========================================================================
    // 10. LOG AMOUNT
    // =========================================================================

    AddLogAmount =
        Table.AddColumn(
            AddAbsoluteAmount,
            "FraudLogAmount",
            each
                if [FraudAbsoluteAmount] > 0
                then
                    Number.Log(
                        1 + [FraudAbsoluteAmount]
                    )
                else 0,
            type number
        ),


    // =========================================================================
    // 11. DATE FEATURES
    // =========================================================================

    AddFraudYear =
        Table.AddColumn(
            AddLogAmount,
            "FraudYear",
            each Date.Year([TransactionDate]),
            Int64.Type
        ),


    AddFraudMonth =
        Table.AddColumn(
            AddFraudYear,
            "FraudMonth",
            each Date.Month([TransactionDate]),
            Int64.Type
        ),


    AddFraudMonthName =
        Table.AddColumn(
            AddFraudMonth,
            "FraudMonthName",
            each Date.MonthName([TransactionDate]),
            type text
        ),


    AddFraudQuarter =
        Table.AddColumn(
            AddFraudMonthName,
            "FraudQuarter",
            each
                "Q"
                &
                Text.From(
                    Date.QuarterOfYear(
                        [TransactionDate]
                    )
                ),
            type text
        ),


    AddFraudPeriodKey =
        Table.AddColumn(
            AddFraudQuarter,
            "FraudPeriodKey",
            each
                Date.Year([TransactionDate]) * 100
                + Date.Month([TransactionDate]),
            Int64.Type
        ),


    // =========================================================================
    // 12. DAY FEATURES
    // =========================================================================

    AddDayOfWeek =
        Table.AddColumn(
            AddFraudPeriodKey,
            "FraudDayOfWeek",
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
            "FraudDayName",
            each
                Date.DayOfWeekName(
                    [TransactionDate]
                ),
            type text
        ),


    AddDayOfMonth =
        Table.AddColumn(
            AddDayName,
            "FraudDayOfMonth",
            each
                Date.Day(
                    [TransactionDate]
                ),
            Int64.Type
        ),


    // =========================================================================
    // 13. WEEKEND FLAG
    // =========================================================================

    AddWeekendFlag =
        Table.AddColumn(
            AddDayOfMonth,
            "FraudWeekendFlag",
            each
                if [FraudDayOfWeek] >= 5
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 14. MONTH-END FLAG
    // =========================================================================

    AddMonthEndFlag =
        Table.AddColumn(
            AddWeekendFlag,
            "FraudMonthEndFlag",
            each
                if
                    [FraudDayOfMonth]
                    >=
                    Date.DaysInMonth(
                        [TransactionDate]
                    ) - 2
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 15. YEAR-END FLAG
    // =========================================================================

    AddYearEndFlag =
        Table.AddColumn(
            AddMonthEndFlag,
            "FraudYearEndFlag",
            each
                if
                    Date.Month(
                        [TransactionDate]
                    ) = 12
                    and
                    Date.Day(
                        [TransactionDate]
                    ) >= 29
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 16. QUARTER-END FLAG
    // =========================================================================

    AddQuarterEndFlag =
        Table.AddColumn(
            AddYearEndFlag,
            "FraudQuarterEndFlag",
            each
                if
                    (
                        Date.Month(
                            [TransactionDate]
                        )
                        = 3
                        or
                        Date.Month(
                            [TransactionDate]
                        )
                        = 6
                        or
                        Date.Month(
                            [TransactionDate]
                        )
                        = 9
                        or
                        Date.Month(
                            [TransactionDate]
                        )
                        = 12
                    )
                    and
                    Date.Day(
                        [TransactionDate]
                    ) >= 28
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 17. TRANSACTION DIRECTION
    // =========================================================================

    AddDirection =
        Table.AddColumn(
            AddQuarterEndFlag,
            "FraudTransactionDirection",
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
    // 18. AMOUNT SIGN
    // =========================================================================

    AddAmountSign =
        Table.AddColumn(
            AddDirection,
            "FraudAmountSign",
            each
                if [Amount] > 0
                then "POSITIVE"
                else if [Amount] < 0
                then "NEGATIVE"
                else "ZERO",
            type text
        ),


    // =========================================================================
    // 19. ZERO VALUE FLAG
    // =========================================================================

    AddZeroValueFlag =
        Table.AddColumn(
            AddAmountSign,
            "FraudZeroValueFlag",
            each
                if [Amount] = 0
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 20. AMOUNT BUCKET
    // =========================================================================

    AddAmountBucket =
        Table.AddColumn(
            AddZeroValueFlag,
            "FraudAmountBucket",
            each
                if [FraudAbsoluteAmount] >= 5000000
                then "EXTREME"

                else if [FraudAbsoluteAmount] >= 1000000
                then "VERY_HIGH"

                else if [FraudAbsoluteAmount] >= 500000
                then "HIGH"

                else if [FraudAbsoluteAmount] >= 100000
                then "MEDIUM"

                else if [FraudAbsoluteAmount] >= 10000
                then "LOW"

                else "VERY_LOW",
            type text
        ),


    // =========================================================================
    // 21. HIGH VALUE FLAG
    // =========================================================================

    AddHighValueFlag =
        Table.AddColumn(
            AddAmountBucket,
            "FraudHighValueFlag",
            each
                if [FraudAbsoluteAmount] >= 500000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 22. VERY HIGH VALUE FLAG
    // =========================================================================

    AddVeryHighValueFlag =
        Table.AddColumn(
            AddHighValueFlag,
            "FraudVeryHighValueFlag",
            each
                if [FraudAbsoluteAmount] >= 5000000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 23. EXTREME VALUE FLAG
    // =========================================================================

    AddExtremeValueFlag =
        Table.AddColumn(
            AddVeryHighValueFlag,
            "FraudExtremeValueFlag",
            each
                if [FraudAbsoluteAmount] >= 10000000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 24. ROUND AMOUNT FLAG
    // =========================================================================

    AddRoundAmountFlag =
        Table.AddColumn(
            AddExtremeValueFlag,
            "FraudRoundAmountFlag",
            each
                if
                    [FraudAbsoluteAmount] <> null
                    and [FraudAbsoluteAmount] > 0
                    and Number.Mod(
                        [FraudAbsoluteAmount],
                        10000
                    ) = 0
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 25. LARGE ROUND AMOUNT
    // =========================================================================

    AddLargeRoundAmountFlag =
        Table.AddColumn(
            AddRoundAmountFlag,
            "FraudLargeRoundAmountFlag",
            each
                if
                    [FraudAbsoluteAmount] >= 100000
                    and
                    Number.Mod(
                        [FraudAbsoluteAmount],
                        100000
                    ) = 0
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 26. PAYMENT METHOD RISK
    // =========================================================================

    AddPaymentMethodRisk =
        Table.AddColumn(
            AddLargeRoundAmountFlag,
            "FraudPaymentMethodRiskFlag",
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
    // 27. DIGITAL PAYMENT METHOD
    // =========================================================================

    AddDigitalPaymentFlag =
        Table.AddColumn(
            AddPaymentMethodRisk,
            "FraudDigitalPaymentFlag",
            each
                if List.Contains(
                    {
                        "UPI",
                        "CARD",
                        "CREDIT_CARD",
                        "DEBIT_CARD",
                        "BANK_TRANSFER",
                        "WIRE",
                        "NEFT",
                        "RTGS",
                        "IMPS",
                        "ONLINE"
                    },
                    [PaymentMethod]
                )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 28. MANUAL TRANSACTION FLAG
    // =========================================================================

    AddManualTransactionFlag =
        Table.AddColumn(
            AddDigitalPaymentFlag,
            "FraudManualTransactionFlag",
            each
                if
                    [PaymentMethod] = "MANUAL"
                    or
                    [TransactionCategory] = "MANUAL"
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 29. UNKNOWN CATEGORY FLAG
    // =========================================================================

    AddUnknownCategoryFlag =
        Table.AddColumn(
            AddManualTransactionFlag,
            "FraudUnknownCategoryFlag",
            each
                if List.Contains(
                    {
                        "UNKNOWN",
                        "OTHER",
                        "MISCELLANEOUS"
                    },
                    [TransactionCategory]
                )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 30. SUSPICIOUS TRANSACTION TYPE
    // =========================================================================

    AddSuspiciousTypeFlag =
        Table.AddColumn(
            AddUnknownCategoryFlag,
            "FraudSuspiciousTypeFlag",
            each
                if List.Contains(
                    {
                        "ADJUSTMENT",
                        "MANUAL",
                        "REVERSAL",
                        "REVERSING",
                        "REFUND",
                        "WRITE_OFF",
                        "WRITE-OFF",
                        "JOURNAL"
                    },
                    [TransactionType]
                )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 31. STATUS RISK
    // =========================================================================

    AddStatusRisk =
        Table.AddColumn(
            AddSuspiciousTypeFlag,
            "FraudStatusRiskFlag",
            each
                if List.Contains(
                    {
                        "FAILED",
                        "REVERSED",
                        "CANCELLED",
                        "VOID",
                        "REJECTED",
                        "DISPUTED"
                    },
                    [Status]
                )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 32. MISSING COUNTERPARTY
    // =========================================================================

    AddMissingCounterparty =
        Table.AddColumn(
            AddStatusRisk,
            "FraudMissingCounterpartyFlag",
            each
                if
                    [CustomerId] = null
                    and [SupplierId] = null
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 33. MISSING ACCOUNT
    // =========================================================================

    AddMissingAccount =
        Table.AddColumn(
            AddMissingCounterparty,
            "FraudMissingAccountFlag",
            each
                if [AccountId] = null
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 34. DESCRIPTION QUALITY
    // =========================================================================

    AddMissingDescription =
        Table.AddColumn(
            AddMissingAccount,
            "FraudMissingDescriptionFlag",
            each
                if
                    [Description] = null
                    or
                    Text.Length(
                        Text.Trim(
                            [Description]
                        )
                    ) = 0
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 35. DUPLICATE TRANSACTION GROUP
    // =========================================================================

    DuplicateGroups =
        Table.Group(
            AddMissingDescription,
            {
                "CompanyId",
                "TransactionDate",
                "Amount",
                "CustomerId",
                "SupplierId",
                "TransactionType"
            },
            {
                {
                    "FraudDuplicateCount",
                    each Table.RowCount(_),
                    Int64.Type
                }
            }
        ),


    JoinDuplicateGroups =
        Table.NestedJoin(
            AddMissingDescription,
            {
                "CompanyId",
                "TransactionDate",
                "Amount",
                "CustomerId",
                "SupplierId",
                "TransactionType"
            },
            DuplicateGroups,
            {
                "CompanyId",
                "TransactionDate",
                "Amount",
                "CustomerId",
                "SupplierId",
                "TransactionType"
            },
            "DuplicateData",
            JoinKind.LeftOuter
        ),


    ExpandDuplicateGroups =
        Table.ExpandTableColumn(
            JoinDuplicateGroups,
            "DuplicateData",
            {
                "FraudDuplicateCount"
            },
            {
                "FraudDuplicateCount"
            }
        ),


    // =========================================================================
    // 36. DUPLICATE FLAG
    // =========================================================================

    AddDuplicateFlag =
        Table.AddColumn(
            ExpandDuplicateGroups,
            "FraudDuplicateFlag",
            each
                if [FraudDuplicateCount] > 1
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 37. CUSTOMER AGGREGATION
    // =========================================================================

    CustomerGroups =
        Table.Group(
            AddDuplicateFlag,
            {
                "CompanyId",
                "CustomerId"
            },
            {
                {
                    "FraudCustomerTransactionCount",
                    each Table.RowCount(_),
                    Int64.Type
                },
                {
                    "FraudCustomerTotalAmount",
                    each
                        List.Sum(
                            List.Transform(
                                [FraudAbsoluteAmount],
                                each
                                    if _ = null
                                    then 0
                                    else _
                            )
                        ),
                    type number
                },
                {
                    "FraudCustomerAverageAmount",
                    each
                        List.Average(
                            List.RemoveNulls(
                                [FraudAbsoluteAmount]
                            )
                        ),
                    type number
                }
            }
        ),


    JoinCustomerGroups =
        Table.NestedJoin(
            AddDuplicateFlag,
            {
                "CompanyId",
                "CustomerId"
            },
            CustomerGroups,
            {
                "CompanyId",
                "CustomerId"
            },
            "CustomerData",
            JoinKind.LeftOuter
        ),


    ExpandCustomerGroups =
        Table.ExpandTableColumn(
            JoinCustomerGroups,
            "CustomerData",
            {
                "FraudCustomerTransactionCount",
                "FraudCustomerTotalAmount",
                "FraudCustomerAverageAmount"
            },
            {
                "FraudCustomerTransactionCount",
                "FraudCustomerTotalAmount",
                "FraudCustomerAverageAmount"
            }
        ),


    // =========================================================================
    // 38. CUSTOMER AMOUNT SHARE
    // =========================================================================

    AddCustomerAmountShare =
        Table.AddColumn(
            ExpandCustomerGroups,
            "FraudCustomerAmountShare",
            each
                SafeDivide(
                    [FraudAbsoluteAmount],
                    [FraudCustomerTotalAmount]
                ),
            type number
        ),


    // =========================================================================
    // 39. CUSTOMER HIGH FREQUENCY
    // =========================================================================

    AddCustomerFrequencyFlag =
        Table.AddColumn(
            AddCustomerAmountShare,
            "FraudCustomerHighFrequencyFlag",
            each
                if
                    [FraudCustomerTransactionCount] <> null
                    and
                    [FraudCustomerTransactionCount] >= 100
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 40. CUSTOMER HIGH VALUE
    // =========================================================================

    AddCustomerHighValueFlag =
        Table.AddColumn(
            AddCustomerFrequencyFlag,
            "FraudCustomerHighValueFlag",
            each
                if
                    [FraudCustomerAverageAmount] <> null
                    and
                    [FraudCustomerAverageAmount] >= 500000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 41. SUPPLIER AGGREGATION
    // =========================================================================

    SupplierGroups =
        Table.Group(
            AddCustomerHighValueFlag,
            {
                "CompanyId",
                "SupplierId"
            },
            {
                {
                    "FraudSupplierTransactionCount",
                    each Table.RowCount(_),
                    Int64.Type
                },
                {
                    "FraudSupplierTotalAmount",
                    each
                        List.Sum(
                            List.Transform(
                                [FraudAbsoluteAmount],
                                each
                                    if _ = null
                                    then 0
                                    else _
                            )
                        ),
                    type number
                },
                {
                    "FraudSupplierAverageAmount",
                    each
                        List.Average(
                            List.RemoveNulls(
                                [FraudAbsoluteAmount]
                            )
                        ),
                    type number
                }
            }
        ),


    JoinSupplierGroups =
        Table.NestedJoin(
            AddCustomerHighValueFlag,
            {
                "CompanyId",
                "SupplierId"
            },
            SupplierGroups,
            {
                "CompanyId",
                "SupplierId"
            },
            "SupplierData",
            JoinKind.LeftOuter
        ),


    ExpandSupplierGroups =
        Table.ExpandTableColumn(
            JoinSupplierGroups,
            "SupplierData",
            {
                "FraudSupplierTransactionCount",
                "FraudSupplierTotalAmount",
                "FraudSupplierAverageAmount"
            },
            {
                "FraudSupplierTransactionCount",
                "FraudSupplierTotalAmount",
                "FraudSupplierAverageAmount"
            }
        ),


    // =========================================================================
    // 42. SUPPLIER AMOUNT SHARE
    // =========================================================================

    AddSupplierAmountShare =
        Table.AddColumn(
            ExpandSupplierGroups,
            "FraudSupplierAmountShare",
            each
                SafeDivide(
                    [FraudAbsoluteAmount],
                    [FraudSupplierTotalAmount]
                ),
            type number
        ),


    // =========================================================================
    // 43. SUPPLIER HIGH FREQUENCY
    // =========================================================================

    AddSupplierFrequencyFlag =
        Table.AddColumn(
            AddSupplierAmountShare,
            "FraudSupplierHighFrequencyFlag",
            each
                if
                    [FraudSupplierTransactionCount] <> null
                    and
                    [FraudSupplierTransactionCount] >= 100
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 44. SUPPLIER HIGH VALUE
    // =========================================================================

    AddSupplierHighValueFlag =
        Table.AddColumn(
            AddSupplierFrequencyFlag,
            "FraudSupplierHighValueFlag",
            each
                if
                    [FraudSupplierAverageAmount] <> null
                    and
                    [FraudSupplierAverageAmount] >= 500000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 45. CONCENTRATION FLAG
    // =========================================================================

    AddConcentrationFlag =
        Table.AddColumn(
            AddSupplierHighValueFlag,
            "FraudConcentrationFlag",
            each
                if
                    (
                        [FraudCustomerAmountShare] <> null
                        and
                        [FraudCustomerAmountShare] >= 0.50
                    )
                    or
                    (
                        [FraudSupplierAmountShare] <> null
                        and
                        [FraudSupplierAmountShare] >= 0.50
                    )
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 46. MULTI-FACTOR SIGNAL COUNT
    // =========================================================================

    AddSignalCount =
        Table.AddColumn(
            AddConcentrationFlag,
            "FraudSignalCount",
            each
                [FraudDuplicateFlag]
                +
                [FraudHighValueFlag]
                +
                [FraudVeryHighValueFlag]
                +
                [FraudExtremeValueFlag]
                +
                [FraudRoundAmountFlag]
                +
                [FraudLargeRoundAmountFlag]
                +
                [FraudWeekendFlag]
                +
                [FraudMonthEndFlag]
                +
                [FraudYearEndFlag]
                +
                [FraudQuarterEndFlag]
                +
                [FraudPaymentMethodRiskFlag]
                +
                [FraudManualTransactionFlag]
                +
                [FraudUnknownCategoryFlag]
                +
                [FraudSuspiciousTypeFlag]
                +
                [FraudStatusRiskFlag]
                +
                [FraudMissingCounterpartyFlag]
                +
                [FraudMissingAccountFlag]
                +
                [FraudMissingDescriptionFlag]
                +
                [FraudCustomerHighFrequencyFlag]
                +
                [FraudSupplierHighFrequencyFlag]
                +
                [FraudConcentrationFlag],
            Int64.Type
        ),


    // =========================================================================
    // 47. SIGNAL STRENGTH
    // =========================================================================

    AddSignalStrength =
        Table.AddColumn(
            AddSignalCount,
            "FraudSignalStrength",
            each
                if [FraudSignalCount] >= 7
                then "VERY_HIGH"

                else if [FraudSignalCount] >= 5
                then "HIGH"

                else if [FraudSignalCount] >= 3
                then "MEDIUM"

                else if [FraudSignalCount] >= 1
                then "LOW"

                else "NONE",
            type text
        ),


    // =========================================================================
    // 48. DETERMINISTIC FRAUD SCORE
    // =========================================================================

    AddRuleScore =
        Table.AddColumn(
            AddSignalStrength,
            "FraudRuleScore",
            each
                Clamp(
                    [FraudDuplicateFlag] * 25
                    +
                    [FraudVeryHighValueFlag] * 15
                    +
                    [FraudExtremeValueFlag] * 20
                    +
                    [FraudHighValueFlag] * 10
                    +
                    [FraudRoundAmountFlag] * 8
                    +
                    [FraudLargeRoundAmountFlag] * 7
                    +
                    [FraudWeekendFlag] * 8
                    +
                    [FraudMonthEndFlag] * 7
                    +
                    [FraudYearEndFlag] * 5
                    +
                    [FraudQuarterEndFlag] * 4
                    +
                    [FraudPaymentMethodRiskFlag] * 5
                    +
                    [FraudManualTransactionFlag] * 7
                    +
                    [FraudUnknownCategoryFlag] * 3
                    +
                    [FraudSuspiciousTypeFlag] * 7
                    +
                    [FraudStatusRiskFlag] * 5
                    +
                    [FraudMissingCounterpartyFlag] * 5
                    +
                    [FraudMissingAccountFlag] * 3
                    +
                    [FraudMissingDescriptionFlag] * 2
                    +
                    [FraudCustomerHighFrequencyFlag] * 5
                    +
                    [FraudSupplierHighFrequencyFlag] * 5
                    +
                    [FraudConcentrationFlag] * 10,
                    0,
                    100
                ),
            type number
        ),


    // =========================================================================
    // 49. RULE RISK BAND
    // =========================================================================

    AddRuleRiskBand =
        Table.AddColumn(
            AddRuleScore,
            "FraudRuleRiskBand",
            each
                if [FraudRuleScore] >= 80
                then "CRITICAL"

                else if [FraudRuleScore] >= 60
                then "HIGH"

                else if [FraudRuleScore] >= 35
                then "MEDIUM"

                else if [FraudRuleScore] >= 15
                then "LOW"

                else "INFO",
            type text
        ),


    // =========================================================================
    // 50. FRAUD EXPLANATION
    // =========================================================================

    AddPrimaryReason =
        Table.AddColumn(
            AddRuleRiskBand,
            "FraudPrimaryReason",
            each
                if [FraudDuplicateFlag] = 1
                then "Duplicate transaction"

                else if [FraudExtremeValueFlag] = 1
                then "Extreme transaction value"

                else if [FraudVeryHighValueFlag] = 1
                then "Very high transaction value"

                else if [FraudHighValueFlag] = 1
                then "High transaction value"

                else if [FraudConcentrationFlag] = 1
                then "High counterparty concentration"

                else if [FraudManualTransactionFlag] = 1
                then "Manual transaction"

                else if [FraudSuspiciousTypeFlag] = 1
                then "Suspicious transaction type"

                else if [FraudStatusRiskFlag] = 1
                then "Risky transaction status"

                else if [FraudWeekendFlag] = 1
                then "Weekend transaction"

                else if [FraudMonthEndFlag] = 1
                then "Month-end transaction"

                else if [FraudRoundAmountFlag] = 1
                then "Round-value transaction"

                else if [FraudMissingCounterpartyFlag] = 1
                then "Missing counterparty"

                else "No material deterministic fraud signal",
            type text
        ),


    // =========================================================================
    // 51. FRAUD EXPLANATION DETAIL
    // =========================================================================

    AddExplanationDetail =
        Table.AddColumn(
            AddPrimaryReason,
            "FraudExplanationDetail",
            each
                Text.Combine(
                    List.RemoveNulls(
                        {
                            if [FraudDuplicateFlag] = 1
                            then "Duplicate"
                            else null,

                            if [FraudExtremeValueFlag] = 1
                            then "ExtremeValue"
                            else null,

                            if [FraudVeryHighValueFlag] = 1
                            then "VeryHighValue"
                            else null,

                            if [FraudHighValueFlag] = 1
                            then "HighValue"
                            else null,

                            if [FraudRoundAmountFlag] = 1
                            then "RoundAmount"
                            else null,

                            if [FraudWeekendFlag] = 1
                            then "Weekend"
                            else null,

                            if [FraudMonthEndFlag] = 1
                            then "MonthEnd"
                            else null,

                            if [FraudYearEndFlag] = 1
                            then "YearEnd"
                            else null,

                            if [FraudPaymentMethodRiskFlag] = 1
                            then "PaymentMethod"
                            else null,

                            if [FraudManualTransactionFlag] = 1
                            then "Manual"
                            else null,

                            if [FraudSuspiciousTypeFlag] = 1
                            then "SuspiciousType"
                            else null,

                            if [FraudStatusRiskFlag] = 1
                            then "StatusRisk"
                            else null,

                            if [FraudMissingCounterpartyFlag] = 1
                            then "MissingCounterparty"
                            else null,

                            if [FraudMissingAccountFlag] = 1
                            then "MissingAccount"
                            else null,

                            if [FraudConcentrationFlag] = 1
                            then "Concentration"
                            else null
                        }
                    ),
                    " | "
                ),
            type text
        ),


    // =========================================================================
    // 52. INVESTIGATION PRIORITY
    // =========================================================================

    AddInvestigationPriority =
        Table.AddColumn(
            AddExplanationDetail,
            "FraudInvestigationPriority",
            each
                if [FraudRuleScore] >= 80
                then "P1 — CRITICAL"

                else if [FraudRuleScore] >= 60
                then "P2 — HIGH"

                else if [FraudRuleScore] >= 35
                then "P3 — MEDIUM"

                else "P4 — LOW",
            type text
        ),


    // =========================================================================
    // 53. INVESTIGATION REQUIRED
    // =========================================================================

    AddInvestigationRequired =
        Table.AddColumn(
            AddInvestigationPriority,
            "FraudInvestigationRequired",
            each
                if [FraudRuleScore] >= 60
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 54. SUPERVISOR AGENT SIGNAL
    // =========================================================================

    AddSupervisorSignal =
        Table.AddColumn(
            AddInvestigationRequired,
            "FraudSupervisorAgentSignal",
            each
                if [FraudRuleScore] >= 80
                then "ESCALATE_TO_SUPERVISOR"

                else if [FraudRuleScore] >= 60
                then "START_FRAUD_INVESTIGATION"

                else if [FraudRuleScore] >= 35
                then "REVIEW_TRANSACTION"

                else "NO_IMMEDIATE_ACTION",
            type text
        ),


    // =========================================================================
    // 55. RAG EVIDENCE TRIGGER
    // =========================================================================

    AddRAGTrigger =
        Table.AddColumn(
            AddSupervisorSignal,
            "FraudRAGEvidenceRequired",
            each
                if [FraudRuleScore] >= 35
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 56. HUMAN REVIEW
    // =========================================================================

    AddHumanReview =
        Table.AddColumn(
            AddRAGTrigger,
            "FraudHumanReviewRequired",
            each
                if [FraudRuleScore] >= 60
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 57. FRAUD DECISION
    // =========================================================================

    AddDecision =
        Table.AddColumn(
            AddHumanReview,
            "FraudDecision",
            each
                if [FraudRuleScore] >= 80
                then "BLOCK_AND_ESCALATE"

                else if [FraudRuleScore] >= 60
                then "HOLD_AND_INVESTIGATE"

                else if [FraudRuleScore] >= 35
                then "MONITOR"

                else "NORMAL",
            type text
        ),


    // =========================================================================
    // 58. FRAUD STATUS
    // =========================================================================

    AddFraudStatus =
        Table.AddColumn(
            AddDecision,
            "FraudTransformationStatus",
            each
                if [FraudRuleScore] >= 80
                then "CRITICAL_FRAUD_SIGNAL"

                else if [FraudRuleScore] >= 60
                then "HIGH_FRAUD_SIGNAL"

                else if [FraudRuleScore] >= 35
                then "FRAUD_WATCH"

                else "NORMAL",
            type text
        ),


    // =========================================================================
    // 59. ML FEATURE READINESS
    // =========================================================================

    AddMLFeatureReadiness =
        Table.AddColumn(
            AddFraudStatus,
            "FraudMLFeatureReady",
            each
                if
                    [TransactionId] <> null
                    and [CompanyId] <> null
                    and [TransactionDate] <> null
                    and [Amount] <> null
                then "YES"
                else "NO",
            type text
        ),


    // =========================================================================
    // 60. FEATURE COUNT
    // =========================================================================

    AddFeatureCount =
        Table.AddColumn(
            AddMLFeatureReadiness,
            "FraudFeatureCount",
            each
                List.Count(
                    {
                        [FraudAbsoluteAmount],
                        [FraudLogAmount],
                        [FraudDayOfWeek],
                        [FraudWeekendFlag],
                        [FraudMonthEndFlag],
                        [FraudYearEndFlag],
                        [FraudQuarterEndFlag],
                        [FraudHighValueFlag],
                        [FraudVeryHighValueFlag],
                        [FraudRoundAmountFlag],
                        [FraudPaymentMethodRiskFlag],
                        [FraudDuplicateFlag],
                        [FraudCustomerTransactionCount],
                        [FraudSupplierTransactionCount],
                        [FraudCustomerAmountShare],
                        [FraudSupplierAmountShare],
                        [FraudSignalCount],
                        [FraudRuleScore]
                    }
                ),
            Int64.Type
        ),


    // =========================================================================
    // 61. DATA COMPLETENESS
    // =========================================================================

    AddDataCompleteness =
        Table.AddColumn(
            AddFeatureCount,
            "FraudDataCompleteness",
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
                        [Currency],
                        [AccountId]
                    },

                    Available =
                        List.Count(
                            List.Select(
                                Fields,
                                each _ <> null
                            )
                        )
                in
                    SafeDivide(
                        Available,
                        List.Count(Fields)
                    ) * 100,
            type number
        ),


    // =========================================================================
    // 62. DATA QUALITY
    // =========================================================================

    AddDataQuality =
        Table.AddColumn(
            AddDataCompleteness,
            "FraudDataQuality",
            each
                if [FraudDataCompleteness] >= 95
                then "HIGH"

                else if [FraudDataCompleteness] >= 80
                then "MEDIUM"

                else if [FraudDataCompleteness] >= 60
                then "LOW"

                else "CRITICAL",
            type text
        ),


    // =========================================================================
    // 63. DATA QUALITY FLAG
    // =========================================================================

    AddDataQualityFlag =
        Table.AddColumn(
            AddDataQuality,
            "FraudDataQualityFlag",
            each
                if [FraudDataCompleteness] >= 80
                then "PASS"
                else "REVIEW",
            type text
        ),


    // =========================================================================
    // 64. FRAUD RECORD KEY
    // =========================================================================

    AddFraudRecordKey =
        Table.AddColumn(
            AddDataQualityFlag,
            "FraudRecordKey",
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
                        else
                            Date.ToText(
                                [TransactionDate],
                                "yyyyMMdd"
                            )
                    },
                    "|"
                ),
            type text
        ),


    // =========================================================================
    // 65. FINAL ROUNDING
    // =========================================================================

    Rounded =
        Table.TransformColumns(
            AddFraudRecordKey,
            {
                {
                    "FraudAbsoluteAmount",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                },
                {
                    "FraudLogAmount",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 4),
                    type number
                },
                {
                    "FraudCustomerAmountShare",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 4),
                    type number
                },
                {
                    "FraudSupplierAmountShare",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 4),
                    type number
                },
                {
                    "FraudRuleScore",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                },
                {
                    "FraudDataCompleteness",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                }
            }
        ),


    // =========================================================================
    // 66. SORT
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
                    "FraudRuleScore",
                    Order.Descending
                },
                {
                    "FraudAbsoluteAmount",
                    Order.Descending
                }
            }
        ),


    // =========================================================================
    // 67. PREFERRED COLUMN ORDER
    // =========================================================================

    PreferredColumns = {
        "FraudRecordKey",
        "FraudTransactionKey",

        "TransactionId",
        "CompanyId",
        "CompanyName",
        "TransactionDate",

        "FraudYear",
        "FraudMonth",
        "FraudMonthName",
        "FraudQuarter",
        "FraudPeriodKey",
        "FraudDayOfWeek",
        "FraudDayName",
        "FraudDayOfMonth",

        "CustomerId",
        "SupplierId",
        "AccountId",

        "TransactionType",
        "TransactionCategory",
        "PaymentMethod",
        "Currency",
        "Status",

        "Amount",
        "FraudAbsoluteAmount",
        "FraudLogAmount",
        "FraudAmountSign",
        "FraudTransactionDirection",
        "FraudAmountBucket",

        "FraudZeroValueFlag",
        "FraudHighValueFlag",
        "FraudVeryHighValueFlag",
        "FraudExtremeValueFlag",
        "FraudRoundAmountFlag",
        "FraudLargeRoundAmountFlag",

        "FraudWeekendFlag",
        "FraudMonthEndFlag",
        "FraudYearEndFlag",
        "FraudQuarterEndFlag",

        "FraudPaymentMethodRiskFlag",
        "FraudDigitalPaymentFlag",
        "FraudManualTransactionFlag",
        "FraudUnknownCategoryFlag",
        "FraudSuspiciousTypeFlag",
        "FraudStatusRiskFlag",

        "FraudMissingCounterpartyFlag",
        "FraudMissingAccountFlag",
        "FraudMissingDescriptionFlag",

        "FraudDuplicateCount",
        "FraudDuplicateFlag",

        "FraudCustomerTransactionCount",
        "FraudCustomerTotalAmount",
        "FraudCustomerAverageAmount",
        "FraudCustomerAmountShare",
        "FraudCustomerHighFrequencyFlag",
        "FraudCustomerHighValueFlag",

        "FraudSupplierTransactionCount",
        "FraudSupplierTotalAmount",
        "FraudSupplierAverageAmount",
        "FraudSupplierAmountShare",
        "FraudSupplierHighFrequencyFlag",
        "FraudSupplierHighValueFlag",

        "FraudConcentrationFlag",

        "FraudSignalCount",
        "FraudSignalStrength",

        "FraudRuleScore",
        "FraudRuleRiskBand",

        "FraudPrimaryReason",
        "FraudExplanationDetail",

        "FraudInvestigationPriority",
        "FraudInvestigationRequired",

        "FraudSupervisorAgentSignal",
        "FraudRAGEvidenceRequired",
        "FraudHumanReviewRequired",

        "FraudDecision",
        "FraudTransformationStatus",

        "FraudMLFeatureReady",
        "FraudFeatureCount",

        "FraudDataCompleteness",
        "FraudDataQuality",
        "FraudDataQualityFlag",

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