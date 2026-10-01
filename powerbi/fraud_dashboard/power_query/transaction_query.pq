// ============================================================================
// FinCo AI — Transaction Query
// File: transaction_query.pq
// ============================================================================
//
// PURPOSE
// -------
// Enterprise transaction fact query for Power BI.
//
// ARCHITECTURE
// ------------
//
//   Transactions
//        ↓
//   common_transformations.pq
//        ↓
//   transaction_query.pq
//        ├── Financial Analytics
//        ├── Fraud Analytics
//        ├── Risk Analytics
//        ├── Forecasting
//        ├── Alerts
//        └── AI Copilot
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
    // 4. ENSURE REQUIRED COLUMNS
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
    // 6. TEXT STANDARDIZATION
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

    ValidRows =
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

    AddTransactionKey =
        Table.AddColumn(
            ValidRows,
            "TransactionKey",
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
            AddTransactionKey,
            "AbsoluteAmount",
            each
                Number.Abs([Amount]),
            type number
        ),


    // =========================================================================
    // 10. TRANSACTION DIRECTION
    // =========================================================================

    AddTransactionDirection =
        Table.AddColumn(
            AddAbsoluteAmount,
            "TransactionDirection",
            each
                if [Amount] > 0
                then "INFLOW"

                else if [Amount] < 0
                then "OUTFLOW"

                else "ZERO",
            type text
        ),


    // =========================================================================
    // 11. TRANSACTION SIGN
    // =========================================================================

    AddTransactionSign =
        Table.AddColumn(
            AddTransactionDirection,
            "TransactionSign",
            each
                if [Amount] > 0
                then 1

                else if [Amount] < 0
                then -1

                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 12. TRANSACTION YEAR
    // =========================================================================

    AddYear =
        Table.AddColumn(
            AddTransactionSign,
            "TransactionYear",
            each
                Date.Year([TransactionDate]),
            Int64.Type
        ),


    // =========================================================================
    // 13. TRANSACTION MONTH
    // =========================================================================

    AddMonth =
        Table.AddColumn(
            AddYear,
            "TransactionMonth",
            each
                Date.Month([TransactionDate]),
            Int64.Type
        ),


    // =========================================================================
    // 14. MONTH NAME
    // =========================================================================

    AddMonthName =
        Table.AddColumn(
            AddMonth,
            "TransactionMonthName",
            each
                Date.MonthName(
                    [TransactionDate]
                ),
            type text
        ),


    // =========================================================================
    // 15. QUARTER
    // =========================================================================

    AddQuarter =
        Table.AddColumn(
            AddMonthName,
            "TransactionQuarter",
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


    // =========================================================================
    // 16. PERIOD KEY
    // =========================================================================

    AddPeriodKey =
        Table.AddColumn(
            AddQuarter,
            "TransactionPeriodKey",
            each
                Date.Year([TransactionDate]) * 100
                +
                Date.Month([TransactionDate]),
            Int64.Type
        ),


    // =========================================================================
    // 17. YEAR-MONTH
    // =========================================================================

    AddYearMonth =
        Table.AddColumn(
            AddPeriodKey,
            "TransactionYearMonth",
            each
                Date.ToText(
                    [TransactionDate],
                    "yyyy-MM"
                ),
            type text
        ),


    // =========================================================================
    // 18. DAY OF WEEK
    // =========================================================================

    AddDayOfWeek =
        Table.AddColumn(
            AddYearMonth,
            "TransactionDayOfWeek",
            each
                Date.DayOfWeek(
                    [TransactionDate],
                    Day.Monday
                ),
            Int64.Type
        ),


    // =========================================================================
    // 19. DAY NAME
    // =========================================================================

    AddDayName =
        Table.AddColumn(
            AddDayOfWeek,
            "TransactionDayName",
            each
                Date.DayOfWeekName(
                    [TransactionDate]
                ),
            type text
        ),


    // =========================================================================
    // 20. WEEKEND FLAG
    // =========================================================================

    AddWeekendFlag =
        Table.AddColumn(
            AddDayName,
            "WeekendTransactionFlag",
            each
                if [TransactionDayOfWeek] >= 5
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 21. MONTH START
    // =========================================================================

    AddMonthStart =
        Table.AddColumn(
            AddWeekendFlag,
            "TransactionMonthStart",
            each
                Date.StartOfMonth(
                    [TransactionDate]
                ),
            type date
        ),


    // =========================================================================
    // 22. MONTH END
    // =========================================================================

    AddMonthEnd =
        Table.AddColumn(
            AddMonthStart,
            "TransactionMonthEnd",
            each
                Date.EndOfMonth(
                    [TransactionDate]
                ),
            type date
        ),


    // =========================================================================
    // 23. MONTH-END FLAG
    // =========================================================================

    AddMonthEndFlag =
        Table.AddColumn(
            AddMonthEnd,
            "MonthEndTransactionFlag",
            each
                if
                    Date.Day(
                        [TransactionDate]
                    )
                    >=
                    Date.DaysInMonth(
                        [TransactionDate]
                    ) - 2
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 24. YEAR-END FLAG
    // =========================================================================

    AddYearEndFlag =
        Table.AddColumn(
            AddMonthEndFlag,
            "YearEndTransactionFlag",
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
    // 25. QUARTER START
    // =========================================================================

    AddQuarterStart =
        Table.AddColumn(
            AddYearEndFlag,
            "TransactionQuarterStart",
            each
                let
                    QuarterMonth =
                        (
                            Date.QuarterOfYear(
                                [TransactionDate]
                            ) - 1
                        ) * 3 + 1
                in
                    #date(
                        Date.Year(
                            [TransactionDate]
                        ),
                        QuarterMonth,
                        1
                    ),
            type date
        ),


    // =========================================================================
    // 26. QUARTER END
    // =========================================================================

    AddQuarterEnd =
        Table.AddColumn(
            AddQuarterStart,
            "TransactionQuarterEnd",
            each
                Date.EndOfMonth(
                    Date.AddMonths(
                        [TransactionQuarterStart],
                        2
                    )
                ),
            type date
        ),


    // =========================================================================
    // 27. FISCAL YEAR
    // =========================================================================

    AddFiscalYear =
        Table.AddColumn(
            AddQuarterEnd,
            "TransactionFiscalYear",
            each
                if Date.Month([TransactionDate]) >= 4
                then Date.Year([TransactionDate])
                else Date.Year([TransactionDate]) - 1,
            Int64.Type
        ),


    // =========================================================================
    // 28. FISCAL YEAR LABEL
    // =========================================================================

    AddFiscalYearLabel =
        Table.AddColumn(
            AddFiscalYear,
            "TransactionFiscalYearLabel",
            each
                "FY"
                &
                Text.From(
                    [TransactionFiscalYear]
                ),
            type text
        ),


    // =========================================================================
    // 29. FISCAL MONTH
    // =========================================================================

    AddFiscalMonth =
        Table.AddColumn(
            AddFiscalYearLabel,
            "TransactionFiscalMonth",
            each
                Number.Mod(
                    Date.Month([TransactionDate]) - 4 + 12,
                    12
                ) + 1,
            Int64.Type
        ),


    // =========================================================================
    // 30. FISCAL QUARTER
    // =========================================================================

    AddFiscalQuarter =
        Table.AddColumn(
            AddFiscalMonth,
            "TransactionFiscalQuarter",
            each
                "FQ"
                &
                Text.From(
                    Number.RoundUp(
                        [TransactionFiscalMonth] / 3
                    )
                ),
            type text
        ),


    // =========================================================================
    // 31. HIGH VALUE TRANSACTION
    // =========================================================================

    AddHighValueFlag =
        Table.AddColumn(
            AddFiscalQuarter,
            "HighValueTransactionFlag",
            each
                if [AbsoluteAmount] >= 500000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 32. VERY HIGH VALUE
    // =========================================================================

    AddVeryHighValueFlag =
        Table.AddColumn(
            AddHighValueFlag,
            "VeryHighValueTransactionFlag",
            each
                if [AbsoluteAmount] >= 5000000
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 33. ZERO VALUE
    // =========================================================================

    AddZeroValueFlag =
        Table.AddColumn(
            AddVeryHighValueFlag,
            "ZeroValueTransactionFlag",
            each
                if [Amount] = 0
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 34. ROUND VALUE
    // =========================================================================

    AddRoundValueFlag =
        Table.AddColumn(
            AddZeroValueFlag,
            "RoundAmountTransactionFlag",
            each
                if
                    [AbsoluteAmount] > 0
                    and
                    Number.Mod(
                        [AbsoluteAmount],
                        10000
                    ) = 0
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 35. PAYMENT METHOD CLASSIFICATION
    // =========================================================================

    AddPaymentMethodClass =
        Table.AddColumn(
            AddRoundValueFlag,
            "PaymentMethodClass",
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
                then "DIGITAL"

                else if List.Contains(
                    {
                        "CASH"
                    },
                    [PaymentMethod]
                )
                then "CASH"

                else if List.Contains(
                    {
                        "CHEQUE",
                        "CHECK"
                    },
                    [PaymentMethod]
                )
                then "CHEQUE"

                else if [PaymentMethod] = null
                then "UNKNOWN"

                else "OTHER",
            type text
        ),


    // =========================================================================
    // 36. PAYMENT METHOD RISK
    // =========================================================================

    AddPaymentRiskFlag =
        Table.AddColumn(
            AddPaymentMethodClass,
            "PaymentMethodRiskFlag",
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
    // 37. MANUAL TRANSACTION
    // =========================================================================

    AddManualFlag =
        Table.AddColumn(
            AddPaymentRiskFlag,
            "ManualTransactionFlag",
            each
                if
                    [PaymentMethod] = "MANUAL"
                    or
                    [TransactionType] = "MANUAL"
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 38. SUSPICIOUS TYPE
    // =========================================================================

    AddSuspiciousTypeFlag =
        Table.AddColumn(
            AddManualFlag,
            "SuspiciousTransactionTypeFlag",
            each
                if List.Contains(
                    {
                        "ADJUSTMENT",
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
    // 39. STATUS CLASSIFICATION
    // =========================================================================

    AddStatusClass =
        Table.AddColumn(
            AddSuspiciousTypeFlag,
            "TransactionStatusClass",
            each
                if List.Contains(
                    {
                        "FAILED",
                        "REJECTED"
                    },
                    [Status]
                )
                then "FAILED"

                else if List.Contains(
                    {
                        "REVERSED",
                        "REVERSAL",
                        "CANCELLED",
                        "VOID"
                    },
                    [Status]
                )
                then "REVERSED_OR_CANCELLED"

                else if List.Contains(
                    {
                        "PENDING",
                        "PROCESSING"
                    },
                    [Status]
                )
                then "PENDING"

                else if List.Contains(
                    {
                        "COMPLETED",
                        "SUCCESS",
                        "POSTED",
                        "SETTLED"
                    },
                    [Status]
                )
                then "COMPLETED"

                else "UNKNOWN",
            type text
        ),


    // =========================================================================
    // 40. STATUS RISK FLAG
    // =========================================================================

    AddStatusRiskFlag =
        Table.AddColumn(
            AddStatusClass,
            "TransactionStatusRiskFlag",
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
    // 41. MISSING CUSTOMER
    // =========================================================================

    AddMissingCustomerFlag =
        Table.AddColumn(
            AddStatusRiskFlag,
            "MissingCustomerFlag",
            each
                if [CustomerId] = null
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 42. MISSING SUPPLIER
    // =========================================================================

    AddMissingSupplierFlag =
        Table.AddColumn(
            AddMissingCustomerFlag,
            "MissingSupplierFlag",
            each
                if [SupplierId] = null
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 43. MISSING ACCOUNT
    // =========================================================================

    AddMissingAccountFlag =
        Table.AddColumn(
            AddMissingSupplierFlag,
            "MissingAccountFlag",
            each
                if [AccountId] = null
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 44. MISSING DESCRIPTION
    // =========================================================================

    AddMissingDescriptionFlag =
        Table.AddColumn(
            AddMissingAccountFlag,
            "MissingDescriptionFlag",
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
    // 45. DUPLICATE TRANSACTION DETECTION
    // =========================================================================

    DuplicateGroups =
        Table.Group(
            AddMissingDescriptionFlag,
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
                    "DuplicateTransactionCount",
                    each Table.RowCount(_),
                    Int64.Type
                }
            }
        ),


    JoinDuplicates =
        Table.NestedJoin(
            AddMissingDescriptionFlag,
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


    ExpandDuplicates =
        Table.ExpandTableColumn(
            JoinDuplicates,
            "DuplicateData",
            {
                "DuplicateTransactionCount"
            },
            {
                "DuplicateTransactionCount"
            }
        ),


    AddDuplicateFlag =
        Table.AddColumn(
            ExpandDuplicates,
            "DuplicateTransactionFlag",
            each
                if [DuplicateTransactionCount] > 1
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 46. TRANSACTION COMPLETENESS
    // =========================================================================

    AddTransactionCompleteness =
        Table.AddColumn(
            AddDuplicateFlag,
            "TransactionDataCompleteness",
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
    // 47. TRANSACTION DATA QUALITY
    // =========================================================================

    AddDataQuality =
        Table.AddColumn(
            AddTransactionCompleteness,
            "TransactionDataQuality",
            each
                if [TransactionDataCompleteness] >= 95
                then "HIGH"

                else if [TransactionDataCompleteness] >= 80
                then "MEDIUM"

                else if [TransactionDataCompleteness] >= 60
                then "LOW"

                else "CRITICAL",
            type text
        ),


    // =========================================================================
    // 48. TRANSACTION QUALITY FLAG
    // =========================================================================

    AddQualityFlag =
        Table.AddColumn(
            AddDataQuality,
            "TransactionDataQualityFlag",
            each
                if [TransactionDataCompleteness] >= 80
                then "PASS"
                else "REVIEW",
            type text
        ),


    // =========================================================================
    // 49. FINANCIAL TRANSACTION CLASSIFICATION
    // =========================================================================

    AddFinancialClass =
        Table.AddColumn(
            AddQualityFlag,
            "FinancialTransactionClass",
            each
                if List.Contains(
                    {
                        "REVENUE",
                        "SALES",
                        "RECEIPT",
                        "COLLECTION",
                        "INCOME"
                    },
                    [TransactionCategory]
                )
                then "REVENUE"

                else if List.Contains(
                    {
                        "EXPENSE",
                        "PURCHASE",
                        "PAYMENT",
                        "COST"
                    },
                    [TransactionCategory]
                )
                then "EXPENSE"

                else if List.Contains(
                    {
                        "TRANSFER",
                        "BANK_TRANSFER",
                        "INTERNAL_TRANSFER"
                    },
                    [TransactionCategory]
                )
                then "TRANSFER"

                else if List.Contains(
                    {
                        "TAX",
                        "GST",
                        "VAT",
                        "DUTY"
                    },
                    [TransactionCategory]
                )
                then "TAX"

                else if List.Contains(
                    {
                        "REFUND",
                        "RETURN"
                    },
                    [TransactionCategory]
                )
                then "REFUND"

                else "OTHER",
            type text
        ),


    // =========================================================================
    // 50. REVENUE TRANSACTION FLAG
    // =========================================================================

    AddRevenueFlag =
        Table.AddColumn(
            AddFinancialClass,
            "RevenueTransactionFlag",
            each
                if [FinancialTransactionClass] = "REVENUE"
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 51. EXPENSE TRANSACTION FLAG
    // =========================================================================

    AddExpenseFlag =
        Table.AddColumn(
            AddRevenueFlag,
            "ExpenseTransactionFlag",
            each
                if [FinancialTransactionClass] = "EXPENSE"
                then 1
                else 0,
            Int64.Type
        ),


    // =========================================================================
    // 52. CASH FLOW CLASSIFICATION
    // =========================================================================

    AddCashFlowClass =
        Table.AddColumn(
            AddExpenseFlag,
            "CashFlowTransactionClass",
            each
                if [TransactionDirection] = "INFLOW"
                then "CASH_INFLOW"

                else if [TransactionDirection] = "OUTFLOW"
                then "CASH_OUTFLOW"

                else "ZERO_FLOW",
            type text
        ),


    // =========================================================================
    // 53. TRANSACTION RISK SIGNAL COUNT
    // =========================================================================

    AddRiskSignalCount =
        Table.AddColumn(
            AddCashFlowClass,
            "TransactionRiskSignalCount",
            each
                [HighValueTransactionFlag]
                +
                [VeryHighValueTransactionFlag]
                +
                [RoundAmountTransactionFlag]
                +
                [WeekendTransactionFlag]
                +
                [MonthEndTransactionFlag]
                +
                [YearEndTransactionFlag]
                +
                [PaymentMethodRiskFlag]
                +
                [ManualTransactionFlag]
                +
                [SuspiciousTransactionTypeFlag]
                +
                [TransactionStatusRiskFlag]
                +
                [DuplicateTransactionFlag]
                +
                [MissingCustomerFlag]
                +
                [MissingSupplierFlag]
                +
                [MissingAccountFlag],
            Int64.Type
        ),


    // =========================================================================
    // 54. TRANSACTION RISK SIGNAL
    // =========================================================================

    AddRiskSignal =
        Table.AddColumn(
            AddRiskSignalCount,
            "TransactionRiskSignal",
            each
                if [TransactionRiskSignalCount] >= 6
                then "HIGH"

                else if [TransactionRiskSignalCount] >= 3
                then "MEDIUM"

                else if [TransactionRiskSignalCount] >= 1
                then "LOW"

                else "NONE",
            type text
        ),


    // =========================================================================
    // 55. TRANSACTION STATUS
    // =========================================================================

    AddTransactionStatus =
        Table.AddColumn(
            AddRiskSignal,
            "TransactionAnalyticsStatus",
            each
                if [TransactionDataQualityFlag] = "REVIEW"
                then "DATA_REVIEW"

                else if [TransactionRiskSignal] = "HIGH"
                then "HIGH_RISK_TRANSACTION"

                else if [TransactionRiskSignal] = "MEDIUM"
                then "TRANSACTION_REVIEW"

                else "NORMAL",
            type text
        ),


    // =========================================================================
    // 56. RECORD KEY
    // =========================================================================

    AddRecordKey =
        Table.AddColumn(
            AddTransactionStatus,
            "TransactionRecordKey",
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
    // 57. ROUND NUMERIC VALUES
    // =========================================================================

    Rounded =
        Table.TransformColumns(
            AddRecordKey,
            {
                {
                    "Amount",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                },
                {
                    "AbsoluteAmount",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                },
                {
                    "TransactionDataCompleteness",
                    each
                        if _ = null
                        then null
                        else Number.Round(_, 2),
                    type number
                }
            }
        ),


    // =========================================================================
    // 58. SORT
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
                    "AbsoluteAmount",
                    Order.Descending
                }
            }
        ),


    // =========================================================================
    // 59. PREFERRED COLUMN ORDER
    // =========================================================================

    PreferredColumns = {
        "TransactionRecordKey",
        "TransactionKey",

        "TransactionId",
        "CompanyId",
        "CompanyName",
        "TransactionDate",

        "TransactionYear",
        "TransactionMonth",
        "TransactionMonthName",
        "TransactionQuarter",
        "TransactionPeriodKey",
        "TransactionYearMonth",

        "TransactionFiscalYear",
        "TransactionFiscalYearLabel",
        "TransactionFiscalMonth",
        "TransactionFiscalQuarter",

        "TransactionDayOfWeek",
        "TransactionDayName",
        "WeekendTransactionFlag",

        "TransactionMonthStart",
        "TransactionMonthEnd",
        "MonthEndTransactionFlag",

        "TransactionQuarterStart",
        "TransactionQuarterEnd",
        "YearEndTransactionFlag",

        "CustomerId",
        "SupplierId",
        "AccountId",

        "TransactionType",
        "TransactionCategory",
        "FinancialTransactionClass",
        "PaymentMethod",
        "PaymentMethodClass",
        "Currency",
        "Status",
        "TransactionStatusClass",

        "Amount",
        "AbsoluteAmount",
        "TransactionDirection",
        "TransactionSign",
        "CashFlowTransactionClass",

        "HighValueTransactionFlag",
        "VeryHighValueTransactionFlag",
        "ZeroValueTransactionFlag",
        "RoundAmountTransactionFlag",

        "PaymentMethodRiskFlag",
        "ManualTransactionFlag",
        "SuspiciousTransactionTypeFlag",
        "TransactionStatusRiskFlag",

        "MissingCustomerFlag",
        "MissingSupplierFlag",
        "MissingAccountFlag",
        "MissingDescriptionFlag",

        "DuplicateTransactionCount",
        "DuplicateTransactionFlag",

        "RevenueTransactionFlag",
        "ExpenseTransactionFlag",

        "TransactionRiskSignalCount",
        "TransactionRiskSignal",
        "TransactionAnalyticsStatus",

        "TransactionDataCompleteness",
        "TransactionDataQuality",
        "TransactionDataQualityFlag",

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