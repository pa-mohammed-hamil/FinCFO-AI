let
    // ============================================================
    // FinCo AI - Risk Dashboard
    // Query: Risk Exposure
    // ============================================================

    // ------------------------------------------------------------
    // 1. DATABASE CONNECTION
    // ------------------------------------------------------------
    DatabaseHost =
        "localhost",

    DatabaseName =
        "finco",

    DatabaseSchema =
        "public",

    SourceDatabase =
        PostgreSQL.Database(
            DatabaseHost,
            DatabaseName,
            [
                CreateNavigationProperties = false,
                CommandTimeout = #duration(0, 0, 30, 0)
            ]
        ),

    SourceSchema =
        SourceDatabase{[Schema = DatabaseSchema]}[Data],

    // ------------------------------------------------------------
    // 2. LOAD SOURCE TABLE
    // ------------------------------------------------------------
    SourceTable =
        try
            SourceSchema{
                [
                    Item = "fact_risk_exposure",
                    Kind = "Table"
                ]
            }[Data]
        otherwise
            #table(
                {
                    "exposure_id",
                    "company_id",
                    "entity_id",
                    "exposure_date",
                    "exposure_type",
                    "counterparty",
                    "exposure_amount",
                    "secured_amount",
                    "unsecured_amount",
                    "currency",
                    "exposure_status",
                    "maturity_date",
                    "probability_of_default",
                    "loss_given_default",
                    "credit_rating",
                    "country",
                    "industry"
                },
                {}
            ),

    // ------------------------------------------------------------
    // 3. SELECT REQUIRED COLUMNS
    // ------------------------------------------------------------
    SelectedColumns =
        Table.SelectColumns(
            SourceTable,
            {
                "exposure_id",
                "company_id",
                "entity_id",
                "exposure_date",
                "exposure_type",
                "counterparty",
                "exposure_amount",
                "secured_amount",
                "unsecured_amount",
                "currency",
                "exposure_status",
                "maturity_date",
                "probability_of_default",
                "loss_given_default",
                "credit_rating",
                "country",
                "industry"
            },
            MissingField.Ignore
        ),

    // ------------------------------------------------------------
    // 4. RENAME COLUMNS
    // ------------------------------------------------------------
    RenamedColumns =
        Table.RenameColumns(
            SelectedColumns,
            {
                {"exposure_id", "ExposureId"},
                {"company_id", "CompanyId"},
                {"entity_id", "EntityId"},
                {"exposure_date", "ExposureDate"},
                {"exposure_type", "ExposureType"},
                {"counterparty", "Counterparty"},
                {"exposure_amount", "ExposureAmount"},
                {"secured_amount", "SecuredAmount"},
                {"unsecured_amount", "UnsecuredAmount"},
                {"currency", "Currency"},
                {"exposure_status", "ExposureStatus"},
                {"maturity_date", "MaturityDate"},
                {"probability_of_default", "ProbabilityOfDefault"},
                {"loss_given_default", "LossGivenDefault"},
                {"credit_rating", "CreditRating"},
                {"country", "Country"},
                {"industry", "Industry"}
            },
            MissingField.Ignore
        ),

    // ------------------------------------------------------------
    // 5. DATA TYPES
    // ------------------------------------------------------------
    TypedColumns =
        Table.TransformColumnTypes(
            RenamedColumns,
            {
                {"ExposureId", type text},
                {"CompanyId", type text},
                {"EntityId", type text},
                {"ExposureDate", type date},
                {"ExposureType", type text},
                {"Counterparty", type text},
                {"ExposureAmount", Currency.Type},
                {"SecuredAmount", Currency.Type},
                {"UnsecuredAmount", Currency.Type},
                {"Currency", type text},
                {"ExposureStatus", type text},
                {"MaturityDate", type date},
                {"ProbabilityOfDefault", type number},
                {"LossGivenDefault", type number},
                {"CreditRating", type text},
                {"Country", type text},
                {"Industry", type text}
            },
            "en-US"
        ),

    // ------------------------------------------------------------
    // 6. CLEAN TEXT FIELDS
    // ------------------------------------------------------------
    CleanTextColumns =
        Table.TransformColumns(
            TypedColumns,
            {
                {
                    "ExposureType",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                },
                {
                    "Counterparty",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Trim(Text.From(_)),
                    type text
                },
                {
                    "Currency",
                    each
                        if _ = null
                        then "UNKNOWN"
                        else Text.Upper(Text.Trim(Text.From(_))),
                    type text
                },
                {
                    "ExposureStatus",
                    each
                        if _ = null
                        then "Unknown"
                        else Text.Proper(Text.Trim(Text.From(_))),
                    type text
                },
                {
                    "CreditRating",
                    each
                        if _ = null
                        then "Not Rated"
                        else Text.Upper(Text.Trim(Text.From(_))),
                    type text
                }
            }
        ),

    // ------------------------------------------------------------
    // 7. REPLACE INVALID NUMERIC VALUES
    // ------------------------------------------------------------
    CleanNumericValues =
        Table.TransformColumns(
            CleanTextColumns,
            {
                {
                    "ExposureAmount",
                    each
                        if _ = null or _ < 0
                        then 0
                        else _,
                    Currency.Type
                },
                {
                    "SecuredAmount",
                    each
                        if _ = null or _ < 0
                        then 0
                        else _,
                    Currency.Type
                },
                {
                    "UnsecuredAmount",
                    each
                        if _ = null or _ < 0
                        then 0
                        else _,
                    Currency.Type
                },
                {
                    "ProbabilityOfDefault",
                    each
                        if _ = null
                        then null
                        else
                            if _ > 1 and _ <= 100
                            then _ / 100
                            else if _ < 0
                            then 0
                            else if _ > 1
                            then 1
                            else _,
                    type number
                },
                {
                    "LossGivenDefault",
                    each
                        if _ = null
                        then null
                        else
                            if _ > 1 and _ <= 100
                            then _ / 100
                            else if _ < 0
                            then 0
                            else if _ > 1
                            then 1
                            else _,
                    type number
                }
            }
        ),

    // ------------------------------------------------------------
    // 8. CALCULATE EXPOSURE BALANCE
    // ------------------------------------------------------------
    AddCalculatedUnsecuredAmount =
        Table.AddColumn(
            CleanNumericValues,
            "CalculatedUnsecuredAmount",
            each
                if [UnsecuredAmount] = null
                then
                    Number.Max(
                        {
                            0,
                            [ExposureAmount] - [SecuredAmount]
                        }
                    )
                else [UnsecuredAmount],
            Currency.Type
        ),

    AddCalculatedSecuredAmount =
        Table.AddColumn(
            AddCalculatedUnsecuredAmount,
            "CalculatedSecuredAmount",
            each
                if [SecuredAmount] = null
                then
                    Number.Min(
                        {
                            [ExposureAmount],
                            [ExposureAmount]
                        }
                    )
                else [SecuredAmount],
            Currency.Type
        ),

    // ------------------------------------------------------------
    // 9. EXPOSURE PERCENTAGES
    // ------------------------------------------------------------
    AddSecuredExposurePct =
        Table.AddColumn(
            AddCalculatedSecuredAmount,
            "SecuredExposurePct",
            each
                if [ExposureAmount] = null
                    or [ExposureAmount] = 0
                then null
                else
                    [CalculatedSecuredAmount]
                    / [ExposureAmount],
            Percentage.Type
        ),

    AddUnsecuredExposurePct =
        Table.AddColumn(
            AddSecuredExposurePct,
            "UnsecuredExposurePct",
            each
                if [ExposureAmount] = null
                    or [ExposureAmount] = 0
                then null
                else
                    [CalculatedUnsecuredAmount]
                    / [ExposureAmount],
            Percentage.Type
        ),

    // ------------------------------------------------------------
    // 10. EXPOSURE RISK LEVEL
    // ------------------------------------------------------------
    AddExposureRiskLevel =
        Table.AddColumn(
            AddUnsecuredExposurePct,
            "ExposureRiskLevel",
            each
                if [UnsecuredExposurePct] = null
                then "Unknown"
                else if [UnsecuredExposurePct] >= 0.75
                then "Critical"
                else if [UnsecuredExposurePct] >= 0.50
                then "High"
                else if [UnsecuredExposurePct] >= 0.25
                then "Medium"
                else "Low",
            type text
        ),

    // ------------------------------------------------------------
    // 11. EXPECTED CREDIT LOSS
    // ECL = Exposure × Probability of Default × Loss Given Default
    // ------------------------------------------------------------
    AddExpectedCreditLoss =
        Table.AddColumn(
            AddExposureRiskLevel,
            "ExpectedCreditLoss",
            each
                if [ExposureAmount] = null
                    or [ProbabilityOfDefault] = null
                    or [LossGivenDefault] = null
                then null
                else
                    [ExposureAmount]
                    * [ProbabilityOfDefault]
                    * [LossGivenDefault],
            Currency.Type
        ),

    // ------------------------------------------------------------
    // 12. REMAINING MATURITY
    // ------------------------------------------------------------
    CurrentDate =
        Date.From(DateTime.LocalNow()),

    AddDaysToMaturity =
        Table.AddColumn(
            AddExpectedCreditLoss,
            "DaysToMaturity",
            each
                if [MaturityDate] = null
                then null
                else
                    Duration.Days(
                        [MaturityDate] - CurrentDate
                    ),
            Int64.Type
        ),

    AddMaturityBucket =
        Table.AddColumn(
            AddDaysToMaturity,
            "MaturityBucket",
            each
                if [DaysToMaturity] = null
                then "Unknown"
                else if [DaysToMaturity] < 0
                then "Matured"
                else if [DaysToMaturity] <= 30
                then "0-30 Days"
                else if [DaysToMaturity] <= 90
                then "31-90 Days"
                else if [DaysToMaturity] <= 180
                then "91-180 Days"
                else if [DaysToMaturity] <= 365
                then "181-365 Days"
                else "Over 1 Year",
            type text
        ),

    // ------------------------------------------------------------
    // 13. EXPOSURE STATUS FLAGS
    // ------------------------------------------------------------
    AddIsActive =
        Table.AddColumn(
            AddMaturityBucket,
            "IsActive",
            each
                List.Contains(
                    {
                        "Active",
                        "Open",
                        "Outstanding",
                        "Current"
                    },
                    [ExposureStatus]
                ),
            type logical
        ),

    AddIsMatured =
        Table.AddColumn(
            AddIsActive,
            "IsMatured",
            each
                [MaturityDate] <> null
                and [MaturityDate] < CurrentDate,
            type logical
        ),

    AddIsHighRisk =
        Table.AddColumn(
            AddIsMatured,
            "IsHighRisk",
            each
                List.Contains(
                    {
                        "Critical",
                        "High"
                    },
                    [ExposureRiskLevel]
                ),
            type logical
        ),

    // ------------------------------------------------------------
    // 14. DATA QUALITY FLAGS
    // ------------------------------------------------------------
    AddDataQualityStatus =
        Table.AddColumn(
            AddIsHighRisk,
            "DataQualityStatus",
            each
                if [ExposureId] = null
                    or [CompanyId] = null
                then "Missing Identifier"
                else if [ExposureAmount] = null
                then "Missing Exposure Amount"
                else if [SecuredAmount] > [ExposureAmount]
                then "Secured Amount Exceeds Exposure"
                else if [UnsecuredAmount] > [ExposureAmount]
                then "Unsecured Amount Exceeds Exposure"
                else if [ExposureDate] = null
                then "Missing Exposure Date"
                else "Valid",
            type text
        ),

    // ------------------------------------------------------------
    // 15. EXPOSURE MONTH
    // ------------------------------------------------------------
    AddExposureMonth =
        Table.AddColumn(
            AddDataQualityStatus,
            "ExposureMonth",
            each
                if [ExposureDate] = null
                then null
                else Date.StartOfMonth([ExposureDate]),
            type date
        ),

    AddExposureYear =
        Table.AddColumn(
            AddExposureMonth,
            "ExposureYear",
            each
                if [ExposureDate] = null
                then null
                else Date.Year([ExposureDate]),
            Int64.Type
        ),

    AddExposureQuarter =
        Table.AddColumn(
            AddExposureYear,
            "ExposureQuarter",
            each
                if [ExposureDate] = null
                then null
                else
                    "Q"
                    & Text.From(
                        Date.QuarterOfYear([ExposureDate])
                    ),
            type text
        ),

    // ------------------------------------------------------------
    // 16. RISK RANKING
    // ------------------------------------------------------------
    AddRiskPriority =
        Table.AddColumn(
            AddExposureQuarter,
            "RiskPriority",
            each
                if [ExposureRiskLevel] = "Critical"
                then 1
                else if [ExposureRiskLevel] = "High"
                then 2
                else if [ExposureRiskLevel] = "Medium"
                then 3
                else if [ExposureRiskLevel] = "Low"
                then 4
                else 5,
            Int64.Type
        ),

    // ------------------------------------------------------------
    // 17. FINAL FILTERING
    // ------------------------------------------------------------
    ValidExposureRows =
        Table.SelectRows(
            AddRiskPriority,
            each
                [ExposureId] <> null
                or [CompanyId] <> null
                or [ExposureAmount] <> null
        ),

    // ------------------------------------------------------------
    // 18. REMOVE DUPLICATES
    // ------------------------------------------------------------
    DeduplicatedRows =
        if Table.HasColumns(
            ValidExposureRows,
            {"ExposureId"}
        )
        then
            Table.Distinct(
                ValidExposureRows,
                {"ExposureId"}
            )
        else
            ValidExposureRows,

    // ------------------------------------------------------------
    // 19. SORT OUTPUT
    // ------------------------------------------------------------
    SortedRows =
        Table.Sort(
            DeduplicatedRows,
            {
                {"RiskPriority", Order.Ascending},
                {"ExposureAmount", Order.Descending},
                {"ExposureDate", Order.Descending}
            }
        )
in
    SortedRows