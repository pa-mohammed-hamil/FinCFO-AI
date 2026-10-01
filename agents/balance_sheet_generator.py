


from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Account:
    """Individual account"""
    account_id: str
    account_name: str
    account_type: str  # asset, liability, equity, revenue, expense
    normal_balance: str  # debit or credit
    balance: Decimal
    parent_category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "account_name": self.account_name,
            "account_type": self.account_type,
            "normal_balance": self.normal_balance,
            "balance": float(self.balance),
            "parent_category": self.parent_category
        }


@dataclass
class JournalEntry:
    """Journal entry with debit and credit"""
    entry_id: str
    date: str
    description: str
    debit_account: str
    credit_account: str
    amount: Decimal
    reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "date": self.date,
            "description": self.description,
            "debit_account": self.debit_account,
            "credit_account": self.credit_account,
            "amount": float(self.amount),
            "reference": self.reference
        }


@dataclass
class BalanceSheet:
    """Complete Balance Sheet"""
    company_name: str
    financial_year: str
    as_of_date: str
    
    # Assets (Debit side)
    current_assets: Dict[str, Decimal]
    non_current_assets: Dict[str, Decimal]
    
    # Liabilities (Credit side)
    current_liabilities: Dict[str, Decimal]
    non_current_liabilities: Dict[str, Decimal]
    
    # Equity (Credit side)
    shareholders_equity: Dict[str, Decimal]
    
    # Totals
    total_assets: Decimal
    total_liabilities: Decimal
    total_equity: Decimal
    total_liabilities_and_equity: Decimal
    
    # Balance check
    is_balanced: bool
    difference: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "financial_year": self.financial_year,
            "as_of_date": self.as_of_date,
            "assets": {
                "current_assets": {k: float(v) for k, v in self.current_assets.items()},
                "non_current_assets": {k: float(v) for k, v in self.non_current_assets.items()},
                "total_current_assets": float(sum(self.current_assets.values())),
                "total_non_current_assets": float(sum(self.non_current_assets.values())),
                "total_assets": float(self.total_assets)
            },
            "liabilities": {
                "current_liabilities": {k: float(v) for k, v in self.current_liabilities.items()},
                "non_current_liabilities": {k: float(v) for k, v in self.non_current_liabilities.items()},
                "total_current_liabilities": float(sum(self.current_liabilities.values())),
                "total_non_current_liabilities": float(sum(self.non_current_liabilities.values())),
                "total_liabilities": float(self.total_liabilities)
            },
            "equity": {
                "shareholders_equity": {k: float(v) for k, v in self.shareholders_equity.items()},
                "total_equity": float(self.total_equity)
            },
            "totals": {
                "total_assets": float(self.total_assets),
                "total_liabilities_and_equity": float(self.total_liabilities_and_equity),
                "is_balanced": self.is_balanced,
                "difference": float(self.difference)
            }
        }


# ============================================================================
# CHART OF ACCOUNTS
# ============================================================================

class ChartOfAccounts:
    """
    Standard Chart of Accounts with proper debit/credit normal balances
    """
    
    def __init__(self):
        self.accounts: Dict[str, Account] = {}
        self._initialize_standard_accounts()
    
    def _initialize_standard_accounts(self):
        """Initialize standard chart of accounts"""
        
        # ASSETS (Debit normal balance)
        asset_accounts = [
            ("1001", "Cash and Cash Equivalents", "asset", "debit", "Current Assets"),
            ("1002", "Accounts Receivable", "asset", "debit", "Current Assets"),
            ("1003", "Inventory", "asset", "debit", "Current Assets"),
            ("1004", "Prepaid Expenses", "asset", "debit", "Current Assets"),
            ("1005", "Short-term Investments", "asset", "debit", "Current Assets"),
            ("1101", "Property, Plant & Equipment", "asset", "debit", "Non-Current Assets"),
            ("1102", "Accumulated Depreciation", "asset", "credit", "Non-Current Assets"),  # Contra-asset
            ("1103", "Intangible Assets", "asset", "debit", "Non-Current Assets"),
            ("1104", "Long-term Investments", "asset", "debit", "Non-Current Assets"),
            ("1105", "Goodwill", "asset", "debit", "Non-Current Assets"),
        ]
        
        # LIABILITIES (Credit normal balance)
        liability_accounts = [
            ("2001", "Accounts Payable", "liability", "credit", "Current Liabilities"),
            ("2002", "Short-term Debt", "liability", "credit", "Current Liabilities"),
            ("2003", "Accrued Expenses", "liability", "credit", "Current Liabilities"),
            ("2004", "Unearned Revenue", "liability", "credit", "Current Liabilities"),
            ("2005", "Current Portion of Long-term Debt", "liability", "credit", "Current Liabilities"),
            ("2101", "Long-term Debt", "liability", "credit", "Non-Current Liabilities"),
            ("2102", "Bonds Payable", "liability", "credit", "Non-Current Liabilities"),
            ("2103", "Deferred Tax Liabilities", "liability", "credit", "Non-Current Liabilities"),
            ("2104", "Pension Obligations", "liability", "credit", "Non-Current Liabilities"),
        ]
        
        # EQUITY (Credit normal balance)
        equity_accounts = [
            ("3001", "Common Stock", "equity", "credit", "Shareholders' Equity"),
            ("3002", "Preferred Stock", "equity", "credit", "Shareholders' Equity"),
            ("3003", "Additional Paid-in Capital", "equity", "credit", "Shareholders' Equity"),
            ("3004", "Retained Earnings", "equity", "credit", "Shareholders' Equity"),
            ("3005", "Treasury Stock", "equity", "debit", "Shareholders' Equity"),  # Contra-equity
            ("3006", "Accumulated Other Comprehensive Income", "equity", "credit", "Shareholders' Equity"),
        ]
        
        # Create account objects
        all_accounts = asset_accounts + liability_accounts + equity_accounts
        
        for acc_id, name, acc_type, normal_bal, category in all_accounts:
            self.accounts[acc_id] = Account(
                account_id=acc_id,
                account_name=name,
                account_type=acc_type,
                normal_balance=normal_bal,
                balance=Decimal("0"),
                parent_category=category
            )
    
    def get_account(self, account_id: str) -> Optional[Account]:
        """Get account by ID"""
        return self.accounts.get(account_id)
    
    def get_accounts_by_type(self, account_type: str) -> List[Account]:
        """Get all accounts of a specific type"""
        return [acc for acc in self.accounts.values() if acc.account_type == account_type]
    
    def get_accounts_by_category(self, category: str) -> List[Account]:
        """Get all accounts in a category"""
        return [acc for acc in self.accounts.values() if acc.parent_category == category]


# ============================================================================
# ACCOUNTING ENGINE
# ============================================================================

class AccountingEngine:
    """
    Core accounting engine that ensures debits = credits
    """
    
    def __init__(self):
        self.chart_of_accounts = ChartOfAccounts()
        self.journal_entries: List[JournalEntry] = []
    
    def post_journal_entry(
        self,
        description: str,
        debit_account_id: str,
        credit_account_id: str,
        amount: Decimal,
        date: Optional[str] = None,
        reference: Optional[str] = None
    ) -> JournalEntry:
        """
        Post a journal entry (automatic debit/credit balancing)
        
        The fundamental rule: Debit = Credit
        Every entry has equal debits and credits
        """
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        debit_account = self.chart_of_accounts.get_account(debit_account_id)
        credit_account = self.chart_of_accounts.get_account(credit_account_id)
        
        if not debit_account or not credit_account:
            raise ValueError("Invalid account ID")
        
        # Create journal entry
        entry = JournalEntry(
            entry_id=str(uuid.uuid4()),
            date=date or datetime.now(timezone.utc).isoformat(),
            description=description,
            debit_account=debit_account.account_name,
            credit_account=credit_account.account_name,
            amount=amount,
            reference=reference
        )
        
        # Update account balances
        # Debit increases: Assets, Expenses | Decreases: Liabilities, Equity, Revenue
        if debit_account.normal_balance == "debit":
            debit_account.balance += amount
        else:
            debit_account.balance -= amount
        
        # Credit increases: Liabilities, Equity, Revenue | Decreases: Assets, Expenses
        if credit_account.normal_balance == "credit":
            credit_account.balance += amount
        else:
            credit_account.balance -= amount
        
        self.journal_entries.append(entry)
        return entry
    
    def verify_trial_balance(self) -> Tuple[Decimal, Decimal, bool]:
        """
        Verify that total debits = total credits (Trial Balance)
        
        Returns: (total_debits, total_credits, is_balanced)
        """
        total_debits = Decimal("0")
        total_credits = Decimal("0")
        
        for account in self.chart_of_accounts.accounts.values():
            if account.normal_balance == "debit":
                total_debits += abs(account.balance)
            else:
                total_credits += abs(account.balance)
        
        is_balanced = abs(total_debits - total_credits) < Decimal("0.01")
        
        return total_debits, total_credits, is_balanced
    
    def close_net_income_to_retained_earnings(self, net_income: Decimal):
        """
        Close net income to retained earnings (year-end closing)
        """
        if net_income > 0:
            # Profit: Debit Income Summary, Credit Retained Earnings
            self.post_journal_entry(
                description="Close net income to retained earnings",
                debit_account_id="3004",  # Retained Earnings (will decrease debit side)
                credit_account_id="3004",  # Retained Earnings (will increase credit side)
                amount=net_income,
                reference="YE-CLOSE"
            )
        elif net_income < 0:
            # Loss: Debit Retained Earnings, Credit Income Summary
            self.post_journal_entry(
                description="Close net loss to retained earnings",
                debit_account_id="3004",  # Retained Earnings
                credit_account_id="3004",  # Retained Earnings
                amount=abs(net_income),
                reference="YE-CLOSE"
            )


# ============================================================================
# BALANCE SHEET GENERATOR
# ============================================================================

class BalanceSheetGenerator:
    """
    Automatically generates a complete, balanced Balance Sheet
    """
    
    def __init__(self, accounting_engine: AccountingEngine):
        self.engine = accounting_engine
    
    def generate_balance_sheet(
        self,
        company_name: str,
        financial_year: str,
        as_of_date: Optional[str] = None
    ) -> BalanceSheet:
        """
        Generate complete balance sheet
        
        Automatically ensures: Assets = Liabilities + Equity
        """
        if not as_of_date:
            as_of_date = datetime.now(timezone.utc).date().isoformat()
        
        # Get accounts by category
        current_assets = self._get_category_balances("Current Assets")
        non_current_assets = self._get_category_balances("Non-Current Assets")
        current_liabilities = self._get_category_balances("Current Liabilities")
        non_current_liabilities = self._get_category_balances("Non-Current Liabilities")
        shareholders_equity = self._get_category_balances("Shareholders' Equity")
        
        # Calculate totals
        total_current_assets = sum(current_assets.values())
        total_non_current_assets = sum(non_current_assets.values())
        total_assets = total_current_assets + total_non_current_assets
        
        total_current_liabilities = sum(current_liabilities.values())
        total_non_current_liabilities = sum(non_current_liabilities.values())
        total_liabilities = total_current_liabilities + total_non_current_liabilities
        
        total_equity = sum(shareholders_equity.values())
        total_liabilities_and_equity = total_liabilities + total_equity
        
        # Check if balanced
        difference = total_assets - total_liabilities_and_equity
        is_balanced = abs(difference) < Decimal("0.01")
        
        balance_sheet = BalanceSheet(
            company_name=company_name,
            financial_year=financial_year,
            as_of_date=as_of_date,
            current_assets=current_assets,
            non_current_assets=non_current_assets,
            current_liabilities=current_liabilities,
            non_current_liabilities=non_current_liabilities,
            shareholders_equity=shareholders_equity,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            total_equity=total_equity,
            total_liabilities_and_equity=total_liabilities_and_equity,
            is_balanced=is_balanced,
            difference=difference
        )
        
        return balance_sheet
    
    def _get_category_balances(self, category: str) -> Dict[str, Decimal]:
        """Get all account balances in a category"""
        accounts = self.engine.chart_of_accounts.get_accounts_by_category(category)
        return {
            acc.account_name: acc.balance
            for acc in accounts
            if acc.balance != Decimal("0")
        }
    
    def format_balance_sheet(self, balance_sheet: BalanceSheet) -> str:
        """
        Format balance sheet as readable text
        """
        output = []
        output.append("=" * 80)
        output.append(f"{balance_sheet.company_name}")
        output.append(f"BALANCE SHEET")
        output.append(f"As of {balance_sheet.as_of_date}")
        output.append(f"Financial Year: {balance_sheet.financial_year}")
        output.append("=" * 80)
        output.append("")
        
        # ASSETS
        output.append("ASSETS")
        output.append("-" * 80)
        output.append("")
        output.append("Current Assets:")
        for name, amount in balance_sheet.current_assets.items():
            output.append(f"  {name:<50} ${amount:>20,.2f}")
        output.append(f"  {'Total Current Assets':<50} ${sum(balance_sheet.current_assets.values()):>20,.2f}")
        output.append("")
        
        output.append("Non-Current Assets:")
        for name, amount in balance_sheet.non_current_assets.items():
            output.append(f"  {name:<50} ${amount:>20,.2f}")
        output.append(f"  {'Total Non-Current Assets':<50} ${sum(balance_sheet.non_current_assets.values()):>20,.2f}")
        output.append("")
        
        output.append(f"{'TOTAL ASSETS':<50} ${balance_sheet.total_assets:>20,.2f}")
        output.append("=" * 80)
        output.append("")
        
        # LIABILITIES
        output.append("LIABILITIES")
        output.append("-" * 80)
        output.append("")
        output.append("Current Liabilities:")
        for name, amount in balance_sheet.current_liabilities.items():
            output.append(f"  {name:<50} ${amount:>20,.2f}")
        output.append(f"  {'Total Current Liabilities':<50} ${sum(balance_sheet.current_liabilities.values()):>20,.2f}")
        output.append("")
        
        output.append("Non-Current Liabilities:")
        for name, amount in balance_sheet.non_current_liabilities.items():
            output.append(f"  {name:<50} ${amount:>20,.2f}")
        output.append(f"  {'Total Non-Current Liabilities':<50} ${sum(balance_sheet.non_current_liabilities.values()):>20,.2f}")
        output.append("")
        
        output.append(f"{'TOTAL LIABILITIES':<50} ${balance_sheet.total_liabilities:>20,.2f}")
        output.append("")
        
        # EQUITY
        output.append("SHAREHOLDERS' EQUITY")
        output.append("-" * 80)
        for name, amount in balance_sheet.shareholders_equity.items():
            output.append(f"  {name:<50} ${amount:>20,.2f}")
        output.append("")
        output.append(f"{'TOTAL EQUITY':<50} ${balance_sheet.total_equity:>20,.2f}")
        output.append("")
        
        output.append(f"{'TOTAL LIABILITIES AND EQUITY':<50} ${balance_sheet.total_liabilities_and_equity:>20,.2f}")
        output.append("=" * 80)
        output.append("")
        
        # Balance Check
        if balance_sheet.is_balanced:
            output.append("✅ BALANCE SHEET IS BALANCED!")
            output.append(f"   Assets = Liabilities + Equity")
            output.append(f"   ${balance_sheet.total_assets:,.2f} = ${balance_sheet.total_liabilities_and_equity:,.2f}")
        else:
            output.append("⚠️  WARNING: BALANCE SHEET IS OUT OF BALANCE!")
            output.append(f"   Difference: ${balance_sheet.difference:,.2f}")
        
        output.append("=" * 80)
        
        return "\n".join(output)


# ============================================================================
# AUTOMATIC BALANCE SHEET AGENT
# ============================================================================

class AutomaticBalanceSheetAgent:
    """
    Main agent that automatically generates complete balance sheets
    with all accounting tasks handled automatically
    """
    
    def __init__(self):
        self.engine = AccountingEngine()
        self.generator = BalanceSheetGenerator(self.engine)
    
    def create_sample_company_data(self):
        """
        Create sample financial data for demonstration
        """
        # Post sample transactions (all automatically balanced)
        
        # 1. Initial capital investment
        self.engine.post_journal_entry(
            "Initial capital investment",
            debit_account_id="1001",  # Cash
            credit_account_id="3001",  # Common Stock
            amount=Decimal("1000000")
        )
        
        # 2. Purchase equipment
        self.engine.post_journal_entry(
            "Purchase equipment",
            debit_account_id="1101",  # PP&E
            credit_account_id="1001",  # Cash
            amount=Decimal("300000")
        )
        
        # 3. Purchase inventory on credit
        self.engine.post_journal_entry(
            "Purchase inventory on credit",
            debit_account_id="1003",  # Inventory
            credit_account_id="2001",  # Accounts Payable
            amount=Decimal("150000")
        )
        
        # 4. Obtain long-term loan
        self.engine.post_journal_entry(
            "Obtain long-term loan",
            debit_account_id="1001",  # Cash
            credit_account_id="2101",  # Long-term Debt
            amount=Decimal("500000")
        )
        
        # 5. Accounts receivable from sales
        self.engine.post_journal_entry(
            "Sales on credit",
            debit_account_id="1002",  # Accounts Receivable
            credit_account_id="3004",  # Retained Earnings (revenue)
            amount=Decimal("400000")
        )
        
        # 6. Depreciation
        self.engine.post_journal_entry(
            "Annual depreciation",
            debit_account_id="3004",  # Retained Earnings (expense)
            credit_account_id="1102",  # Accumulated Depreciation
            amount=Decimal("30000")
        )
        
        # 7. Pay accounts payable
        self.engine.post_journal_entry(
            "Pay suppliers",
            debit_account_id="2001",  # Accounts Payable
            credit_account_id="1001",  # Cash
            amount=Decimal("100000")
        )
        
        # 8. Accrued expenses
        self.engine.post_journal_entry(
            "Accrued salary expenses",
            debit_account_id="3004",  # Retained Earnings (expense)
            credit_account_id="2003",  # Accrued Expenses
            amount=Decimal("50000")
        )
    
    def generate_complete_balance_sheet(
        self,
        company_name: str = "Acme Corporation",
        financial_year: str = "2024",
        include_sample_data: bool = False
    ) -> BalanceSheet:
        """
        Automatically generate complete balance sheet
        
        Args:
            company_name: Company name
            financial_year: Financial year
            include_sample_data: If True, creates sample data
            
        Returns:
            Complete, balanced Balance Sheet
        """
        if include_sample_data:
            self.create_sample_company_data()
        
        # Verify trial balance
        total_debits, total_credits, is_balanced = self.engine.verify_trial_balance()
        
        print(f"\n📊 Trial Balance Verification:")
        print(f"   Total Debits:  ${total_debits:,.2f}")
        print(f"   Total Credits: ${total_credits:,.2f}")
        print(f"   Balanced: {'✅ YES' if is_balanced else '❌ NO'}\n")
        
        # Generate balance sheet
        balance_sheet = self.generator.generate_balance_sheet(
            company_name=company_name,
            financial_year=financial_year
        )
        
        return balance_sheet
    
    def print_balance_sheet(self, balance_sheet: BalanceSheet):
        """Print formatted balance sheet"""
        formatted = self.generator.format_balance_sheet(balance_sheet)
        print(formatted)
    
    def get_journal_entries(self) -> List[JournalEntry]:
        """Get all journal entries"""
        return self.engine.journal_entries


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_automatic_balance_sheet(
    company_name: str = "Acme Corporation",
    financial_year: str = "2024",
    with_sample_data: bool = True
) -> Dict[str, Any]:
    """
    Generate automatic balance sheet with one function call
    
    Returns complete balance sheet as dictionary
    """
    agent = AutomaticBalanceSheetAgent()
    
    balance_sheet = agent.generate_complete_balance_sheet(
        company_name=company_name,
        financial_year=financial_year,
        include_sample_data=with_sample_data
    )
    
    agent.print_balance_sheet(balance_sheet)
    
    return {
        "balance_sheet": balance_sheet.to_dict(),
        "journal_entries": [entry.to_dict() for entry in agent.get_journal_entries()],
        "is_balanced": balance_sheet.is_balanced,
        "message": "Balance Sheet generated automatically with all debits and credits balanced!"
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FinCo AI - Automatic Balance Sheet Generator")
    print("Generates complete Balance Sheet with Credits and Debits Balanced")
    print("="*80 + "\n")
    
    result = generate_automatic_balance_sheet(
        company_name="Acme Corporation",
        financial_year="2024",
        with_sample_data=True
    )
    
    print(f"\n\n✅ {result['message']}")
    print(f"📝 Generated {len(result['journal_entries'])} journal entries")
    print(f"⚖️  Balance Status: {'BALANCED ✅' if result['is_balanced'] else 'OUT OF BALANCE ❌'}")
