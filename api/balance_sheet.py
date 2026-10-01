"""
Balance Sheet API Endpoints
===========================

API endpoints for automatic balance sheet generation
with credits and debits automatically balanced
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from decimal import Decimal

from backend.app.agents.balance_sheet_generator import (
    AutomaticBalanceSheetAgent,
    generate_automatic_balance_sheet
)

router = APIRouter(prefix="/balance-sheet", tags=["Balance Sheet"])


# ============================================================================
# REQUEST MODELS
# ============================================================================

class GenerateBalanceSheetRequest(BaseModel):
    """Request to generate balance sheet"""
    company_name: str = Field(default="Company", description="Company name")
    financial_year: str = Field(default="2024", description="Financial year")
    include_sample_data: bool = Field(default=False, description="Include sample transactions")


class JournalEntryRequest(BaseModel):
    """Request to post a journal entry"""
    description: str = Field(..., description="Entry description")
    debit_account_id: str = Field(..., description="Debit account ID (e.g., '1001')")
    credit_account_id: str = Field(..., description="Credit account ID (e.g., '2001')")
    amount: float = Field(..., gt=0, description="Amount (must be positive)")
    date: Optional[str] = Field(None, description="Transaction date")
    reference: Optional[str] = Field(None, description="Reference number")


# ============================================================================
# GLOBAL AGENT INSTANCE
# ============================================================================

# In production, this would be stored in database per company/session
_agent_instance = None

def get_agent() -> AutomaticBalanceSheetAgent:
    """Get or create agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AutomaticBalanceSheetAgent()
    return _agent_instance


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/generate")
async def generate_balance_sheet(request: GenerateBalanceSheetRequest):
    """
    Automatically generate complete balance sheet
    
    This endpoint:
    1. Creates/uses accounting engine
    2. Posts sample transactions (if requested)
    3. Verifies trial balance (debits = credits)
    4. Generates complete balance sheet
    5. Ensures Assets = Liabilities + Equity
    
    Returns:
        Complete balance sheet with all accounts balanced
    """
    try:
        result = generate_automatic_balance_sheet(
            company_name=request.company_name,
            financial_year=request.financial_year,
            with_sample_data=request.include_sample_data
        )
        
        return {
            "status": "success",
            "data": result,
            "message": "Balance sheet generated successfully with all accounts balanced"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating balance sheet: {str(e)}")


@router.post("/journal-entry")
async def post_journal_entry(request: JournalEntryRequest):
    """
    Post a journal entry (automatic debit/credit balancing)
    
    Every entry maintains the accounting equation:
    - Debit = Credit
    - Assets = Liabilities + Equity
    """
    try:
        agent = get_agent()
        
        entry = agent.engine.post_journal_entry(
            description=request.description,
            debit_account_id=request.debit_account_id,
            credit_account_id=request.credit_account_id,
            amount=Decimal(str(request.amount)),
            date=request.date,
            reference=request.reference
        )
        
        # Verify balance after posting
        total_debits, total_credits, is_balanced = agent.engine.verify_trial_balance()
        
        return {
            "status": "success",
            "journal_entry": entry.to_dict(),
            "trial_balance": {
                "total_debits": float(total_debits),
                "total_credits": float(total_credits),
                "is_balanced": is_balanced
            },
            "message": f"Journal entry posted successfully. Trial balance: {'BALANCED ✅' if is_balanced else 'OUT OF BALANCE ❌'}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error posting journal entry: {str(e)}")


@router.get("/trial-balance")
async def get_trial_balance():
    """
    Get trial balance (verify debits = credits)
    """
    try:
        agent = get_agent()
        total_debits, total_credits, is_balanced = agent.engine.verify_trial_balance()
        
        return {
            "status": "success",
            "trial_balance": {
                "total_debits": float(total_debits),
                "total_credits": float(total_credits),
                "difference": float(abs(total_debits - total_credits)),
                "is_balanced": is_balanced
            },
            "message": f"Trial balance: {'BALANCED ✅' if is_balanced else 'OUT OF BALANCE ❌'}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trial balance: {str(e)}")


@router.get("/chart-of-accounts")
async def get_chart_of_accounts():
    """
    Get complete chart of accounts
    """
    try:
        agent = get_agent()
        accounts = [acc.to_dict() for acc in agent.engine.chart_of_accounts.accounts.values()]
        
        return {
            "status": "success",
            "accounts": accounts,
            "total_accounts": len(accounts),
            "message": "Chart of accounts retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chart of accounts: {str(e)}")


@router.get("/journal-entries")
async def get_journal_entries():
    """
    Get all journal entries
    """
    try:
        agent = get_agent()
        entries = [entry.to_dict() for entry in agent.engine.journal_entries]
        
        return {
            "status": "success",
            "journal_entries": entries,
            "total_entries": len(entries),
            "message": "Journal entries retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting journal entries: {str(e)}")


@router.post("/reset")
async def reset_agent():
    """
    Reset the agent (clear all data)
    """
    global _agent_instance
    _agent_instance = None
    
    return {
        "status": "success",
        "message": "Agent reset successfully. All data cleared."
    }


@router.get("/health")
async def health_check():
    """Health check for balance sheet service"""
    return {
        "status": "healthy",
        "service": "Balance Sheet Generator",
        "version": "1.0.0",
        "features": [
            "Automatic balance sheet generation",
            "Credit/Debit balancing",
            "Trial balance verification",
            "Journal entry posting",
            "Chart of accounts",
            "Financial year-end closing"
        ]
    }
