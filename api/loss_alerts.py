"""
Loss Alert API Endpoints
========================

API endpoints for automatic loss detection and CFO alerting
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from backend.app.agents.loss_alert_agent import (
    LossAlertAgent,
    check_financial_performance
)
from backend.app.services.notification_service import notification_service

router = APIRouter(prefix="/api/loss-alerts", tags=["Loss Alerts"])


# ============================================================================
# REQUEST MODELS
# ============================================================================

class FinancialDataRequest(BaseModel):
    """Request model for financial data monitoring"""
    revenue: float = Field(..., description="Total revenue", ge=0)
    expenses: float = Field(..., description="Total expenses", ge=0)
    company_name: str = Field(default="Company", description="Company name")
    period: str = Field(default="Current Period", description="Time period")
    previous_profit: Optional[float] = Field(None, description="Previous period profit")
    revenue_breakdown: Optional[Dict[str, float]] = Field(None, description="Revenue by category")
    expense_breakdown: Optional[Dict[str, float]] = Field(None, description="Expenses by category")
    cfo_name: str = Field(default="Chief Financial Officer", description="CFO name")
    cfo_email: str = Field(default="cfo@finco.com", description="CFO email")
    cfo_phone: str = Field(default="+1-555-0100", description="CFO phone")


class QuickCheckRequest(BaseModel):
    """Quick check request"""
    revenue: float = Field(..., ge=0)
    expenses: float = Field(..., ge=0)
    company_name: Optional[str] = "Company"
    period: Optional[str] = "Current Period"


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/monitor")
async def monitor_financial_performance(data: FinancialDataRequest):
    """
    Monitor financial performance and automatically alert CFO if loss detected
    
    This endpoint:
    1. Analyzes revenue vs expenses
    2. Detects losses automatically
    3. Identifies root causes
    4. Generates recommendations
    5. Sends alert to CFO if loss detected
    
    Returns:
        Alert details if loss detected, or success status if profitable
    """
    try:
        agent = LossAlertAgent(
            cfo_name=data.cfo_name,
            cfo_email=data.cfo_email,
            cfo_phone=data.cfo_phone
        )
        
        alert = agent.monitor_and_alert(
            revenue=data.revenue,
            expenses=data.expenses,
            company_name=data.company_name,
            period=data.period,
            previous_profit=data.previous_profit,
            revenue_breakdown=data.revenue_breakdown,
            expense_breakdown=data.expense_breakdown
        )
        
        profit = data.revenue - data.expenses
        
        if alert is None:
            return {
                "status": "success",
                "profitable": True,
                "profit": profit,
                "revenue": data.revenue,
                "expenses": data.expenses,
                "alert_sent": False,
                "message": f"Company is profitable: ${profit:,.2f}"
            }
        
        # Send multi-channel notifications
        notification_results = notification_service.send_loss_alert(
            cfo_email=data.cfo_email,
            cfo_name=data.cfo_name,
            cfo_phone=data.cfo_phone,
            company_name=data.company_name,
            loss_amount=alert.detection.loss_amount,
            period=data.period,
            severity=alert.detection.severity,
            root_causes=alert.root_causes,
            recommendations=alert.recommendations,
            alert_message=alert.message_body
        )
        
        return {
            "status": "success",
            "profitable": False,
            "loss": abs(profit),
            "revenue": data.revenue,
            "expenses": data.expenses,
            "alert_sent": True,
            "notifications": notification_results,
            "alert": alert.to_dict(),
            "message": f"Loss detected: ${abs(profit):,.2f} - CFO has been automatically notified via email, SMS, and dashboard"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error monitoring financial performance: {str(e)}")


@router.post("/quick-check")
async def quick_loss_check(data: QuickCheckRequest):
    """
    Quick check for profit/loss
    
    Simple endpoint to check if company is profitable or has loss
    """
    try:
        result = check_financial_performance(
            revenue=data.revenue,
            expenses=data.expenses,
            company_name=data.company_name or "Company",
            period=data.period or "Current Period"
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking financial performance: {str(e)}")


@router.get("/alert-history")
async def get_alert_history():
    """
    Get history of all loss alerts sent to CFO
    """
    try:
        # In production, this would query from database
        # For now, return empty list
        return {
            "status": "success",
            "alerts": [],
            "total": 0,
            "message": "Alert history retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving alert history: {str(e)}")


@router.post("/test-alert")
async def test_alert_system():
    """
    Test the alert system with sample loss data
    
    Useful for testing CFO notification system
    """
    try:
        result = check_financial_performance(
            revenue=100000,
            expenses=250000,
            company_name="Test Company",
            period="Test Period",
            cfo_name="Test CFO",
            cfo_email="test.cfo@finco.com"
        )
        
        return {
            "status": "success",
            "test_result": result,
            "message": "Test alert system executed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing alert system: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check for loss alert system"""
    return {
        "status": "healthy",
        "service": "Loss Alert Agent",
        "version": "1.0.0",
        "features": [
            "Automatic loss detection",
            "Root cause analysis",
            "CFO alerting",
            "Email notifications",
            "SMS notifications",
            "Dashboard notifications"
        ]
    }
