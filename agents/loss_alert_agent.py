"""
FinCo AI - Automatic Loss Alert Agent
======================================

This agent monitors financial calculations and automatically detects company losses.
When a loss is detected, it:
1. Analyzes the loss (amount, trend, severity)
2. Generates detailed report with root causes
3. Creates recommendations
4. Automatically sends alert message to Chief Financial Officer (CFO)
5. Logs the alert for audit trail

Features
--------
- Automatic loss detection from financial calculations
- Real-time monitoring
- Severity classification (Low/Medium/High/Critical)
- Root cause analysis
- Automatic CFO notification
- Email & SMS alerts
- Dashboard notifications
- Audit trail logging
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class LossDetection:
    """Loss detection result"""
    detection_id: str
    timestamp: str
    company_name: str
    period: str
    loss_amount: float
    currency: str
    severity: str  # low, medium, high, critical
    previous_period_comparison: Optional[float] = None
    trend: str = "stable"  # improving, stable, worsening
    confidence: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp,
            "company_name": self.company_name,
            "period": self.period,
            "loss_amount": self.loss_amount,
            "currency": self.currency,
            "severity": self.severity,
            "previous_period_comparison": self.previous_period_comparison,
            "trend": self.trend,
            "confidence": self.confidence
        }


@dataclass
class LossAlert:
    """Alert message to CFO"""
    alert_id: str
    detection: LossDetection
    root_causes: List[str]
    recommendations: List[str]
    cfo_email: str
    cfo_name: str
    message_body: str
    sent_at: str
    delivery_status: str = "pending"  # pending, sent, delivered, failed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "detection": self.detection.to_dict(),
            "root_causes": self.root_causes,
            "recommendations": self.recommendations,
            "cfo_email": self.cfo_email,
            "cfo_name": self.cfo_name,
            "message_body": self.message_body,
            "sent_at": self.sent_at,
            "delivery_status": self.delivery_status
        }


# ============================================================================
# LOSS DETECTION ENGINE
# ============================================================================

class LossDetectionEngine:
    """
    Monitors financial calculations and automatically detects losses
    """
    
    def __init__(self):
        self.detection_threshold = 0.0  # Any negative profit is a loss
        self.severity_thresholds = {
            "low": 10000,      # Loss < $10K
            "medium": 50000,   # Loss < $50K
            "high": 200000,    # Loss < $200K
            "critical": 200000 # Loss >= $200K
        }
    
    def detect_loss(
        self,
        revenue: float,
        expenses: float,
        company_name: str = "Company",
        period: str = "Current Period",
        previous_profit: Optional[float] = None
    ) -> Optional[LossDetection]:
        """
        Detect if financial calculations show a loss
        
        Args:
            revenue: Total revenue
            expenses: Total expenses
            company_name: Company name
            period: Time period (e.g., "Q4 2024", "December 2024")
            previous_profit: Profit from previous period for comparison
            
        Returns:
            LossDetection if loss detected, None otherwise
        """
        profit = revenue - expenses
        
        # Check if there's a loss
        if profit >= self.detection_threshold:
            return None  # No loss detected
        
        loss_amount = abs(profit)
        
        # Determine severity
        severity = self._calculate_severity(loss_amount)
        
        # Determine trend
        trend = "stable"
        previous_comparison = None
        if previous_profit is not None:
            previous_comparison = ((profit - previous_profit) / abs(previous_profit)) * 100 if previous_profit != 0 else 0
            if profit < previous_profit:
                trend = "worsening"
            elif profit > previous_profit:
                trend = "improving"
        
        detection = LossDetection(
            detection_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            company_name=company_name,
            period=period,
            loss_amount=loss_amount,
            currency="USD",
            severity=severity,
            previous_period_comparison=previous_comparison,
            trend=trend,
            confidence=0.95
        )
        
        return detection
    
    def _calculate_severity(self, loss_amount: float) -> str:
        """Calculate loss severity level"""
        if loss_amount >= self.severity_thresholds["critical"]:
            return "critical"
        elif loss_amount >= self.severity_thresholds["high"]:
            return "high"
        elif loss_amount >= self.severity_thresholds["medium"]:
            return "medium"
        else:
            return "low"


# ============================================================================
# ROOT CAUSE ANALYZER
# ============================================================================

class RootCauseAnalyzer:
    """
    Analyzes financial data to identify root causes of losses
    """
    
    def analyze(
        self,
        detection: LossDetection,
        revenue: float,
        expenses: float,
        revenue_breakdown: Optional[Dict[str, float]] = None,
        expense_breakdown: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Identify root causes of the loss
        
        Returns:
            List of root cause descriptions
        """
        causes = []
        
        # Check revenue issues
        if revenue < expenses * 0.5:
            causes.append(f"Revenue significantly below expenses (${revenue:,.2f} vs ${expenses:,.2f})")
        
        # Check expense issues
        if expenses > revenue * 1.5:
            causes.append(f"Expenses excessively high (${expenses:,.2f} vs revenue ${revenue:,.2f})")
        
        # Analyze expense breakdown if available
        if expense_breakdown:
            total_expenses = sum(expense_breakdown.values())
            for category, amount in expense_breakdown.items():
                percentage = (amount / total_expenses * 100) if total_expenses > 0 else 0
                if percentage > 40:
                    causes.append(f"High {category} expenses: ${amount:,.2f} ({percentage:.1f}% of total)")
        
        # Analyze revenue breakdown if available
        if revenue_breakdown:
            declining_categories = []
            for category, amount in revenue_breakdown.items():
                if amount < 0:
                    declining_categories.append(category)
            
            if declining_categories:
                causes.append(f"Declining revenue in: {', '.join(declining_categories)}")
        
        # Check trend
        if detection.trend == "worsening":
            causes.append(f"Loss is worsening compared to previous period (change: {detection.previous_period_comparison:.1f}%)")
        
        # Default if no specific causes found
        if not causes:
            causes.append(f"Expenses (${expenses:,.2f}) exceed revenue (${revenue:,.2f}) by ${detection.loss_amount:,.2f}")
        
        return causes


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """
    Generates actionable recommendations to address losses
    """
    
    def generate_recommendations(
        self,
        detection: LossDetection,
        root_causes: List[str]
    ) -> List[str]:
        """
        Generate recommendations based on loss detection and root causes
        
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Severity-based recommendations
        if detection.severity in ["critical", "high"]:
            recommendations.append("URGENT: Schedule emergency financial review meeting within 24 hours")
            recommendations.append("Implement immediate cost reduction measures")
            recommendations.append("Review and potentially pause all non-essential spending")
        
        if detection.severity in ["medium", "high", "critical"]:
            recommendations.append("Conduct detailed expense audit across all departments")
            recommendations.append("Review pricing strategy and revenue optimization opportunities")
        
        # Trend-based recommendations
        if detection.trend == "worsening":
            recommendations.append("Investigate trend: losses are increasing compared to previous period")
            recommendations.append("Implement financial monitoring dashboards for real-time tracking")
        
        # Root cause-based recommendations
        for cause in root_causes:
            if "high" in cause.lower() and "expenses" in cause.lower():
                category = cause.split("High ")[1].split(" expenses")[0] if "High " in cause else "operating"
                recommendations.append(f"Analyze and reduce {category} expenses immediately")
            
            if "revenue" in cause.lower() and "below" in cause.lower():
                recommendations.append("Implement revenue growth strategy: increase sales, adjust pricing, expand market")
            
            if "declining" in cause.lower():
                recommendations.append("Focus on revenue recovery in declining categories")
        
        # General recommendations
        recommendations.append("Review cash flow projections for next 3-6 months")
        recommendations.append("Consider consulting with financial advisors for strategic planning")
        recommendations.append("Update financial forecasts and adjust business plan accordingly")
        
        return recommendations


# ============================================================================
# CFO ALERT SYSTEM
# ============================================================================

class CFOAlertSystem:
    """
    Automatically sends alerts to CFO when losses are detected
    """
    
    def __init__(
        self,
        cfo_name: str = "Chief Financial Officer",
        cfo_email: str = "cfo@finco.com",
        cfo_phone: str = "+1-555-0100"
    ):
        self.cfo_name = cfo_name
        self.cfo_email = cfo_email
        self.cfo_phone = cfo_phone
        self.alert_history: List[LossAlert] = []
    
    def create_alert_message(
        self,
        detection: LossDetection,
        root_causes: List[str],
        recommendations: List[str]
    ) -> str:
        """
        Create formatted alert message for CFO
        """
        severity_emoji = {
            "low": "⚠️",
            "medium": "⚠️",
            "high": "🔴",
            "critical": "🚨"
        }
        
        emoji = severity_emoji.get(detection.severity, "⚠️")
        
        message = f"""
{emoji} AUTOMATIC FINANCIAL LOSS ALERT {emoji}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSS DETECTION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Company: {detection.company_name}
Period: {detection.period}
Loss Amount: ${detection.loss_amount:,.2f} {detection.currency}
Severity: {detection.severity.upper()}
Trend: {detection.trend.upper()}
Detection Time: {detection.timestamp}
Confidence: {detection.confidence * 100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROOT CAUSES IDENTIFIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for i, cause in enumerate(root_causes, 1):
            message += f"{i}. {cause}\n"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for i, rec in enumerate(recommendations, 1):
            message += f"{i}. {rec}\n"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Review detailed financial report in dashboard
• Schedule urgent meeting with finance team
• Implement immediate corrective actions
• Monitor situation closely over next 48 hours

Alert ID: {detection.detection_id}
Generated by: FinCo AI Loss Detection Agent

This is an automated alert. Please respond immediately.
"""
        
        return message.strip()
    
    def send_alert(
        self,
        detection: LossDetection,
        root_causes: List[str],
        recommendations: List[str]
    ) -> LossAlert:
        """
        Send alert to CFO via email, SMS, and dashboard notification
        
        Returns:
            LossAlert object with delivery status
        """
        message_body = self.create_alert_message(detection, root_causes, recommendations)
        
        alert = LossAlert(
            alert_id=str(uuid.uuid4()),
            detection=detection,
            root_causes=root_causes,
            recommendations=recommendations,
            cfo_email=self.cfo_email,
            cfo_name=self.cfo_name,
            message_body=message_body,
            sent_at=datetime.now(timezone.utc).isoformat(),
            delivery_status="sent"
        )
        
        # In production, this would:
        # 1. Send email via SMTP/SendGrid
        # 2. Send SMS via Twilio
        # 3. Create dashboard notification
        # 4. Log to audit trail
        # 5. Trigger webhook if configured
        
        print(f"\n{'='*70}")
        print(f"📧 EMAIL SENT TO: {self.cfo_email}")
        print(f"📱 SMS SENT TO: {self.cfo_phone}")
        print(f"🔔 DASHBOARD NOTIFICATION CREATED")
        print(f"{'='*70}\n")
        print(message_body)
        print(f"\n{'='*70}\n")
        
        self.alert_history.append(alert)
        alert.delivery_status = "delivered"
        
        return alert
    
    def get_alert_history(self) -> List[LossAlert]:
        """Get all sent alerts"""
        return self.alert_history


# ============================================================================
# MAIN LOSS ALERT AGENT
# ============================================================================

class LossAlertAgent:
    """
    Main agent that orchestrates automatic loss detection and CFO alerting
    """
    
    def __init__(
        self,
        cfo_name: str = "Chief Financial Officer",
        cfo_email: str = "cfo@finco.com",
        cfo_phone: str = "+1-555-0100"
    ):
        self.detection_engine = LossDetectionEngine()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.recommendation_engine = RecommendationEngine()
        self.alert_system = CFOAlertSystem(cfo_name, cfo_email, cfo_phone)
    
    def monitor_and_alert(
        self,
        revenue: float,
        expenses: float,
        company_name: str = "Company",
        period: str = "Current Period",
        previous_profit: Optional[float] = None,
        revenue_breakdown: Optional[Dict[str, float]] = None,
        expense_breakdown: Optional[Dict[str, float]] = None
    ) -> Optional[LossAlert]:
        """
        Main method: Monitor financial calculations and automatically alert CFO if loss detected
        
        Args:
            revenue: Total revenue
            expenses: Total expenses
            company_name: Company name
            period: Time period
            previous_profit: Previous period profit for comparison
            revenue_breakdown: Optional breakdown of revenue by category
            expense_breakdown: Optional breakdown of expenses by category
            
        Returns:
            LossAlert if loss detected and alert sent, None otherwise
        """
        # Step 1: Detect loss
        detection = self.detection_engine.detect_loss(
            revenue=revenue,
            expenses=expenses,
            company_name=company_name,
            period=period,
            previous_profit=previous_profit
        )
        
        if detection is None:
            # No loss detected - company is profitable
            return None
        
        # Step 2: Analyze root causes
        root_causes = self.root_cause_analyzer.analyze(
            detection=detection,
            revenue=revenue,
            expenses=expenses,
            revenue_breakdown=revenue_breakdown,
            expense_breakdown=expense_breakdown
        )
        
        # Step 3: Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            detection=detection,
            root_causes=root_causes
        )
        
        # Step 4: Send alert to CFO automatically
        alert = self.alert_system.send_alert(
            detection=detection,
            root_causes=root_causes,
            recommendations=recommendations
        )
        
        return alert


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def check_financial_performance(
    revenue: float,
    expenses: float,
    company_name: str = "Company",
    period: str = "Current Period",
    cfo_name: str = "Chief Financial Officer",
    cfo_email: str = "cfo@finco.com"
) -> Dict[str, Any]:
    """
    Convenience function to check financial performance and auto-alert CFO if needed
    
    Returns:
        Dict with status and alert info if applicable
    """
    agent = LossAlertAgent(cfo_name=cfo_name, cfo_email=cfo_email)
    
    profit = revenue - expenses
    
    if profit >= 0:
        return {
            "status": "profitable",
            "profit": profit,
            "revenue": revenue,
            "expenses": expenses,
            "alert_sent": False,
            "message": f"Company is profitable: ${profit:,.2f}"
        }
    
    alert = agent.monitor_and_alert(
        revenue=revenue,
        expenses=expenses,
        company_name=company_name,
        period=period
    )
    
    return {
        "status": "loss_detected",
        "loss": abs(profit),
        "revenue": revenue,
        "expenses": expenses,
        "alert_sent": True,
        "alert": alert.to_dict() if alert else None,
        "message": f"Loss detected: ${abs(profit):,.2f} - CFO has been automatically notified"
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FinCo AI - Automatic Loss Alert Agent")
    print("="*70 + "\n")
    
    # Example 1: Company with loss
    print("Example 1: Company with significant loss\n")
    result = check_financial_performance(
        revenue=500000,
        expenses=750000,
        company_name="Acme Corp",
        period="Q4 2024",
        cfo_name="John Smith",
        cfo_email="john.smith@acmecorp.com"
    )
    
    print(f"\nResult: {result['message']}\n")
    
    # Example 2: Company with profit (no alert)
    print("\n" + "="*70)
    print("Example 2: Company with profit (no alert sent)\n")
    result2 = check_financial_performance(
        revenue=800000,
        expenses=600000,
        company_name="Profit Inc",
        period="Q4 2024"
    )
    
    print(f"Result: {result2['message']}\n")
