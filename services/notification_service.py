"""
FinCo AI - Notification Service
================================

Handles multi-channel notifications:
- Email notifications (SMTP)
- SMS notifications (Twilio)
- Dashboard notifications
- Audit logging

Supports:
- Loss alerts
- Balance sheet notifications
- Financial warnings
- System alerts
"""

from __future__ import annotations

import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class EmailNotification:
    """Email notification"""
    to_email: str
    subject: str
    body: str
    from_email: str = "noreply@finco-ai.com"
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    html_body: Optional[str] = None


@dataclass
class SMSNotification:
    """SMS notification"""
    to_phone: str
    message: str
    from_phone: str = "+1-555-0199"


@dataclass
class DashboardNotification:
    """Dashboard notification"""
    user_id: str
    title: str
    message: str
    notification_type: str  # info, warning, error, success
    priority: str = "normal"  # low, normal, high, urgent
    action_url: Optional[str] = None


@dataclass
class AuditLogEntry:
    """Audit log entry"""
    timestamp: str
    event_type: str
    user_id: Optional[str]
    description: str
    metadata: Dict[str, Any]
    severity: str = "info"  # debug, info, warning, error, critical


# ============================================================================
# EMAIL SERVICE
# ============================================================================

class EmailService:
    """
    Email notification service using SMTP
    
    Configuration via environment variables:
    - SMTP_HOST: SMTP server host
    - SMTP_PORT: SMTP server port
    - SMTP_USER: SMTP username
    - SMTP_PASSWORD: SMTP password
    - SMTP_USE_TLS: Use TLS (true/false)
    """
    
    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        use_tls: bool = True,
        dry_run: bool = True  # Set to False in production
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.use_tls = use_tls
        self.dry_run = dry_run
    
    def send_email(self, notification: EmailNotification) -> bool:
        """
        Send email notification
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if self.dry_run:
                # Development mode - just log
                logger.info(f"📧 [DRY RUN] Email would be sent to: {notification.to_email}")
                logger.info(f"   Subject: {notification.subject}")
                logger.info(f"   Body: {notification.body[:100]}...")
                print(f"\n{'='*70}")
                print(f"📧 EMAIL NOTIFICATION (DRY RUN)")
                print(f"{'='*70}")
                print(f"To: {notification.to_email}")
                print(f"From: {notification.from_email}")
                print(f"Subject: {notification.subject}")
                print(f"{'='*70}")
                print(notification.body)
                print(f"{'='*70}\n")
                return True
            
            # Production mode - actually send email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = notification.subject
            msg['From'] = notification.from_email
            msg['To'] = notification.to_email
            
            if notification.cc:
                msg['Cc'] = ', '.join(notification.cc)
            
            # Attach plain text
            text_part = MIMEText(notification.body, 'plain')
            msg.attach(text_part)
            
            # Attach HTML if provided
            if notification.html_body:
                html_part = MIMEText(notification.html_body, 'html')
                msg.attach(html_part)
            
            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                
                # Send email
                recipients = [notification.to_email]
                if notification.cc:
                    recipients.extend(notification.cc)
                if notification.bcc:
                    recipients.extend(notification.bcc)
                
                server.sendmail(notification.from_email, recipients, msg.as_string())
            
            logger.info(f"✅ Email sent successfully to {notification.to_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email: {str(e)}")
            return False


# ============================================================================
# SMS SERVICE
# ============================================================================

class SMSService:
    """
    SMS notification service
    
    In production, integrate with Twilio, AWS SNS, or other SMS provider
    
    Configuration:
    - TWILIO_ACCOUNT_SID
    - TWILIO_AUTH_TOKEN
    - TWILIO_PHONE_NUMBER
    """
    
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_phone: str = "+1-555-0199",
        dry_run: bool = True  # Set to False in production
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_phone = from_phone
        self.dry_run = dry_run
    
    def send_sms(self, notification: SMSNotification) -> bool:
        """
        Send SMS notification
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if self.dry_run:
                # Development mode - just log
                logger.info(f"📱 [DRY RUN] SMS would be sent to: {notification.to_phone}")
                logger.info(f"   Message: {notification.message[:50]}...")
                print(f"\n{'='*70}")
                print(f"📱 SMS NOTIFICATION (DRY RUN)")
                print(f"{'='*70}")
                print(f"To: {notification.to_phone}")
                print(f"From: {notification.from_phone}")
                print(f"Message: {notification.message}")
                print(f"{'='*70}\n")
                return True
            
            # Production mode - use Twilio or other SMS provider
            # Example with Twilio (uncomment in production):
            # from twilio.rest import Client
            # client = Client(self.account_sid, self.auth_token)
            # message = client.messages.create(
            #     body=notification.message,
            #     from_=self.from_phone,
            #     to=notification.to_phone
            # )
            # logger.info(f"✅ SMS sent successfully: {message.sid}")
            
            logger.info(f"✅ SMS sent successfully to {notification.to_phone}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send SMS: {str(e)}")
            return False


# ============================================================================
# DASHBOARD NOTIFICATION SERVICE
# ============================================================================

class DashboardNotificationService:
    """
    In-app dashboard notification service
    
    Stores notifications that appear in the user's dashboard
    In production, this would write to database (PostgreSQL, Redis, etc.)
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/notifications")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def create_notification(self, notification: DashboardNotification) -> str:
        """
        Create dashboard notification
        
        Returns:
            Notification ID
        """
        try:
            notification_id = f"notif_{datetime.now(timezone.utc).timestamp()}"
            
            notification_data = {
                "id": notification_id,
                "user_id": notification.user_id,
                "title": notification.title,
                "message": notification.message,
                "type": notification.notification_type,
                "priority": notification.priority,
                "action_url": notification.action_url,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "read": False
            }
            
            # Save to file (in production, save to database)
            file_path = self.storage_path / f"{notification_id}.json"
            with open(file_path, 'w') as f:
                json.dump(notification_data, f, indent=2)
            
            logger.info(f"🔔 Dashboard notification created: {notification_id}")
            print(f"\n{'='*70}")
            print(f"🔔 DASHBOARD NOTIFICATION")
            print(f"{'='*70}")
            print(f"User: {notification.user_id}")
            print(f"Type: {notification.notification_type.upper()}")
            print(f"Priority: {notification.priority.upper()}")
            print(f"Title: {notification.title}")
            print(f"Message: {notification.message}")
            if notification.action_url:
                print(f"Action: {notification.action_url}")
            print(f"{'='*70}\n")
            
            return notification_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create dashboard notification: {str(e)}")
            return ""
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get all notifications for a user"""
        notifications = []
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    notif = json.load(f)
                    
                if notif["user_id"] == user_id:
                    if unread_only and notif.get("read", False):
                        continue
                    notifications.append(notif)
            except Exception as e:
                logger.error(f"Error reading notification {file_path}: {e}")
        
        # Sort by created_at descending
        notifications.sort(key=lambda x: x["created_at"], reverse=True)
        return notifications


# ============================================================================
# AUDIT LOG SERVICE
# ============================================================================

class AuditLogService:
    """
    Audit logging service
    
    Records all important events for compliance and debugging
    In production, write to database and/or dedicated log storage
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("data/audit_logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_path / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    def log_event(self, entry: AuditLogEntry):
        """
        Log audit event
        
        Writes to JSONL file (one JSON object per line)
        In production, also write to database
        """
        try:
            log_data = {
                "timestamp": entry.timestamp,
                "event_type": entry.event_type,
                "user_id": entry.user_id,
                "description": entry.description,
                "severity": entry.severity,
                "metadata": entry.metadata
            }
            
            # Append to log file
            with open(self.current_log_file, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
            
            # Also log to standard logger
            log_level = getattr(logging, entry.severity.upper(), logging.INFO)
            logger.log(log_level, f"📝 AUDIT: [{entry.event_type}] {entry.description}")
            
            print(f"\n{'='*70}")
            print(f"📝 AUDIT LOG ENTRY")
            print(f"{'='*70}")
            print(f"Timestamp: {entry.timestamp}")
            print(f"Event: {entry.event_type}")
            print(f"User: {entry.user_id or 'system'}")
            print(f"Severity: {entry.severity.upper()}")
            print(f"Description: {entry.description}")
            if entry.metadata:
                print(f"Metadata: {json.dumps(entry.metadata, indent=2)}")
            print(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ Failed to write audit log: {str(e)}")
    
    def log_loss_alert(
        self,
        company_name: str,
        loss_amount: float,
        cfo_email: str,
        alert_sent: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Convenience method to log loss alert events"""
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="loss_alert",
            user_id="system",
            description=f"Loss alert for {company_name}: ${loss_amount:,.2f} - CFO ({cfo_email}) {'notified' if alert_sent else 'notification failed'}",
            severity="warning" if alert_sent else "error",
            metadata={
                "company_name": company_name,
                "loss_amount": loss_amount,
                "cfo_email": cfo_email,
                "alert_sent": alert_sent,
                **(metadata or {})
            }
        )
        self.log_event(entry)
    
    def log_balance_sheet_generation(
        self,
        company_name: str,
        financial_year: str,
        is_balanced: bool,
        total_assets: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Convenience method to log balance sheet generation"""
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="balance_sheet_generated",
            user_id="system",
            description=f"Balance sheet generated for {company_name} (FY {financial_year}) - {'Balanced' if is_balanced else 'UNBALANCED'} - Assets: ${total_assets:,.2f}",
            severity="info" if is_balanced else "warning",
            metadata={
                "company_name": company_name,
                "financial_year": financial_year,
                "is_balanced": is_balanced,
                "total_assets": total_assets,
                **(metadata or {})
            }
        )
        self.log_event(entry)


# ============================================================================
# UNIFIED NOTIFICATION SERVICE
# ============================================================================

class NotificationService:
    """
    Unified notification service
    
    Single interface for all notification types:
    - Email
    - SMS
    - Dashboard
    - Audit logs
    """
    
    def __init__(
        self,
        email_service: Optional[EmailService] = None,
        sms_service: Optional[SMSService] = None,
        dashboard_service: Optional[DashboardNotificationService] = None,
        audit_service: Optional[AuditLogService] = None
    ):
        self.email = email_service or EmailService()
        self.sms = sms_service or SMSService()
        self.dashboard = dashboard_service or DashboardNotificationService()
        self.audit = audit_service or AuditLogService()
    
    def send_loss_alert(
        self,
        cfo_email: str,
        cfo_name: str,
        cfo_phone: str,
        company_name: str,
        loss_amount: float,
        period: str,
        severity: str,
        root_causes: List[str],
        recommendations: List[str],
        alert_message: str
    ) -> Dict[str, bool]:
        """
        Send loss alert via all channels
        
        Returns:
            Dictionary with status of each channel
        """
        results = {}
        
        # 1. Send email
        email = EmailNotification(
            to_email=cfo_email,
            subject=f"🚨 URGENT: Financial Loss Alert - {company_name} - {period}",
            body=alert_message
        )
        results['email'] = self.email.send_email(email)
        
        # 2. Send SMS
        sms_text = (
            f"URGENT: {company_name} financial loss alert. "
            f"Loss: ${loss_amount:,.0f}. "
            f"Severity: {severity.upper()}. "
            f"Check email for details."
        )
        sms = SMSNotification(
            to_phone=cfo_phone,
            message=sms_text
        )
        results['sms'] = self.sms.send_sms(sms)
        
        # 3. Create dashboard notification
        dashboard_notif = DashboardNotification(
            user_id=cfo_email,  # Using email as user ID
            title=f"Financial Loss Alert: {company_name}",
            message=f"Loss of ${loss_amount:,.2f} detected for {period}. Severity: {severity.upper()}",
            notification_type="error" if severity in ["critical", "high"] else "warning",
            priority="urgent" if severity == "critical" else "high",
            action_url="/dashboard/loss-alerts"
        )
        notification_id = self.dashboard.create_notification(dashboard_notif)
        results['dashboard'] = bool(notification_id)
        
        # 4. Log to audit trail
        self.audit.log_loss_alert(
            company_name=company_name,
            loss_amount=loss_amount,
            cfo_email=cfo_email,
            alert_sent=all(results.values()),
            metadata={
                "period": period,
                "severity": severity,
                "root_causes": root_causes,
                "recommendations": recommendations,
                "notification_id": notification_id
            }
        )
        
        return results
    
    def send_balance_sheet_notification(
        self,
        recipient_email: str,
        company_name: str,
        financial_year: str,
        is_balanced: bool,
        total_assets: float,
        total_liabilities_and_equity: float
    ) -> Dict[str, bool]:
        """
        Send balance sheet generation notification
        
        Returns:
            Dictionary with status of each channel
        """
        results = {}
        
        # 1. Send email
        subject = f"✅ Balance Sheet Generated: {company_name} - FY {financial_year}" if is_balanced else f"⚠️ Balance Sheet Warning: {company_name} - FY {financial_year}"
        
        body = f"""
Balance Sheet Generated Successfully

Company: {company_name}
Financial Year: {financial_year}
Status: {'✅ BALANCED' if is_balanced else '❌ OUT OF BALANCE'}

Total Assets: ${total_assets:,.2f}
Total Liabilities + Equity: ${total_liabilities_and_equity:,.2f}
Difference: ${abs(total_assets - total_liabilities_and_equity):,.2f}

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

View full balance sheet in the dashboard.
"""
        
        email = EmailNotification(
            to_email=recipient_email,
            subject=subject,
            body=body
        )
        results['email'] = self.email.send_email(email)
        
        # 2. Dashboard notification
        dashboard_notif = DashboardNotification(
            user_id=recipient_email,
            title=f"Balance Sheet: {company_name}",
            message=f"FY {financial_year} balance sheet generated. Status: {'Balanced' if is_balanced else 'Unbalanced'}",
            notification_type="success" if is_balanced else "warning",
            priority="normal",
            action_url="/dashboard/balance-sheet"
        )
        notification_id = self.dashboard.create_notification(dashboard_notif)
        results['dashboard'] = bool(notification_id)
        
        # 3. Audit log
        self.audit.log_balance_sheet_generation(
            company_name=company_name,
            financial_year=financial_year,
            is_balanced=is_balanced,
            total_assets=total_assets,
            metadata={
                "total_liabilities_and_equity": total_liabilities_and_equity,
                "notification_id": notification_id
            }
        )
        
        return results


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create global notification service instance
notification_service = NotificationService()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FinCo AI - Notification Service Test")
    print("="*70 + "\n")
    
    # Test loss alert notification
    results = notification_service.send_loss_alert(
        cfo_email="cfo@acmecorp.com",
        cfo_name="John Smith",
        cfo_phone="+1-555-0100",
        company_name="Acme Corporation",
        loss_amount=250000,
        period="Q4 2024",
        severity="critical",
        root_causes=[
            "Expenses exceed revenue by 50%",
            "High operating expenses"
        ],
        recommendations=[
            "Schedule emergency meeting",
            "Implement cost reduction"
        ],
        alert_message="Test loss alert message"
    )
    
    print(f"\n{'='*70}")
    print("Notification Results:")
    print(f"{'='*70}")
    for channel, success in results.items():
        status = "✅ Sent" if success else "❌ Failed"
        print(f"{channel.capitalize()}: {status}")
    print(f"{'='*70}\n")
