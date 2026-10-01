# backend/app/core/audit.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.database.models.audit_log import AuditLog


class AuditService:
    """
    Centralized audit logging service for FinCo AI.

    Records important security, financial, AI-agent, and
    human-review actions.

    Typical events:
        LOGIN
        LOGOUT
        CREATE
        UPDATE
        DELETE
        DOCUMENT_UPLOAD
        DOCUMENT_PROCESSED
        TRANSACTION_CREATED
        ALERT_CREATED
        ALERT_REVIEWED
        AGENT_EXECUTION
        RECOMMENDATION_CREATED
        RECOMMENDATION_APPROVED
        RECOMMENDATION_REJECTED
        WHAT_IF_EXECUTED
        REPORT_GENERATED
    """

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # CORE AUDIT LOG
    # ------------------------------------------------------------------

    def log(
        self,
        *,
        action: str,
        company_id: Optional[int] = None,
        user_id: Optional[int] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[int] = None,
        description: Optional[str] = None,
        status: str = "success",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditLog:
        """
        Create an audit log entry.

        This method intentionally does not contain business logic.
        It only records what happened.
        """

        audit_log = AuditLog(
            company_id=company_id,
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            description=description,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
        )

        self.db.add(audit_log)
        self.db.flush()

        return audit_log

    # ------------------------------------------------------------------
    # AUTHENTICATION
    # ------------------------------------------------------------------

    def log_login(
        self,
        *,
        user_id: int,
        company_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        success: bool = True,
    ) -> AuditLog:
        """
        Record a login attempt.
        """

        return self.log(
            action="LOGIN",
            company_id=company_id,
            user_id=user_id,
            entity_type="user",
            entity_id=user_id,
            description=(
                "User login successful"
                if success
                else "User login failed"
            ),
            status="success" if success else "failed",
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
        )

    def log_logout(
        self,
        *,
        user_id: int,
        company_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> AuditLog:
        """
        Record user logout.
        """

        return self.log(
            action="LOGOUT",
            company_id=company_id,
            user_id=user_id,
            entity_type="user",
            entity_id=user_id,
            description="User logged out",
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # CRUD EVENTS
    # ------------------------------------------------------------------

    def log_create(
        self,
        *,
        entity_type: str,
        entity_id: int,
        user_id: Optional[int] = None,
        company_id: Optional[int] = None,
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditLog:
        return self.log(
            action="CREATE",
            company_id=company_id,
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
            description=description,
            metadata=metadata,
        )

    def log_update(
        self,
        *,
        entity_type: str,
        entity_id: int,
        user_id: Optional[int] = None,
        company_id: Optional[int] = None,
        description: Optional[str] = None,
        changes: Optional[dict[str, Any]] = None,
    ) -> AuditLog:
        return self.log(
            action="UPDATE",
            company_id=company_id,
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
            description=description,
            metadata={
                "changes": changes or {},
            },
        )

    def log_delete(
        self,
        *,
        entity_type: str,
        entity_id: int,
        user_id: Optional[int] = None,
        company_id: Optional[int] = None,
        description: Optional[str] = None,
    ) -> AuditLog:
        return self.log(
            action="DELETE",
            company_id=company_id,
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
            description=description,
        )

    # ------------------------------------------------------------------
    # DOCUMENT EVENTS
    # ------------------------------------------------------------------

    def log_document_upload(
        self,
        *,
        document_id: int,
        company_id: int,
        user_id: Optional[int] = None,
        filename: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditLog:
        return self.log(
            action="DOCUMENT_UPLOAD",
            company_id=company_id,
            user_id=user_id,
            entity_type="document",
            entity_id=document_id,
            description="Financial document uploaded",
            request_id=request_id,
            metadata={
                "filename": filename,
            },
        )

    def log_document_processed(
        self,
        *,
        document_id: int,
        company_id: int,
        user_id: Optional[int] = None,
        chunks: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> AuditLog:
        return self.log(
            action="DOCUMENT_PROCESSED",
            company_id=company_id,
            user_id=user_id,
            entity_type="document",
            entity_id=document_id,
            description="Financial document processed",
            request_id=request_id,
            metadata={
                "chunks": chunks,
            },
        )

    # ------------------------------------------------------------------
    # TRANSACTION EVENTS
    # ------------------------------------------------------------------

    def log_transaction_created(
        self,
        *,
        transaction_id: int,
        company_id: int,
        user_id: Optional[int] = None,
        amount: Optional[float] = None,
        transaction_type: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditLog:
        return self.log(
            action="TRANSACTION_CREATED",
            company_id=company_id,
            user_id=user_id,
            entity_type="transaction",
            entity_id=transaction_id,
            description="Financial transaction created",
            request_id=request_id,
            metadata={
                "amount": amount,
                "transaction_type": transaction_type,
            },
        )

    # ------------------------------------------------------------------
    # ALERT EVENTS
    # ------------------------------------------------------------------

    def log_alert_created(
        self,
        *,
        alert_id: int,
        company_id: int,
        alert_type: str,
        severity: str,
        risk_score: Optional[float] = None,
        user_id: Optional[int] = None,
    ) -> AuditLog:
        return self.log(
            action="ALERT_CREATED",
            company_id=company_id,
            user_id=user_id,
            entity_type="alert",
            entity_id=alert_id,
            description=f"{severity} financial alert created",
            metadata={
                "alert_type": alert_type,
                "severity": severity,
                "risk_score": risk_score,
            },
        )

    def log_alert_reviewed(
        self,
        *,
        alert_id: int,
        company_id: int,
        user_id: int,
        decision: str,
        comment: Optional[str] = None,
    ) -> AuditLog:
        return self.log(
            action="ALERT_REVIEWED",
            company_id=company_id,
            user_id=user_id,
            entity_type="alert",
            entity_id=alert_id,
            description="Financial alert reviewed by human",
            metadata={
                "decision": decision,
                "comment": comment,
            },
        )

    # ------------------------------------------------------------------
    # AGENT EVENTS
    # ------------------------------------------------------------------

    def log_agent_execution(
        self,
        *,
        company_id: int,
        user_id: Optional[int],
        agent_name: str,
        task: str,
        status: str = "success",
        tools_used: Optional[list[str]] = None,
        execution_time_ms: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> AuditLog:
        """
        Record an Agentic AI execution.

        Important for explaining:
            Which agent acted?
            What task was performed?
            Which tools were used?
            Did execution succeed?
        """

        return self.log(
            action="AGENT_EXECUTION",
            company_id=company_id,
            user_id=user_id,
            entity_type="agent",
            description=f"{agent_name} executed a task",
            status=status,
            request_id=request_id,
            metadata={
                "agent_name": agent_name,
                "task": task,
                "tools_used": tools_used or [],
                "execution_time_ms": execution_time_ms,
            },
        )

    # ------------------------------------------------------------------
    # RECOMMENDATION EVENTS
    # ------------------------------------------------------------------

    def log_recommendation_created(
        self,
        *,
        recommendation_id: int,
        company_id: int,
        recommendation_type: str,
        user_id: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> AuditLog:
        return self.log(
            action="RECOMMENDATION_CREATED",
            company_id=company_id,
            user_id=user_id,
            entity_type="recommendation",
            entity_id=recommendation_id,
            description="AI recommendation generated",
            metadata={
                "recommendation_type": recommendation_type,
                "confidence": confidence,
            },
        )

    def log_recommendation_decision(
        self,
        *,
        recommendation_id: int,
        company_id: int,
        user_id: int,
        decision: str,
        comment: Optional[str] = None,
    ) -> AuditLog:
        """
        Record human approval/rejection of an AI recommendation.
        """

        action = (
            "RECOMMENDATION_APPROVED"
            if decision.lower() == "approved"
            else "RECOMMENDATION_REJECTED"
        )

        return self.log(
            action=action,
            company_id=company_id,
            user_id=user_id,
            entity_type="recommendation",
            entity_id=recommendation_id,
            description="Human reviewed AI recommendation",
            metadata={
                "decision": decision,
                "comment": comment,
            },
        )

    # ------------------------------------------------------------------
    # WHAT-IF EVENTS
    # ------------------------------------------------------------------

    def log_what_if_execution(
        self,
        *,
        company_id: int,
        user_id: Optional[int],
        scenario_name: str,
        inputs: dict[str, Any],
        results: dict[str, Any],
        request_id: Optional[str] = None,
    ) -> AuditLog:
        return self.log(
            action="WHAT_IF_EXECUTED",
            company_id=company_id,
            user_id=user_id,
            entity_type="scenario",
            description="What-if financial scenario executed",
            request_id=request_id,
            metadata={
                "scenario_name": scenario_name,
                "inputs": inputs,
                "results": results,
            },
        )

    # ------------------------------------------------------------------
    # REPORT EVENTS
    # ------------------------------------------------------------------

    def log_report_generated(
        self,
        *,
        company_id: int,
        user_id: Optional[int],
        report_type: str,
        report_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> AuditLog:
        return self.log(
            action="REPORT_GENERATED",
            company_id=company_id,
            user_id=user_id,
            entity_type="report",
            entity_id=report_id,
            description=f"{report_type} report generated",
            request_id=request_id,
            metadata={
                "report_type": report_type,
            },
        )

    # ------------------------------------------------------------------
    # SECURITY EVENTS
    # ------------------------------------------------------------------

    def log_security_event(
        self,
        *,
        action: str,
        description: str,
        company_id: Optional[int] = None,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditLog:
        """
        Record security-sensitive activity.

        Examples:
            PERMISSION_DENIED
            RATE_LIMIT_EXCEEDED
            INVALID_TOKEN
            SUSPICIOUS_ACTIVITY
        """

        return self.log(
            action=action,
            company_id=company_id,
            user_id=user_id,
            entity_type="security",
            description=description,
            status="security_event",
            ip_address=ip_address,
            request_id=request_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # QUERY AUDIT HISTORY
    # ------------------------------------------------------------------

    def get_by_company(
        self,
        company_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        """
        Retrieve audit history for a company.
        """

        return (
            self.db.query(AuditLog)
            .filter(AuditLog.company_id == company_id)
            .order_by(AuditLog.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_by_user(
        self,
        user_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        """
        Retrieve actions performed by a specific user.
        """

        return (
            self.db.query(AuditLog)
            .filter(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_by_entity(
        self,
        *,
        entity_type: str,
        entity_id: int,
        company_id: Optional[int] = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        """
        Retrieve the complete audit trail for an entity.
        """

        query = self.db.query(AuditLog).filter(
            AuditLog.entity_type == entity_type,
            AuditLog.entity_id == entity_id,
        )

        if company_id is not None:
            query = query.filter(
                AuditLog.company_id == company_id
            )

        return (
            query
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
            .all()
        )