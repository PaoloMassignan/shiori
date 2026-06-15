"""
Quality eval cases for Shiori compression — Claude Code patterns.

Each QualityCase separates the large structured context (compressed by Shiori)
from the short question (never compressed). The runner builds:
  original:   user = context + question
  compressed: user = compress(context) + question

Cases (QUALITY_CASES — direct compressor, no API routing):
  json_config    — k8s deployment JSON → replica count + memory limit
  stack_trace    — pytest failure traceback → file + line number
  python_class   — Python class with docstrings (AST strips them) → method return value
  log_analysis   — nginx access log → request count for a specific IP
  multi_turn     — two DB query results → subscription tier from first result

Cases (COMPARISON_CASES — routed via safe vs aggressive mode):
  meeting_minutes — 22-round meeting transcript → Q1 revenue figure
                    safe=lossless, aggressive=llmlingua (summarize instruction triggers it)
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityCase:
    name: str
    context: str              # large structured content — this is what Shiori compresses
    question: str             # short question appended after context — never compressed
    expected_facts: list[str]
    compressor_name: str
    description: str


# ---------------------------------------------------------------------------
# Case 1: JSON config (k8s deployment)
# ---------------------------------------------------------------------------

_K8S_CONFIG = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "name": "api-gateway",
        "namespace": "production",
        "labels": {"app": "api-gateway", "version": "2.4.1"},
        "annotations": {"kubectl.kubernetes.io/last-applied-configuration": None},
    },
    "spec": {
        "replicas": 3,
        "selector": {"matchLabels": {"app": "api-gateway"}},
        "template": {
            "metadata": {"labels": {"app": "api-gateway"}},
            "spec": {
                "containers": [
                    {
                        "name": "api-gateway",
                        "image": "gcr.io/myproject/api-gateway:2.4.1",
                        "ports": [
                            {"containerPort": 8443, "protocol": "TCP"},
                            {"containerPort": 8080, "protocol": "TCP"},
                        ],
                        "resources": {
                            "requests": {"memory": "256Mi", "cpu": "100m"},
                            "limits": {"memory": "512Mi", "cpu": "500m"},
                        },
                        "env": [
                            {"name": "LOG_LEVEL", "value": "info"},
                            {"name": "MAX_CONNECTIONS", "value": "1000"},
                            {"name": "TIMEOUT_MS", "value": "30000"},
                            {"name": "DB_POOL_SIZE", "value": "20"},
                            {"name": "CACHE_TTL_SECONDS", "value": "300"},
                            {"name": "ENABLE_TRACING", "value": "true"},
                            {"name": "RATE_LIMIT_RPM", "value": "10000"},
                            {"name": "RETRY_MAX_ATTEMPTS", "value": "3"},
                        ],
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8080},
                            "initialDelaySeconds": 15,
                            "periodSeconds": 20,
                            "timeoutSeconds": 5,
                            "failureThreshold": 3,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/ready", "port": 8080},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 10,
                            "timeoutSeconds": 3,
                            "failureThreshold": 3,
                        },
                        "volumeMounts": [
                            {"name": "config-volume", "mountPath": "/etc/config", "readOnly": True},
                            {"name": "tls-certs", "mountPath": "/etc/ssl/certs", "readOnly": True},
                        ],
                    }
                ],
                "volumes": [
                    {"name": "config-volume", "configMap": {"name": "api-gateway-config"}},
                    {"name": "tls-certs", "secret": {"secretName": "api-gateway-tls"}},
                ],
                "affinity": {
                    "podAntiAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": [
                            {
                                "labelSelector": {
                                    "matchExpressions": [
                                        {"key": "app", "operator": "In", "values": ["api-gateway"]}
                                    ]
                                },
                                "topologyKey": "kubernetes.io/hostname",
                            }
                        ]
                    }
                },
                "terminationGracePeriodSeconds": 30,
                "serviceAccountName": "api-gateway-sa",
            },
        },
        "strategy": {
            "type": "RollingUpdate",
            "rollingUpdate": {"maxSurge": 1, "maxUnavailable": 0},
        },
    },
}

# ---------------------------------------------------------------------------
# Case 2: Stack trace
# ---------------------------------------------------------------------------

_STACK_TRACE_CONTEXT = """\
============================= test session starts ==============================
platform linux -- Python 3.11.4, pytest-7.4.0
collecting ... collected 47 items

tests/unit/test_auth.py ..........                                      [ 21%]
tests/unit/test_models.py .......                                       [ 36%]
tests/integration/test_api.py ..                                        [ 40%]
tests/integration/test_user_service.py FAILED                          [ 42%]

=================================== FAILURES ===================================
____________ test_get_user_profile_with_inactive_account ____________

    def test_get_user_profile_with_inactive_account():
        user = create_test_user(active=False)
>       result = UserService.get_profile(user.id)

tests/integration/test_user_service.py:31:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

src/services/user_service.py:47: in get_profile
    return {"email": account.email, "tier": account.subscription.tier}
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

src/services/user_service.py:47: AttributeError: 'NoneType' object has no attribute 'email'

=========================== short test summary info ============================
FAILED tests/integration/test_user_service.py::test_get_user_profile_with_inactive_account
======================== 1 failed, 46 passed in 3.42s ========================="""

# ---------------------------------------------------------------------------
# Case 3: Python class with docstrings (AST will strip them)
# ---------------------------------------------------------------------------

_PYTHON_CLASS_CONTEXT = '''\
"""Payment processing module for subscription billing."""
from decimal import Decimal
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_DISCOUNT_RATES = {
    "SAVE10": Decimal("0.10"),
    "SAVE20": Decimal("0.20"),
    "PARTNER50": Decimal("0.50"),
    "NEWUSER": Decimal("0.15"),
}


class PaymentProcessor:
    """Handles payment processing and discount calculation for subscriptions.

    All monetary values are handled as Decimal to avoid floating point
    precision issues.

    Attributes:
        _gateway: The payment gateway client instance.
        _currency: The ISO 4217 currency code (default: USD).
    """

    def __init__(self, gateway, currency: str = "USD"):
        """Initialize the payment processor.

        Args:
            gateway: Payment gateway client (Stripe, Braintree, etc.)
            currency: ISO 4217 currency code for all transactions.
        """
        self._gateway = gateway
        self._currency = currency

    def calculate_discount(self, order_total: Decimal, coupon_code: Optional[str]) -> Decimal:
        """Calculate the discount amount for a given order total and coupon.

        Looks up the coupon code in the discount rates table. Unknown or None
        coupon codes result in a zero discount. The discount is calculated as
        a percentage of the order total.

        Args:
            order_total: The pre-discount order total as a Decimal.
            coupon_code: The coupon code string, or None if no coupon applied.

        Returns:
            The discount amount as a Decimal. Returns Decimal("0") if the
            coupon code is invalid or not provided.

        Raises:
            ValueError: If order_total is negative.
        """
        if order_total < Decimal("0"):
            raise ValueError(f"order_total must be non-negative, got {order_total}")
        rate = _DISCOUNT_RATES.get(coupon_code, Decimal("0"))
        return order_total * rate

    def apply_charge(self, customer_id: str, amount: Decimal, description: str) -> dict:
        """Submit a charge to the payment gateway.

        Args:
            customer_id: The gateway customer identifier.
            amount: Amount to charge in the processor currency.
            description: Human-readable charge description for the invoice.

        Returns:
            Gateway response dict with keys: transaction_id, status, timestamp.
        """
        logger.info("Charging customer %s amount %s %s", customer_id, amount, self._currency)
        return self._gateway.charge(
            customer_id=customer_id,
            amount=int(amount * 100),
            currency=self._currency,
            description=description,
        )
'''

# ---------------------------------------------------------------------------
# Case 4: Nginx access log (7 requests from 10.0.0.42)
# ---------------------------------------------------------------------------

_LOG_LINES = [
    '10.0.0.1 - - [14/Jun/2026:10:01:22 +0000] "GET /api/v1/users HTTP/1.1" 200 1240 "-" "Mozilla/5.0"',
    '10.0.0.7 - - [14/Jun/2026:10:02:45 +0000] "POST /api/v1/products HTTP/1.1" 201 892 "-" "curl/7.88.1"',
    '10.0.0.15 - - [14/Jun/2026:10:03:11 +0000] "GET /api/v2/orders HTTP/1.1" 200 3421 "-" "axios/1.4.0"',
    '10.0.0.42 - - [14/Jun/2026:10:15:00 +0000] "GET /api/v2/health HTTP/1.1" 200 843 "-" "HealthChecker/1.0"',
    '192.168.1.100 - - [14/Jun/2026:10:05:33 +0000] "GET /static/app.js HTTP/1.1" 304 0 "-" "Chrome/114"',
    '172.16.0.8 - - [14/Jun/2026:10:07:18 +0000] "DELETE /api/v1/users/99 HTTP/1.1" 204 0 "-" "curl/7.88.1"',
    '10.0.0.42 - - [14/Jun/2026:10:16:00 +0000] "GET /api/v2/health HTTP/1.1" 200 843 "-" "HealthChecker/1.0"',
    '10.0.0.1 - - [14/Jun/2026:10:09:02 +0000] "GET /api/v1/products HTTP/1.1" 200 5632 "-" "Mozilla/5.0"',
    '10.0.0.7 - - [14/Jun/2026:10:11:44 +0000] "GET /api/v2/orders HTTP/1.1" 200 1120 "-" "axios/1.4.0"',
    '10.0.0.42 - - [14/Jun/2026:10:17:00 +0000] "GET /api/v2/health HTTP/1.1" 200 843 "-" "HealthChecker/1.0"',
    '192.168.1.100 - - [14/Jun/2026:10:13:09 +0000] "POST /api/v1/users HTTP/1.1" 201 644 "-" "Chrome/114"',
    '10.0.0.42 - - [14/Jun/2026:10:18:00 +0000] "GET /api/v2/health HTTP/1.1" 200 843 "-" "HealthChecker/1.0"',
    '172.16.0.8 - - [14/Jun/2026:10:16:55 +0000] "GET /api/v2/health HTTP/1.1" 200 843 "-" "HealthChecker/1.0"',
    '10.0.0.15 - - [14/Jun/2026:10:18:30 +0000] "PUT /api/v1/users/12 HTTP/1.1" 200 980 "-" "curl/7.88.1"',
    '10.0.0.42 - - [14/Jun/2026:10:19:00 +0000] "GET /api/v2/health HTTP/1.1" 200 843 "-" "HealthChecker/1.0"',
    '10.0.0.1 - - [14/Jun/2026:10:20:17 +0000] "GET /api/v1/products HTTP/1.1" 200 4211 "-" "Mozilla/5.0"',
    '10.0.0.42 - - [14/Jun/2026:10:20:00 +0000] "GET /api/v2/health HTTP/1.1" 200 843 "-" "HealthChecker/1.0"',
    '10.0.0.7 - - [14/Jun/2026:10:22:01 +0000] "GET /api/v1/users HTTP/1.1" 200 2100 "-" "axios/1.4.0"',
    '10.0.0.42 - - [14/Jun/2026:10:21:00 +0000] "GET /api/v2/health HTTP/1.1" 200 843 "-" "HealthChecker/1.0"',
    '192.168.1.100 - - [14/Jun/2026:10:23:44 +0000] "GET /api/v2/orders HTTP/1.1" 200 8821 "-" "Chrome/114"',
]

# ---------------------------------------------------------------------------
# Case 5: Two DB query results
# ---------------------------------------------------------------------------

_DB_RESULT = json.dumps({
    "query": "SELECT * FROM users WHERE user_id = 'usr_7f3a9b'",
    "rows": 1,
    "result": [
        {
            "user_id": "usr_7f3a9b",
            "email": "marco.ferrari@acme.com",
            "full_name": "Marco Ferrari",
            "subscription_tier": "enterprise",
            "seats": 150,
            "created_at": "2024-01-15T09:30:00Z",
            "last_login": "2026-06-14T08:45:22Z",
            "billing_cycle": "annual",
            "renewal_date": "2027-01-15",
            "region": "eu-west-1",
            "account_manager": "sales-team-b",
            "notes": None,
            "feature_flags": None,
            "custom_domain": None,
        }
    ],
    "execution_time_ms": 4,
}, indent=2)

_BILLING_RESULT = json.dumps({
    "invoice_id": "inv_8823xz",
    "amount_due": 18000.00,
    "currency": "USD",
    "status": "paid",
    "period_start": "2026-01-15",
    "period_end": "2027-01-15",
    "line_items": [
        {"description": "Enterprise plan (150 seats)", "amount": 18000.00},
        {"discount_applied": None, "promo_code": None, "tax": None},
    ],
    "payment_method": "wire_transfer",
    "paid_at": "2026-01-20T14:22:00Z",
    "next_invoice": None,
    "notes": None,
}, indent=2)

_MULTI_TURN_CONTEXT = json.dumps([
    {
        "query_label": "user_profile",
        "query": "SELECT * FROM users WHERE user_id = 'usr_7f3a9b'",
        "rows": 1,
        "result": [
            {
                "user_id": "usr_7f3a9b",
                "email": "marco.ferrari@acme.com",
                "full_name": "Marco Ferrari",
                "subscription_tier": "enterprise",
                "seats": 150,
                "created_at": "2024-01-15T09:30:00Z",
                "last_login": "2026-06-14T08:45:22Z",
                "billing_cycle": "annual",
                "renewal_date": "2027-01-15",
                "region": "eu-west-1",
                "account_manager": "sales-team-b",
                "notes": None,
                "feature_flags": None,
                "custom_domain": None,
            }
        ],
        "execution_time_ms": 4,
    },
    {
        "query_label": "latest_invoice",
        "query": "SELECT * FROM invoices WHERE user_id = 'usr_7f3a9b' ORDER BY created_at DESC LIMIT 1",
        "rows": 1,
        "result": [
            {
                "invoice_id": "inv_8823xz",
                "amount_due": 18000.00,
                "currency": "USD",
                "status": "paid",
                "period_start": "2026-01-15",
                "period_end": "2027-01-15",
                "line_items": [
                    {"description": "Enterprise plan (150 seats)", "amount": 18000.00},
                    {"discount_applied": None, "promo_code": None, "tax": None},
                ],
                "payment_method": "wire_transfer",
                "paid_at": "2026-01-20T14:22:00Z",
                "next_invoice": None,
                "notes": None,
            }
        ],
        "execution_time_ms": 6,
    },
], indent=2)

# ---------------------------------------------------------------------------
# Case 6: Meeting transcript (for safe vs aggressive comparison)
# 22 rounds × ~120 words ≈ 2000+ tokens — triggers is_long_prose
# Question contains "summarize" — triggers has_summarize_instruction
# → safe mode: lossless | aggressive mode: llmlingua
# ---------------------------------------------------------------------------

_MEETING_ROUND_TEMPLATE = (
    "Alice: Q{i} revenue came in at {rev} million, up {pct}% versus the same quarter last year. "
    "The enterprise segment led with strong bookings from financial services and healthcare. "
    "SMB continued to face headwinds from budget cycles but showed improvement late in the quarter.\n\n"
    "Bob: On the product side, we shipped {features} new features in Q{i}. "
    "The dashboard redesign saw 34% adoption after three weeks, ahead of our 30% target. "
    "The mobile application is on track for general availability next quarter.\n\n"
    "Carol: Engineering velocity improved after the team restructuring. "
    "We reduced mean time to deploy from 45 minutes to 12 minutes. "
    "Technical debt reduction accounted for 20% of engineering capacity this quarter.\n\n"
    "David: For next quarter I recommend three priorities: expanding the enterprise sales team "
    "by four headcount, launching the partner programme targeting system integrators, "
    "and improving onboarding to reduce SMB churn which currently stands at 8% monthly.\n"
)

_MEETING_TRANSCRIPT = "\n".join(
    _MEETING_ROUND_TEMPLATE.format(
        i=q + 1,
        rev=round(3.8 + q * 0.4, 1),
        pct=8 + q * 2,
        features=12 + q * 3,
    )
    for q in range(22)  # 22 rounds ≈ 2000 tokens — triggers is_long_prose threshold
)

# ---------------------------------------------------------------------------
# Assembled cases
# ---------------------------------------------------------------------------

QUALITY_CASES: list[QualityCase] = [
    QualityCase(
        name="json_config",
        context=json.dumps(_K8S_CONFIG, indent=2),
        question="How many replicas is the api-gateway deployment configured to run, and what is its memory limit?",
        expected_facts=["3", "512"],
        compressor_name="json",
        description="Extract replica count and memory limit from compressed k8s JSON",
    ),
    QualityCase(
        name="stack_trace",
        context=_STACK_TRACE_CONTEXT,
        question="In which source file and at which line number does the AttributeError occur?",
        expected_facts=["user_service.py", "47"],
        compressor_name="lossless",
        description="Identify error file and line from compressed stack trace",
    ),
    QualityCase(
        name="python_class",
        context=_PYTHON_CLASS_CONTEXT,
        question="What does calculate_discount return when an unknown coupon code is passed?",
        expected_facts=["0", "Decimal"],
        compressor_name="ast",
        description="Infer method return value from code after docstring removal by AST compressor",
    ),
    QualityCase(
        name="log_analysis",
        context="\n".join(_LOG_LINES),
        question="How many total requests did IP address 10.0.0.42 make?",
        expected_facts=["7"],
        compressor_name="lossless",
        description="Count requests from a specific IP in compressed nginx log",
    ),
    QualityCase(
        name="multi_turn",
        context=_MULTI_TURN_CONTEXT,
        question="What is the subscription tier of the user from the first query result?",
        expected_facts=["enterprise"],
        compressor_name="json",
        description="Recall subscription tier from first of two compressed DB query results",
    ),
]

COMPARISON_CASES: list[QualityCase] = [
    *QUALITY_CASES,
    QualityCase(
        name="meeting_minutes",
        context=_MEETING_TRANSCRIPT,
        question="Please summarize the key financial results. What was the exact Q1 revenue figure?",
        expected_facts=["3.8"],
        compressor_name="lossless",  # used only in normal mode; comparison uses router
        description="22-round meeting transcript: safe=lossless, aggressive=llmlingua (summarize trigger)",
    ),
]
