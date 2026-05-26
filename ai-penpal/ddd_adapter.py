"""
ddd_adapter.py
--------------
195B Transport Layer — DDD gRPC Integration.
REWRITTEN based on real DDD repo (Codex verified).

HOW DDD WORKS (corrected):
─────────────────────────────────────────
1. YOUR server runs a gRPC service (ServiceAdapterService)
2. DDD Bundle Server connects TO your server and calls exchangeADUs()
3. You also call checkAdapterRegistration() ON DDD to register your URL

TWO gRPC roles:
  YOU are a gRPC SERVER  → DDD calls exchangeADUs() and pendingDataCheck() on you
  YOU are a gRPC CLIENT  → You call checkAdapterRegistration() on DDD to register

MESSAGE FORMAT (from ServiceAdapterService.proto):
  ExchangeADUsRequest {
    clientId          ← DDD client identifier
    adus[]            ← incoming messages
      aduId           ← sequential message ID
      data            ← raw bytes (WE define the format inside here)
    lastADUIdReceived ← ack from DDD of what it received from us
  }

  ExchangeADUsResponse {
    adus[]            ← our replies back to client
      aduId           ← sequential ID
      data            ← raw bytes of our reply
    lastADUIdReceived ← ack of what we received
  }

DATA FORMAT inside adus[].data — we define as JSON:
  { "email": "user@example.com", "subject": "...", "body": "..." }

IMPORTANT: App ID must be pre-registered by Prof. Carlos in DDD
           database before checkAdapterRegistration() returns code=0.

Based on: EchoDDDAdapter.java, BundleServerAduDeliverer.java,
          ServiceAdapterService.proto, AdapterRegisterService.java
"""

import json
import grpc
import logging
import threading
from concurrent import futures

from processor import process_message
from config import (
    DDD_GRPC_HOST,
    DDD_GRPC_PORT,
    DDD_APP_NAME,
    DDD_OUR_GRPC_URL,
    DDD_OUR_GRPC_PORT,
)

# Generated from ServiceAdapterService.proto — run this first:
# python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ServiceAdapterService.proto
try:
    import ServiceAdapterService_pb2 as sa_pb2
    import ServiceAdapterService_pb2_grpc as sa_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("[DDD] WARNING: gRPC stubs not generated yet.")
    print("[DDD] Run: python -m grpc_tools.protoc -I../proto "
          "--python_out=. --grpc_python_out=. ServiceAdapterService.proto")

logger = logging.getLogger(__name__)


# ── STEP 1: YOUR gRPC SERVER (DDD calls you) ──────────────────────────────────

class AIPenpalAdapter(sa_pb2_grpc.ServiceAdapterServiceServicer):
    """
    YOUR gRPC service that DDD connects to.
    Mirrors EchoDDDAdapter.java but routes through Ollama instead of echoing.

    DDD calls:
      exchangeADUs()     → deliver messages to us, collect our replies
      pendingDataCheck() → ask if we have anything queued to send
    """

    def __init__(self):
        # Track last ADU id per client (mirrors Echo adapter pattern)
        self._client_last_adu_id: dict = {}
        self._lock = threading.Lock()

    def exchangeADUs(self, request, context):
        """
        Core integration point — DDD delivers messages here and
        collects our replies in the same call.
        """
        client_id = request.clientId
        reply_adus = []

        logger.info(
            f"[DDD] exchangeADUs from clientId={client_id} "
            f"with {len(request.adus)} ADU(s)"
        )

        for adu in request.adus:
            adu_id = adu.aduId
            raw_data = adu.data

            logger.info(f"[DDD] Processing ADU id={adu_id} from {client_id}")

            # Parse our app-level payload from raw bytes
            message = _parse_adu_data(raw_data)
            if message is None:
                logger.warning(f"[DDD] Could not parse ADU {adu_id}, skipping")
                continue

            email_addr = message.get("email", client_id)
            subject    = message.get("subject", "(no subject)")
            body       = message.get("body", "")

            if not body:
                logger.warning(f"[DDD] Empty body in ADU {adu_id}, skipping")
                continue

            # Run through core pipeline — same as 195A smtp_server.py
            result = process_message(email_addr, subject, body)

            # Package reply as ADU bytes
            reply_data = _build_adu_data(
                email=email_addr,
                subject=result["reply_subject"],
                body=result["reply_body"]
            )

            reply_adus.append(
                sa_pb2.AppDataUnit(
                    aduId=adu_id,
                    data=reply_data
                )
            )

            logger.info(
                f"[DDD] Reply ADU prepared for {email_addr} "
                f"(success={result['success']})"
            )

        # Update last received ADU id tracking
        if request.adus:
            last_id = request.adus[-1].aduId
            with self._lock:
                current = self._client_last_adu_id.get(client_id, 0)
                self._client_last_adu_id[client_id] = max(current, last_id)

        with self._lock:
            last_received = self._client_last_adu_id.get(client_id, 0)

        return sa_pb2.ExchangeADUsResponse(
            adus=reply_adus,
            lastADUIdReceived=last_received
        )

    def pendingDataCheck(self, request, context):
        """
        DDD asks: do you have pending data for any clients?
        Returning empty for now — we reply reactively when messages arrive.
        """
        logger.debug("[DDD] pendingDataCheck called")
        return sa_pb2.PendingDataCheckResponse(clientId=[])


# ── STEP 2: REGISTER WITH DDD (you call DDD) ─────────────────────────────────

def register_with_ddd() -> bool:
    """
    Call checkAdapterRegistration() on DDD Bundle Server.

    IMPORTANT: Prof. Carlos must pre-register your appName + url
    in the DDD database BEFORE this returns code=0.
    Based on AdapterRegisterService.java.
    """
    if not GRPC_AVAILABLE:
        logger.error("[DDD] Cannot register — generate gRPC stubs first")
        return False

    try:
        channel = grpc.insecure_channel(f"{DDD_GRPC_HOST}:{DDD_GRPC_PORT}")
        stub = sa_pb2_grpc.ServiceAdapterRegistryServiceStub(channel)

        response = stub.checkAdapterRegistration(
            sa_pb2.ConnectionData(
                appName=DDD_APP_NAME,    # Must match what Prof. Carlos registered
                url=DDD_OUR_GRPC_URL     # Our gRPC server address DDD will connect to
            )
        )

        channel.close()

        if response.code == 0:
            logger.info(f"[DDD] Registered! {response.message}")
            return True
        else:
            logger.error(f"[DDD] Registration failed: code={response.code} {response.message}")
            logger.error(f"[DDD] Ask Prof. Carlos to register appName='{DDD_APP_NAME}' url='{DDD_OUR_GRPC_URL}'")
            return False

    except grpc.RpcError as e:
        logger.error(f"[DDD] gRPC error: {e.code()} — {e.details()}")
        return False
    except Exception as e:
        logger.error(f"[DDD] Registration error: {e}")
        return False


# ── STEP 3: START YOUR gRPC SERVER ───────────────────────────────────────────

def run_ddd_adapter():
    """
    Start your gRPC server, then register with DDD Bundle Server.
    DDD will connect to your server and call exchangeADUs() when messages arrive.
    """
    if not GRPC_AVAILABLE:
        logger.error("[DDD] Cannot start — generate stubs first:")
        logger.error("  python -m grpc_tools.protoc -I../proto "
                     "--python_out=. --grpc_python_out=. ServiceAdapterService.proto")
        return

    # Start gRPC server — DDD connects to this
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sa_pb2_grpc.add_ServiceAdapterServiceServicer_to_server(AIPenpalAdapter(), server)
    server.add_insecure_port(f"[::]:{DDD_OUR_GRPC_PORT}")
    server.start()
    logger.info(f"[DDD] gRPC adapter listening on port {DDD_OUR_GRPC_PORT}")

    # Register our URL with DDD
    registered = register_with_ddd()
    if not registered:
        logger.warning("[DDD] Not registered with DDD yet — coordinate with Prof. Carlos")

    logger.info("[DDD] Waiting for DDD connections...")
    server.wait_for_termination()


# ── HELPERS: ADU DATA ENCODING ────────────────────────────────────────────────

def _parse_adu_data(raw: bytes) -> dict:
    """Parse raw ADU bytes into a message dict. We use JSON encoding."""
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        logger.warning(f"[DDD] JSON parse failed: {e}, trying plain text fallback")
        try:
            return {"email": "unknown", "subject": "Message", "body": raw.decode("utf-8")}
        except Exception:
            return None


def _build_adu_data(email: str, subject: str, body: str) -> bytes:
    """Encode reply as raw ADU bytes (JSON format)."""
    return json.dumps({"email": email, "subject": subject, "body": body}).encode("utf-8")