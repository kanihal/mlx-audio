import threading
import time

from mlx_audio.server_inference import (
    BaseModelExecutionAdapter,
    InferenceBroker,
    InferenceRequest,
)


def _collect(handle, timeout=2.0):
    deadline = time.time() + timeout
    chunks = []
    while time.time() < deadline:
        chunk = handle.result_queue.get(timeout=timeout)
        chunks.append(chunk)
        if chunk.kind == "done":
            return chunks
    raise TimeoutError("timed out waiting for broker results")


class SerializedAdapter(BaseModelExecutionAdapter):
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def run_serial(self, request: InferenceRequest) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        time.sleep(0.05)
        request.emit_data(request.payload)

        with self.lock:
            self.active -= 1

        request.emit_done()


class ContinuousSession:
    def __init__(self):
        self.requests = []
        self.submitted_payloads = []
        self.failed = None

    @property
    def idle(self):
        return not self.requests

    def submit(self, request: InferenceRequest) -> None:
        self.submitted_payloads.append(request.payload)
        self.requests.append(request)

    def step(self) -> None:
        if not self.requests:
            return
        request = self.requests.pop(0)
        request.emit_data(("continuous", request.payload))
        request.emit_done()

    def fail(self, error: BaseException) -> None:
        self.failed = error
        for request in self.requests:
            request.emit_error(error)
            request.emit_done()
        self.requests.clear()


class ContinuousAdapter(BaseModelExecutionAdapter):
    max_batch_size = 4

    def __init__(self):
        self.sessions = []
        self.serial_payloads = []

    def supports_continuous_batch(self, request: InferenceRequest) -> bool:
        del request
        return True

    def continuous_batch_key(self, request: InferenceRequest):
        del request
        return "continuous"

    def create_continuous_batch_session(
        self,
        request: InferenceRequest,
    ) -> ContinuousSession:
        del request
        session = ContinuousSession()
        self.sessions.append(session)
        return session

    def run_serial(self, request: InferenceRequest) -> None:
        self.serial_payloads.append(request.payload)
        request.emit_data(("serial", request.payload))
        request.emit_done()


class BlockingSerializedAdapter(BaseModelExecutionAdapter):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def run_serial(self, request: InferenceRequest) -> None:
        self.started.set()
        self.release.wait(timeout=2.0)
        request.emit_data(request.payload)
        request.emit_done()


class BlockingContinuousSession:
    def __init__(self):
        self.requests = []
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def idle(self):
        return not self.requests

    def submit(self, request: InferenceRequest) -> None:
        self.requests.append(request)
        self.started.set()

    def step(self) -> None:
        if not self.requests or not self.release.is_set():
            return
        request = self.requests.pop(0)
        request.emit_data(("continuous", request.payload))
        request.emit_done()

    def fail(self, error: BaseException) -> None:
        for request in self.requests:
            request.emit_error(error)
            request.emit_done()
        self.requests.clear()


class BlockingContinuousAdapter(BaseModelExecutionAdapter):
    def __init__(self):
        self.session = BlockingContinuousSession()

    def supports_continuous_batch(self, request: InferenceRequest) -> bool:
        del request
        return True

    def continuous_batch_key(self, request: InferenceRequest):
        del request
        return "continuous"

    def create_continuous_batch_session(
        self,
        request: InferenceRequest,
    ) -> BlockingContinuousSession:
        del request
        return self.session

    def run_serial(self, request: InferenceRequest) -> None:
        request.emit_data(("serial", request.payload))
        request.emit_done()


def _wait_for(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_inference_broker_serializes_requests():
    broker = InferenceBroker()
    adapter = SerializedAdapter()
    broker.register_adapter("serial", adapter)

    try:
        first = broker.submit(
            endpoint_kind="serial",
            model_name="model-a",
            payload="first",
        )
        second = broker.submit(
            endpoint_kind="serial",
            model_name="model-a",
            payload="second",
        )

        assert _collect(first)[0].payload == "first"
        assert _collect(second)[0].payload == "second"
        assert adapter.max_active == 1
    finally:
        broker.stop_and_join()


def test_inference_broker_runs_idle_callback_after_serial_work_drains():
    callback_calls = []
    broker = InferenceBroker(
        idle_poll_s=0.01,
        idle_callback=lambda: callback_calls.append(time.monotonic()),
        idle_callback_interval_s=0.01,
    )
    adapter = BlockingSerializedAdapter()
    broker.register_adapter("serial", adapter)

    try:
        handle = broker.submit(
            endpoint_kind="serial",
            model_name="model-a",
            payload="first",
        )

        assert adapter.started.wait(timeout=1.0)
        time.sleep(0.05)
        assert callback_calls == []

        adapter.release.set()
        assert _collect(handle)[0].payload == "first"
        assert _wait_for(lambda: len(callback_calls) >= 1)
    finally:
        broker.stop_and_join()


def test_inference_broker_routes_continuous_batch_sessions():
    broker = InferenceBroker(idle_poll_s=0.01)
    adapter = ContinuousAdapter()
    broker.register_adapter("continuous", adapter)

    try:
        first = broker.submit(
            endpoint_kind="continuous",
            model_name="model-a",
            payload="first",
        )
        second = broker.submit(
            endpoint_kind="continuous",
            model_name="model-a",
            payload="second",
        )

        assert _collect(first)[0].payload == ("continuous", "first")
        assert _collect(second)[0].payload == ("continuous", "second")
        assert adapter.serial_payloads == []
        assert [
            payload
            for session in adapter.sessions
            for payload in session.submitted_payloads
        ] == [
            "first",
            "second",
        ]
    finally:
        broker.stop_and_join()


def test_inference_broker_runs_idle_callback_after_continuous_work_drains():
    callback_calls = []
    broker = InferenceBroker(
        idle_poll_s=0.01,
        idle_callback=lambda: callback_calls.append(time.monotonic()),
        idle_callback_interval_s=0.01,
    )
    adapter = BlockingContinuousAdapter()
    broker.register_adapter("continuous", adapter)

    try:
        handle = broker.submit(
            endpoint_kind="continuous",
            model_name="model-a",
            payload="first",
        )

        assert adapter.session.started.wait(timeout=1.0)
        time.sleep(0.05)
        assert callback_calls == []

        adapter.session.release.set()
        assert _collect(handle)[0].payload == ("continuous", "first")
        assert _wait_for(lambda: len(callback_calls) >= 1)
    finally:
        broker.stop_and_join()
