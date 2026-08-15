import onnxruntime as ort


def providers_for(provider: str) -> list[str]:
    if provider == "CPU":
        return ["CPUExecutionProvider"]
    if provider == "GPU":
        available = ort.get_available_providers()
        if "DmlExecutionProvider" not in available:
            raise RuntimeError(f"ONNX Runtime DirectML is not available. Providers: {available}")
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"Unsupported model provider: {provider}")


def session_options_for(provider: str) -> ort.SessionOptions:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 限制 ORT 线程池：默认每个 session 按 CPU 核数建线程池（16线程/会话，实测 6 个会话共 100 线程，
    # fallback 到 CPU 的 op 会全速并行吃满多核）。DML 推理是单图执行，inter-op 并行无收益。
    session_options.inter_op_num_threads = 1
    if provider == "CPU":
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = 2
    else:
        # GPU 会话：fallback op 少量线程足够，GPU 推理不受影响
        session_options.intra_op_num_threads = 4
    return session_options
