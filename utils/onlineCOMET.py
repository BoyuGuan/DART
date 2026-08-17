# comet_cluster_server_fast.py
# ------------------------------------------------------------
# 目标：
# 1) GPU0 启动 4 个进程：监听 10080-10083
# 2) GPU1 启动 4 个进程：监听 10084-10087
# 3) 仅允许来自 172.18.31.61 的请求，其它返回空响应（默认 403）
# 4) 低延迟推理：方案 B（prepare_for_inference + predict_step）+ warmup + 限制 CPU 线程
#
# 依赖：
#   pip install fastapi uvicorn comet
# ------------------------------------------------------------

import os
import argparse
import multiprocessing as mp
from typing import List, Set, Dict
from copy import deepcopy

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response
from pydantic import BaseModel

# ----------------------------
# 端口 -> GPU 映射
# ----------------------------
PORT_GPU_MAP: Dict[int, int] = {
    10080: 0, 10081: 0, 10082: 0, 10083: 0,  # GPU 0 (4 procs)
    10084: 1, 10085: 1, 10086: 1, 10087: 1,  # GPU 1 (4 procs)
}

DEFAULT_MODEL_PATH = "./huggingface/Unbabel/wmt22-comet-da/checkpoints/model.ckpt"
DEFAULT_ALLOWED_CLIENT = "172.18.31.61"


class CometRequest(BaseModel):
    src: List[str]
    preds: List[str]
    refs: List[str]


def _get_client_ip(request: Request) -> str:
    """
    获取客户端 IP：
    - 若经过反代/网关，优先使用 X-Forwarded-For / X-Real-IP
    - 否则使用 request.client.host
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()

    xri = request.headers.get("x-real-ip", "")
    if xri:
        return xri.strip()

    if request.client and request.client.host:
        return request.client.host

    return ""


def build_app(
    model_path: str,
    allowed_clients: Set[str],
    batch_size: int,
    warmup: bool,
) -> FastAPI:
    """
    注意：该函数在子进程内调用，并且在设置 CUDA_VISIBLE_DEVICES 后再 import torch/comet，
    确保每个子进程只“看到”自己绑定的那张 GPU。
    """
    # 子进程内 import（必须在 CUDA_VISIBLE_DEVICES 设置后）
    import torch
    from comet import load_from_checkpoint

    app = FastAPI()

    # ---- IP 白名单中间件：非允许 IP 直接拒绝 ----
    @app.middleware("http")
    async def ip_allowlist_middleware(request: Request, call_next):
        client_ip = _get_client_ip(request)
        if client_ip not in allowed_clients:
            # “不予回应”的工程实现：返回空 body
            # 如需更隐蔽可改成 404
            return Response(status_code=403)
        return await call_next(request)

    # ---- 设备与性能相关设置 ----
    DEVICE = torch.device("cuda:0")  # 由于限制了 CUDA_VISIBLE_DEVICES，这里的 cuda:0 就是“映射后的第0张”
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 对小 batch 推理通常有利；若遇到形状变化频繁或不稳定，可改回 False
    torch.backends.cudnn.benchmark = True

    # ---- 加载模型（常驻显存）----
    try:
        comet_model = load_from_checkpoint(model_path)
        comet_model = comet_model.to(DEVICE)
        comet_model.eval()
    except Exception as e:
        raise RuntimeError(f"CRITICAL ERROR: 模型加载失败: {e}") from e

    @torch.inference_mode()
    def comet_score_fast(samples: List[Dict[str, str]]):
        """
        samples: List[{"src":..., "mt":..., "ref":...}]
        返回：(system_score: float, scores: List[float])
        """
        # 1) 构造 batch（绕开 Lightning Trainer / DataLoader）
        batch = comet_model.prepare_for_inference(samples)

        # 2) 张量搬到 GPU
        if isinstance(batch, dict):
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE, non_blocking=True)

        # 3) 直接 predict_step 前向
        out = comet_model.predict_step(batch)

        # 4) 兼容不同版本返回格式
        if isinstance(out, dict):
            # 取第一个 tensor
            out = next(v for v in out.values() if torch.is_tensor(v))
        elif isinstance(out, (list, tuple)) and len(out) > 0:
            out = out[0]

        if not torch.is_tensor(out):
            raise RuntimeError(f"Unexpected predict_step output type: {type(out)}")

        scores_t = out.detach().float().view(-1).cpu()
        scores = scores_t.tolist()
        system_score = float(sum(scores) / max(len(scores), 1))
        return system_score, scores

    # ---- 可选：warmup，减少首次请求的 CUDA kernel / cuDNN autotune 冷启动 ----
    if warmup:
        try:
            _ = comet_score_fast([{"src": "a", "mt": "a", "ref": "a"}])
            _ = comet_score_fast([{"src": "b", "mt": "b", "ref": "b"}])
        except Exception:
            pass

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/compute_score")
    async def compute_score(request: CometRequest):
        if not (len(request.src) == len(request.preds) == len(request.refs)):
            raise HTTPException(status_code=400, detail="输入列表长度不一致")

        data = [
            {"src": s, "mt": p, "ref": r}
            for s, p, r in zip(request.src, request.preds, request.refs)
        ]

        try:
            system_score, scores = comet_score_fast(data)
            return {"system_score": system_score, "scores": scores}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推理出错: {str(e)}")

    return app


def _build_uvicorn_log_config_with_time() -> dict:
    """
    为 uvicorn 的 default/access 日志加上时间戳（含毫秒）。
    重点是 uvicorn.access 的那条：
    INFO:     172.18... - "POST ..." 200 OK
    """
    from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

    log_config = deepcopy(UVICORN_LOGGING_CONFIG)

    # 统一时间格式（logging 不支持 %f，毫秒用 %(msecs)03d）
    datefmt = "%Y-%m-%d %H:%M:%S"

    # default 日志格式（例如启动、异常等）
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s.%(msecs)03d %(levelprefix)s %(message)s"
    log_config["formatters"]["default"]["datefmt"] = datefmt

    # access 日志格式（请求行那条）
    log_config["formatters"]["access"]["fmt"] = (
        '%(asctime)s.%(msecs)03d %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    log_config["formatters"]["access"]["datefmt"] = datefmt

    return log_config


def serve_one(
    gpu_id: int,
    port: int,
    model_path: str,
    allowed_clients: Set[str],
    batch_size: int,
    warmup: bool,
):
    """
    每个子进程：
    - 绑定指定 GPU（通过 CUDA_VISIBLE_DEVICES）
    - 限制 CPU 线程（降低多进程争抢导致的抖动与延迟）
    - 启动一个 uvicorn 实例监听指定端口
    """
    # ----------------------------
    # 关键：必须在 import torch/comet 前设置
    # ----------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 限制 CPU 线程（必须尽量早设置）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 子进程内导入
    import torch
    import uvicorn

    # 同步设置 PyTorch 线程数
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    print(
        f"[PID={os.getpid()}] Start COMET server on port={port}, "
        f"CUDA_VISIBLE_DEVICES={gpu_id} (mapped cuda:0), allowed={allowed_clients}"
    )

    app = build_app(
        model_path=model_path,
        allowed_clients=allowed_clients,
        batch_size=batch_size,
        warmup=warmup,
    )

    # 关键改动：为 uvicorn access log 加时间戳
    log_config = _build_uvicorn_log_config_with_time()

    # uvicorn 单 worker（一个进程一个模型实例），避免额外复制模型
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        log_config=log_config,
    )


def main():
    parser = argparse.ArgumentParser(description="Low-latency COMET Multi-Process Server (2 GPUs, 8 ports)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="COMET checkpoint 路径")
    parser.add_argument("--allowed_client", type=str, default=DEFAULT_ALLOWED_CLIENT, help="仅允许访问的客户端 IP")
    parser.add_argument("--batch_size", type=int, default=8, help="保留参数（如需分块可用）")
    parser.add_argument("--warmup", action="store_true", help="启动时进行 warmup（建议开启）")
    args = parser.parse_args()

    allowed_clients = {args.allowed_client}

    # 用 spawn：避免父进程 import 影响子进程 CUDA 可见卡设置
    ctx = mp.get_context("spawn")
    procs: List[mp.Process] = []

    for port, gpu in sorted(PORT_GPU_MAP.items()):
        p = ctx.Process(
            target=serve_one,
            args=(gpu, port, args.model_path, allowed_clients, args.batch_size, args.warmup),
            daemon=False,
        )
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
