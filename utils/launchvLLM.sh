#!/bin/bash

# 启动Qwen3-30B-A3B
# ./utils/lanuchvLLM.sh

# 启动Qwen3-VL-32B-Instruct
# ./utils/lanuchvLLM.sh -m ./huggingface/Qwen/Qwen3-VL-32B-Instruct -l 16384

# 设置默认值
DEFAULT_MODEL_PATH="./huggingface/Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_CUDA_DEVICES="0,1"
DEFAULT_PORT="8000"
DEFAULT_MAX_MODEL_LEN="8192"
DEFAULT_MAX_NUM_SEQS="16"
DEFAULT_GPU_MEMORY_UTILIZATION="0.9"

# 帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项："
    echo "  -m, --model-path             模型路径 (默认: $DEFAULT_MODEL_PATH)"
    echo "  -c, --cuda-devices           CUDA设备 (默认: $DEFAULT_CUDA_DEVICES)"
    echo "  -p, --port                   端口号 (默认: $DEFAULT_PORT)"
    echo "  -l, --max-model-len          最大模型长度 (默认: $DEFAULT_MAX_MODEL_LEN)"
    echo "  -s, --max-num-seqs           最大序列数 (默认: $DEFAULT_MAX_NUM_SEQS)"
    echo "  -g, --gpu-memory-utilization GPU内存利用率 (默认: $DEFAULT_GPU_MEMORY_UTILIZATION)"
    echo "  -h, --help                   显示帮助信息"
    echo ""
    echo "示例："
    echo "  $0 -m /path/to/model -c 0,1,2,3 -p 8001 -l 8192 -s 512 -g 0.8"
    echo "  $0 --model-path /path/to/model --cuda-devices 0,1 --port 8002 --max-model-len 2048 --max-num-seqs 128 --gpu-memory-utilization 0.95"
}



# 解析命令行参数
MODEL_PATH="$DEFAULT_MODEL_PATH"
CUDA_DEVICES="$DEFAULT_CUDA_DEVICES"
PORT="$DEFAULT_PORT"
MAX_MODEL_LEN="$DEFAULT_MAX_MODEL_LEN"
MAX_NUM_SEQS="$DEFAULT_MAX_NUM_SEQS"
GPU_MEMORY_UTILIZATION="$DEFAULT_GPU_MEMORY_UTILIZATION"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -l|--max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -s|--max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        -g|--gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 计算tensor-parallel-size（CUDA设备数量）
if [[ -z "$CUDA_DEVICES" ]]; then
    echo "错误: CUDA设备不能为空"
    exit 1
fi

# 计算CUDA设备数量
TENSOR_PARALLEL_SIZE=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

echo "配置信息："
echo "  模型路径: $MODEL_PATH"
echo "  CUDA设备: $CUDA_DEVICES"
echo "  端口: $PORT"
echo "  最大模型长度: $MAX_MODEL_LEN"
echo "  并行数: $TENSOR_PARALLEL_SIZE"
echo "  最大序列数: $MAX_NUM_SEQS"
echo "  GPU内存利用率: $GPU_MEMORY_UTILIZATION"
echo ""

# 检查GPU可用性
# nvidia-smi || { echo "GPU不可用"; exit 1; }

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "端口$PORT已被占用"
    exit 1
fi

# 检查模型路径是否存在
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 启动服务
echo "正在启动vLLM服务..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
vllm serve "$MODEL_PATH" \
    --dtype bfloat16 \
    --max-model-len $MAX_MODEL_LEN \
    --host 127.0.0.1 \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-num-seqs $MAX_NUM_SEQS \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --enable-prefix-caching \
    --allowed-local-media-path /home
# 记录PID
# echo $! > /tmp/vllm.pid

# # 等待服务启动
# echo "等待服务启动..."
# sleep 10

# # 健康检查
# echo "进行健康检查..."
# curl -f http://127.0.0.1:$PORT/health || { echo "服务启动失败"; exit 1; }
