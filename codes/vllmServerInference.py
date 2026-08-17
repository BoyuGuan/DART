from openai import AsyncOpenAI
import asyncio
from tqdm import tqdm
import argparse
import json
from utils.prompts import getUserPrompt, getSystemPrompt
import os
import time
import random
from datetime import datetime
import logging

logger = logging.getLogger('vllmServerInference')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

# 所有支持的 cue 类型
ALL_CUE_TYPES = ["baseline", "people", "objects", "actions", "ocr", "spatial_relations", "pointing_gaze", "all_cues"]

async def process_video_message(args_tuple):
    video_mess, model_path, base_url, clipID, fps = args_tuple
    # 在每个协程中创建 AsyncOpenAI client 实例
    async_process_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=base_url,
        timeout=3600
    )
    
    try:
        # 使用官方最新的调用方式，直接传递 video_url 格式的消息
        chat_response = await async_process_client.chat.completions.create(
            model=model_path,
            messages=video_mess,
            extra_body={"mm_processor_kwargs": {"fps": fps}},
            max_tokens=8192,
            temperature=0.6,
        )
        return clipID, chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing video message: {e}")
        return clipID, f"Error: {str(e)}"
    finally:
        await async_process_client.close()
        
async def process_text_message(args_tuple):
    text_mess, model_path, base_url, task_id = args_tuple
    
    # 如果传入的消息是 None，说明 Prompt 为空，直接返回空字符串，不请求 API
    if text_mess is None:
        return task_id, ""
    
    async_process_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=base_url,
    )
    try:
        chat_response = await async_process_client.chat.completions.create(
            model=model_path,
            messages=text_mess, 
            max_tokens=4906, 
            temperature=0.6,
        )
        return task_id, chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing text message: {e}")
        return task_id, f"Error: {str(e)}"
    finally:
        await async_process_client.close()

# 主函数修改为异步
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filePath", type=str, required=True)
    parser.add_argument("--promptType", type=str, required=True, choices=["textReasoning", "videoCaption", "videoInfoExtraction", "mmPromptTranslation"])
    parser.add_argument("--model_path", type=str, default="./huggingface/Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="multimodal", choices=['text', 'multimodal'])
    parser.add_argument("--dataset_type", type=str, default="video-text", choices=['text', 'video-text', 'image-text'])
    parser.add_argument("-sl", "--srcLanguage", type=str, default="en", choices=['zh', 'en'])
    parser.add_argument("-tl", "--tgtLanguage", type=str, default="zh", choices=['zh', 'en'])
    parser.add_argument("-pl", "--promptLanguage", type=str, default="en", choices=['zh', 'en'])
    parser.add_argument("--mmCueTypes", type=str, default="all",
                        help="要使用的 cue 类型，逗号分隔或 'all' 表示全部。可选: baseline,people,objects,actions,ocr,spatial_relations,pointing_gaze,all_cues")
    parser.add_argument("-s", "--startIndex", type=int, default=None, help="Start index of the data to process, unit 10K")
    parser.add_argument("-e", "--endIndex", type=int, default=None, help="End index of the data to process, unit 10K")
    parser.add_argument("--num_concurrent_requests", type=int, default=10)
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=8000)
    args = parser.parse_args()
    
    # log 设置
    while True:
        logDirName = f'./eval/eval-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        if not os.path.exists(logDirName):
            os.makedirs(logDirName)
            break
        else:
            logger.info("Please waiting\n")
            time.sleep(random.randint(1, 10))
    
    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]
            
    fileHandler = logging.FileHandler(f'{logDirName}/eval.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    # 解析 mmCueTypes 参数
    if args.mmCueTypes.lower() == "all":
        selected_cue_types = ALL_CUE_TYPES
    else:
        selected_cue_types = [t.strip() for t in args.mmCueTypes.split(",")]
        for t in selected_cue_types:
            if t not in ALL_CUE_TYPES:
                logger.error(f"未知的 cue 类型: {t}. 可选: {ALL_CUE_TYPES}")
                return
    
    args2Log = "Script arguments: \n"
    for key, value in vars(args).items():
        args2Log += f"{key}: {value} \n"
    args2Log += f"Selected cue types: {selected_cue_types}\n"
    logger.info(args2Log)
    
    vllmURL = f"http://{args.ip}:{args.port}/v1"
    
    with open(args.filePath, "r") as f:
        allData = json.load(f)
    if args.endIndex is not None and args.startIndex is not None:
        allData = allData[args.startIndex*10000:args.endIndex*10000]
    
    semaphore = asyncio.Semaphore(args.num_concurrent_requests)
    clipIDs = [f"{item['video_id']}_{item['clip_id']}" for item in allData]

    all_message_contents = [] 
    task_ids = [] 
    target_process_function = None

    fps = 2 
    if args.dataset_type == "video-text":
        for clipID in clipIDs:
            systemPrompt = getSystemPrompt(args.model_name, args.model_type, args.promptType)
            userPrompt = getUserPrompt(args.promptLanguage, args.srcLanguage, args.tgtLanguage, None, 0, args.dataset_type, args.promptType)
            video_message_content_for_api = []
            if systemPrompt is not None:
                video_message_content_for_api.append(systemPrompt)

            video_path = os.path.abspath(f"./data/TriFine/videoClips/{clipID[:11]}/{clipID}.mp4")
            userMessage = {"role": "user", "content": [
                    {"type": "text", "text": userPrompt},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"file://{video_path}"
                        }
                    }]
                }
            video_message_content_for_api.append(userMessage)
            all_message_contents.append(video_message_content_for_api)
            task_ids.append(clipID)
        target_process_function = process_video_message

    elif args.dataset_type == "text":
        if args.promptType == "mmPromptTranslation":
            system_prompt_text = (
                "You are a professional translator.\n"
                "Translate the source sentence into the target language.\n"
                "Use the provided video cue ONLY as auxiliary context when it helps disambiguate meaning.\n"
                "Do NOT invent any details beyond the cue.\n"
                "Output ONLY the final translation, with no explanation."
            )
            
            logger.info(f"将对以下 cue 类型进行推理: {selected_cue_types}")
            logger.info(f"总任务数: {len(allData)} 条数据 × {len(selected_cue_types)} 种 cue = {len(allData) * len(selected_cue_types)} 个推理任务")
            
            for idx, clipData in enumerate(allData):
                clipID = clipIDs[idx]
                
                # 对每种 cue 类型创建一个推理任务
                for cue_type in selected_cue_types:
                    prompt_field = f"mm_prompt_{cue_type}"
                    # 默认取空字符串，如果字段不存在则认为是空
                    mm_prompt = clipData.get(prompt_field, "")
                    
                    # 如果 prompt 为空字符串，我们不再生成 API 消息，而是传入 None
                    # 这样 process_text_message 会直接返回 ""
                    if not mm_prompt or mm_prompt.strip() == "":
                        text_message_content_for_api = None
                    else:
                        text_message_content_for_api = [
                            {"role": "system", "content": system_prompt_text},
                            {"role": "user", "content": [{"type": "text", "text": mm_prompt}]}
                        ]
                    
                    all_message_contents.append(text_message_content_for_api)
                    # task_id 格式: clipID|||cue_type
                    task_ids.append(f"{clipID}|||{cue_type}")
        else:
            for clipData in allData:
                srcSent = clipData[f"{args.srcLanguage.upper()}_sentence"]
                userPrompt = getUserPrompt(args.promptLanguage, args.srcLanguage, args.tgtLanguage, srcSent, 0, args.dataset_type, args.promptType)
                text_message_content_for_api = [
                    {"role": "user", "content": [{"type": "text", "text": userPrompt}]}
                ]
                all_message_contents.append(text_message_content_for_api)
            task_ids = clipIDs
            
        target_process_function = process_text_message
    else:
        logger.error(f"Unsupported dataset_type: {args.dataset_type}. Please use 'video-text' or 'text'. Exiting.")
        return

    if not all_message_contents:
        logger.error("No messages were prepared to process. Exiting.")
        return

    if args.dataset_type == "video-text":
        message_args_list = [(content, args.model_path, vllmURL, task_id, fps) for content, task_id in zip(all_message_contents, task_ids)]
    else:
        message_args_list = [(content, args.model_path, vllmURL, task_id) for content, task_id in zip(all_message_contents, task_ids)]
    
    async def process_item_with_semaphore(arg_tuple_for_processor, actual_processor_func):
        async with semaphore:
            return await actual_processor_func(arg_tuple_for_processor)

    tasks = [process_item_with_semaphore(m_args, target_process_function) for m_args in message_args_list]

    result_dict = {}
    for future in tqdm(asyncio.as_completed(tasks), total=len(message_args_list), desc="Processing"):
        task_id, result = await future
        result_dict[task_id] = result
    
    # 按照原始allData的顺序生成输出
    outputs = []
    
    if args.promptType == "mmPromptTranslation":
        for i, item in enumerate(allData):
            clipID = f"{item['video_id']}_{item['clip_id']}"
            output_item = item.copy()
            
            for cue_type in selected_cue_types:
                task_id = f"{clipID}|||{cue_type}"
                output_field = f"translation_{cue_type}"
                
                # result_dict 中一定会有该 task_id 的结果（无论是 API 返回的还是直接返回的空字符串）
                if task_id in result_dict:
                    output_item[output_field] = result_dict[task_id]
                else:
                    output_item[output_field] = "Error" # 理论上不应到达这里
            
            outputs.append(output_item)
    else:
        for i, item in enumerate(allData):
            clipID = f"{item['video_id']}_{item['clip_id']}"
            output_item = item.copy()
            if clipID in result_dict:
                output_item[args.promptType] = result_dict[clipID]
            else:
                output_item[args.promptType] = "None"
            outputs.append(output_item)
    
    with open(f"{logDirName}/results.json", "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    logger.info(f"Results saved to {logDirName}/results.json")
    
    if args.promptType == "mmPromptTranslation":
        logger.info(f"处理完成！")
        logger.info(f"  - 数据条数: {len(allData)}")
        logger.info(f"  - cue 类型数: {len(selected_cue_types)}")
        logger.info(f"  - 总任务数: {len(message_args_list)}")

if __name__ == '__main__':
    asyncio.run(main())