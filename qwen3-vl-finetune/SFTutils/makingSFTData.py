import argparse
import logging
import json
import random
import os

import numpy as np

from utils.prompts import getUserPrompt

logger = logging.getLogger('cleanTrainData')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def filterClips(clips, scores, alpha, number, srcLanguage):
    """过滤clips，选取分数高于阈值的前number条数据"""
    
    clips_with_scores = list(zip(clips, scores))
    # clips_with_scores.sort(key=lambda x: x[1], reverse=True) # 按分数排序，从高到低
    
    # 过滤出分数高于阈值的clips
    high_score_clips = [clip for clip, score in clips_with_scores if score > alpha]
    
    logger.info(f"{srcLanguage} 总clips数量: {len(clips)}")
    logger.info(f"{srcLanguage} 高分clips数量: {len(high_score_clips)}")
    
    # 取前number条clips
    final_clips = high_score_clips[:number]
    
    if len(final_clips) < number:
        logger.warning(f"{srcLanguage} 高分clips不足{number}条，实际只有{len(final_clips)}条")
    
    return final_clips

def makeSFTData(clips, srcLanguage, tgtLanguage):
    """生成SFT数据集，使用默认的video-text prompt"""
    
    sftData = []
    
    for clip in clips:
        videoPath = f"./data/TriFine/videoClips/{clip['video_id']}/{clip['video_id']}_{clip['clip_id']}.mp4"
        tgtSent = clip[f"{tgtLanguage.upper()}_sentence"]
        
        # 使用默认的video-text prompt
        videoTextTranslatePrompt = getUserPrompt(
            promptLanguage="en", 
            srcLanguage=srcLanguage, 
            tgtLanguage=tgtLanguage,
            srcSent=clip[f"{srcLanguage.upper()}_sentence"], 
            dataset_type="video-text"
        )
        
        sftItem = {
            "video": videoPath, 
            "conversations": [ 
                {
                    "from": "human", 
                    "value": f"<video>\n{videoTextTranslatePrompt}"
                },
                {
                    "from": "gpt",
                    "value": tgtSent
                }
            ]
        }
        sftData.append(sftItem)
    
    return sftData

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, default=50000, help="目标数据条数")
    parser.add_argument("--alpha", type=float, default=0.6, help="COMET分数阈值")
    args = parser.parse_args()
    
    fileHandler = logging.FileHandler('./log/cleanTrainData.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    args2Log = "数据构造参数: \n"
    for key, value in vars(args).items():
        args2Log += f"{key}: {value} \n"
    logger.info(args2Log)

    # 检查输出文件是否已存在
    output_file = f"./data/work3/sftData/sftData_{args.number}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 加载原始数据和分数
    logger.info("加载原始数据和COMET分数...")
    enScores = np.load("./data/work3/dataClean/train_en_comet_scores.npy").tolist()
    zhScores = np.load("./data/work3/dataClean/train_zh_comet_scores.npy").tolist()
    enTrainClips = json.load(open("./data/TriFine/Train_clips_en.json", "r"))
    zhTrainClips = json.load(open("./data/TriFine/Train_clips_zh.json", "r"))
    
    # 过滤clips
    logger.info("过滤clips...")
    enClipsFiltered = filterClips(
        enTrainClips, enScores, args.alpha, args.number, "en"
    )
    zhClipsFiltered = filterClips(
        zhTrainClips, zhScores, args.alpha, args.number, "zh"
    )
    
    # 确保两种语言的数据量相同
    min_clips = min(len(enClipsFiltered), len(zhClipsFiltered))
    enClipsFiltered = enClipsFiltered[:min_clips]
    zhClipsFiltered = zhClipsFiltered[:min_clips]
    
    logger.info(f"最终使用的数据量: 英文{len(enClipsFiltered)}条，中文{len(zhClipsFiltered)}条")
    
    # 生成英文到中文的数据
    logger.info("生成英文到中文的SFT数据...")
    en_sftData = makeSFTData(enClipsFiltered, "en", "zh")
    
    # 生成中文到英文的数据
    logger.info("生成中文到英文的SFT数据...")
    zh_sftData = makeSFTData(zhClipsFiltered, "zh", "en")
    
    # 合并数据并打乱
    logger.info("合并数据并打乱...")
    all_sftData = en_sftData + zh_sftData
    
    random.seed(42)
    random.shuffle(all_sftData)
    
    # 保存数据
    logger.info("保存数据...")
    json.dump(all_sftData, open(output_file, "w"),
                ensure_ascii=False, indent=4)
    
    logger.info(f"数据生成完成！")
    logger.info(f"SFT数据: {len(all_sftData)}条")
    logger.info(f"保存路径: {output_file}")
