"""
检查视频中的视频存在情况，并输出不存在视频的clip_id.
发现只有train_clips_en.json中存在视频不存在的情况，train_clips_zh.json中不存在视频不存在的情况。
"""

import os
import json
import argparse

from tqdm import tqdm
import numpy as np



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file_path", type=str, default="./data/TriFine/Train_clips_en.json")
    parser.add_argument('-s', "--score_path", type=str, default="./data/work3/dataClean/train_en_comet_scores.npy")
    parser.add_argument("--video_path", type=str, default="./data/TriFine/videoClips")
    # parser.add_argument('-s', "--start", type=int, default=0)
    # parser.add_argument('-e', "--end", type=int, default=1000000000)
    args = parser.parse_args()


    allClips = json.load(open(args.file_path, "r"))
    # if args.start != 0:
    #     allClips = allClips[args.start*10000:]
    # if args.end != 1000000000:
    #     allClips = allClips[:args.end*10000]

    notFoundVideoClipID = set()
    for clip in tqdm(allClips):
        video_path = os.path.join(args.video_path, clip["video_id"], f"{clip['video_id']}_{clip['clip_id']}.mp4")
        if not os.path.exists(video_path):
            notFoundVideoClipID.add(f"{clip['video_id']}_{clip['clip_id']}")

    print(len(notFoundVideoClipID))
    # print(notFoundVideoClipID)
    
    

    scoresOfClips = np.load(args.score_path)

    # 使用末尾合格样例替换不合格样例的方式进行筛选
    # 首先找出所有不合格的位置
    invalidIndices = []
    for i, clip in enumerate(allClips):
        clip_key = f"{clip['video_id']}_{clip['clip_id']}"
        if clip_key in notFoundVideoClipID:
            invalidIndices.append(i)
    
    # 从末尾开始找合格的样例用于替换
    validReplaceIndices = []
    for i in range(len(allClips) - 1, -1, -1):
        if len(validReplaceIndices) >= len(invalidIndices):
            break
        clip = allClips[i]
        clip_key = f"{clip['video_id']}_{clip['clip_id']}"
        if clip_key not in notFoundVideoClipID and i not in invalidIndices:
            validReplaceIndices.append(i)
    
    # 执行替换
    filteredClips = allClips.copy()
    filteredScores = scoresOfClips.copy()
    
    for i, invalidIdx in enumerate(invalidIndices):
        if i < len(validReplaceIndices):
            replaceIdx = validReplaceIndices[i]
            filteredClips[invalidIdx] = allClips[replaceIdx]
            filteredScores[invalidIdx] = scoresOfClips[replaceIdx]
    
    # 截断数组，去掉末尾被用来替换的部分
    finalLength = len(allClips) - len(invalidIndices)
    filteredClips = filteredClips[:finalLength]
    filteredScores = filteredScores[:finalLength]
    
    print(f"原始clip数量: {len(allClips)}")
    print(f"不存在视频的clip数量: {len(notFoundVideoClipID)}")
    print(f"无效位置数量: {len(invalidIndices)}")
    print(f"找到的替换样例数量: {len(validReplaceIndices)}")
    print(f"筛选后clip数量: {len(filteredClips)}")
    print(f"筛选后score数量: {len(filteredScores)}")
    
    # 保存筛选后的结果
    outPutResult = filteredClips
            
    print(len(outPutResult))
    with open(args.file_path.replace(".json", "_filtered.json"), "w") as f:
        json.dump(outPutResult, f, ensure_ascii=False, indent=4)
    
    # 保存筛选后的scores
    np.save(args.score_path.replace(".npy", "_filtered.npy"), filteredScores)















