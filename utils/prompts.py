import json

# 增加了多模态模型的systemprompt
def getSystemPrompt(modelName, modelType, promptType):
    if promptType == "translation":
        if modelType == "text":
            return  {"role": "system", "content": "You are an expert translator who is fluent in English and Chinese."}
        elif modelType == "multimodal":
            return {"role": "system","content": [{"type": "text", "text": "You are an expert translator who is fluent in English and Chinese."},],}
        else:
            raise TypeError("Model type format error!")
    if promptType == "videoInfoExtraction":
        videoInfoExtractionSystemPrompt = """You are a multimodal information extraction engine for video.
Your job: extract factual, observable cues from the given video/frames. Do NOT narrate.
Rules:
1) Only report what is visible/legible/audible in the video.
2) Never hallucinate names, brands, or text. OCR must match exactly what is readable.
3) Use stable IDs across the clip: persons P1,P2,... objects O1,O2,... regions R1,R2,...
4) Output MUST be valid JSON and NOTHING ELSE."""

        return  {"role": "system","content": [{"type": "text", "text": videoInfoExtractionSystemPrompt}]}

    modelName2Type = {"Qwen2-7B-Instruct": "qwen2", "Qwen2.5-7B-Instruct": "qwen2", "Qwen2.5-14B-Instruct": "qwen2", "Llama-3-8B-Instruct": "llama3", \
        "LLaVA-NeXT-Video-7B-hf":"llava", "Qwen2-VL-7B-Instruct":"qwen2-vl","MiniCPM-V-2_6":"minicpm","MiniCPM-V-4_5":"minicpm","Qwen2.5-VL-7B-Instruct":"qwen2.5-vl","Qwen2.5-VL-32B-Instruct":"qwen2.5-vl",\
            "Qwen2.5-VL-3B-Instruct":"qwen2.5-vl", "internlm3-8b-instruct":"internlm3", "Qwen2.5-3B-Instruct": "qwen2", "Qwen3-4B": "qwen2", "Qwen3-8B": "qwen2", "Qwen3-30B-A3B":"qwen2",\
        "InternVideo2_5_Chat_8B":"InternVideo2_5","Llama-3.2-11B-Vision-Instruct":"llama3.2-vision","Llama-3.1-8B-Instruct":"llama3.1", "Qwen3-32B": "qwen2", "InternVL3-14B":"internvl3", \
            "Qwen3-VL-8B-Instruct":"qwen3-vl", "Qwen3-VL-4B-Instruct":"qwen3-vl", "Qwen3-VL-8B-Thinking":"qwen3-vl", "Qwen3-VL-4B-Thinking":"qwen3-vl", "InternVL3_5-8B": "internvl3", "InternVL3_5-4B": "internvl3", \
            "Qwen3-4B-Thinking-2507": "qwen2", "Qwen3-4B-Instruct-2507": "qwen2", "DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill", "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill"}
    systemPrompts = dict()
    systemPrompts["qwen2"] = {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
    systemPrompts["llama3"] = None    
    systemPrompts["llava"] = None  
    systemPrompts["qwen2-vl"] = None
    systemPrompts["minicpm"] = None
    systemPrompts["qwen2.5-vl"] = None
    systemPrompts["internlm3"] = {"role": "system", "content": """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""}
    systemPrompts["InternVideo2_5"] = None
    systemPrompts["llama3.2-vision"] = None
    systemPrompts["llama3.1"] = None
    systemPrompts["internvl3"] = None
    systemPrompts["qwen3-vl"] = None
    systemPrompts["DeepSeek-R1-Distill"] = None

    return systemPrompts[modelName2Type[modelName]]

def getUserPrompt(promptLanguage, srcLanguage, tgtLanguage, srcSent, shotNum=0, dataset_type="text", prompt_type=None):
    userPrompts = dict()
    sentencePairsOfshot = [
        {'zh': '请问最近的地铁站在哪里？', 'en': 'Excuse me, where is the nearest subway station?'},\
        {'zh': '很高兴见到你。', 'en': "It's nice to meet you."},\
        {'zh': '祝你有美好的一天！', 'en': "I wish you a wonderful day!"},\
        {'zh': '不要放弃，坚持就是胜利。', 'en': "Don't give up; persistence is victory."},\
        {'zh': '你让我很开心。', 'en': "You make me very happy."},\
        {'zh': '我想点一份牛排。', 'en': "I would like to order a steak."},\
        {'zh': '今天的会议几点开始？', 'en': "What time does today’s meeting start?"},\
        {'zh': '当然可以，我很乐意帮助你。', 'en': "Of course, I'd be happy to help you."},\
    ]
    languageID2text = {"zh": {"zh": "中文", "en": "英文"}, "en": {"zh": "Chinese", "en": "English"}}
    if dataset_type == "text":
        userPrompts["zh"] = f"请把以下输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
        userPrompts["en"] = f"Please translate the following input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]}. ONLY output the translated sentence.\n"
        for i in range(shotNum):
            userPrompts["zh"] += f"输入句子:\n{sentencePairsOfshot[i][srcLanguage]}\n翻译：\n{sentencePairsOfshot[i][tgtLanguage]}\n"
            userPrompts["en"] += f"Input sentence:\n{sentencePairsOfshot[i][srcLanguage]}\nTranslation sentence:\n{sentencePairsOfshot[i][tgtLanguage]}\n"
        userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
        userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
            

    elif dataset_type == "video-text":
        if shotNum != 0:
            raise TypeError("Only zero shot is supported now in video-text")
        userPrompts["zh"] = f"请根据输入的视频内容，把以下输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
        userPrompts["en"] = f"Please translate the following input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]} according to the video. ONLY output the translated sentence.\n"
        userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
        userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
        if prompt_type == "videoTranslationWithSelfReasoningCue":
            userPrompts["en"] = f"I will give you an input sentence, which is a subtitle of a video clip, and I will also input the corresponding video clip.\n"
            userPrompts["en"] += f"I need to translate this input sentence from {srcLanguage} to {tgtLanguage}. Please refer to thevisual cues in the video, such as people, objects, actions, OCR, spatial relations, and pointing/gaze cues when producing the translation of this sentence.\n"
            userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
        elif prompt_type == "videoInfoExtraction":
            userPrompts["en"] = """Given the input video, extract the following 6 cue types:
(1) Who-is-who: person/entity identity registry with stable IDs
(2) Object category + object attributes
(3) Action category + action–object binding
(4) OCR on-screen text
(5) Spatial relations (in/on/under + direction)
(6) Pointing/gaze target grounding (this/that target)

Return JSON with this schema:

{
  "people": [
    {
      "person_id":"P1",
      "role_guess":{"role":"<e.g., cashier/teacher/driver/unknown>"},
    }
  ],

  "objects": [
    {"object_id":"O1","category":"<e.g., cup/knife/phone/unknown>","attributes":{"color":"...","state":"..."}}
  ],

  "actions": [
    {
      "action_id":"A1",
      "predicate":"<verb label, e.g., pour/cut/open/hand_over>",
      "agent_id":"P1|O1|unknown",
      "patient_id":"O2|P2|unknown",
      "instrument_id":"O3|unknown"
    }
  ],

  "ocr": [
    {
      "text":"<exact string>"
    }
  ],

  "spatial_relations": [
    {
      "subject_id":"O1|P1",
      "relation":"in|on|under|left_of|right_of|in_front_of|behind|near|towards|away_from|upward|downward",
      "object_id":"O2|P2|R1",
    }
  ],

  "pointing_gaze": [
    {
      "source_id":"P1",
      "type":"pointing|gaze|head_turn",
      "target_id":"O2|P2|R1|unknown",
      "target_description":"<if target_id unknown, describe the region/object>",
    }
  ]
}

Additional constraints:
- Prefer fewer, high-precision items over many low-confidence items.
- Use "unknown" rather than guessing.
- If no cue of a type exists, return an empty list for that field.
"""

    elif dataset_type == 'image-text':
        if shotNum != 0:
            raise TypeError("Only zero shot is supported now in video-text")
        userPrompts["zh"] = f"请根据输入的图片内容，把以下输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
        userPrompts["en"] = f"Please translate the following input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]} according to the image. ONLY output the translated sentence.\n"
        userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
        userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
    elif dataset_type == 'images-text':   # 给模型多张图片，让模型自己选择图片进行推理
        if shotNum != 0:
            raise TypeError("Only zero shot is supported now in images-text")
        userPrompts["zh"] = f"我将给你一个输入句子，它是一个视频片段的字幕，我还会同时输入这个视频片段的十帧画面。"
        userPrompts["en"] = f"I will give you an input sentence, which is a subtitle of a video clip, and I will also input the uniformly sampled frames of this video clip. "
        userPrompts["zh"] += f"请从输入的十帧画面中选取对翻译有用的画面信息，以将输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
        userPrompts["en"] += f"Please select the useful image information for translation from the input frames to translate the input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]}. ONLY output the translated sentence.\n"
        userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
        userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
    else:
        raise TypeError("Dataset type error!")
    return userPrompts[promptLanguage]

if __name__ == '__main__':
    print(getUserPrompt('en', 'zh', 'en', "我爱中国", 3))
