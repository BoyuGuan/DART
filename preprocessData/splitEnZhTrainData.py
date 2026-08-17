import json

with open("/home/byguan/3vmt/data/TriFine/Train_clips.json", "r") as f:
    data = json.load(f)

enData, zhData = [], []

for item in data:
    if "en" in item["language"]:
        enData.append(item)
    elif "zh" in item["language"]:
        zhData.append(item)
    else:
        raise ValueError(f"Language {item['language']} not supported")

with open("/home/byguan/3vmt/data/TriFine/Train_clips_en.json", "w") as f:
    json.dump(enData, f, ensure_ascii=False, indent=4)
with open("/home/byguan/3vmt/data/TriFine/Train_clips_zh.json", "w") as f:
    json.dump(zhData, f, ensure_ascii=False, indent=4)
