import json
import argparse
import logging

logger = logging.getLogger('processReasoningData')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def processReasoningData(filePath):
    with open(filePath, "r") as f:
        data = json.load(f)

    error_count = 0
    output_data = []
    for clip in data:
        
        output = clip["textReasoning"]
        if "</think>" in output:
            try:    
                output = output.split("</think>")[-1].strip()
                clip["textReasoning"] = json.loads(output)
                output_data.append(clip)
            except:
                logger.error(f"Error processing reasoning data: {output}")
                error_count += 1
        else:
            error_count += 1 

    logger.info(f"Error count: {error_count}")
    logger.info(f"Success count: {len(data) - error_count}")

    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filePath", type=str, required=True)
    args = parser.parse_args()
    
    
    fileHandler = logging.FileHandler('./log/processReasoningData.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    logger.info(f"Process {args.filePath}")

    data = processReasoningData(args.filePath)
    
    logger.info(f"Process {args.filePath} done\n")
    
    with open(args.filePath.replace(".json", "_processed.json"), "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)














