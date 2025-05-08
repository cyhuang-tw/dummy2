import json
import argparse
from tqdm import tqdm
from pathlib import Path

import os
import pandas as pd

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from models import MODELS
from prompt import INSTRUCTION_REFINE_COT_SINGLE_AND_MULTI_CHOICE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_csv", type=Path, default='./DynamicSUPERB_ConfirmedTasks.csv')
    parser.add_argument("-llm", "--llm_name", choices=MODELS, required=True)
    parser.add_argument("-r", "--result_dir", type=Path, required=True)
    parser.add_argument("-s", "--save_dir", type=Path, required=True)
    parser.add_argument("-d", "--device", type=str, default="auto")
    return parser.parse_args()

def aggregate_labels(example):
    labels = [example['label']]
    
    index = 2
    while True:
        key = 'label{}'.format(index)
        if key not in example or example[key] == None:
            break

        labels.append(example[key])
        index += 1

    return ",".join(labels)

def eval(
    llm_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    p_result: Path,
    save_dir: Path,      
):
    sllm_name = p_result.parent.name
    taskname = p_result.name
    p_save = save_dir / sllm_name / taskname
    if p_save.exists():
        print("{} already exists".format(p_save))
        return

    # Load results
    raw_data = json.load(p_result.open(mode='r'))

    eval_data = {}
    for pwav, example in tqdm(raw_data.items()):
        # Ensure that the label is available
        assert 'label' in example, "Missing label in taskname: {}".format(taskname)
        # assert 'sllm_response' in example, "Missing sllm_response in taskname: {}".format(taskname)
        if 'sllm_response' not in example:
            return

        if taskname == 'HEARSoundEventDetection_DCASE2016Task2':
            import ast
            example['label'] = ast.literal_eval(example['label'])
            example['label'] = set(example['label'])
            example['label'] = '[{}]'.format(', '.join(example['label']))

        # Generate text
        if taskname == 'Dialogue_Act_Classification_SLUE-HVB':
            text = INSTRUCTION_REFINE_COT_SINGLE_AND_MULTI_CHOICE.format(
                instruction=example['instruction'],
                label=aggregate_labels(example),
                response=example['sllm_response']
            )
        else:  
            text = INSTRUCTION_REFINE_COT_SINGLE_AND_MULTI_CHOICE.format(
                instruction=example['instruction'],
                label=example['label'],
                response=example['sllm_response']
            )

        text_input = tokenizer.apply_chat_template([
            {"role": "user", "content": text},
        ], tokenize=False, add_generation_prompt=True)

        enc_text = tokenizer(text_input, return_tensors="pt").to(model.device)
        enc_text_len = enc_text.input_ids.shape[1]

        # Evaluate with LLM
        outputs = model.generate(
            **enc_text,
            max_new_tokens=512,
            do_sample=False
        )

        outputs = tokenizer.decode(outputs[0][enc_text_len:], skip_special_tokens=True).strip()

        # Record
        eval_data[pwav] = raw_data[pwav]
        # eval_data[pwav]['llm_name'] = llm_name
        eval_data[pwav]['messages'] = [{"role": "user", "content": text}]
        eval_data[pwav]['evaluation_response'] = outputs

    # Save results
    p_save.parent.mkdir(parents=True, exist_ok=True)
    json.dump(eval_data, p_save.open(mode='w'), indent=4, ensure_ascii=False)        


def main(args):
    # df = pd.read_csv(args.task_csv)
    # cls_tasks = df[df['Category']=='Classification']['HuggingfacePath'].apply(lambda x: os.path.basename(x)).to_list()
    cls_tasks = [
        # 'DynamicSuperb/SuperbER_RAVDESS',
        # 'DynamicSuperb/HEARMusicSpeechClassification_MAESTRO_Librispeech',
        # 'DynamicSuperb/IntentClassification_SLURP_MINDS14-Intent',
        # 'DynamicSuperb/IntentClassification_SLURP_MINDS14-Action',
        # 'DynamicSuperb/SceneFakeDetection_SceneFake_ASPIRE',
        # 'DynamicSuperb/HEARMusicGenreClassification_ISMIR04',
        # 'DynamicSuperb/SuperbSV_SuperbHiddenSet',
        # 'DynamicSuperb/IntentClassification_SLURP_MINDS14',

        # 'DynamicSuperb/MARBLEInstrumentClassification_MTGInstrument-Fold1',
        # 'DynamicSuperb/MARBLEInstrumentClassification_MTGInstrument-Fold2',
        # 'DynamicSuperb/MARBLEInstrumentClassification_MTGInstrument-Fold3',
        # 'DynamicSuperb/MARBLEInstrumentClassification_MTGInstrument-Fold4',
        # 'DynamicSuperb/MARBLEInstrumentClassification_MTGInstrument-Fold5',
        # 'DynamicSuperb/MARBLEEmotionDetection_MTGMoodTheme-Fold1',
        # 'DynamicSuperb/MARBLEEmotionDetection_MTGMoodTheme-Fold2',
        # 'DynamicSuperb/MARBLEEmotionDetection_MTGMoodTheme-Fold3',
        # 'DynamicSuperb/MARBLEEmotionDetection_MTGMoodTheme-Fold4',
        # 'DynamicSuperb/MARBLEEmotionDetection_MTGMoodTheme-Fold5',
        # 'DynamicSuperb/MARBLEGenreClassification_MTG-Genre-Fold1',
        # 'DynamicSuperb/MARBLEGenreClassification_MTG-Genre-Fold2',
        # 'DynamicSuperb/MARBLEGenreClassification_MTG-Genre-Fold3',
        # 'DynamicSuperb/MARBLEGenreClassification_MTG-Genre-Fold4',
        # 'DynamicSuperb/MARBLEGenreClassification_MTG-Genre-Fold5',
        # 'DynamicSuperb/MARBLEMusicTagging_MTGTop50-Fold1',
        # 'DynamicSuperb/MARBLEMusicTagging_MTGTop50-Fold2',
        # 'DynamicSuperb/MARBLEMusicTagging_MTGTop50-Fold3',
        # 'DynamicSuperb/MARBLEMusicTagging_MTGTop50-Fold4',
        # 'DynamicSuperb/MARBLEMusicTagging_MTGTop50-Fold5',
        # 'DynamicSuperb/SpeakerVerification_LibriSpeech-TestOther',

        # 'DynamicSuperb/IntentClassification_SLURP_MINDS14',
        # 'DynamicSuperb/EmergencyTrafficDetection_Large-Scale-Audio-dataset',
        # 'DynamicSuperb/SoundEffectDetection_RemFx',
        # 'DynamicSuperb/SpeechSentimentAnalysis_MELD',        
        'DynamicSuperb/SpeechTextMatching_LibriSpeech-TestClean'
    ]
    cls_tasks = [os.path.basename(x) for x in cls_tasks]

    save_dir = args.save_dir / args.llm_name

    # Load pretrained models
    model = AutoModelForCausalLM.from_pretrained(args.llm_name, device_map=args.device, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)

    import random
    random.shuffle(cls_tasks)  

    for taskname in tqdm(cls_tasks):
        p_result = args.result_dir / "{}.json".format(taskname)
        if not p_result.exists():
            print(p_result, " does not exist")
            continue

        print("Evaluating {}".format(taskname))
        eval(
            llm_name=args.llm_name,
            model=model,
            tokenizer=tokenizer,
            p_result=p_result,
            save_dir=save_dir
        )

if __name__ == "__main__":
    args = parse_args()
    main(args)

