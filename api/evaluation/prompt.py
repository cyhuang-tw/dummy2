# -*- coding: utf-8 -*-
INSTRUCTION_REFINE_COT_SINGLE_AND_MULTI_CHOICE = """
You will be given a question, a corresponding correct answer(s), and a response from a model.
The model's response is a reply to the question. Your task is to judge if the "Model's Response" aligns with the "Ground Truth Answer" based on the "Question."
Please strictly follow the guidelines below:
- Briefly explain the reasons for your judgment.
- Answer with the format "Result: <YES or NO>" at the end.
- Output "YES" if the response aligns with the ground truth answer; 
output "NO" if the response does not match the ground truth answer, selects incorrect or irrelevant options, or provides more answers than required.
- The questions would be single-choice or multi-choice:
For single-choice questions, the model's response should contain one and only one answer. If the model's response selects more than one answer or does not clearly indicate a single answer, you should mark it as incorrect and output "NO." 
For multi-choice questions, the model's response must exactly match all applicable correct choices. If the model's response selects too many, too few, or any incorrect answers, you should mark it as incorrect and output "NO."
- Since the question is short answer, the model's response does not need to mention the content of the question. You only need to check if the model's response has the same meaning as the ground truth answer(s).

Input Format:
Question: {instruction}
Ground Truth Answer: {label}
Model's Response: {response}
""".strip()

INSTRUCTION_REFINE_COT_ALIGN_QA = """
You will be given a question, a corresponding correct answer and a response from a model. 
Model's Response is a reply to the Question. Your task is to judge if "Model's Response" aligns with the "Ground Truth Answer" based on the "Question". 
Please strictly follow the guidelines below:
- Briefly explain the reasons for your judgment.
- Answer with the format "Result: <YES or NO>" at the end.
- Output "YES" if the response aligns with the ground truth answer; output "NO" if the response does not match the ground truth answer or contain more than one answer.
- The question is single-choice, so model's response should contain one and only one answer. Therefore, if the model's response does not clearly indicate only an answer or select more than one answer, you should mark it as incorrect, and output "NO".
- Since the question is short answer, model's response does not need to mention the content of the question. You only need to check if the model's response has the same meaning as the ground truth answer.
Question: {instruction}
Ground Truth Answer: {label}
Model's Response: {response}
""".strip()

INSTRUCTION_INCLUDED_PROMPT = """
Evaluate the audio-language model's response to the following task instruction and compare it with the ground truth:

Task Instruction:
{instruction}

Model's Response:
{response}

Ground Truth:
{label}

Your answer should be yes or no.
""".strip()


TASK_NAME_INCLUDED_PROMPT = """
You will be provided with a ground truth label and a corresponding output 
from a generative model engaged in the {task_name} task.
Your tasks is to determine if the response matches the given ground truth label.
Ground truth label: {label}
Response: {response}
Your answer should be yes or no.
""".strip()


SIMPLE_PROMPT = """
Is the sentence "{response}" aligned with the ground truth "{label}"?
""".strip()
