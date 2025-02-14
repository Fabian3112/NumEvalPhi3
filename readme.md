## NumEval: Evaluating Numeral-Aware Language Models

This repository contains the implementation and evaluation code for the work "Advancing Numeral-Aware Language Understanding: A Comparative Study of FlanT5 and Phi3-mini Models". The study focuses on enhancing numeral-aware language understanding through advanced model architectures and innovative prompting techniques, as part of the NumEval Challenge.

Numeral comprehension is a critical component of natural language understanding, particularly in applications like finance, healthcare, and legal analysis, where accurate numerical reasoning is essential. This repository provides the resources to reproduce our results, comparing the performance of FlanT5 (baseline) and Phi3-mini (proposed) models across two tasks from the NumEval Challenge:

1.   Quantitative Predictions (QP): Predicting the magnitude of masked numbers in text.

2.   Quantitative Question Answering (QQA): Answering questions that require numerical reasoning.

For each model and task evaluated in this work, there is one python notebook to replicate the results. 

```
.
├── flant5_qp.ipynb              # Fine-tuning for FlanT5 on QP
├── flant5_qqa.ipynb             # Fine-tuning for FlanT5 on QQA
├── phi3_qp.ipynb                # Fine-tuning for Phi3-mini on QP
├── phi3_qqa.ipynb               # Fine-tuning for Phi3-mini on QQA
├── phi3_chain_of_thought.ipynb  # phi3-mini on QQA with cahin of thought and few shot prompting
├── phi3_cot_training.ipynb      # Fine-tuning for phi3 on QQA with step by step solutions 
├── gpt_answers.ipynb            # generating Gpt-4o chain of thought solutions
├── data
│   ├── QQA_train_cot.json       # new dataset with step by step solutions generated by Gpt-4o
```

The detailed finding and work can be found in numeval.pdf. For details to the challenge please refer to https://sites.google.com/view/numeval/numeval 