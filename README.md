## Exploring the Potential of Large Language Models in Generating Code-Tracing Questions for Introductory Programming Courses
Source code for EMNLP 2023 Findings paper [Exploring the Potential of Large Language Models in Generating Code-Tracing Questions for Introductory Programming Courses](https://aclanthology.org/2023.findings-emnlp.496.pdf).

## Folder Structure
- emnlp_analyze.py: will run all the analyses mentioned in the paper.

### our_data
- E1, E2, E3, and E4.csv: tracing question evaluations from 4 ananymous annotators.
- evaluation_questions.csv: evaluation criteria.
- final_result_for_paper_gpt3.5.csv: generation results using GPT3.5.
- final_result_for_paper_gpt4.csv: generation results using GPT4.
- tracing_data_truth.csv: ground truth of the authors (human or LLM).
- tracing_question.csv: human created tracing questions collected.

### stat_analysis
- descriptive.py: statistical descriptive analyses.
- emnlp_dataloader.py: data loader.


## Citation
```bibtex
@inproceedings{fan2023exploring,
  title={Exploring the Potential of Large Language Models in Generating Code-Tracing Questions for Introductory Programming Courses},
  author={Fan, Aysa and Zhang, Haoran and Paquette, Luc and Zhang, Rui},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={7406--7421},
  year={2023}
}
``` 