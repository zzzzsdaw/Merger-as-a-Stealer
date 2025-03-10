# Merger-as-a-Stealer: Stealing Targeted PII from Aligned LLMs with Model Merging
This project is the official open-source content for the paper "Merger-as-a-Stealer: Stealing Targeted PII from Aligned LLMs with Model Merging" submitted to ACL 2025. The paper reveals a security vulnerability in the model merging process, where malicious mergers can extract targeted Personally Identifiable Information (PII) from aligned Large Language Models (LLMs) through model merging, and proposes a corresponding attack framework, Merger-as-a-Stealer.

## Paper Link

[Merger-as-a-Stealer: Stealing Targeted PII from Aligned LLMs with Model Merging](https://arxiv.org/pdf/2502.16094)

## Project Overview

As model merging becomes a popular method for updating large language models, this study uncovers serious security risks that could lead to PII leakage. This project opens up relevant datasets and evaluation codes to help researchers gain a deeper understanding of and defend against such attacks.

## Adopted PII Datasets

**Enron PII**: A publicly available dataset containing 3,333 non - Enron data subjects, each with a name and email pair. It is commonly used to evaluate PII leakage. In this project, it is used to construct the expert dataset and evaluate the attack effectiveness under different experimental settings.

**LeakPII**: A more comprehensive dataset introduced in this study, consisting of 1,000 PII data items to simulate the PII of victim users. Each data item contains multiple PII attributes, such as name, position, phone number, fax number, birthday, Social Security Number (SSN), address, email, Bitcoin address, and UUID. All data are synthetically generated in accordance with ethical policies and do not contain real - world personal information.

## Evaluation Codes

This project provides evaluation codes for PII extraction attacks in the context of model merging for different large language models (such as LLaMA-2-13B-Chat, DeepSeek-R1-DistillQwen-14B, Qwen1.5-14B-Chat, etc.). The codes implement support for different attack settings (such as Naive and Practical), different model merging algorithms (such as Slerp and Task Arithmetic), and different evaluation metrics (such as Exact Match, Memorization Score, Prompt Overlap).

## Contribution and Feedback

If you have any questions, suggestions, or want to contribute code while using this project, please contact us in the following ways:

**Submit an Issue**: Create a new issue in the GitHub repository of this project, and describe the problem you encountered or the suggestion you have in detail.

**Submit Code**: Fork this repository, make modifications, and submit a Pull Request, and we will review it in a timely manner.

## License

This project follows the \[Specific License Name] license. Please refer to the `LICENSE` file in the project for details. When using the code and datasets of this project, please be sure to comply with the provisions of the license.