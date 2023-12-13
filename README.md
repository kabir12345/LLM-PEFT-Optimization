# LLM-PEFT-Optimization

## Author
Kabir Jaiswal <br>
New York University<br>
kj2294@nyu.edu

## Abstract
In the era of big data and sophisticated machine learning architectures, Large Language Models (LLMs) have set unprecedented benchmarks in multiple natural language processing tasks. However, their massive size often poses computational and memory challenges, especially during real-time inference and in memory-constrained environments. This research presents a comprehensive exploration into Parameter Efficient Fine-Tuning (PEFT) as a solution for optimizing LLMs to achieve faster inference and reduce memory footprint without compromising on performance. This project aims to investigate three primary techniques: LORA (Low-Rank Adaptation), and IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations). LORA makes fine-tuning more efficient by drastically reducing the number of trainable parameters. Meanwhile, IA3 offers an architecture-focused adaptation, ensuring optimal utilization of the underlying hardware. The project aims to determine which PEFT technique yields the most optimized results in terms of throughput latency, accuracy, and model storage. The proposed applications of PEFT techniques open new horizons for deploying state-of-the-art LLMs in diverse applications, from edge devices to large-scale platforms, making sophisticated language models more accessible and practical for a broader range of use cases.

## LLM Overview
### Transformers
Transformers are a type of neural network architecture that has become the backbone of most modern Large Language Models (LLMs). Introduced in the paper "Attention Is All You Need" by Vaswani et al., transformers are distinguished by their use of self-attention mechanisms, which allow them to process input data (like text) in parallel and capture complex contextual relationships between words. This architecture has proven to be highly effective for a range of tasks in natural language processing, leading to significant improvements in machine translation, text generation, and language understanding.
![Alt text](URL_of_the_image)


### GPT-2
GPT-2, developed by OpenAI, is a transformer-based model known for its effectiveness in natural language understanding and generation. It features a layered architecture with 12 to 48 transformer blocks, depending on the model variant, and a varying number of parameters from 117 million in the smallest variant to 1.5 billion in the largest. Each block consists of multi-headed self-attention mechanisms and fully connected layers. GPT-2's architecture allows it to generate coherent and contextually relevant text, making it a powerful tool for a variety of NLP tasks.

### T5-Small
T5, or Text-to-Text Transfer Transformer, introduced by Google, adopts a unified approach to NLP tasks, framing them as a text-to-text problem. The T5-Small variant is a scaled-down version of the original T5 model, containing approximately 60 million parameters. It consists of an encoder-decoder structure with 6 layers each, leveraging self-attention and feed-forward networks. This architecture enables T5-Small to perform a wide range of tasks, including translation, summarization, and question-answering, with efficiency and a relatively smaller footprint compared to its larger counterparts.

## Code Structure
The repository includes various directories and files essential for the project:
- **peft**: Contains environment files.
- **src**: Source code for GPT-2 and T5-Small experiments.
- **utils**: Utility functions for dataset loading and model downloading.
- **results**: Performance results of baseline and optimized models.

### Key Files and Their Functions
- `baseline_gpt2.py`: Baseline performance code for GPT-2.
- `optimized_gpt2.py`: GPT-2 optimized with LORA and IA3.
- `baseline_t5_small.py`: Baseline code for t5-small.
- `optimized_t5_small.py`: t5-small optimized with Lora and IA3.

## Steps to Reproduce the Code and Run It
1. Clone the repository.
2. Install the required dependencies from `requirements.txt`.
3. Follow the instructions in each script to run the experiments for GPT-2 and T5-Small.

## Results
The results directory contains images showing the performance metrics for both the baseline and optimized models of GPT-2 and t5-small. These results demonstrate the effectiveness of PEFT techniques in optimizing LLMs.

## References
- [Relevant Paper or Article 1]
- [Relevant Paper or Article 2]
- [Any other relevant reference]

---

*This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.*
