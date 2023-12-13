# LLM-PEFT-Optimization

## Author
Kabir Jaiswal <br>
New York University<br>
kj2294@nyu.edu

## Abstract
In the era of big data and sophisticated machine learning architectures, Large Language Models (LLMs) have set unprecedented benchmarks in multiple natural language processing tasks. However, their massive size often poses computational and memory challenges, especially during real-time inference and in memory-constrained environments. This research presents a comprehensive exploration into Parameter Efficient Fine-Tuning (PEFT) as a solution for optimizing LLMs to achieve faster inference and reduce memory footprint without compromising on performance. This project aims to investigate three primary techniques: LORA (Low-Rank Adaptation), and IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations). LORA makes fine-tuning more efficient by drastically reducing the number of trainable parameters. Meanwhile, IA3 offers an architecture-focused adaptation, ensuring optimal utilization of the underlying hardware.  The proposed applications of PEFT techniques open new horizons for deploying state-of-the-art LLMs in diverse applications, from edge devices to large-scale platforms, making sophisticated language models more accessible and practical for a broader range of use cases.

## LLM Overview
### Transformers
Transformers are a type of neural network architecture that has become the backbone of most modern Large Language Models (LLMs). Introduced in the paper "Attention Is All You Need" by Vaswani et al., transformers are distinguished by their use of self-attention mechanisms, which allow them to process input data (like text) in parallel and capture complex contextual relationships between words. This architecture has proven to be highly effective for a range of tasks in natural language processing, leading to significant improvements in machine translation, text generation, and language understanding.

![Transformers Architecture](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/static/transformers.png)


### GPT-2
GPT-2, developed by OpenAI, is a transformer-based model known for its effectiveness in natural language understanding and generation. It features a layered architecture with 12 to 48 transformer blocks, depending on the model variant, and a varying number of parameters from 117 million in the smallest variant to 1.5 billion in the largest. Each block consists of multi-headed self-attention mechanisms and fully connected layers. GPT-2's architecture allows it to generate coherent and contextually relevant text, making it a powerful tool for a variety of NLP tasks.

### T5-Small
T5, or Text-to-Text Transfer Transformer, introduced by Google, adopts a unified approach to NLP tasks, framing them as a text-to-text problem. The T5-Small variant is a scaled-down version of the original T5 model, containing approximately 60 million parameters. It consists of an encoder-decoder structure with 6 layers each, leveraging self-attention and feed-forward networks. This architecture enables T5-Small to perform a wide range of tasks, including translation, summarization, and question-answering, with efficiency and a relatively smaller footprint compared to its larger counterparts.

### LORA (Low-Rank Adaptation)
LORA, or Low-Rank Adaptation, is a technique designed to make the fine-tuning of large language models more parameter-efficient. Instead of updating all parameters during the fine-tuning process, LORA focuses on modifying a small set of additional parameters that are introduced into the model. These parameters are used to approximate the changes to the original weights, effectively reducing the number of trainable parameters. This approach allows for significant reductions in memory and computational requirements, making it easier to deploy large models in resource-constrained environments while maintaining high performance.

### IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
IA3 stands for Infused Adapter by Inhibiting and Amplifying Inner Activations. It is a novel approach that focuses on optimizing the internal architecture of language models for better performance and efficiency. IA3 works by selectively inhibiting or amplifying the activations within the model's layers. This method allows for a more targeted and efficient use of the model's capacity, leading to improvements in both inference speed and model accuracy. By fine-tuning the model's internal dynamics, IA3 provides a pathway to harness the full potential of large language models in a more computationally efficient manner.

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


### 1. Clone the Repository
Open your terminal and run the following command to clone the repository:

```bash
git clone https://github.com/kabir12345/LLM-PEFT-Optimization.git
cd LLM-PEFT-Optimization
```

This command clones the repository and changes your current directory to the cloned repository.

### 2. Install Required Dependencies
Ensure you have activated the Python environment and libraries installed on your system:

```bash
source peft/bin/activate
```
post this install the required libraries using:

```bash
pip install -r requirements.txt
```

This command installs all the necessary Python packages listed in `requirements.txt`.

### 3. Run the Experiments
Navigate to the script you want to run and execute it using Python. For example, to run the baseline GPT-2 experiment:

```bash
cd src/gpt2
python baseline_gpt2.py
```

Similarly, to run the optimized GPT-2 experiment:

```bash
python optimized_gpt2.py
```

For T5-Small experiments, navigate to the `t5_small` directory and run the respective scripts:

```bash
cd ../t5_small
python baseline_t5_small.py
python optimized_t5_small.py
```

### Additional Notes
- Ensure you are in the correct directory (`src/gpt2` or `src/t5_small`) before running the scripts.
- The scripts may take some time to execute, depending on your system's capabilities and the complexity of the tasks.

## Results
### GPT-2
---
This section details the quantitative outcomes of our experiments, focusing on loss, ROUGE, and BLEU scores across epochs for the baseline GPT-2 model, and after applying LORA and IA3 optimizations.

#### Baseline GPT-2 Model Results
- **Loss**: Showed a substantial decrease from about 6.0 to 2.0 over the epochs.
- **ROUGE Scores**: ROUGE-1 scores ranged from 0.05 to 0.30, ROUGE-2 scores from 0.05 to 0.25, and ROUGE-L scores from 0.10 to 0.25.
- **BLEU Score**: Demonstrated an initial decline from around 0.0335 to 0.0315, followed by a rebound to approximately 0.0345.

![GPT2 Baseline](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/results/gpt2/baseline_gpt2.png)

#### GPT-2 Optimized with LORA
- **Loss**: Exhibited a decrease from roughly 12.0 to just below 10.0.
- **ROUGE Scores**: ROUGE-1 varied between 0.05 and 0.35, ROUGE-2 between 0.00 and 0.30, and ROUGE-L between 0.05 and 0.15.
- **BLEU Score**: Started at about 0.0290, dropped slightly, then experienced a substantial increase to above 0.0305.

![GPT2 with LORA](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/results/gpt2/optimised_gpt2_lora.png)

#### GPT-2 Optimized with IA3
- **Loss**: Declined from approximately 12.2 to under 11.8.
- **ROUGE Scores**: ROUGE-1 and ROUGE-2 scores fluctuated between 0.00 and 0.20, while ROUGE-L scores were between 0.05 and 0.15.
- **BLEU Score**: Displayed fluctuations around 0.02886 with a noticeable increase to just over 0.0289 towards the final epoch.

![GPT2 with IA3](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/results/gpt2/optimised_gpt2_ia3.png)

### Conclusion
The numerical data underscore that both LORA and IA3 optimizations lead to a decrease in loss, indicating improved learning efficiency. However, the variability in ROUGE and BLEU scores suggests that while the models may be becoming more efficient, language generation quality can fluctuate. Notably, the BLEU score shows a significant increase at the end of training with LORA optimization, which may imply a late-stage improvement in language quality.

### T5-Small Model Optimization Results
---
This section conveys the quantitative findings from the application of LORA and IA3 optimization techniques to the T5-Small model, measured by loss, ROUGE, and BLEU scores over successive epochs.

### Baseline T5-Small Model Results
- **Loss**: Significantly reduced from approximately 3.0 to just above 1.0.
- **ROUGE Scores**: Varied widely with ROUGE-1 nearly reaching 0.8, ROUGE-2 oscillating around 0.6, and ROUGE-L staying within 0.2 to 0.4.
- **BLEU Score**: Showed significant fluctuations, spanning from 0.045 to 0.075.

![T5-Small Baseline](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/results/t5-small/baseline_t5.png)

### T5-Small Optimized with LORA
- **Loss**: Experienced fluctuations within a range of 4.40 to 4.20.
- **ROUGE Scores**: Both ROUGE-1 and ROUGE-2 showed high variability, with peaks close to 0.8, while ROUGE-L scores showed a more moderate variation between 0.2 to 0.6.
- **BLEU Score**: Demonstrated noticeable variability, with the lowest point at around 0.076 and the highest above 0.084.

![T5-Small with LORA](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/results/t5-small/optimized_t5_lora.png)

### T5-Small Optimized with IA3
- **Loss**: Oscillated with a general downward trend, with the highest peaks just below 4.35 and troughs around 4.20.
- **ROUGE Scores**: ROUGE-1 and ROUGE-2 scores showed significant variability, with the former nearly hitting 0.8 and the latter peaking near 0.7, whereas ROUGE-L fluctuated between approximately 0.1 to 0.6.
- **BLEU Score**: Displayed a general upward trend in scores, starting near 0.076 and occasionally reaching above 0.086.

![T5-Small with IA3](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/results/t5-small/optimized_t5_ia3.png)

### Conclusion
Upon comparing the baseline and optimized models, we observed that while the loss for the baseline model reduced more significantly, the optimized models with LORA and IA3 maintained more stable loss values over epochs. The optimized models also demonstrated higher variability in ROUGE and BLEU scores, suggesting potential trade-offs between stability in loss reduction and variability in language quality metrics.


## References
1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008) 

2. Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C. (2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning. arXiv preprint arXiv:2205.05638.


3. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.

4. Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2022). PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods. Retrieved from https://github.com/huggingface/peft

---
*This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.*
