arXiv:2605.15984v1 [cs.SD] 15 May 2026

# Beyond Content: A Comprehensive Speech Toxicity Dataset and Detection Framework Incorporating Paralinguistic Cues

**Zhongjie Ba$^{1,2}$, Liang Yi$^{1}$, Peng Cheng$^{1,2,*}$, Qingcao Li$^{3,1}$, Qinglong Wang$^{1,2}$, Li Lu$^{1,2}$**  
$^1$The State Key Laboratory of Blockchain and Data Security, Zhejiang University  
$^2$Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security, Hangzhou, Zhejiang, China  
$^3$School of Cyber Science and Engineering, Nanjing University of Science and Technology  
`{zhongjieba, yiliang, peng_cheng}@zju.edu.cn`, `liqingcao@njust.edu.cn`, `{qinglong.wang, li.lu}@zju.edu.cn`

---

## Abstract

Toxic speech detection has become a crucial challenge in maintaining safe online communication environments. However, existing approaches to toxic speech detection often neglect the contribution of paralinguistic cues, such as emotion, intonation, and speech rate, which are key to detecting speech toxicity. Moreover, current toxic speech datasets are predominantly text-based, limiting the development of models that can capture paralinguistic cues. To address these challenges, we present **ToxiAlert-Bench**, a large-scale audio dataset comprising over 30,000 audio clips annotated with seven major toxic categories and twenty fine-grained toxic labels. Uniquely, our dataset annotates toxicity sources—distinguishing between textual content and paralinguistic origins—for comprehensive toxic speech analysis. Furthermore, we propose a dual-head neural network with a multi-stage training strategy tailored for toxic speech detection. This architecture features two task-specific classification headers: one for identifying the source of sensitivity (textual or paralinguistic), and the other for categorizing the specific toxic type. The training process involves independent head training followed by joint fine-tuning to reduce task interference. To mitigate data class imbalance, we incorporate class-balanced sampling and weighted loss functions. Our experimental results show that leveraging paralinguistic features significantly improves detection performance. Our method consistently outperforms existing baselines across multiple evaluation metrics, with a 21.1% relative improvement in Macro-F1 score and a 13.0% relative gain in accuracy over the strongest baseline, highlighting its enhanced effectiveness and practical applicability.

**Code:** [https://github.com/yiliang-la/ToxiAlert](https://github.com/yiliang-la/ToxiAlert)

---

## Introduction

Toxic speech, as part of toxic behaviors, can occur virtually and physically, resulting in a negative psychological impact (Nada, Latif, and Qadir 2023). "Toxic speech" often includes hostile intent that is threatening, abusive, discriminatory, etc. (Garg et al. 2023; Fortuna, Soler, and Wanner 2020). Such behavior can target individuals or groups, leading to severe outcomes such as cyberbullying, harassment, and the spread of discriminatory ideas.

Voice-based social platforms have surged in recent years, amplifying the risk of spreading toxicity through audio. Popular social platforms like Twitter and Facebook have mature text-based content moderation systems (Zeng et al. 2024; Inan et al. 2023). However, voice-involving social platforms for live streaming, multiplayer online gaming, or voice/video chatting, such as Twitch and Slack, require more than text-based moderation (Hamilton, Garretson, and Kerne 2014). Researchers have discovered that when it comes to detecting intents embedded within audio signals, purely text-based models are not sufficient (Lin and Emmanouilidou 2022). The combination of verbal and non-verbal cues can express some toxic intentions. For example, explicit adult content can be conveyed through non-verbal cues such as moaning or Autonomous Sensory Meridian Response (ASMR)-like sounds, which can bypass text-based detection. Therefore, how to leverage paralinguistic cues for effective detection of toxic speech urges more study.

Recent studies have realized the importance of acoustic features other than semantic content in toxic speech identification. Relevant studies can be divided into three categories: Generic Acoustic-Based (Yousefi and Emmanouilidou 2021; Ghosh et al. 2022), Feature Fusion-Based (Lin and Emmanouilidou 2022; Rana and Jha 2022; Mandal et al. 2024), and Textual Task-Assisted Multi-task Learning (Nada, Latif, and Qadir 2023; Liu et al. 2024b; Kumar Nandwana et al. 2024).

Despite these advances, several critical limitations hinder progress in paralinguistic-aware toxic speech detection:

* **Lack of Suitable Datasets.** The scarcity of publicly available datasets poses a significant barrier to research development. Among existing works, only DeToxy (Ghosh et al. 2022) has released a public dataset (i.e., DeToxy-B), but its toxicity classification is solely based on textual content. It lacks samples where toxicity originates from paralinguistic cues (Scherer, London, and Wolf 1973) alone or from both textual and paralinguistic sources combined. Consequently, existing public datasets are insufficient to support the development of detection systems capable of identifying paralinguistic-based toxicity. While some studies focus on non-textual information of audio signals as sources of toxicity and develop corresponding detection systems, their datasets remain private and undisclosed, with unclear and non-transparent construction methodologies.
* **Technical Limitations of Existing Methods.** Existing MTL and acoustic feature-based approaches are highly dependent on textual information. This text-dependency bias may inherently limit the applicability of these methods when semantics are benign but paralinguistic properties, such as intonation and emotion, convey harmful intent. Current feature fusion methods focus on combining specific acoustic dimensions, which may miss subtle paralinguistic toxic signals that exist beyond their explicitly extracted dimensions. Some acoustic feature-based methods rely on traditional handcrafted features for toxicity detection, which may fail to capture rich dimensions representing harmful intent. DeToxy (Ghosh et al. 2022) applies Self-Supervised Learning (SSL) (Liu et al. 2022; Gong et al. 2022) pre-trained foundation models, but underutilizes their representational capabilities through simple feature extraction without sophisticated architectural design. Moreover, DeToxy focuses solely on textual content analysis, failing to address toxicity that originates from paralinguistic sources.
* **Evaluation and Reproducibility Limitations.** The benchmarking practices in this field are incomplete and inconsistent. Existing works primarily compare against their own baselines, lacking broader evaluations. The lack of open-source code further hinders reproducibility and collaboration in the field. These limitations collectively create substantial obstacles to advancing research in paralinguistic-aware toxic speech detection, highlighting the urgent need for comprehensive datasets, transparent methodologies, and reproducible evaluation frameworks.

In this work, we address these limitations from both data and methodological perspectives:
1. We develop and open-source **ToxiAlert-Bench**, the first large-scale toxic speech dataset specifically designed for paralinguistic-aware detection, comprising over 60 hours of annotated audio clips. It features comprehensive toxicity source annotations, including four distinct categories: safe for both textual and paralinguistic sources, textually toxic but paralinguistically safe, textually safe but paralinguistically toxic, and toxic for both sources. The dataset encompasses seven toxic categories and a safe category, with twenty fine-grained toxic labels. We systematically document and open-source the complete dataset construction pipeline, enabling reproducible research and facilitating future dataset-building studies in this domain.
2. We propose a novel dual-head neural network architecture built upon pre-trained SSL foundation models for robust toxic speech detection. Our model leverages large-scale pre-trained representations to capture both semantic and paralinguistic features effectively. The architecture incorporates two specialized classification heads with a multi-stage training strategy. To address data imbalance challenges, we integrate class-balanced sampling and weighted loss functions. Extensive experimental results, including benchmarking and ablation results, validate the effectiveness of our model architecture and training strategies.

Our contributions are summarized as follows:
1. We fill the gap in the research domain of toxic speech detection, with the documentation and open source **ToxiAlert-Bench**, a comprehensive paralinguistic-aware toxic speech dataset.
2. We design a dual-head speech detection framework, employing a multi-stage training strategy with class-balanced sampling, weighted loss functions, and sequential head-specific training followed by joint fine-tuning.
3. Through comprehensive benchmarking against established baselines, including DeToxy and state-of-the-art (SOTA) multimodal large language models (MLLM), our approach demonstrates significant improvements, achieving a 21.1% relative improvement in Macro-F1 and a 13.0% relative gain in accuracy over the strongest baseline.

---

## Related Work

### Textual Content Moderation
Conventional toxic speech detection often ignores the non-verbal properties of speech signals. Some early content moderation (CM) methods heavily depend on manual examination, which is costly and non-scalable. Platforms commonly employ automated CM to ensure that content aligns with behavioral standards by removing inappropriate posts and spam. Most moderators on social platforms utilize conventionally text-based frameworks (Lin and Emmanouilidou 2022; Nada, Latif, and Qadir 2023; Koratana and Hu 2018). They identify whether a post or comment contains toxic information by analyzing the textual features.

### Audio-Based Toxic Speech Detection
Studies in this area can be divided into three categories:
* **General Acoustic Features:** Yousefi et al. (Yousefi and Emmanouilidou 2021) propose a self-attentive Convolutional Neural Networks framework to detect audio-based toxic language. DeToxy (Ghosh et al. 2022) proposes to use acoustic features (F-Bank and wav2vec2.0) for classification. Liu et al. (Liu et al. 2024a) propose a cross-modal learning to incorporate semantic information of text into audio feature representatives, facilitating speech toxicity classification only requiring audio.
* **Multi-Task Learning with text information:** Nada et al. (Nada, Latif, and Qadir 2023) applies an Automatic Speech Recognition (ASR) task to assist toxicity detection. Liu et al. (Liu et al. 2024b) predict the toxicity labels of a speech signal with the assistance of text information alignment. Nandwana et al. (Kumar Nandwana et al. 2024) utilize multi-task learning to predict the toxicity of speech with the assistance of an auxiliary Audio Keyword detection task.
* **Feature Fusion:** Lin et al. (Lin and Emmanouilidou 2022) explore the relationship between speech emotion and toxic speech and propose a framework combining speech emotion recognition (SER) and audio-based CM models. Rana et al. (Rana and Jha 2022) combines acoustic features representing emotion and text features for hate speech detection. Attentive fusion (Mandal et al. 2024) fuses audio and text representation for hate speech identification.

---

## ToxiAlert-Bench

### Overview
ToxiAlert-Bench is a comprehensive English toxic speech dataset comprising 32,561 audio samples totaling 60.82 hours. It uniquely combines both real-world and synthesized audio, including 19,745 samples from established speech corpora and 12,816 samples specifically synthesized for toxicity analysis. To facilitate rigorous experimentation, ToxiAlert-Bench is split into training, validation, and test sets using a 7:1:2 ratio. Each audio sample is annotated with three key attributes:
1. **Toxicity classification** across 7 major categories—**C1:** Sarcasm, **C2:** Horror, **C3:** Sexual, **C4:** Mental Health & Risk Behavior (Mental & Risk), **C5:** Political & Ideological Sensitivity (Ideology), **C6:** Violence & Harm, and **C7:** Discrimination—plus a **Safe** category;
2. **Toxicity source identification**, distinguishing between textual-only toxic, paralinguistic-only toxic, both textual and paralinguistic toxic, or safe; and
3. **Fine-grained categorization**, using 20 specific toxicity labels within the taxonomy. 

Notably, our dataset focuses on paralinguistic toxicity detection—6,728 samples exhibit toxicity solely through paralinguistic cues, addressing a critical gap in existing datasets that primarily focus on textual toxicity.

Compared to the existing DeToxy-B dataset (Ghosh et al. 2022), ToxiAlert-Bench demonstrates advantages in both scale and annotation comprehensiveness. While DeToxy-B contains 20,271 utterances (24.4 hours), ToxiAlert-Bench includes 60% more samples and 150% more total duration. Importantly, DeToxy-B defines toxicity purely based on textual content, with all 5,077 toxic samples labeled through text-only analysis. In contrast, ToxiAlert-Bench introduces a novel toxicity source annotation framework, explicitly labeling samples as textual-only toxic (6,953), paralinguistic-only toxic (6,728), both textual and paralinguistic toxic (2,551), and safe content (16,329). This granular source-based annotation enables researchers to develop models capable of identifying paralinguistic toxicity. Furthermore, while DeToxy-B does not provide detailed toxicity type labels, our dataset provides comprehensive coverage with 7 major toxic categories and 20 fine-grained labels, supporting more nuanced toxic speech research. Please refer to Appendix A for more details on ToxiAlert-Bench.

> **[Figure 1: Overview of the ToxiAlert-Bench dataset construction framework.]**  
> *Pipeline 1 (Real-world Data Construction)* illustrates the collection from 8 datasets (TIMIT, VCTK, VoxCeleb1, Common Voice, MELD, IEMOCAP, LibriSpeech, LJSpeech), followed by Large Multimodal Model Data Filtering, Large Language Model Label Extraction, and Hybrid Verification (Automatic Grouping + Human Review).  
> *Pipeline 2 (Synthetic Data Construction)* depicts Text Generation via LLM prompts followed by expressive Speech Synthesis.

### Dataset Construction Framework

**Bonafide Data Sources.** We collect bonafide speech samples from eight datasets widely accepted in the domain of speech-related studies. They are: (1) TIMIT (Garofolo et al. 1993), (2) IEMOCAP (Busso et al. 2008), (3) VCTK (Veaux et al. 2017), (4) LibriSpeech (Panayotov et al. 2015), (5) VoxCeleb1 (Nagrani, Chung, and Zisserman 2017), (6) LJSpeech-1.1 (Ito and Johnson 2017), (7) CommonVoice (Ardila et al. 2019), and (8) MELD (Poria et al. 2018). For each one, we collect both non-toxic and toxic samples, and these samples are further categorized with a multi-stage annotation pipeline.

**Annotation Pipeline for Bonafide Data.** Our annotation pipeline for bonafide data follows a systematic multi-stage procedure, as shown in Figure 1. We first employ two large multimodal models, Gemini-1.5-Flash (Team 2024) and R1-AQA (Li et al. 2025)[^1], for initial data filtering and preliminary toxicity assessment. Each audio sample is processed using a structured query that determines whether it contains harmful content (see Appendix B for the complete prompt and question-answer pairs). During this stage, both multimodal models analyze each audio sample and its description along three dimensions: toxicity (toxic/non-toxic), major toxic category, and source. When harmful content is detected, the sample is assigned to one of four major toxic categories that serve as our coarse taxonomy: *Sarcasm*, *Horror*, *Sexual Content*, and *Other Harmful Content (Class D)*. This taxonomy is based on our observation that the first three categories are the only ones that consistently show paralinguistic-only toxicity, such as sarcastic tone. Other harmful behaviors lack stable patterns, and defining additional categories would introduce manual bias. To avoid introducing such bias, all remaining harmful instances are grouped into a single catch-all class.

In the next phase, GPT-4o is used to extract structured label suggestions regarding toxicity, coarse category, and source. These suggestions are combined with the multimodal model outputs for a consistency check. When both models reach consensus on all three aspects, the corresponding labels are automatically assigned and subsequently verified through light human review. If any disagreement arises, the sample is forwarded to re-annotation for validation. During this process, the annotator also provides a free-form description for each sample categorized as Class D, characterizing their specific toxic traits. These descriptions are subsequently clustered using unsupervised algorithms (Likas, Vlassis, and Verbeek 2003) to reveal latent fine-grained toxicity types, and the annotator corrects cluster assignments to ensure semantic coherence (see Appendix A for clustering details). This hybrid clustering approach results in 20 distinct toxic labels plus one safe category. These labels are organized into 7 major toxic classes with hierarchical grouping accomplished by the annotator.

**Synthesized Data Construction.** To enhance diversity and ensure comprehensive coverage of paralinguistic toxicity patterns, we implement a synthetic data generation pipeline, as illustrated in the right portion of Figure 1. We utilize the text-to-speech (TTS) method for synthesizing data (Eskimez et al. 2024; Chen et al. 2024; Anastassiou et al. 2024). Our synthesis begins with GPT-4o generating emotionally charged sentences across toxic categories like psychological horror and subtle sexual tension. Carefully designed prompts (see Appendix B) emphasize subtle and context-dependent expressions of toxicity while avoiding explicit language. GPT-4o's strong safety mechanisms ensure all generated sentences are non-toxic on the textual level, allowing paralinguistic cues to be the sole carriers of toxicity.

Following text generation, we use DubbingX (DubbingX 2025) to synthesize audio. The TTS engine features extensive character personality configurations, enabling speech generation with distinct vocal styles by simply providing the input text and selecting a specific character role. We strategically select personas likely to produce speech with different toxic paralinguistic characteristics. The synthesis process enables us to produce naturalistic speech with varying paralinguistic features, ensuring that the toxicity of the resulting audio manifests itself through non-textual cues, such as intonation, rhythm, and emotional expression.

**Expert Proofreading.** After constructing both bonafide and synthetic data, human annotators assess each sample for detailed evaluation: (1) whether the content contains harmful elements, (2) the source of toxicity (textual content, paralinguistic cues, or both), and (3) the specific toxic category. To ensure annotation quality, two domain experts independently labeled all samples, reaching Cohen's $\kappa = 0.82$ (Cohen 1960); disagreements were resolved through discussion.

[^1]: R1-AQA is based on Qwen2-Audio-7B-Instruct (Chu et al. 2024), optimized through reinforcement learning (RL), achieving SOTA performance on the MMAU benchmark (Sakshi et al. 2024) with only 38k post-training samples.

---

## ToxiAlert

To address the limitations of existing approaches, we propose **ToxiAlert**, a unified detection model designed to identify toxic speech where toxicity may arise from textual content, paralinguistic cues, or their combination.

### Design Principles

We desire to utilize the multi-dimensional information of speech signals to perform toxicity detection. Inspired by recent advancements in the domain of deepfake detection (Tak et al. 2022), SOTA methods explore the use of self-supervised learning to obtain better representations trained on diverse speech data and other tasks with only bonafide samples for the purpose of generalization improvement. The pre-trained SSL model, combined with a classifier, is then finetuned with the downstream task dataset, achieving leading performance. Specifically, we adopt Wav2Vec 2.0 (Baevski et al. 2020) as the speech encoder $f_{\theta}: \mathcal{X} \rightarrow \mathbb{R}^{d}$ where input audio waveform $x \in \mathcal{X}$ is mapped to a latent representation:

$$h = f_{\theta}(x) \tag{1}$$

This representation $h \in \mathbb{R}^{T \times d}$ is passed through two classification heads:

* **Source Head ($g_{\phi}^{(s)}$):** This is a multi-label classification head designed for toxicity source identification. It predicts whether the toxicity in the audio arises from textual content, paralinguistic cues, or both.

$$\hat{y}^{(s)} = \sigma\left(g_{\phi}^{(s)}(h)\right) \tag{2}$$

where $\hat{y}^{(s)} \in [0,1]^{2}$ represents the independent probabilities assigned to the two binary toxicity sources—textual and paralinguistic. $\sigma(\cdot)$ represents the element-wise sigmoid activation function.

* **Category Head ($g_{\phi}^{(c)}$):** This is a multi-class classification head designed for toxic category classification. It determines the specific type of toxicity in the input audio.

$$\hat{y}^{(c)} = \text{softmax}\left(g_{\phi}^{(c)}(h)\right) \tag{3}$$

where $\hat{y}^{(c)} \in [0,1]^{K}$ represents the softmax-normalized likelihoods across $K = 8$ mutually exclusive classes, including seven toxic categories and one safe category.

### Multi-Stage Training Strategy

Our training method employs a multi-stage approach designed to optimize both task-specific performance and inter-task coordination, as illustrated in Figure 2. Let $\mathcal{D}^{(s)}$, $\mathcal{D}^{(c)}$, and $\mathcal{D}^{(full)}$ represent the datasets used in each stage.

> **[Figure 2: Overview of the ToxiAlert training framework.]**  
> *Stage 1* trains the SourceHead while freezing the ToxicCategoryHead on C1-C3 + Safe samples.  
> *Stage 2* trains the ToxicCategoryHead while freezing the SourceHead on C1-C7 + Safe samples.  
> *Stage 3* jointly fine-tunes the Encoder, SourceHead, and ToxicCategoryHead using a class-balanced sampler and combined loss $\mathcal{L}_{total}$.

**Stage 1: Source Head Training.** We first train the source head $g_{\phi}^{(s)}$, freezing the category head $g_{\phi}^{(c)}$. This stage focuses on learning to distinguish between textual and paralinguistic toxicity sources. Therefore, the source-training dataset $\mathcal{D}^{(s)}$ is constructed exclusively from the categories Sarcasm, Horror, and Sexual (C1–C3 in Figure 2), since these three categories contain examples across the complete range of source conditions. To achieve class balance, we supplement the toxic samples with safe samples equivalent to approximately $1/3$ of the total toxic sample count from these three categories. The objective is a binary cross-entropy loss:

$$\mathcal{L}_{source} = -\sum_{i=1}^{N}\sum_{j=1}^{2}\left[y_{i,j}^{(s)}\log\hat{y}_{i,j}^{(s)} + (1 - y_{i,j}^{(s)})\log(1 - \hat{y}_{i,j}^{(s)})\right] \tag{4}$$

where $y_{i,j}^{(s)} \in \{0,1\}$ are the ground-truth binary labels.

**Stage 2: Category Head Training.** In the second stage, we freeze the source head $g_{\phi}^{(s)}$ and train the category head $g_{\phi}^{(c)}$ on $\mathcal{D}^{(c)}$. The dataset is derived from all toxic categories (C1–C7 in Figure 2) and includes textually toxic but paralinguistically safe samples, allowing the model to focus on textual discrimination. We add safe speech samples of approximately $1/7$ of the toxic subset to maintain balance. The training minimizes a weighted cross-entropy loss:

$$\mathcal{L}_{category} = -\sum_{i=1}^{N}\sum_{k=1}^{K}w_{k} \cdot y_{i,k}^{(c)}\log\hat{y}_{i,k}^{(c)} \tag{5}$$

where $w_{k}$ is the inverse frequency of class $k$ for balancing, and $y_{i,k}^{(c)} \in \{0,1\}$ are one-hot toxicity category labels.

**Stage 3: Joint Fine-tuning.** The final stage performs end-to-end joint training of both heads using the complete dataset $\mathcal{D}^{(full)}$, and a composite objective function is optimized:

$$\mathcal{L}_{total} = \lambda \cdot \mathcal{L}_{source} + (1 - \lambda) \cdot \mathcal{L}_{category} \tag{6}$$

where $\lambda = 0.2$ to reflect the relatively auxiliary nature of the source task.

To ensure training stability and mitigate label imbalance, we employ a class-balanced sampler that selects $m$ samples per category for every batch of size $B = m \cdot K$. In our experiments, we use $m = 3$, resulting in $B = 24$.

---

## Experiments

### Settings

**Baselines.** We compare ToxiAlert with several SOTA open-source and commercial systems. DeToxy and NetEase Yidun Audio Moderation API (YIDUN) (NetEase 2025) are specifically built for toxic speech detection. DeToxy is an open-source toxicity classifier, while YIDUN is a commercial platform supporting real-time moderation in multiple languages. In contrast, Qwen2-Audio, GPT-4o Audio, and Gemini-2.5-Flash (Comanici et al. 2025) are general-purpose MLLMs not explicitly trained for toxicity detection; these models have demonstrated strong capabilities in speech comprehension and multimodal reasoning, due to their large parameter scale and training on vast datasets.

**Evaluation Setup.** We train ToxiAlert and DeToxy on the ToxiAlert-Bench training set and directly evaluate other baselines. Toxicity classification performance is assessed at both the category level (7 toxic categories) and the label level (20 fine-grained labels). In addition to overall performance, we emphasize a challenging subset of the benchmark where the toxicity is conveyed solely through paralinguistic cues. This setting remains underexplored in prior work, yet it is highly relevant for real-world applications. For generalization evaluation, all models are tested on the DeToxy-B test set[^2]. Evaluation prompts are detailed in Appendix B.

**Metrics.** To comprehensively evaluate model performance, we adopt metrics from two tasks:
1. **Toxicity category classification:** reporting overall accuracy, per-category accuracy, and Macro-F1 to capture both global and class-specific performance. For binary classifiers like DeToxy, we compute binary accuracy by merging all toxic classes for fair comparison.
2. **Toxicity source identification:** formulated as a multi-label task, evaluated with label-level metrics—accuracy, F1 score, Macro-F1, and Micro-F1. We also report subset accuracy at the sample-level, which measures the percentage of samples with all labels predicted correctly.

**Implementation Details.** We adopt `wav2vec2-large-960h` as the audio encoder, followed by three fully connected layers for toxicity classification and source identification. All audio samples are resampled to 16kHz, and truncated to a maximum length of 25 seconds. All experiments are conducted on NVIDIA A100 GPUs using PyTorch.

[^2]: After excluding CMU-MOSEI, CMU-MOSI, MSP-Improv, MSP-Podcast, Social-IQ, and SwitchBoard due to their discontinued open access, the test set contains 2,035 samples.

---

### Toxic Speech Classification

| Model | Sarcasm | Horror | Sexual | Mental & Risk | Ideology | Violence & Harm | Discrim. | ACC | Macro-F1 | Binary ACC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeToxy** | - | - | - | - | - | - | - | - | - | 85.70 |
| **YIDUN** | - | - | 0.50 | - | 0.50 | 0.65 | - | - | - | 50.49 |
| **Qwen2-Audio** | 4.42 | 0.00 | 12.21 | 0.00 | 2.51 | 26.83 | 9.73 | 55.15 | 19.24 | 60.41 |
| **GPT-4o Audio** | 27.08 | 12.22 | 20.17 | 29.27 | 18.09 | 34.88 | 21.01 | 61.89 | 39.91 | 64.52 |
| **Gemini-2.5-Flash** | 53.00 | 58.89 | 34.32 | 47.15 | 21.61 | 61.64 | 36.19 | 70.84 | 57.55 | 75.38 |
| **ToxiAlert** | **81.10** | **90.94** | **81.85** | **48.78** | **52.76** | **65.95** | **39.30** | **80.04** | **69.69** | **86.33** |

**Table 1:** Comparison of ToxiAlert with baselines on ToxiAlert-Bench. We report per-category accuracy across seven toxicity categories. Note that models without category-level predictions leave corresponding entries blank (-).

**Category Level:** As shown in Table 1, ToxiAlert consistently achieves the best overall performance across all 7 toxicity categories. Compared with the strongest baseline, Gemini-2.5-Flash, it improves Macro-F1 by 21.1% and overall accuracy by 13.0%. While DeToxy reports high binary accuracy, being a binary classifier, it lacks the capacity to distinguish between toxicity types, making it less suitable for fine-grained moderation. In contrast, ToxiAlert delivers both higher binary accuracy and comprehensive multi-class prediction.

**Label Level:** Given that Gemini-2.5-Flash achieves the best performance at the category level, we adopt it as the baseline for assessing fine-grained classification capabilities. As shown in Figure 4, ToxiAlert outperforms Gemini-2.5-Flash on the majority of labels. These improvements suggest that ToxiAlert is better equipped to distinguish subtle differences among overlapping or co-occurring toxic behaviors.

> **[Figure 4: Fine-grained comparison of ToxiAlert and Gemini-2.5-Flash on ToxiAlert-Bench.]**  
> Per-category accuracy (%) across twenty fine-grained toxicity labels (Sarcasm, Horror, Sexual, Self-harm, Extremism, Drugs, Historical, Social, Political, Religious, Violence, Illegal Acts, Pers. Abuse, Threaten, Death, Child Safe, Vulgar Lang., Gender Disc., Racial Disc., Other Disc.). ToxiAlert substantially outperforms Gemini-2.5-Flash across nearly all labels, especially Sarcasm (~81% vs ~43%), Horror (~90% vs ~48%), and Sexual (~81% vs ~36%).

---

### Source-Specific Toxicity Detection

We further assess model performance under varied source conditions of toxic expression. Specifically, we focus on three challenging categories—Sarcasm, Horror, and Sexual—and evaluate classification accuracy when the toxic signal is conveyed through paralinguistic cues (Para.), textual content (Tex.), or both (Para.&Tex.). Results are presented in Figure 3. ToxiAlert consistently outperforms all baselines across all categories and source types. In cases where toxic intent is expressed exclusively through Para., ToxiAlert achieves 91.56% on Sarcasm, 97.60% on Horror, and 98.13% on Sexual. In contrast, baselines show notable performance degradation, as they typically overlook non-verbal signals during training or inference.

| Setting | Qwen2 | GPT-4o | Gemini | **ToxiAlert** |
| :--- | :---: | :---: | :---: | :---: |
| **Sarcasm-Para.** | 4.98 | 31.17 | 62.12 | **91.56** |
| **Sarcasm-Tex.** | 0.00 | 6.67 | 0.00 | **13.33** |
| **Sarcasm-Para.&Tex.** | 2.25 | 9.09 | 14.61 | **38.20** |
| **Horror-Para.** | 0.00 | 11.42 | 63.20 | **97.60** |
| **Horror-Tex.** | 0.00 | 16.98 | 26.42 | **43.40** |
| **Horror-Para.&Tex.** | 0.00 | 19.05 | 38.10 | **52.38** |
| **Sexual-Para.** | 0.53 | 10.96 | 23.20 | **98.13** |
| **Sexual-Tex.** | 31.14 | 35.53 | 52.63 | **54.82** |
| **Sexual-Para.&Tex.** | 33.33 | 0.00 | 33.33 | **66.67** |

**Figure 3 (Data Table):** Performance comparison on source-specific toxicity detection across three toxicity types and three source settings. Values denote per-class accuracy (%).

---

### Toxicity Source Identification

| Model | Type | Label-Level ACC | Label-Level F1 | Label-Level Macro F1 | Label-Level Micro F1 | Sample-Level Subset ACC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2** | Para. | 71.84 | 3.79 | 19.28 | 20.72 | 55.35 |
| | Tex. | 77.00 | 34.77 | | | |
| **Gemini** | Para. | 69.48 | 19.57 | 31.11 | 31.31 | 52.90 |
| | Tex. | 77.48 | 42.66 | | | |
| **GPT-4o** | Para. | 71.50 | 0.32 | 13.81 | 15.04 | 53.20 |
| | Tex. | 75.06 | 27.30 | | | |
| **ToxiAlert** | Para. | **91.18** | **83.30** | **79.48** | **79.34** | **80.21** |
| | Tex. | **86.21** | **75.66** | | | |

**Table 2:** Comparison of model performance on the source identification task. Both label-level and sample-level results are reported.

We assess the model's ability to identify toxicity sources—textual or paralinguistic. As shown in Table 2, ToxiAlert consistently outperforms all baselines. For paralinguistic cues, which are inherently subtle and challenging to detect, ToxiAlert achieves an accuracy of 91.18% and an F1 score of 83.30%, significantly surpassing all competing models. For textual content sources, it also delivers strong results with 86.21% accuracy and 75.66% F1 score. Moreover, ToxiAlert achieves the highest overall performance on Subset Accuracy, improving over the strongest baseline by 44.9%, underscoring its robustness in capturing both explicit and implicit forms of toxic expression.

---

### Generalization Evaluation

| Model | Balanced ACC | F1-Binary | Toxic ACC |
| :--- | :---: | :---: | :---: |
| **DeToxy** | 50.33 | **66.95** | 67.78 |
| **YIDUN** | 49.97 | 0.40 | 0.20 |
| **Qwen2-Audio** | 12.50 | 52.91 | 6.88 |
| **Gemini-2.5-Flash** | 37.17 | 59.89 | 29.47 |
| **GPT-4o Audio** | 69.20 | 54.32 | 48.51 |
| **ToxiAlert** | **72.29** | 55.83 | **80.94** |

**Table 3:** Comparison of model generalization performance. Balanced ACC mitigates the effect of class imbalance; Toxic ACC is the accuracy on toxic samples.

To assess the generalization ability of ToxiAlert, we evaluate it on DeToxy-B. For fair comparison, we train a binary version of ToxiAlert using the training set of ToxiAlert-Bench and evaluate it directly on the DeToxy-B test set without any additional fine-tuning. As shown in Table 3, ToxiAlert surpasses the best-performing baseline, GPT-4o Audio, by 4.5% in balanced accuracy, 2.8% in F1 score, and 32.9% in accuracy on toxic samples. These results demonstrate that ToxiAlert generalizes effectively to out-of-distribution data.

---

### Ablation Study

| Model | Toxic Cls. ACC | Toxic Cls. Macro-F1 | Toxic Cls. Binary ACC | Source ID Macro-F1 | Source ID Subset ACC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **w/o SourceHead** | 75.04 | 66.01 | 81.67 | - | - |
| **w/o Multi-stage** | 78.25 | 68.79 | 84.72 | 78.35 | 77.80 |
| **w/o Sampler** | 78.34 | 68.00 | 85.47 | 79.05 | 79.51 |
| **ToxiAlert** | **80.04** | **69.69** | **86.33** | **79.48** | **80.21** |

**Table 4:** Ablation study on the effectiveness of ToxiAlert components. Performance is shown for toxicity category classification (Toxic Cls.) and toxicity source identification (Source ID).

To investigate the impact of each core component in ToxiAlert, we conduct an ablation study focusing on three core modules: the dual-head architecture, the multi-stage training strategy, and the class-balanced sampler. As shown in Table 4, removing any of these components results in a noticeable performance decline:
* Removing the **source head** leads to a significant drop in classification accuracy and Macro-F1, which drops to 75.04% and 66.01%, respectively. Moreover, the model is no longer capable of performing source identification, highlighting the necessity of this joint modeling approach.
* Removing the **multi-stage training strategy** results in consistent degradation across metrics, with overall accuracy reduced to 78.25% and subset accuracy to 77.80%. These results underscore the importance of progressive training in improving convergence and generalization.
* Finally, without the **class-balanced sampler**, performance in both tasks degrades. This result highlights the importance of structured sampling in improving model stability in multi-class classification tasks.

---

## Conclusion

In this work, we propose ToxiAlert-Bench, the first paralinguistic-toxic-aware speech toxicity dataset, covering diverse toxicity-source combinations and capable of facilitating the development of a more comprehensive toxic speech detection system. We also present an SSL-based model that predicts toxic/safe labels, major toxicity category, and toxicity source. With a dual-head design and multi-stage training strategy, our model outperforms existing academic methods and commercial MLLM-based solutions.

---

## Acknowledgments

This paper is supported in part by the Zhejiang Provincial Natural Science Foundation of China under Grant (LD24F020010), the National Natural Science Foundation of China (62472372, 62172359, 62441238, 62072395 and U20A20178), the Key Research and Development Program of Hangzhou City (2024SZD1A27), and the Key R&D Programme of Zhejiang Province (2025C02264).

---

## References

* Anastassiou, P.; Chen, J.; Chen, J.; Chen, Y.; Chen, Z.; Chen, Z.; Cong, J.; Deng, L.; Ding, C.; Gao, L.; et al. 2024. Seed-tts: A family of high-quality versatile speech generation models. *arXiv preprint arXiv:2406.02430*.
* Ardila, R.; Branson, M.; Davis, K.; Henretty, M.; Kohler, M.; Meyer, J.; Morais, R.; Saunders, L.; Tyers, F. M.; and Weber, G. 2019. Common voice: A massively-multilingual speech corpus. *arXiv preprint arXiv:1912.06670*.
* Baevski, A.; Zhou, H.; Mohamed, A.; and Auli, M. 2020. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *arXiv:2006.11477*.
* Busso, C.; Bulut, M.; Lee, C.-C.; Kazemzadeh, A.; Mower, E.; Kim, S.; Chang, J. N.; Lee, S.; and Narayanan, S. S. 2008. IEMOCAP: Interactive emotional dyadic motion capture database. *Language resources and evaluation*, 42(4): 335–359.
* Chen, Y.; Niu, Z.; Ma, Z.; Deng, K.; Wang, C.; Zhao, J.; Yu, K.; and Chen, X. 2024. F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching. *arXiv preprint arXiv:2410.06885*.
* Chu, Y.; Xu, J.; Yang, Q.; Wei, H.; Wei, X.; Guo, Z.; Leng, Y.; Lv, Y.; He, J.; Lin, J.; Zhou, C.; and Zhou, J. 2024. Qwen2-Audio Technical Report. *arXiv preprint arXiv:2407.10759*.
* Cohen, J. 1960. A coefficient of agreement for nominal scales. *Educational and psychological measurement*, 20(1): 37–46.
* Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. *arXiv preprint arXiv:2507.06261*.
* DubbingX. 2025. DubbingX TTS.
* Eskimez, S. E.; Wang, X.; Thakker, M.; Li, C.; Tsai, C.-H.; Xiao, Z.; Yang, H.; Zhu, Z.; Tang, M.; Tan, X.; et al. 2024. E2 tts: Embarrassingly easy fully non-autoregressive zero-shot tts. In *2024 IEEE Spoken Language Technology Workshop (SLT)*, 682–689. IEEE.
* Fortuna, P.; Soler, J.; and Wanner, L. 2020. Toxic, hateful, offensive or abusive? what are we really classifying? an empirical analysis of hate speech datasets. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, 6786–6794.
* Garg, T.; Masud, S.; Suresh, T.; and Chakraborty, T. 2023. Handling bias in toxic speech detection: A survey. *ACM Computing Surveys*, 55(13s): 1–32.
* Garofolo, J. S.; Lamel, L. F.; Fisher, W. M.; Fiscus, J. G.; and Pallett, D. S. 1993. DARPA TIMIT acoustic-phonetic continuous speech corpus CD-ROM. NIST speech disc 1-1.1. *NASA STI/Recon technical report n*, 93: 27403.
* Ghosh, S.; Lepcha, S.; Sakshi, S.; Shah, R. R.; and Umesh, S. 2022. DeToxy: A Large-Scale Multimodal Dataset for Toxicity Classification in Spoken Utterances. In *Interspeech 2022*, 5185–5189.
* Gong, Y.; Lai, C.-I.; Chung, Y.-A.; and Glass, J. 2022. Ssast: Self-supervised audio spectrogram transformer. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, 10699–10709.
* Hamilton, W. A.; Garretson, O.; and Kerne, A. 2014. Streaming on twitch: fostering participatory communities of play within live mixed media. In *Proceedings of the SIGCHI conference on human factors in computing systems*, 1315–1324.
* Inan, H.; Upasani, K.; Chi, J.; Rungta, R.; Iyer, K.; Mao, Y.; Tontchev, M.; Hu, Q.; Fuller, B.; Testuggine, D.; et al. 2023. Llama guard: Llm-based input-output safeguard for human-ai conversations. *arXiv preprint arXiv:2312.06674*.
* Ito, K.; and Johnson, L. 2017. The LJ Speech Dataset. [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/).
* Koratana, A.; and Hu, K. 2018. Toxic speech detection. URL: `https://web.stanford.edu/class/archive/cs/cs224n/cs224n`, 1194.
* Kumar Nandwana, M.; He, Y.; Liu, J.; Yu, X.; Shang, C.; Du Bois, E.; McGuire, M.; and Bhat, K. 2024. Voice Toxicity Detection Using Multi-Task Learning. In *ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 331–335.
* Li, G.; Liu, J.; Dinkel, H.; Niu, Y.; Zhang, J.; and Luan, J. 2025. Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering. *arXiv preprint arXiv:2503.11197*.
* Likas, A.; Vlassis, N.; and Verbeek, J. J. 2003. The global k-means clustering algorithm. *Pattern recognition*, 36(2): 451–461.
* Lin, W.-C.; and Emmanouilidou, D. 2022. Toxic Speech and Speech Emotions: Investigations of Audio-based Modeling and Intercorrelations. In *2022 30th European Signal Processing Conference (EUSIPCO)*, 115–119.
* Liu, J.; Li, G.; Zhang, J.; Dinkel, H.; Wang, Y.; Yan, Z.; Wang, Y.; and Wang, B. 2024a. Enhancing Automated Audio Captioning via Large Language Models with Optimized Audio Encoding. In *Interspeech 2024*, 1135–1139.
* Liu, J.; Nandwana, M. K.; Pylkkänen, J.; Heikinheimo, H.; and McGuire, M. 2024b. Enhancing multilingual voice toxicity detection with speech-text alignment. *arXiv preprint arXiv:2406.10325*.
* Liu, S.; Mallol-Ragolta, A.; Parada-Cabaleiro, E.; Qian, K.; Jing, X.; Kathan, A.; Hu, B.; and Schuller, B. W. 2022. Audio self-supervised learning: A survey. *Patterns*, 3(12).
* Mandal, A.; Roy, G.; Barman, A.; Dutta, I.; and Naskar, S. K. 2024. Attentive Fusion: A Transformer-based Approach to Multimodal Hate Speech Detection. *arXiv:2401.10653*.
* Nada, A. H. A.; Latif, S.; and Qadir, J. 2023. Lightweight Toxicity Detection in Spoken Language: A Transformer-based Approach for Edge Devices. *arXiv:2304.11408*.
* Nagrani, A.; Chung, J. S.; and Zisserman, A. 2017. Voxceleb: a large-scale speaker identification dataset. *arXiv preprint arXiv:1706.08612*.
* NetEase. 2025. NetEase Yidun: AI-Powered Business Security Platform. [https://dun.163.com/](https://dun.163.com/). Accessed: 2025-08-01.
* Panayotov, V.; Chen, G.; Povey, D.; and Khudanpur, S. 2015. Librispeech: an asr corpus based on public domain audio books. In *2015 IEEE international conference on acoustics, speech and signal processing (ICASSP)*, 5206–5210. IEEE.
* Poria, S.; Hazarika, D.; Majumder, N.; Naik, G.; Cambria, E.; and Mihalcea, R. 2018. Meld: A multimodal multi-party dataset for emotion recognition in conversations. *arXiv preprint arXiv:1810.02508*.
* Rana, A.; and Jha, S. 2022. Emotion Based Hate Speech Detection using Multimodal Learning. *arXiv:2202.06218*.
* Sakshi, S.; Tyagi, U.; Kumar, S.; Seth, A.; Selvakumar, R.; Nieto, O.; Duraiswami, R.; Ghosh, S.; and Manocha, D. 2024. MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark. *arXiv:2410.19168*.
* Scherer, K. R.; London, H.; and Wolf, J. J. 1973. The voice of confidence: Paralinguistic cues and audience evaluation. *Journal of Research in Personality*, 7(1): 31–44.
* Tak, H.; Todisco, M.; Wang, X.; weon Jung, J.; Yamagishi, J.; and Evans, N. 2022. Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation. *arXiv:2202.12233*.
* Team, G. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv:2403.05530*.
* Veaux, C.; Yamagishi, J.; MacDonald, K.; et al. 2017. CSTR VCTK corpus: English multi-speaker corpus for CSTR voice cloning toolkit. *University of Edinburgh. The Centre for Speech Technology Research (CSTR)*, 6: 15.
* Yousefi, M.; and Emmanouilidou, D. 2021. Audio-based Toxic Language Classification using Self-attentive Convolutional Neural Network. In *2021 29th European Signal Processing Conference (EUSIPCO)*, 11–15.
* Zeng, W.; Liu, Y.; Mullins, R.; Peran, L.; Fernandez, J.; Harkous, H.; Narasimhan, K.; Proud, D.; Kumar, P.; Radharapu, B.; et al. 2024. Shieldgemma: Generative ai content moderation based on gemma. *arXiv preprint arXiv:2407.21772*.

---

## Appendix A: Dataset Details

### Overview of ToxiAlert-Bench
Figure 6 presents a visual overview of ToxiAlert-Bench, illustrating the full taxonomy of toxicity categories along with representative examples. The dataset is hierarchically structured into 7 major toxic categories—**Sarcasm**, **Horror**, **Sexual**, **Mental & Risk**, **Ideology**, **Violence & Harm**, and **Discrimination**. Across these categories, the dataset includes a total of 20 fine-grained labels, capturing nuanced forms of toxic speech such as self-harm, religious content, vulgar language, or drug-related expressions. Specifically, the 20 fine-grained labels include:
1. Sarcasm
2. Horror
3. Sexual Content (Sexual)
4. Self-harm & Suicide (Self-harm)
5. Hate & Extremist Mentality (Extremism)
6. Drugs
7. Historical Sensitivity (Historical)
8. Social Sensitivity (Social)
9. Political Sensitivity (Political)
10. Religious Sensitivity (Religious)
11. Violence
12. Illegal Acts
13. Personal Abuse
14. Threats
15. Death
16. Child Safety
17. Vulgar Language
18. Gender Discrimination
19. Racial Discrimination
20. Other Discrimination

For each fine-grained label, ToxiAlert-Bench includes curated audio samples accompanied by real or synthesized utterances.

> **[Figure 6: Overview of ToxiAlert-Bench taxonomy and examples.]**  
> Representative audio utterance examples from the dataset wheel:
> * *Self-harm & Suicide (C4):* "This man got up to look for him and found him hanging from the bars of a neighboring room."
> * *Horror (C2):* "My mom's boiling their babies for food and stuff. It was really, really horrific. I felt like..."
> * *Gender Disc. (C7):* "But the Brotherhood are a bit like a golf club or something, you know, they don't like women there."
> * *Vulgar Language (C6):* "I've done 63 motherfucking films, 259 television shows and four Broadway shows."
> * *Historical (C5):* "All I could hear in my head was this Nazi villain buttering me up before the big showdown, which was the scene we were."
> * *Religious (C5):* "In their attack on the hierarchy, they still more openly passed all bounds of moderation, as supposing no doubt..."
> * *Violence (C6):* "He hit me across the head with his club and handcuffed me and taken me away."
> * *Child Safety (C6):* "He was a pedophile, a guy who was a suspect in the Sodiak killings. Certainly a lot of the circumstantial evidence points."

### Annotation Format

Each audio sample in ToxiAlert-Bench is annotated using a structured JSON format, as illustrated in Figure 5. The annotation schema is designed to support detailed and consistent labeling across multiple dimensions:

* `file_name`: The filename of the audio sample.
* `source`: Metadata indicating the origin of the sample. It includes the `type` (either `"real"` or `"synthetic"`) and the name of the source `dataset` (e.g., `"LibriSpeech_train"`).
* `sensitivity`: A structured label indicating whether the sample contains toxic content. It includes:
  * `overall`: A boolean value indicating the presence of any toxicity.
  * `paralinguistic`: Whether the toxicity arises solely from paralinguistic cues such as prosody or tone.
  * `textual`: Whether the toxicity is present in the textual content of the speech.
* `category`: Hierarchical toxicity labels, including:
  * `category`: One of the seven major categories (e.g., `"Discrimination"`). For non-toxic content, this is `"Safe"`.
  * `label`: A fine-grained subcategory (e.g., `"Racial Discrimination"`). For non-toxic content, this is also `"Safe"`.
* `description`: A free-form textual description that summarizes the specific toxic theme, aiding interpretability and human validation (e.g., `"Racial Discrimination, Slavery"`).

```json
{
  "file_name": "8051-118101-0010.wav",
  "source": {
    "type": "real",
    "dataset": "LibriSpeech_train"
  },
  "sensitivity": {
    "overall": true,
    "paralinguistic": false,
    "textual": true
  },
  "category": {
    "category": "Discrimination",
    "label": "Racial Discrimination"
  },
  "description": "Racial Discrimination, Slavery"
}
```
**Figure 5:** An example annotation from ToxiAlert-Bench in JSON format. Both toxic and non-toxic samples share the same format, with the `category` and `label` fields set to `Safe` when the audio is non-toxic.

### Dataset Statistics

| Split | C1 Sarcasm | C2 Horror | C3 Sexual | C4 Mental & Risk | C5 Ideology | C6 Violence & Harm | C7 Discrimination | Safe | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1976 | 2006 | 2118 | 428 | 693 | 3239 | 897 | 11430 | 22787 |
| **Dev** | 282 | 287 | 302 | 61 | 99 | 463 | 128 | 1633 | 3255 |
| **Test** | 566 | 574 | 606 | 123 | 199 | 928 | 257 | 3266 | 6519 |

**Table 5:** Statistics of samples per category and Safe class across data splits. All seven toxic categories and the Safe class are included in each subset. The dataset is partitioned with an approximate 70:10:20 split ratio.

### Composition of Bonafide Audio Sources

| Dataset | Split | Total | Toxic | Non-Toxic |
| :--- | :--- | :---: | :---: | :---: |
| **TIMIT** | - | 6300 | 194 | 6106 |
| **LibriSpeech** | Train-100 | 28539 | 1940 | 26599 |
| | Dev | 2703 | 121 | 2582 |
| | Test | 2620 | 128 | 2492 |
| **LJSpeech-1.1** | - | 13100 | 744 | 12356 |
| **MELD** | Train | 9988 | 525 | 9463 |
| | Dev | 1112 | 52 | 1060 |
| | Test | 2747 | 141 | 2606 |
| **IEMOCAP** | - | 10039 | 439 | 9600 |
| **VoxCeleb1** | Dev | 148642 | 4369 | 144273 |
| | Test | 4874 | 198 | 4676 |
| **VCTK** | - | 44257 | 703 | 43554 |
| **Common Voice** | 21.0-delta | 21796 | 367 | 21429 |
| **Total** | | **296717** | **9921** | **286796** |

**Table 6:** Statistics of toxic and non-toxic utterances from each of the eight source datasets used in constructing the bonafide portion of ToxiAlert-Bench. Toxicity annotations are derived using the Dataset Construction Framework.

### Toxic Label Distribution in Bonafide Speech Data

| Category | Label | Count | Cat. Total |
| :--- | :--- | :---: | :---: |
| **C1: Sarcasm** | Sarcasm | 681 | 681 |
| **C2: Horror** | Horror | 501 | 501 |
| **C3: Sexual** | Sexual Content | 1127 | 1127 |
| **C4: Mental & Risk** | Self-harm & Suicide | 204 | 612 |
| | Hate & Extremist Mentality | 143 | |
| | Drugs | 265 | |
| **C5: Ideology** | Historical Sensitivity | 68 | 991 |
| | Social Sensitivity | 143 | |
| | Political Sensitivity | 537 | |
| | Religious Sensitivity | 243 | |
| **C6: Violence & Harm** | Violence | 2372 | 4630 |
| | Illegal Acts | 364 | |
| | Personal Abuse | 625 | |
| | Threats | 208 | |
| | Death | 370 | |
| | Child Safety | 224 | |
| | Vulgar Language | 467 | |
| **C7: Discrim.** | Gender Discrimination | 506 | 1282 |
| | Racial Discrimination | 562 | |
| | Other Discrimination | 214 | |
| **Uncategorized** | Uncategorized | 97 | 97 |
| **Total** | | | **9921** |

**Table 7:** Distribution of toxic samples across 7 major categories and 20 fine-grained labels within the bonafide portion of ToxiAlert-Bench. The temporary "Uncategorized" group is excluded from final benchmark usage.

### Composition of Synthetic Toxic Speech

| C1 Sarcasm | C2 Horror | C3 Sexual | Safe | Total |
| :---: | :---: | :---: | :---: | :---: |
| 2143 | 2366 | 1899 | 6408 | 12816 |

**Table 8:** Sample counts across the three toxic categories and the Safe class in the synthetic speech subset of ToxiAlert-Bench. All toxic samples emphasize paralinguistic cues without explicit lexical toxicity.

### Composition of Overall Dataset

| Category | Label | Tex. | Para. | Tex.&Para. | Total |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C1: Sarcasm** | Sarcasm | 93 | 2355 | 376 | 2824 |
| **C2: Horror** | Horror | 288 | 2448 | 131 | 2867 |
| **C3: Sexual** | Sexual Content | 1064 | 1925 | 37 | 3026 |
| **C4: Mental & Risk** | Self-harm & Suicide | 135 | 0 | 69 | 204 |
| | Extremism | 99 | 0 | 44 | 143 |
| | Drugs | 216 | 0 | 49 | 265 |
| **C5: Ideology** | Historical Sens. | 56 | 0 | 12 | 68 |
| | Social Sens. | 122 | 0 | 21 | 143 |
| | Political Sens. | 420 | 0 | 117 | 537 |
| | Religious Sens. | 182 | 0 | 61 | 243 |
| **C6: Violence & Harm** | Violence | 1851 | 0 | 521 | 2372 |
| | Illegal Acts | 308 | 0 | 56 | 364 |
| | Personal Abuse | 301 | 0 | 324 | 625 |
| | Threats | 116 | 0 | 92 | 208 |
| | Death | 283 | 0 | 87 | 370 |
| | Child Safety | 172 | 0 | 52 | 224 |
| | Vulgar Language | 242 | 0 | 225 | 467 |
| **C7: Discrim.** | Gender Discrim. | 423 | 0 | 83 | 506 |
| | Racial Discrim. | 432 | 0 | 130 | 562 |
| | Other Discrim. | 150 | 0 | 64 | 214 |
| **Toxic Total** | | **6953** | **6728** | **2551** | **16232** |
| **Safe** | | | | | **16329** |
| **Total** | | | | | **32561** |

**Table 9:** Distribution of ToxiAlert-Bench samples across toxicity categories, fine-grained labels, and toxicity sources: textual (Tex.), paralinguistic (Para.), or both (Tex.&Para.).

### Clustering Analysis of Toxic Labels
To better understand the semantic structure and diversity of toxic expressions in our dataset, we conduct unsupervised clustering on label-level text descriptions. Each Class-D audio sample is first annotated with detailed natural language descriptions of its toxic characteristics by human annotators. These human-written annotations are then embedded into a dense vector space using a pre-trained text embedding model from the Sentence Transformer framework.

To facilitate visualization, we project the high-dimensional embeddings into two dimensions using t-SNE with cosine distance, with parameters tuned to preserve local neighborhood structure. Subsequently, we apply KMeans clustering with $k = 20$ to identify semantically coherent groups of toxicity descriptions. Figure 7 visualizes the resulting clusters, each representing a distinct toxicity pattern.

---

## Appendix B: Prompt Designs

### Prompt for MLLM-Based Data Filtering
> **Audio Transcript:** *"I've done 63 motherfucking films, 259 television shows and four Broadway shows."*  
> **Prompt:**  
> Analyze the audio and determine if it contains harmful content.  
> If harmful content is detected, classify it into one or more of the following categories: Sarcasm, Horror, Sexual Content, Other harmful type.  
> Indicate whether your judgment is based on the tone or content of the audio.  
> If the audio does not contain any harmful content, respond explicitly with: Not-Harmful.  
>
> **Model Output:**  
> Harmful content detected. **Category:** Other harmful type (profanity) **Basis:** The audio contains explicit profanity ("motherfucking"). The tone is aggressive and frustrated.

*(Figure 8: Unified multimodal prompt used for large model-based pre-annotation in the bonafide data pipeline.)*

### Prompt for Description-Based Extraction
> **Prompt:**  
> The following is a description of an audio clip. Classify harmful content only based on the given description—do not infer or add personal judgments. Harmful content categories are:  
> A) Sarcasm B) Horror C) Sexual content D) Other harmful categories  
> Does the audio clip contain harmful content? Answer yes or no only.  
> If harmful content is detected, what category does it belong to? Answer only A, B, C, or D.  
> Is the harm judged by content, tone, or both? Answer only content or tone or both.  
>
> **Input Description:** Harmful content detected. **Category:** Other harmful type (profanity) **Basis:** The audio contains explicit profanity ("motherfucking"). The tone is aggressive and frustrated.  
>
> **Model Output:**  
> Yes  
> D  
> Both

*(Figure 9: Structured prompt used for description-based annotation.)*

### Prompt for Synthetic Sample Generation
* **Horror Prompt:**  
  > Generate multiple unsettling sentences with subtle psychological horror. The sentences should be emotionally charged, atmospheric, and use sensory or emotional cues to imply fear or dread, while avoiding any explicit violence, or gore.
* **Sexual Prompt:**  
  > Generate multiple sensual sentences with subtle sexual tension. The sentences should be emotionally charged, atmospheric, and use physical or emotional cues to imply desire, but avoid any explicit or graphic language.
* **Sarcasm Prompt:**  
  > The sentences should be emotionally charged, atmospheric, and use irony, contrast, or exaggerated politeness to imply criticism, absurdity, or mockery, while avoiding overt name-calling or crude humor.
* **Example Model Output:**  
  > "I found a dusty old diary in the attic, but the last entry was dated tomorrow."

*(Figure 10: Category-specific prompt examples used for generating synthetic toxic utterances.)*

### Prompt for Category-Level Toxicity Classification
> **Audio Transcript:** *"...of my failed life. The misery that swirls around me and end only with a sweet release."*  
> **Prompt:**  
> Analyze the audio and determine if it contains harmful content.  
> If the audio does not contain any harmful content, respond explicitly with: Not-Harmful.  
> If harmful content is detected, classify it into one of the following categories:  
> C1: Sarcasm; C2: Horror; C3: Sexual; C4: Mental Health & Risk Behavior; C5: Political & Ideological Sensitivity; C6: Violence & Harm; C7: Discrimination.  
>
> **Model Responses:**  
> * **Qwen2:** Not-Harmful.  
> * **GPT-4o:** C4: Mental Health & Risk Behavior *(Note: printed as C7 in visual graphic)*  
> * **Gemini:** C4: Mental Health & Risk Behavior *(Note: printed as C7 in visual graphic)*

*(Figure 11: Prompt and model responses for category-level toxicity classification.)*

### Prompt for Label-Level Toxicity Classification
> **Audio Transcript:** *"...of my failed life. The misery that swirls around me and end only with a sweet release."*  
> **Prompt:**  
> Analyze the audio and determine if it contains harmful content.  
> If the audio does not contain any harmful content, respond explicitly with: Not-Harmful.  
> If harmful content is detected, classify it into one of the following categories:  
> C1: Sarcasm; C2: Horror; C3: Sexual Content; C4: Self-harm & Suicide; C5: Hate & Extremist Mentality; C6: Drugs; C7: Historical Sensitivity; C8: Social Sensitivity; C9: Political Sensitivity; C10: Religious Sensitivity; C11: Violence; C12: Illegal Acts; C13: Personal Abuse; C14: Threaten; C15: Death; C16: Child Safe; C17: Vulgar Language; C18: Gender Discrimination; C19: Racial Discrimination; C20: Other Discrimination.  
>
> **Gemini Response:** C4: Self-harm & Suicide

*(Figure 12: Prompt and Gemini-2.5-Flash response for fine-grained toxicity classification.)*

### Prompt for Toxicity Source Identification
> **Audio Transcript:** *"...of my failed life. The misery that swirls around me and end only with a sweet release."*  
> **Prompt:**  
> Analyze the audio and determine if it contains harmful content.  
> If the audio does not contain any harmful content, respond explicitly with: Not-Harmful.  
> If harmful content is detected, indicate whether your judgment is based on the textual content, the paralinguistic cues, or both in the audio.  
>
> **Model Responses:**  
> * **Qwen2:** Not-Harmful.  
> * **GPT-4o:** The textual content  
> * **Gemini:** Harmful content detected based on both textual content and paralinguistic cues.

*(Figure 13: Prompt and responses from Qwen2, GPT-4o, and Gemini for toxicity source identification.)*

### Prompt for Generalization Evaluation
> **Audio Transcript:** *"He is a right bastard."*  
> **Prompt:**  
> Analyze the audio and determine if it contains harmful content.  
> If the audio contains harmful content, respond explicitly with: Harmful.  
> If the audio does not contain any harmful content, respond explicitly with: Not-Harmful.  
>
> **Model Responses:**  
> * **Qwen2:** Harmful.  
> * **GPT-4o:** Harmful.  
> * **Gemini:** Harmful.

*(Figure 14: Example interaction from the generalization evaluation setup.)*