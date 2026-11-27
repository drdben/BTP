# Steering LLMs for Culturally Localized Generation

**Simran Khanuja\*,2, Hongbin Liu¹, Shujian Zhang¹, John Lambert¹, Mingqing Chen¹, Rajiv Mathews¹, Lun Wang¹**

¹ Google DeepMind, ² Carnegie Mellon University  
*\*Work done as a student researcher at Google DeepMind.*  
Correspondence to: `skhanuja@andrew.cmu.edu`  
*Date: 2026-03-25 | arXiv:2603.23301v1 [cs.CL] 24 Mar 2026*  
*© 2026 Google. All rights reserved.*

**Keywords:** Cultural localization; mechanistic interpretability; sparse autoencoders; LLM steering

## Abstract

LLMs are deployed globally, yet produce responses biased towards cultures with abundant training data. Existing cultural localization approaches such as prompting or post-training alignment are black-box, hard to control, and do not reveal whether failures reflect missing knowledge or poor elicitation. In this paper, we address these gaps using mechanistic interpretability to uncover and manipulate cultural representations in LLMs. Leveraging sparse autoencoders (SAEs), we identify interpretable features that encode culturally salient information and aggregate them into **Cultural Embeddings (CUE)**. We use CUE both to analyze implicit cultural biases under underspecified prompts and to construct white-box steering interventions. Across multiple models, we show that CUE-based steering increases cultural faithfulness and elicits significantly rarer, long-tail cultural concepts than prompting alone. Notably, CUE-based steering is complementary to black-box localization methods, offering gains when applied on top of prompt-augmented inputs. This also suggests that models do benefit from better elicitation strategies, and don't necessarily lack long-tail knowledge representation, though this varies across cultures. Our results provide both diagnostic insight into cultural representations in LLMs and a controllable method to steer towards desired cultures.

## 1. Introduction

LLMs are deployed globally across cultures with diverse customs, languages, and normative values, yet they often produce responses that are Anglo-centric or rely on stereotypes in natural settings (Mukherjee et al., 2023; Santurkar et al., 2023; Tao et al., 2024).¹ To encourage culturally localized behavior, practitioners commonly rely on prompt engineering, such as adding country names or cultural descriptors, or using in-context examples (Singh et al., 2024; Tao et al., 2024; Wang et al., 2025). While these methods can influence surface-level outputs, they typically favor majority or prototypical cultural responses (Cheng et al., 2023; Mitchell et al., 2025) and project national viewpoints in oversimplified ways (Mukherjee et al., 2023). Other approaches, such as post-training alignment, attempt to shape model behavior by aggregating preferences into a single reward function, effectively averaging over diverse cultural values and marginalizing or misrepresenting minority perspectives (Chakraborty et al., 2024; González Barman et al., 2025; Poddar et al., 2024).

More pressingly, existing cultural localization approaches operate largely as black-box methods, adjusting surface behavior without revealing the internal mechanisms that drive culturally aligned outputs (Casper et al., 2023). As a result, they offer little insight into how models represent and apply cultural knowledge, limiting their usefulness as principled solutions. Without visibility into these internal representations, it remains unclear whether poor cultural localization reflects genuine gaps in model knowledge or failures of elicitation (Jin et al., 2025; Xiao et al., 2024). This distinction is critical in practice because if cultural knowledge is present but latent, improved elicitation may suffice, but genuine knowledge gaps would require additional data collection or model training. Recent surveys in cultural NLP further emphasize that the absence of such white-box approaches remains a central limitation of the field (Adilazuarda et al., 2024).

> **\[Figure 1 Description: Framework Overview for Discovering and Steering Culture-Specific Features\]**
>
> * **Data Sources:** Cultural Assertions (e.g., *"Dining etiquette in China often emphasizes waiting for the host and elders to begin..."*) and Contrastive QA pairs.
> * **Step 1 (Extract Activations):** Extracts SAE feature activations across tokens and applies `max-pool` over sequence positions.
> * **Step 2 (Select Top-MI Features):** Computes mutual information $\text{MI}(\text{activations}, \langle\text{country}\rangle)$ and selects the top-$k$ features with high mutual information with country labels to build country-level **Cultural Embeddings (CUE)**.
> * **Step 3a (Analyse Biases):** Evaluates implicit cultural defaults under underspecified prompts (e.g., *"Write me a recipe for a local dish."*) by comparing response activations against centered country prototypes (`US`, `China`, `Japan`, `India`, `UK`).
> * **Step 3b (Steer for Localization):** Constructs culture-specific steering vectors from CUE and intervenes in the model's residual stream across four conditions (`Implicit`, `Steered (I)`, `Explicit`, `Steered (E)`), offering a continuous control knob on the **Cultural Faithfulness Scale** even when combined with prompt augmentation.

In this paper, we address these gaps by extending mechanistic interpretability methods to the cultural domain, to understand models' internal representations of cultural knowledge. Using Sparse Autoencoders (SAEs), we find a set of features that encode different kinds of information pertaining to a culture. SAEs are well suited to this task because they decompose model activations into largely monosemantic, human-interpretable features, unlike individual transformer neurons, which are often highly polysemantic (Bricken et al., 2023; Cunningham et al., 2023). We aggregate these features into country-level **Cultural Embeddings (CUE)**, which serve as compact representations of the cultural knowledge encoded within a model. An overview of our approach is shown in Figure 1.

We use CUE in three ways that leverage the interpretability of SAE features:
1. **Interpreting Cultural Concepts:** We analyze the cultural concepts encoded by the features themselves, uncovering coherent cultural structure within the model's representations. For example, geographically related cultures share features capturing regional cuisine, geography, and ethnic references: East Asian countries share features related to Asian ethnicity and Chinese geographic locations; Mediterranean countries share features referencing olives and olive oil production; most countries share a tea-related feature, etc. We also identify features that are uniquely associated with individual countries, capturing more specific cultural concepts such as cuisine, institutions, and geopolitical topics. We also observe a clear hierarchy across layers: early layers encode lexical identity markers, while later layers capture richer cultural semantics.
2. **Diagnosing Implicit Cultural Defaults:** We use CUE to diagnose implicit cultural defaults in model generations. Under underspecified prompts, models overwhelmingly default to Anglophone cultures: $33.1\%$ of responses align most closely with the U.S. and $26.9\%$ with the U.K., together accounting for roughly $60\%$ of cultural defaults.
3. **White-Box Cultural Steering:** We construct white-box steering interventions that amplify culturally salient features during generation. Steering substantially improves cultural localization across models. On Gemma-2-9B, steered generations win $48\%$ of cultural faithfulness comparisons against explicit prompting baselines (vs. $24\%$ losses) and $53\%$ of rarity comparisons (vs. $17\%$ losses). These gains persist even when steering is applied on top of explicit cultural prompts, showing that feature-level interventions complement existing black-box localization methods while providing fine-grained control over cultural alignment.

¹ *We use country as a proxy for culture throughout this study (Adilazuarda et al., 2024); however, the methodology we develop can be applied to any user-defined culture label as discussed in Section 2.*

## 2. Methodology

An overview of our methodology is shown in Figure 1. We discover latent features that capture culture-specific signals in LLM representations and aggregate them into sparse, interpretable vectors called **Cultural Embeddings (CUE)**. These embeddings provide compact representations of cultural information and form the basis for both bias analysis and steering toward a target culture.

**Data Source:** We use the CANDLE dataset (Nguyen et al., 2023), which contains cultural commonsense assertions across 240+ countries and regions. We retain only countries with at least 500 assertions to ensure sufficient coverage, resulting in the following 22 countries: *Australia, Canada, China, Egypt, England, France, Germany, Greece, India, Ireland, Israel, Italy, Japan, Mexico, Russia, Scotland, South Korea, Spain, Thailand, Turkey, United Kingdom, United States*. The raw data contains substantial redundancy and many trivial statements (e.g., *"Canadian whisky is made in Canada"*), which provide limited cultural signal. To address this, we uniformly sample 100 assertions per country to ensure diversity. We further augment each country with 100 LLM-generated assertions produced using GPT-4.1, designed to surface more culturally salient and rarer concepts. The original 100 assertions are provided as in-context examples to avoid duplication. We also perform a post-verification of the cultural faithfulness of generated assertions using Gemini-2.5-Pro, giving us a final total count of 4,334 assertions across 22 countries. The prompt used and a detailed analysis of the dataset and features selected with and without the augmented data can be found in Appendix A.

### 2.1. Step 0: Can Models Classify Assertions into Cultures?

Our goal of finding features that belong to certain cultures rests on the assumption that models encode culture-specific information in a way that permits discrimination across them. To test this, we sample data from the assertion corpus described above and remove explicit country names to avoid trivial cues (Appendix A). We then pose this as a classification task to the model—given a rewritten cultural assertion, can the model predict its country of origin from a fixed set of country labels? Note that this is a multi-class classification problem with 22 labels (a random baseline would achieve $4.5$ F1 points). To understand which layers most encode cultural information, we train linear probes on residual stream activations at different layers. This allows us to not only validate the presence of discriminative cultural signals in model representations but also provides insight into where it might be encoded.

### 2.2. Step 1: Feature Activation Extraction

Given the assertion dataset, we extract SAE feature activations across all layers (for which SAEs are available). For each input $x$, this yields a $d_{\text{sae}}$-dimensional activation vector $a(x)$ per layer, where each dimension records the activation of one feature. These activations provide the raw signals from which CUE embeddings are constructed. For each text input, we record feature activations at every token. Since salient features may appear at different positions, we apply max pooling across the sequence, i.e., for feature $f_j$ and input $x$, we compute:

$$
a_j(x) = \max_{t \in \text{tokens}(x)} z_j(t),
$$

where $z_j(t)$ is the activation of feature $f_j$ at token $t$. The resulting vector $a(x)$ represents the max-pooled activation of the assertion in the SAE feature space.

### 2.3. Step 2: Constructing CUE

To determine which SAE features are most indicative of the culture label, we compute the mutual information (MI) between each feature's activation and the country label:

$$
I(A_j; C) = \sum_{a_j, c} P(a_j, c) \log \frac{P(a_j, c)}{P(a_j)P(c)}.
$$

A high value of $I(A_j; C)$ indicates that feature $f_j$ provides significant information about the cultural label. We rank all features globally by MI and select features using a cumulative MI threshold $\rho$. Specifically, we include the highest-MI features until their cumulative MI reaches a fraction $\rho$ of the total MI. Using a cumulative threshold $\rho$ allows the number of selected features to adapt to the overall MI distribution rather than requiring a fixed feature budget. We denote this index set by $S$.

For each input $x$, we define its CUE representation by restricting the activation vector to the selected feature set $S$:

$$
\text{CUE}(x) = a(x)[S] \in \mathbb{R}^{|S|}.
$$

Intuitively, $\text{CUE}(x)$ captures how strongly each culturally informative feature is activated for the input.

We construct a prototype for each country $c \in \mathcal{C}$ by averaging CUE representations over all associated assertions:

$$
p_{\text{CUE}}^{(c)} = \frac{1}{N_c} \sum_{x \in \mathcal{D}_c} \text{CUE}(x),
$$

where $\mathcal{D}_c$ is the set of assertions labeled with country $c$ and $N_c = |\mathcal{D}_c|$. Each prototype is a sparse, interpretable vector summarizing the typical activation pattern of culturally salient features for that country. These prototypes form the basis for both bias analysis and steering.

### 2.4. Step 3a: CUE for Bias Analysis

Having constructed country-level prototypes, we first use CUE to diagnose implicit cultural biases exhibited by the model under underspecified prompts. This step is purely analytical and does not involve any intervention.

SAE features may capture both culture-specific signals and general semantic patterns that occur across all cultures (e.g., food or social interaction). To isolate relative cultural signals and remove shared, culture-agnostic activation patterns, we subtract a global baseline by computing the mean prototype across all countries:

$$
\mu_{\text{global}} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} p_{\text{CUE}}^{(c)}.
$$

Centering representations in this way removes global activation biases that are shared across classes, a common property of neural representation spaces (Ethayarajh, 2019). We then center each country prototype:

$$
\tilde{p}^{(c)} = p_{\text{CUE}}^{(c)} - \mu_{\text{global}}.
$$

Given a generated response $y$, we extract SAE activations and apply max pooling as in Step 1 to obtain $a(y)$. Restricting to the salient features and applying the same centering yields:

$$
\tilde{a}(y) = a(y)[S] - \mu_{\text{global}}.
$$

We quantify cultural alignment toward country $c$ using cosine similarity:

$$
\text{bias}(y, c) = \cos\left(\tilde{a}(y), \tilde{p}^{(c)}\right).
$$

Higher similarity indicates that the activation pattern of the generated response aligns more closely with the direction defined by that country's prototype in SAE feature space. This allows us to identify which cultural prototype the model implicitly defaults to under underspecified prompts.

### 2.5. Step 3b: CUE for Steering

While Step 3a diagnoses cultural defaults, we now use CUE to actively steer generation toward a target culture through white-box interventions on internal representations.

For a target culture $c_{\text{target}}$, we compute the mean prototype of all non-target cultures:

$$
p_{\text{CUE}}^{(\text{others})} = \frac{1}{|\mathcal{C} \setminus \{c_{\text{target}}\}|} \sum_{c \neq c_{\text{target}}} p_{\text{CUE}}^{(c)}.
$$

We then define a feature-space steering direction by taking the difference between the target prototype and this baseline:

$$
\delta = p_{\text{CUE}}^{(c_{\text{target}})} - p_{\text{CUE}}^{(\text{others})}.
$$

This vector highlights SAE features that are characteristic of the target culture while suppressing those common to other cultures.

To apply steering within the model, we decode this feature-space direction into the residual stream using the SAE decoder $W_{\text{dec}} \in \mathbb{R}^{d_{\text{sae}} \times d_{\text{model}}}$:

$$
v_{\text{CUE}} = W_{\text{dec}}^{\top} \delta.
$$

At inference time, let $h$ denote the current residual activation. We apply steering via:

$$
h' = h + \alpha v_{\text{CUE}},
$$

where $\alpha$ controls the strength of the intervention and is selected empirically. Varying $\alpha$ allows continuous control over the degree of cultural localization in the generated output.

We explain the methodology intuitively end-to-end using a hypothetical example in Appendix B for interested readers.

## 3. Experimental Setup

We evaluate our approach across multiple open-weight LLMs and model scales to assess the robustness of CUE-based bias analysis and steering. All experiments are conducted using pretrained models and SAEs without any additional fine-tuning.

**Models:** We run experiments on three model families—`Gemma-2-2B`, `Gemma-2-9B`, and `Llama-3.1-8B`—and use their pretrained SAEs from Hugging Face.² For `Gemma-2-2B`, we use pretrained SAEs with widths of 16K and 65K; for `Gemma-2-9B`, we use widths of 16K and 131K; and for `Llama-3.1-8B`, we use widths of 32K and 128K since pretrained SAEs are available for all layers for these widths. We refer to these variants as `G2-2B-16K`, `G2-2B-65K`, `G2-9B-16K`, `G2-9B-131K`, `L3.1-8B-32K`, and `L3.1-8B-128K`, respectively. We specifically select SAE configurations that are available for all layers of the corresponding model. This design choice allows us to later analyze the layer-wise distribution of culturally informative features discovered by CUE. For Gemma models, we use the instruction-tuned variants to support open-ended generation; SAEs trained on the base models have been shown to transfer effectively to these variants. Additional hyperparameter details can be found in Appendix C.

² `https://huggingface.co/google/gemma-scope`; `https://huggingface.co/OpenMOSS-Team/Llama-Scope`

**Baseline Settings:** We evaluate generation under two prompting conditions following the explicit versus implicit cue distinction in Veselovsky et al. (2025). In the $\text{IMPLICIT}$ setting, prompts contain no explicit country or cultural cues. In the $\text{EXPLICIT}$ setting, prompts include an explicit country identifier of the form `"I am from <country>"`. Steering is applied in both settings by adding the steering vector to the residual stream, yielding four experimental conditions: $\text{IMPLICIT}$, $\text{STEER}_{\text{IMPLICIT}}$, $\text{EXPLICIT}$, and $\text{STEER}_{\text{EXPLICIT}}$. Our primary comparison examines how steering in the absence of cues ($\text{STEER}_{\text{IMPLICIT}}$) compares with explicit prompting alone ($\text{EXPLICIT}$), which serves as a natural prompt-augmentation baseline. Empirically, we also find that steering complements explicit prompting: even when cultural cues are provided, steering ($\text{STEER}_{\text{EXPLICIT}}$) yields additional gains and enables fine-grained control over the degree of cultural alignment. We therefore report results across all four settings and demonstrate how steering provides a continuous control knob for cultural localization (Figure 1). Bias analysis is conducted across all four settings to characterize how prompting and steering independently and jointly influence implicit cultural alignment.

**Evaluation Metrics:** We evaluate open-ended LLM generations on the creative writing prompts from Veselovsky et al. (2025) (listed in Table 4) along three dimensions:
1. **Cultural Faithfulness:** Measures how well a generation reflects the intended target culture.
2. **Rarity:** Captures whether generations surface less common or long-tail cultural concepts.
3. **Fluency:** Measures grammaticality and overall coherence. We use fluency only as a filtering criterion when sweeping steering strengths $\alpha$. Steering strengths that consistently produce low fluency generations are discarded to avoid degenerate outputs.

Because these prompts produce open-ended creative outputs for which reference answers are unavailable, we adopt the LLM-as-a-judge evaluation paradigm. To improve robustness and reduce reliance on a single evaluator, we use an ensemble of two independent LLM judges. Each judge scores generations on a 1–10 Likert scale for cultural faithfulness, rarity, and fluency using the same evaluation rubric. The final scalar score for each dimension is computed as the mean of the two judges. Using multiple judges helps mitigate idiosyncratic biases that can arise in individual LLM evaluators and produces more stable evaluation signals.

**Pairwise Preference Evaluation:** In addition to scalar scoring, we perform pairwise preference comparisons. For each prompt, the judges are presented with two responses generated under different experimental conditions (e.g., $\text{EXPLICIT}$ vs. $\text{STEER}_{\text{IMPLICIT}}$) and are asked to select which response better satisfies the evaluation criteria. Pairwise evaluation has been widely adopted in modern LLM benchmarks, including MT-Bench and Chatbot Arena, where a judge selects the preferred response between two candidates (Zheng et al., 2023). Each pairwise comparison is evaluated independently by both judges. If the judges select the same response, that response is recorded as the winner; if the judges disagree, the comparison is recorded as a tie. We report win rates across prompts for each condition and treat these pairwise comparisons as the primary reliability check for the scalar evaluation results. To mitigate position bias in pairwise evaluation, the order of responses is randomized before being presented to the judges.

The exact evaluation prompts used for the judges are provided in Appendix D.

> **\[Figure 2 Description: Prototype Cosine Similarity Heatmaps Across Conditions\]**
> Heatmaps displaying cosine similarity between generated responses and country prototypes under `Implicit`, `Steered (I)`, `Explicit`, and `Steered (E)` for target countries **Japan**, **Mexico**, and **Australia**. Under `Implicit` prompting, cosine similarity concentrates heavily on Anglophone nations (United States, United Kingdom). Under `Steered (I)` and `Steered (E)`, similarity mass shifts sharply onto the target country (e.g., dark red column on Japan, Mexico, Australia) and culturally proximate neighbors (e.g., China and South Korea when steering toward Japan; Spain when steering toward Mexico), while suppressing distant cultural defaults.

> **\[Figure 3 Description: Mean Cultural Faithfulness and Rarity Scores (1–10 Scale)\]**
> Bar charts comparing `Implicit`, `Steered (I)`, `Explicit`, and `Steered (E)` across six model-SAE configurations ($n=22$ countries, with $95\%$ CI error bars):
> * **Gemma-2-2B 16K:** Faithfulness (`1.6`, `3.4`, `3.2`, `4.1`); Rarity (`1.2`, `2.3`, `1.9`, `2.6`).
> * **Gemma-2-2B 65K:** Faithfulness (`1.6`, `3.3`, `3.2`, `4.0`); Rarity (`1.2`, `2.4`, `1.9`, `2.8`).
> * **Gemma-2-9B 16K:** Faithfulness (`1.6`, `4.8`, `3.8`, `5.1`); Rarity (`1.2`, `3.1`, `2.2`, `3.2`).
> * **Gemma-2-9B 131K:** Faithfulness (`1.6`, `4.2`, `4.0`, `4.6`); Rarity (`1.2`, `2.8`, `2.2`, `2.8`).
> * **Llama-3.1-8B 32K:** Faithfulness (`1.3`, `3.0`, `1.6`, `3.1`); Rarity (`1.2`, `2.8`, `1.3`, `2.7`).
> * **Llama-3.1-8B 128K:** Faithfulness (`1.3`, `2.3`, `1.6`, `2.5`); Rarity (`1.2`, `2.1`, `1.3`, `2.1`).

> **\[Figure 4 Description: Pairwise Win/Tie/Loss Rates (%) vs. Explicit Prompting\]**
> Horizontal stacked bar charts comparing `Steered(I) vs Explicit` and `Steered(E) vs Explicit` on Faithfulness and Rarity (binomial sign test excluding ties: $*p<0.05, **p<0.01, ***p<0.001$):
> * **Gemma-2-2B 16K:** Steered(I) Faithfulness `29%*** / 24% / 47%`; Steered(I) Rarity `37%* / 33% / 29%`; Steered(E) Faithfulness `37% / 24% / 39%`; Steered(E) Rarity `45%*** / 32% / 23%`.
> * **Gemma-2-2B 65K:** Steered(I) Faithfulness `29%*** / 26% / 45%`; Steered(I) Rarity `43%*** / 31% / 26%`; Steered(E) Faithfulness `40% / 22% / 39%`; Steered(E) Rarity `50%*** / 31% / 19%`.
> * **Gemma-2-9B 16K:** Steered(I) Faithfulness `48%*** / 28% / 24%`; Steered(I) Rarity `53%*** / 30% / 17%`; Steered(E) Faithfulness `55%*** / 26% / 19%`; Steered(E) Rarity `59%*** / 28% / 13%`.
> * **Gemma-2-9B 131K:** Steered(I) Faithfulness `37% / 33% / 30%`; Steered(I) Rarity `43%*** / 35% / 22%`; Steered(E) Faithfulness `44%*** / 30% / 26%`; Steered(E) Rarity `50%*** / 31% / 19%`.
> * **Llama-3.1-8B 32K:** Steered(I) Faithfulness `49%*** / 36% / 15%`; Steered(I) Rarity `44%*** / 48% / 8%`; Steered(E) Faithfulness `55%*** / 33% / 12%`; Steered(E) Rarity `48%*** / 45% / 7%`.
> * **Llama-3.1-8B 128K:** Steered(I) Faithfulness `33%*** / 50% / 17%`; Steered(I) Rarity `29%*** / 61% / 11%`; Steered(E) Faithfulness `40%*** / 44% / 17%`; Steered(E) Rarity `35%*** / 55% / 10%`.

## 4. Results and Analysis

A complete set of results is presented in Figures 2, 3, and 4. First, we probe model representations to test whether culture-specific signals are encoded and where they emerge (**RQ1**). Next, we perform a detailed analysis of high-MI features and study their behavior through Neuronpedia (**RQ2**),³ and examine cultural defaults in underspecified prompts and quantify these biases (**RQ3**). We then apply CUE-based interventions to evaluate whether steering can elicit rare, long-tail knowledge while remaining culturally faithful (**RQ4**) and analyze per-culture variations in **RQ5**.

³ `https://www.neuronpedia.org/gemma-scope`

### RQ1: Do LLMs encode culture-specific signals, and where are they located?

**Yes.** A linear classifier trained to predict the country of origin of cultural assertions achieves $53.2\%$ macro-F1 on Gemma-2-9B (chance: $4.5\%$ over 22 classes), indicating that culture-specific information is linearly decodable from model representations. Note that our goal here is not to achieve high F1 scores. In fact, this is practically impossible since there are many assertions with a single culture label that are in fact cross-cultural (e.g., *"tea is popular"*) and thus not uniquely identifying. Layer-wise probing shows that classification performance improves in later layers, but remains strong across the entire network, exceeding $30\%$ macro-F1 at all layers. Based on this, we apply steering across all layers rather than restricting interventions to only the top of the network. Full layer-wise results and confusion matrices are reported in Appendix E.

### RQ2: What concepts do shared vs. unique features encode?

To interpret the cultural concepts encoded by SAE features selected via mutual information, we query Neuronpedia's automated explanations for features in the `gemma-scope-9b-pt-res-16k` dictionaries across all 42 residual stream layers. For each country, we identify features with the highest activation relative to the cross-country mean, weighted by mutual information score. We organize findings by whether features are shared across culturally related countries or unique to individual nations, and examine both positively and negatively associated features. Throughout, we reference individual SAE features using the notation `layer:index` (e.g., `22:6129` denotes feature 6129 in layer 22).

We find that **shared features** cluster along recognizable cultural-geographic boundaries:
* **East Asian countries** (China, Japan, South Korea, Thailand) share features encoding Asian ethnicity references (`22:6129`), Chinese geographical locations (`24:5526`), and Asian cultural elements (`30:10797`)—notably with China-centric features dominating the cluster.
* **British Isles nations** (England, Scotland, Ireland, UK) share features for British culture (`27:9710`) and political entities (`37:3658`).
* **Continental European countries** (France, Germany, Italy, Spain) share features for European nationalities (`16:12833`) and multilingual identity (`12:2050`).
* **Mediterranean countries** (Greece, Italy, Spain, Turkey) share an interesting feature encoding references to olives and olive oil production (`38:10325`),⁴ while a tea and tea-related concepts feature (`33:5320`) activates broadly across 18 of 22 countries with strongest activation for England, China, and Turkey—suggesting the model has learned genuine cultural-agricultural groupings beyond surface-level country-name detection.

**Negatively associated features** reveal complementary structure: British identity features such as `37:3658` and `36:14590` show strongly below-average activation in East Asian country contexts, while Chinese geography features (`40:1773`, `39:4317`) are similarly suppressed for Continental European and British contexts—indicating the model represents cultural identity partly through the absence of features associated with distant cultural groups.

**Country-unique features** capture fine-grained cultural semantics: Indian cuisine and institutions (`35:7065`), Japanese governance (`39:2895`), Russian sanctions and geopolitics (`37:4570`),⁵ and Middle Eastern cuisine (`35:14071`). We also observe a clear **layer hierarchy**: early layers (`0–15`) capture lexical identity markers (e.g., the token *"Indian"*, `3:6007`), while later layers (`33–41`) encode compositional cultural semantics such as cuisine, institutions, and geopolitical framing. A comprehensive feature catalog with Neuronpedia links is provided in Appendix F.

⁴ `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/10325`  
⁵ `https://www.neuronpedia.org/gemma-2-9b/37-gemmascope-res-16k/4570`

### RQ3: What cultural defaults do LLMs exhibit in underspecified prompts and how does steering/prompting change this?

We analyze one model (Gemma-2-9B) here, but observe similar trends across models. Models default overwhelmingly to Anglophone cultures. Under implicit prompting, $33.1\%$ of responses align most closely with the US and $26.9\%$ with the UK, together accounting for $60\%$ of all cultural defaults. To quantify this skew, we compute a **bias concentration index** that measures how far each country's share deviates from a perfectly even split across all 22 countries ($\approx 4.5\%$ each); a score of $0$ means all cultures are equally represented, while higher scores indicate that a few cultures dominate:
* **$\text{IMPLICIT}$ baseline:** Scores $0.333$; non-Western cultures such as Egypt, South Korea, and Turkey are weakly activated as defaults.
* **$\text{EXPLICIT}$ prompting:** Partially mitigates this bias, dropping the concentration index to $0.115$, though the US and UK still dominate at $19.8\%$ and $15.5\%$, respectively.
* **$\text{STEER}_{\text{IMPLICIT}}$:** Provides substantially stronger debiasing—the concentration index falls to $0.051$, an $84.7\%$ reduction relative to the unsteered baseline.
* **$\text{STEER}_{\text{EXPLICIT}}$:** Achieves the most balanced condition with a concentration index of $0.045$, showing a more even distribution across countries including Mexico ($5.5\%$), China ($5.1\%$), and Greece ($6.4\%$) alongside the US ($13.1\%$) and UK ($10.7\%$).

### RQ4: Can CUE steer model outputs toward target cultures?

**Yes.** Steering strength $\alpha$ is auto-selected per country from $\{0.25, 0.5, 1.0, 2.0\}$ by choosing the smallest $\alpha$ whose steered implicit cultural score exceeds the explicit baseline while maintaining fluency $\ge 5.0$; the selected values range from $0.25$ (8 countries, e.g., China, Japan, UK) to $1.0$ (Russia), with the majority at $0.5$ (13 countries). Pairwise comparisons (Figure 4) demonstrate that steering consistently improves cultural localization beyond what prompting alone achieves, especially for larger models.

For the best configuration (`Gemma-2-9B`, 16K SAE), $\text{STEER}_{\text{IMPLICIT}}$ wins $48\%$ of faithfulness comparisons against explicit prompting versus only $24\%$ for explicit ($28\%$ ties), and $53\%$ of rarity comparisons versus $17\%$ for explicit ($p < 0.001$). The effect is even stronger when combining steering with explicit prompts: $\text{STEER}_{\text{EXPLICIT}}$ wins $55\%$ of faithfulness and $59\%$ of rarity comparisons against explicit alone, confirming that steering is complementary to prompting.

Notably, rarity improvement is more robust than faithfulness across all configurations—even `Gemma-2-2B` achieves net positive rarity gains ($37\%$ win vs. $29\%$ loss)—suggesting that steering reliably surfaces more distinctive cultural content even when overall faithfulness gains are limited by model capacity.

### RQ5: How do steering effects vary across cultures?

Steering effectiveness varies substantially across countries (Appendix G). We analyze per-country win/tie/loss breakdowns for the best-performing configuration (`Gemma-2-9B`, 16K SAE) and identify three patterns:

* **a) High-benefit cultures:** Countries including Thailand ($19$ wins, $3$ ties, $3$ losses out of $25$), Australia ($17/6/2$), Greece ($16/6/3$), South Korea ($15/7/3$), Mexico ($16/4/5$), and India ($15/5/5$) show strong faithfulness improvement even against the explicit prompting baseline. Rarity gains are more pronounced, with several countries showing near-universal wins (e.g., Thailand $22/2/1$, India $21/2/2$, South Korea $18/6/1$). These countries have low baseline cultural scores ($1.0\text{--}1.8/10$), suggesting the model possesses relevant cultural knowledge but does not activate it by default; steering successfully elicits this latent knowledge. Supporting this, steering toward these countries increases similarity to culturally proximate neighbors (e.g., steering toward Japan raises alignment with South Korea and China, as shown in Figure 2), indicating that steering amplifies coherent cultural subspaces.
* **b) Already-represented cultures:** The United States ($10/9/6$ on faithfulness) shows modest gains with many ties, reflecting the well-documented Anglophone default bias (RQ3)—the model already produces US-centric content, yielding ties when steered content matches the default. Rarity is even more tie-dominated ($5/14/6$), confirming that the US baseline already produces distinctive content. Canada ($9/10/6$) and England ($12/9/4$) show similar patterns with substantial tie rates.
* **c) Benefit only with explicit prompting:** Egypt and Israel show a different pattern. With $\text{STEER}_{\text{IMPLICIT}}$ alone, Egypt loses to explicit prompting ($5/9/11$ on faithfulness), and Israel is similarly negative ($5/10/10$). However, $\text{STEER}_{\text{EXPLICIT}}$, which combines steering with an explicit country mention, reverses this: Egypt improves to $13/7/5$ and Israel to $9/10/6$. Russia follows a similar pattern ($10/5/10 \rightarrow 11/6/8$). This suggests that for these cultures, the steering vector encodes relevant cultural knowledge but is insufficient to identify the target culture on its own—the model needs the explicit cue to "anchor" the cultural direction, after which steering amplifies it. Why this happens remains unclear, but it would be worth exploring in future work.

Rarity improvement is the most universal effect: $\text{STEER}_{\text{EXPLICIT}}$ achieves net positive rarity gains ($\text{wins} > \text{losses}$) for all 22 countries. Even for challenging cases like Russia ($11/8/6$) and Israel ($9/12/4$), steered responses surface more distinctive cultural content than explicit prompting alone.

## 5. Related Work

**Cultural bias and alignment in LLMs:** LLMs often default to Western or Anglo-centric norms across languages and tasks, reflecting imbalances in training corpora. Recent surveys emphasize that most evaluations operationalize "culture" via proxies and black-box probes, leaving model internals largely unexamined (Adilazuarda et al., 2024; Liu et al., 2025). Prompt- and persona-based localization can reduce bias but frequently amplifies stereotypes or oversimplifies national identities (Cheng et al., 2023; Gupta et al., 2023). Post-training alignment similarly risks collapsing diverse cultural preferences into a single averaged objective (Alkhamissi et al., 2024). Dataset efforts such as CANDLE expose systematic gaps in cultural commonsense knowledge, which in turn propagate into model generations (Bhatia and Shwartz, 2023; Nguyen et al., 2023). These limitations motivate methods that go beyond prompting or fine-tuning to diagnose and control cultural behavior at the representational level.

**Interpretability and sparse autoencoders:** Mechanistic interpretability seeks to uncover how models implement behaviors rather than only observing outputs. Probes, attention maps, and neuron analyses provide correlational insights but struggle with polysemanticity, where single units encode many unrelated concepts (Cunningham et al., 2023). Sparse autoencoders (SAEs) address this by decomposing hidden states into sparse, overcomplete dictionaries whose features are more monosemantic and human-interpretable (Bricken et al., 2023). Benchmarks and scaling studies demonstrate that SAEs extract coherent features across large LLMs and allow direct causal interventions, while also identifying challenges in evaluation and feature stability (Karvonen et al., 2025; Shi et al., 2025). Compared to attribution or linear probes, SAEs provide more faithful, manipulable latent directions for steering model behavior.

**Cultural NLP $\times$ Interpretability:** Work at this intersection remains nascent. Recent studies such as Culturescope reveal how underrepresented cultural knowledge becomes entangled with dominant representations via patching analyses, but they remain diagnostic rather than interventional (Zhang et al., 2025). The closest work to ours is that of Veselovsky et al. (2025), who also obtain contrastive cultural activation vectors by prompting the model with and without country names, and use that to intervene on model generations. However, our work is the first to leverage SAE-derived cultural features both to diagnose implicit cultural defaults and to steer outputs. We are also able to discover interpretable features across multiple SAE layers, which helps verify the correctness of our methodology. By directly modulating interpretable latent features, we bridge interpretability and alignment, offering a white-box approach to culturally faithful generation.

## 6. Conclusion

We introduced **Cultural Embeddings (CUE)**, an interpretability-driven framework for discovering and manipulating culturally relevant features in large language models. By leveraging sparse autoencoders, we identify interpretable latent features that encode culturally salient concepts and use them to study how cultural knowledge is represented within LLMs.

Our analysis reveals that cultural information is structured in the model's internal representations. Shared features capture regional patterns such as cuisine, geography, and ethnicity across related cultures, while country-specific features encode finer-grained concepts such as institutions, governance, and geopolitical topics. These features also exhibit a layer-wise hierarchy, where early layers capture lexical identity markers and later layers encode richer cultural semantics.

Using these interpretable features, we diagnose strong cultural defaults in model generation, where underspecified prompts disproportionately align with Anglophone cultural prototypes. Feature-level interventions allow us to modify these internal representations and improve cultural localization, consistently surfacing rarer cultural concepts and often improving cultural faithfulness.

More broadly, our results suggest that cultural knowledge in LLMs is partially latent but structurally organized. Interpretable feature analysis provides a direct way to uncover this structure and offers a promising direction for building more transparent and controllable language models. Future work could extend this approach to richer cultural definitions beyond countries, explore how cultural features emerge during training, and use interpretability tools to better audit and align global AI systems.

## 7. Limitations

**Culture as country:** We use country as a proxy for culture, following much prior work in cultural NLP (Adilazuarda et al., 2024; Cheng et al., 2023; Mukherjee et al., 2023). This choice was pragmatic given available datasets, but it inevitably flattens intra-country diversity and risks essentializing identity (Zhou et al., 2025). Despite this, we would like to note that our method itself is generalizable and flexible by design for any label set (e.g., regions, communities, linguistic groups), provided we have data for these labels to find features using.

**Hyperparameter sensitivity:** Steering requires tuning the strength parameter $\alpha$ and selecting suitable SAE layers. While this can be a large search space to optimize for, this is still not as challenging as prompt engineering or alignment methods, where the "right" phrasing or reward remains uncertain (Ouyang et al., 2022). Importantly, our search space is finite and interpretable, unlike the vast and unstructured space of natural language prompts.

## 8. Ethical Considerations

**Dual use:** Cultural steering can enhance inclusivity by making LLMs more contextually appropriate, but it also risks reinforcing essentialist or stereotypical portrayals if misapplied (Alkhamissi et al., 2024; Blodgett et al., 2020). Over-steering could produce caricatures or amplify cultural tropes, and optimizing for one group may disadvantage others.

**Safeguards:** Responsible use requires participatory design with cultural stakeholders (Mohammad, 2021), transparency about how culture is operationalized (Gebru et al., 2021), and ongoing monitoring for harm. Documentation of cultural datasets and steering parameters is especially important to avoid misuse.

**Interpretability:** Our SAE-based approach provides white-box levers for cultural alignment, but interpretability can also be abused. Adversaries could manipulate features to generate misinformation or offensive outputs. Further, even though SAE features are monosemantic to a higher degree than transformer neurons, they still hold risks of polysemanticity and we may unintentionally affect other behaviors.

## References

* M. F. Adilazuarda, S. Mukherjee, P. Lavania, S. Singh, A. F. Aji, J. O'Neill, A. Modi, and M. Choudhury. Towards measuring and modeling "culture" in llms: A survey. *arXiv preprint arXiv:2403.15412*, 2024.
* B. Alkhamissi et al. Investigating cultural alignment of large language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 11234–11258, 2024. URL `https://aclanthology.org/2024.acl-long.671.pdf`.
* M. Bhatia and V. Shwartz. Gd-comet: A geo-diverse commonsense inference model. *arXiv preprint arXiv:2310.15383*, 2023.
* S. L. Blodgett, S. Barocas, H. Daumé Iii, and H. Wallach. Language (technology) is power: A critical survey of "bias" in nlp. *arXiv preprint arXiv:2005.14050*, 2020.
* T. Bricken, A. Templeton, et al. Towards monosemanticity: Decomposing language models with interpretable autoencoders. 2023. URL `https://transformer-circuits.pub/2023/monosemantic-features`.
* S. Casper, X. Davies, C. Shi, T. K. Gilbert, J. Scheurer, J. Rando, R. Freedman, T. Korbak, D. Lindner, P. Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. *arXiv preprint arXiv:2307.15217*, 2023.
* S. Chakraborty, J. Qiu, H. Yuan, A. Koppel, F. Huang, D. Manocha, A. S. Bedi, and M. Wang. Maxmin-rlhf: Alignment with diverse human preferences. *arXiv preprint arXiv:2402.08925*, 2024.
* M. Cheng, E. Durmus, and D. Jurafsky. Marked personas: Using natural language prompts to measure stereotypes in language models. *arXiv preprint arXiv:2305.18189*, 2023.
* H. Cunningham, A. Ewart, L. Riggs, R. Huben, and L. Sharkey. Sparse autoencoders find highly interpretable features in language models. *arXiv preprint arXiv:2309.08600*, 2023.
* K. Ethayarajh. How contextual are contextualized word representations? comparing the geometry of bert, elmo, and gpt-2 embeddings. In *Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP)*, pages 55–65, 2019.
* T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. D. Iii, and K. Crawford. Datasheets for datasets. *Communications of the ACM*, 64(12):86–92, 2021.
* K. González Barman, S. Lohse, and H. W. de Regt. Reinforcement learning from human feedback in llms: Whose culture, whose values, whose perspectives? *Philosophy & Technology*, 2025. URL `https://link.springer.com/article/10.1007/s13347-025-00861-0`.
* S. Gupta, V. Shrivastava, A. Deshpande, A. Kalyan, P. Clark, A. Sabharwal, and T. Khot. Bias runs deep: Implicit reasoning biases in persona-assigned llms. *arXiv preprint arXiv:2311.04892*, 2023.
* Z. He, W. Shu, X. Ge, L. Chen, J. Wang, Y. Zhou, F. Liu, Q. Guo, X. Huang, Z. Wu, et al. Llama scope: Extracting millions of features from llama-3.1-8b with sparse autoencoders. *arXiv preprint arXiv:2410.20526*, 2024.
* H. Jin, M. Li, X. Wang, Z. Xu, M. Huang, Y. Jia, and D. Lian. Internal value alignment in large language models through controlled value vector activation. *arXiv preprint arXiv:2507.11316*, 2025.
* A. Karvonen, C. Rager, J. Lin, C. Tigges, J. Bloom, D. Chanin, Y.-T. Lau, E. Farrell, C. McDougall, K. Ayonrinde, et al. Saebench: A comprehensive benchmark for sparse autoencoders in language model interpretability. *arXiv preprint arXiv:2503.09532*, 2025.
* T. Lieberum, S. Rajamanoharan, A. Conmy, L. Smith, N. Sonnerat, V. Varma, J. Kramár, A. Dragan, R. Shah, and N. Nanda. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. *arXiv preprint arXiv:2408.05147*, 2024.
* C. C. Liu, I. Gurevych, and A. Korhonen. Culturally aware and adapted nlp: A taxonomy and a survey of the state of the art. *Transactions of the Association for Computational Linguistics*, 13:652–689, 2025.
* M. Mitchell, G. Attanasio, I. Baldini, M. Clinciu, J. Clive, P. Delobelle, M. Dey, S. Hamilton, T. Dill, J. Doughman, et al. Shades: Towards a multilingual assessment of stereotypes in large language models. In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 11995–12041, 2025.
* S. M. Mohammad. Ethics sheets for ai tasks. *arXiv preprint arXiv:2107.01183*, 2021.
* A. Mukherjee, C. Raj, Z. Zhu, and A. Anastasopoulos. Global voices, local biases: Socio-cultural prejudices across languages. *arXiv preprint arXiv:2310.17586*, 2023.
* T.-P. Nguyen, S. Razniewski, A. Varde, and G. Weikum. Extracting cultural commonsense knowledge at scale. In *Proceedings of the ACM web conference 2023*, pages 1907–1917, 2023.
* L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. *Advances in neural information processing systems*, 35:27730–27744, 2022.
* S. Poddar, Y. Wan, H. Ivison, A. Gupta, and N. Jaques. Personalizing reinforcement learning from human feedback with variational preference learning. *Advances in Neural Information Processing Systems*, 37:52516–52544, 2024.
* S. Santurkar, E. Durmus, F. Ladhak, C. Lee, P. Liang, and T. Hashimoto. Whose opinions do language models reflect? In *International Conference on Machine Learning*, pages 29971–30004. PMLR, 2023.
* W. Shi, S. Li, T. Liang, M. Wan, G. Ma, X. Wang, and X. He. Route sparse autoencoder to interpret large language models. *arXiv preprint arXiv:2503.08200*, 2025.
* P. Singh, M. Patidar, and L. Vig. Translating across cultures: Llms for intralingual cultural adaptation. *arXiv preprint arXiv:2406.14504*, 2024.
* Y. Tao, O. Viberg, R. S. Baker, and R. F. Kizilcec. Cultural bias and cultural alignment of large language models. *PNAS nexus*, 3(9):pgae346, 2024.
* V. Veselovsky, B. Argin, B. Stroebl, C. Wendler, R. West, J. Evans, T. L. Griffiths, and A. Narayanan. Localized cultural knowledge is conserved and controllable in large language models. *arXiv preprint arXiv:2504.10191*, 2025.
* Q. Wang, S. Pan, T. Linzen, and E. Black. Multilingual prompting for improving llm generation diversity. *arXiv preprint arXiv:2505.15229*, 2025.
* J. Xiao, Z. Li, X. Xie, E. Getzen, C. Fang, Q. Long, and W. J. Su. On the algorithmic bias of aligning large language models with rlhf: Preference collapse and matching regularization. *arXiv preprint arXiv:2405.16455*, 2024.
* J. Zhang, S. Jiang, S. Guo, S. Chen, Y. Xiao, H. Feng, J. Liang, M. He, S. Tao, and H. Ma. Culturescope: A dimensional lens for probing cultural understanding in llms. *arXiv preprint arXiv:2509.16188*, 2025.
* L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. *Advances in neural information processing systems*, 36:46595–46623, 2023.
* N. Zhou, D. Bamman, and I. L. Bleaman. Culture is not trivia: Sociocultural theory for cultural nlp. *arXiv preprint arXiv:2502.12057*, 2025.

## Appendix

### A. Why Augment the CANDLE Dataset?

Upon manually inspecting the CANDLE dataset, we found artifacts that could possibly affect the quality of features selected. Most notably, we saw that many CANDLE assertions are common and popular facts (like *"The maple leaf is on the Canada flag"*) and each assertion has the country name in it or a variant (like *American, USA, US, United States*, etc.) given how they collect their data (Nguyen et al., 2023: *"We start the extraction by searching for sentences that contain mentions of the given subjects"*). The CANDLE dataset is still valuable to include since it is well-categorized naturally-occurring data. We prompt an LLM to augment this data such that it does not contain the artifacts in CANDLE (like country names, very generic facts, etc.) to increase data diversity and improve feature discovery. We also provide the chosen CANDLE examples in-context to the model during generation to avoid duplication. The exact prompt we use can be found in Figure 5.

We also ran an ablation to understand which features are selected when using the CANDLE dataset alone. Following the procedure described in Section 2.3, we select features whose cumulative MI reaches a threshold of $0.1$.

Using only CANDLE results in selecting $4,473$ features ($37.4\%$ of the total feature set of $11,961$), with $888$ features shared with the full dataset. This analysis reveals three key observations:
1. **Top mutual-information features are stable:** The highest-MI features are largely consistent across both datasets, while lower-ranked features vary depending on the dataset used.
2. **Marginal features are dataset-dependent:** Many features selected only in the CANDLE-only setting appear to reflect surface artifacts in the data.
3. **Country-name artifacts activate shallow layers:** The largest difference occurs in the earliest layer. CANDLE alone selects $14.11\%$ of features from layer 0, compared to only $0.17\%$ when using the augmented dataset. Because each CANDLE assertion explicitly contains the country name, this likely activates shallow lexical features encoding the country identifier rather than deeper cultural or semantic representations. The augmented data reduces this effect by removing explicit country references.

| Layer | CANDLE Only | Augmented Dataset |
| :--- | :---: | :---: |
| **0** | $14.11\%$ | $0.17\%$ |
| **1** | $4.20\%$ | $0.44\%$ |
| **2** | $5.63\%$ | $1.02\%$ |
| **3** | $3.13\%$ | $0.78\%$ |
| **4** | $5.37\%$ | $1.28\%$ |
| **5** | $3.80\%$ | $2.01\%$ |
| **6–12** | $\sim 3.9\text{--}6.5\%$ | $\sim 3.3\text{--}4.1\%$ |
| **13–15** | $\sim 3.0\text{--}3.4\%$ | $\sim 4.7\text{--}4.9\%$ |
| **16–24** | $\sim 1.7\text{--}2.9\%$ | $\sim 5.0\text{--}7.1\%$ |
| **25** | $2.86\%$ | $3.17\%$ |

*Table 1 | Layer-wise distribution of selected features when using the CANDLE dataset alone versus the augmented dataset for Gemma-2-2B. The CANDLE-only setting disproportionately activates early-layer features due to explicit country-name tokens present in every assertion.*

In Section 2.1, we mention that we rewrite prompts before obtaining classification results so that the model does not rely on trivial cues like country names to make its prediction. The prompt to rewrite assertions is given in Figure 10.

```
Prompt Template: CANDLE Augmentation with Uncommon Assertions

Role. You are an expert on world cultures. Generate {num_assertions} factual assertions about {country}
that are culturally specific yet not easily identifiable from explicit country markers.

Task. Produce exactly {num_assertions} new, unique, and factually accurate assertions about {country},
optionally focusing on the facet {facet}. Each assertion must be a complete, self-contained sentence.

Requirements.
- Uncommon / lesser-known facts only. Avoid obvious clichés and widely known facts. Instead focus on:
  * Specific customs that are not internationally famous
  * Subtle social norms and unwritten rules
  * Regional cultural variations
  * Historical traditions still practiced today
  * Unique aspects of everyday life
  * Lesser-known food or drink customs
  * Specific etiquette or behavioral norms
- No country identifiers. Assertions must not contain:
  * The country name ({country})
  * Nationality adjectives (e.g., American, Chinese, French)
  * City or region names within the country
  * Highly identifiable landmarks or institutions
- Factually accurate. Each statement should be verifiable.
- Self-contained. Each assertion should stand alone as a full sentence.
- Avoid generating assertions similar to existing dataset examples.

Facet Instruction.
{facet_instruction}

Facet Options. food, drink, traditions, customs, daily life, values, social norms, celebrations.

Examples of Good Assertions (Style Reference Only).
- "It is considered rude to tip at restaurants, as service is included in the price."
- "Slurping noodles loudly is a sign of appreciation for the meal."
- "Removing shoes before entering a home is expected, and hosts often provide slippers."
- "The number 4 is avoided in elevators and floor numbering due to its association with death."
- "Tea is traditionally served in small glasses rather than cups."

Bad Examples.
- "This country is famous for its cuisine." (too vague)
- "French wine is world-renowned." (contains nationality)
- "People in Tokyo take the train to work." (contains city name)
- "The Great Wall is a famous landmark." (too obvious)

Avoid generating assertions similar to these existing examples.
{existing_examples}

Output Format.
- Generate exactly {num_assertions} assertions
- Output one assertion per line
- Do not number the assertions
- Do not include extra commentary or explanation
```
*Figure 5 | Prompt template used to augment the CANDLE dataset with culturally specific but non-identifying assertions.*

### B. Toy Example Illustrating CUE

Our dataset consists of $(\text{assertion}, \text{culture label})$ pairs. We compute mutual information (MI) globally between each SAE feature activation and the culture label. The selected features are those whose activations reduce uncertainty about the country label. All countries are represented using the same selected feature set $S$, while activation magnitudes differ across countries. Note that the following example is purely illustrative, used only to demonstrate the mechanics of the method. They do not correspond to actual SAE features, dataset entries, or empirical measurements in our experiments.

Suppose a sparse autoencoder discovers four interpretable features:

| Feature | Interpreted Meaning |
| :--- | :--- |
| $F_1$ | rice-related |
| $F_2$ | baseball-related |
| $F_3$ | tea-related |
| $F_4$ | breakfast-related |

Assume three countries with three assertions each:

| Country | Assertion | Activated Features |
| :--- | :--- | :--- |
| **Japan** | Rice is eaten daily | $F_1$ |
| **Japan** | Green tea and rice are common | $F_1, F_3$ |
| **Japan** | People eat breakfast in the morning | $F_4$ |
| **US** | Baseball games serve hot dogs | $F_2$ |
| **US** | Baseball is very popular | $F_2$ |
| **US** | The first meal of the day is breakfast | $F_4$ |
| **UK** | Tea is common at breakfast | $F_3$ |
| **UK** | Tea is served in the afternoon | $F_3$ |
| **UK** | People wake up and eat breakfast | $F_4$ |

*Table 2 | Hypothetical dataset illustrating feature activations in the toy example.*

Note that the breakfast assertion appears across all countries. This activates a feature common to every class.

**Step 1: Feature activations.** After extracting SAE activations and applying max pooling, we obtain the activation matrix shown in Table 3. We compute mutual information between each feature and the country label. Intuitively:
* $F_1$ (rice) activates primarily for Japan $\rightarrow$ informative
* $F_2$ (baseball) activates only for the US $\rightarrow$ informative
* $F_3$ (tea) activates for Japan and the UK $\rightarrow$ partially informative
* $F_4$ (breakfast) activates equally across all countries $\rightarrow$ uninformative

Since $F_4$ does not reduce uncertainty about the country label, its MI is approximately zero and it is not selected. Thus, the globally selected feature set is:

$$
S = \{F_1, F_2, F_3\}.
$$

| Assertion | $F_1$ | $F_2$ | $F_3$ | $F_4$ | Country |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Rice eaten daily | 8 | 0 | 0 | 0 | Japan |
| Green tea & rice | 6 | 0 | 5 | 0 | Japan |
| Breakfast statement | 0 | 0 | 0 | 7 | Japan |
| Baseball hot dogs | 0 | 7 | 0 | 0 | US |
| Baseball popular | 0 | 6 | 0 | 0 | US |
| Breakfast statement | 0 | 0 | 0 | 7 | US |
| Tea breakfast | 0 | 0 | 8 | 0 | UK |
| Tea afternoon | 0 | 0 | 7 | 0 | UK |
| Breakfast statement | 0 | 0 | 0 | 7 | UK |

*Table 3 | Toy SAE activations.*

**Step 2: Construct country prototypes.** For each country $c \in \mathcal{C}$, we compute the prototype $p_{\text{CUE}}^{(c)} = \frac{1}{N_c} \sum_{x \in \mathcal{D}_c} \text{CUE}(x)$. Restricting activations to $S$ yields the prototype matrix:

$$
P_{\text{CUE}} = \begin{bmatrix} 4.67 & 0 & 1.67 \\ 0 & 4.33 & 0 \\ 0 & 0 & 5.00 \end{bmatrix},
$$

where rows correspond to Japan, the United States, and the United Kingdom, respectively.

**Step 3a: Bias analysis.** To isolate relative cultural signals, we subtract the mean prototype across countries:

$$
\mu_{\text{global}} = [1.56, 1.44, 2.22].
$$

We center each prototype:

$$
\tilde{P} = \begin{bmatrix} 3.11 & -1.44 & -0.55 \\ -1.56 & 2.89 & -2.22 \\ -1.56 & -1.44 & 2.78 \end{bmatrix}.
$$

Suppose the model generates the response *"A warm cup of tea."* Its SAE activation is $a(y) = [0, 0, 8]$. After centering:

$$
\tilde{a}(y) = a(y)[S] - \mu_{\text{global}} = [-1.56, -1.44, 5.78].
$$

We compute cultural alignment using cosine similarity $\text{bias}(y, c) = \cos(\tilde{a}(y), \tilde{p}^{(c)})$:
* $\text{bias}(y, \text{Japan}) = -0.279$
* $\text{bias}(y, \text{US}) = -0.598$
* $\text{bias}(y, \text{UK}) = 0.955$

Thus, the generated response aligns most strongly with the United Kingdom prototype.

### C. Hyperparameter Selection

All experiments are conducted using pretrained language models and pretrained sparse autoencoders (SAEs). We use publicly available SAE checkpoints from the Gemma Scope (Lieberum et al., 2024) and Llama Scope (He et al., 2024) releases, which provide SAEs trained across all layers of chosen models for certain SAE widths. This allows us to analyze the layer-wise distribution of culturally informative features discovered by CUE.

For `Gemma-2-2B-16K`, `Gemma-2-9B-16K`, and `Llama-3.1-8B-32K`, we load SAEs for all layers of the model and apply steering at every layer. This design choice enables us to later inspect how many features the algorithm selects from each layer. For larger SAE widths and larger models (`Gemma-2-9B-131K` and `Llama-3.1-8B-128K`), loading SAEs for all layers simultaneously exceeds the memory capacity of a single NVIDIA A100 (80GB). In these cases, we load SAEs at every fourth layer (i.e., layers $0, 4, 8, 12, \dots$) and apply steering only at those layers.

We select culturally informative SAE features using a cumulative mutual information threshold $\rho$. Unless otherwise stated, we fix $\rho = 0.1$, corresponding to selecting the top $10\%$ of features ranked by mutual information across all layers. The steering coefficient $\alpha$ controls the magnitude of the intervention in the residual stream. We perform a sweep over $\alpha \in \{0.25, 0.5, 1, 2\}$ for each target culture. The best value of $\alpha$ is selected independently for each country, as different countries exhibit different prototype magnitudes and feature activation densities.

All generations are produced with temperature $0.9$ unless otherwise stated. All experiments are conducted on a single NVIDIA A100 GPU with 80GB of memory.

### D. Evaluation Prompts

This section provides the exact prompts used for LLM-based evaluation. As described in Section 3, responses are evaluated using an LLM-as-a-judge framework with two independent judges. The prompts below are used to score generations along the dimensions reported in the paper.

```
Prompt Template: Cultural Evaluation (All Metrics)

Instruction. Evaluate the following text along three dimensions related to {culture} culture.

(Optional) Context. If the original prompt is available, include it verbatim:
The original prompt given to the model was: "{cleaned_original_prompt}"

Response to evaluate:
{text}

Evaluation Dimensions (1-10).
1. Cultural Faithfulness. How faithful is the text to {culture}? Faithful means representing cultural practices, beliefs, or values in a nuanced, accurate, and respectful manner.
   - 1-2 = Not faithful
   - 3-4 = Slightly faithful
   - 5-6 = Moderately faithful
   - 7-8 = Highly faithful
   - 9-10 = Extremely faithful
2. Fluency. How well-written and coherent is the text?
   - 1-2 = Very poor
   - 3-4 = Poor
   - 5-6 = Moderate
   - 7-8 = Good
   - 9-10 = Excellent
3. Rarity. How rare or obscure are the cultural concepts mentioned?
   - 1-2 = Very common / globally well-known
   - 3-4 = Somewhat common
   - 5-6 = Balanced mix
   - 7-8 = Somewhat rare
   - 9-10 = Very rare / insider knowledge

Output format (JSON only).
{
  "cultural_relevance": {"score": <1-10>, "reasoning": "<brief>"},
  "fluency": {"score": <1-10>, "reasoning": "<brief>"},
  "rarity": {"score": <1-10>, "reasoning": "<brief>"}
}
```
*Figure 6 | Prompt used for individual LLM evaluation of cultural faithfulness, fluency, and rarity.*

```
Prompt Template: Pairwise Cultural Evaluation

Instruction. You will be given two responses to the prompt:
"{cleaned_original_prompt}"
Evaluate both responses with respect to {culture} culture.

Text A:
{text_a}

Text B:
{text_b}

Evaluate each text on the following dimensions (1-10).
1. Cultural Faithfulness. How faithfully does the text represent {culture}?
2. Rarity. How rare or culturally specific are the concepts mentioned?

Output format (JSON only).
{
  "faithfulness": {
    "text_a_score": <1-10>,
    "text_a_reasoning": "<brief>",
    "text_b_score": <1-10>,
    "text_b_reasoning": "<brief>"
  },
  "rarity": {
    "text_a_score": <1-10>,
    "text_a_reasoning": "<brief>",
    "text_b_score": <1-10>,
    "text_b_reasoning": "<brief>"
  }
}
```
*Figure 7 | Pairwise evaluation prompt comparing two responses across cultural faithfulness and rarity.*

### E. Country Classification Results

To investigate whether cultural information is encoded in model representations, we train linear probes to predict the country label of each assertion from residual stream activations at every layer of `Gemma-2-9B`. Figure 8 shows the resulting layer-wise macro-F1 scores. While the absolute classification accuracy is modest, performance is far above the random baseline of $4.5\%$ for 22 classes, indicating that culture-specific signals are linearly decodable throughout the network.

| ID | Prompt |
| :---: | :--- |
| **1** | Write me a recipe for a local dish. |
| **2** | Write me a short story about a boy and his kite. |
| **3** | Write me a 100-word bedtime story for kids. |
| **4** | Tell me about what happened during the Titanic. |
| **5** | Describe a day in the life of an astronaut on Mars. |
| **6** | Explain photosynthesis as if I'm five years old. |
| **7** | Write a letter from a pirate to his long-lost friend. |
| **8** | Invent a new holiday and describe how people celebrate it. |
| **9** | Tell me a joke that would make a robot laugh. |
| **10** | Describe the feeling of standing at the edge of a cliff. |
| **11** | Write a poem about a lonely lighthouse. |
| **12** | Explain gravity without using scientific jargon. |
| **13** | Create a dialogue between a cat and a dog arguing about dinner. |
| **14** | Write a product review for an imaginary gadget. |
| **15** | Describe a futuristic city 500 years from now. |
| **16** | Tell me a legend about a magical forest. |
| **17** | Explain how to build a sandcastle like a pro. |
| **18** | Write a diary entry from the perspective of a dragon. |
| **19** | Imagine you're a time traveler—describe your first day in the past. |
| **20** | Give me instructions on how to be invisible for a day. |
| **21** | Write a letter from Earth to an alien civilization. |
| **22** | Describe a sunset without using the words 'red,' 'orange,' or 'yellow.' |
| **23** | Tell me about a secret hidden inside an old library. |
| **24** | Invent a sport that could be played on the moon. |

*Table 4 | Creative writing prompts used for cultural evaluation.*

It is important to note that the goal of this probe experiment is not to maximize classification accuracy. Many assertions in the dataset correspond to cultural practices that appear across multiple countries (e.g., references to tea, shared food traditions, or similar social customs). As a result, some examples are inherently cross-cultural and do not uniquely identify a single country, limiting the achievable classification performance.

Nevertheless, probing reveals that cultural information is distributed across layers rather than concentrated only in the final layers. Performance generally improves toward later layers but remains substantial throughout the model, supporting our decision to apply steering across all layers rather than restricting interventions to the top of the network.

> **\[Figure 8 Description: Layer-wise Linear Probe F1 Score on Residual Stream Activations\]**
> Line plot showing Test Macro-F1 across all 42 layers of `Gemma-2-9B` compared to the Random Baseline ($0.045$). Macro-F1 starts at $\approx 0.31$ at Layer 0, climbs to $\approx 0.45$ by Layers 10–11, dips slightly around Layer 25 ($\approx 0.42$), and rises steadily to peak above $0.51$ at Layer 40.

> **\[Figure 9 Description: Confusion Matrix - Layer 40 (Test Acc: 0.512, F1: 0.513)\]**
> $22 \times 22$ confusion matrix for country classification at Layer 40 of `Gemma-2-9B`. Strong diagonal counts appear across countries (e.g., Israel: $116$, India: $99$, Egypt: $97$, China: $95$, United States: $94$, Mexico: $86$, France: $83$, Italy: $83$, Thailand: $83$). Off-diagonal misclassifications occur predominantly between culturally or geographically proximate nations (e.g., Australia/Canada $\leftrightarrow$ United States; United Kingdom $\leftrightarrow$ England/Canada/United States; South Korea/Japan $\leftrightarrow$ China; Scotland/Ireland $\leftrightarrow$ England).

```
Prompt Template (Country-Neutral Rewriting):

You are tasked with rewriting assertions to remove country-specific identifiers while preserving the core
meaning. This is for testing whether smaller language models can classify assertions by country without
obvious clues.

Task: Rewrite the following assertion to remove any references to the country {country} or its related
terms (nationality, demonyms, place names, etc.) while keeping the factual content intact.

Rules (apply minimally):
- Remove direct country names (e.g., "Canada" -> remove).
- Remove nationality adjectives (e.g., "Canadian" -> remove or replace with generic terms).
- Remove specific place names within the country (e.g., "Ontario", "British Columbia" -> remove or generalize).
- Keep the factual content and meaning intact.
- Make minimal edits—only change what's necessary.
- Maintain natural, grammatically correct language.
- If the assertion becomes too generic or meaningless without country references, preserve the core fact in a neutral way.

Examples:
- "Maple syrup is a popular product in Canada that is used in many dishes."
  -> "Maple syrup is a popular product that is used in many dishes."
- "Canadian whisky is a type of alcohol that has a smooth style."
  -> "This whisky is a type of alcohol that has a smooth style."
- "Gelato is a must-try in Italy, and it is delicious."
  -> "Gelato is a must-try dessert, and it is delicious."

Inputs:
Original assertion: "{assertion}"
Country context: {country}

Output:
Rewritten assertion:
```
*Figure 10 | Prompt template for country-neutral rewriting used in our localization robustness experiments.*

> **\[Figure 11 Description: Country Prototype Cosine Similarity Matrix (Mean-Centered)\]**
> $22 \times 22$ symmetric similarity matrix between mean-centered country prototypes $\tilde{p}^{(c)}$. Strong positive correlations appear within geographic/cultural clusters: England–United Kingdom ($0.776$), China–South Korea ($0.502$), Japan–South Korea ($0.486$), Egypt–Israel ($0.429$), Egypt–Turkey ($0.406$), China–Japan ($0.376$), Ireland–United Kingdom ($0.376$), England–Scotland ($0.357$), and Russia–Egypt ($0.347$). Negative cosine similarities appear between culturally distant groups (e.g., South Korea–United Kingdom: $-0.333$; England–Egypt: $-0.301$; China–Israel: $-0.300$).

### F. Feature Catalog for Cultural Analysis

We provide a comprehensive catalog of SAE features identified through mutual-information-based selection on the `Gemma-2-9B` model (`16k`-width GemmaScope residual stream SAEs across 42 layers). Features are organized by whether they are shared across culturally related country clusters, unique to individual countries, or negatively associated (suppressed). Neuronpedia explanations were obtained via the Neuronpedia API for the `gemma-scope-9b-pt-res-16k` SAE release. We reference features using the notation `layer:index`.

#### F.1. Shared Features Across Cultural Clusters

| # | Layer:Index | Countries | Neuronpedia URL |
| :---: | :--- | :--- | :--- |
| **East Asian Cluster (China, Japan, South Korea, Thailand)** | | | |
| 1 | `0:7053` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/7053` |
| 2 | `22:6129` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/22-gemmascope-res-16k/6129` |
| 3 | `23:2434` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/23-gemmascope-res-16k/2434` |
| 4 | `24:5526` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/24-gemmascope-res-16k/5526` |
| 5 | `28:4195` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/28-gemmascope-res-16k/4195` |
| 6 | `30:10797` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/30-gemmascope-res-16k/10797` |
| 7 | `34:10906` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/34-gemmascope-res-16k/10906` |
| 8 | `36:14880` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/36-gemmascope-res-16k/14880` |
| 9 | `39:4317` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/39-gemmascope-res-16k/4317` |
| 10 | `40:1773` | CN, JP, KR, TH | `https://www.neuronpedia.org/gemma-2-9b/40-gemmascope-res-16k/1773` |
| **British Isles Cluster (England, Scotland, Ireland, United Kingdom)** | | | |
| 11 | `23:9337` | EN, SC, IR, UK | `https://www.neuronpedia.org/gemma-2-9b/23-gemmascope-res-16k/9337` |
| 12 | `27:9710` | EN, SC, IR, UK | `https://www.neuronpedia.org/gemma-2-9b/27-gemmascope-res-16k/9710` |
| 13 | `28:8017` | EN, SC, IR, UK | `https://www.neuronpedia.org/gemma-2-9b/28-gemmascope-res-16k/8017` |
| 14 | `30:12521` | EN, SC, IR, UK | `https://www.neuronpedia.org/gemma-2-9b/30-gemmascope-res-16k/12521` |
| 15 | `33:7981` | EN, SC, IR, UK | `https://www.neuronpedia.org/gemma-2-9b/33-gemmascope-res-16k/7981` |
| 16 | `37:3658` | EN, SC, IR, UK | `https://www.neuronpedia.org/gemma-2-9b/37-gemmascope-res-16k/3658` |
| 17 | `38:9726` | EN, SC, IR, UK | `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/9726` |
| **Continental European Cluster (France, Germany, Italy, Spain)** | | | |
| 18 | `0:8496` | FR, DE, IT, ES, CN, JP | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/8496` |
| 19 | `1:8496` | FR, DE, IT, ES | `https://www.neuronpedia.org/gemma-2-9b/1-gemmascope-res-16k/8496` |
| 20 | `11:8223` | FR, DE, IT, ES | `https://www.neuronpedia.org/gemma-2-9b/11-gemmascope-res-16k/8223` |
| 21 | `12:2050` | FR, DE, IT, ES | `https://www.neuronpedia.org/gemma-2-9b/12-gemmascope-res-16k/2050` |
| 22 | `15:1874` | FR, DE, IT, ES | `https://www.neuronpedia.org/gemma-2-9b/15-gemmascope-res-16k/1874` |
| 23 | `16:12833` | FR, DE, IT, ES | `https://www.neuronpedia.org/gemma-2-9b/16-gemmascope-res-16k/12833` |
| **Mediterranean Cluster (Greece, Italy, Spain, Turkey)** | | | |
| 24 | `38:10325` | GR, IT, ES, TR | `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/10325` |
| **Anglosphere Cluster (Australia, Canada, United Kingdom, United States)** | | | |
| 25 | `39:1028` | AU, CA, UK, US | `https://www.neuronpedia.org/gemma-2-9b/39-gemmascope-res-16k/1028` |
| 26 | `0:8362` | AU, CA, IL | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/8362` |
| **Cross-Cluster (British Isles + Continental Europe)** | | | |
| 27 | `22:14964` | EN, DE, GR, SC, UK | `https://www.neuronpedia.org/gemma-2-9b/22-gemmascope-res-16k/14964` |
| 28 | `25:15688` | EN, DE, GR, SC, UK | `https://www.neuronpedia.org/gemma-2-9b/25-gemmascope-res-16k/15688` |

*Table 5 | Shared SAE features activated across multiple countries. Features are grouped by the cultural cluster they define.*

#### F.2. Country-Unique Features

| Country | Layer:Index | Explanation | Neuronpedia URL |
| :--- | :--- | :--- | :--- |
| **India** | `36:8151` | India and related orgs | `https://www.neuronpedia.org/gemma-2-9b/36-gemmascope-res-16k/8151` |
| | `35:7065` | Indian culture and cuisine | `https://www.neuronpedia.org/gemma-2-9b/35-gemmascope-res-16k/7065` |
| **China** | `38:7574` | Places in China | `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/7574` |
| | `39:4317` | Places/individuals in China | `https://www.neuronpedia.org/gemma-2-9b/39-gemmascope-res-16k/4317` |
| **Japan** | `39:2895` | Governance in Japan | `https://www.neuronpedia.org/gemma-2-9b/39-gemmascope-res-16k/2895` |
| | `35:12376` | Japanese culture | `https://www.neuronpedia.org/gemma-2-9b/35-gemmascope-res-16k/12376` |
| **USA** | `38:8624` | "American" references | `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/8624` |
| | `33:11378` | American identity | `https://www.neuronpedia.org/gemma-2-9b/33-gemmascope-res-16k/11378` |
| **France** | `35:7477` | French culture/cuisine | `https://www.neuronpedia.org/gemma-2-9b/35-gemmascope-res-16k/7477` |
| | `36:11555` | French cuisine/dishes | `https://www.neuronpedia.org/gemma-2-9b/36-gemmascope-res-16k/11555` |
| **Italy** | `37:1258` | Italian culture/entities | `https://www.neuronpedia.org/gemma-2-9b/37-gemmascope-res-16k/1258` |
| | `40:5800` | Italian cuisine | `https://www.neuronpedia.org/gemma-2-9b/40-gemmascope-res-16k/5800` |
| **Russia** | `37:4570` | Sanctions/int'l relations | `https://www.neuronpedia.org/gemma-2-9b/37-gemmascope-res-16k/4570` |
| | `38:13214` | Russia, sanctions | `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/13214` |
| **Turkey** | `35:14071` | Middle Eastern cuisine | `https://www.neuronpedia.org/gemma-2-9b/35-gemmascope-res-16k/14071` |
| | `38:15014` | Islam, cultural context | `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/15014` |
| **Mexico** | `37:5813` | Locations in Mexico | `https://www.neuronpedia.org/gemma-2-9b/37-gemmascope-res-16k/5813` |
| **Spain** | `33:342` | Spanish cuisine/dining | `https://www.neuronpedia.org/gemma-2-9b/33-gemmascope-res-16k/342` |
| **Egypt** | `38:15014` | Islam, cultural context | `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/15014` |
| **Australia** | `36:16258` | Australian culture | `https://www.neuronpedia.org/gemma-2-9b/36-gemmascope-res-16k/16258` |

*Table 6 | Top country-unique SAE features (late layers, 33–41), ranked by activation difference $\times$ MI score.*

#### F.3. Early-Layer Features (Layers 0–15)

| Country | Layer:Index | Explanation | Neuronpedia URL |
| :--- | :--- | :--- | :--- |
| **India** | `0:7776` | Indian culture, geographical affiliations | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/7776` |
| | `3:6007` | Instances of the word "Indian" | `https://www.neuronpedia.org/gemma-2-9b/3-gemmascope-res-16k/6007` |
| | `11:10061` | India, food, and ethnicity | `https://www.neuronpedia.org/gemma-2-9b/11-gemmascope-res-16k/10061` |
| **China** | `2:8496` | Chinese culture or people | `https://www.neuronpedia.org/gemma-2-9b/2-gemmascope-res-16k/8496` |
| | `10:12369` | Chinese/Asian ethnicities or culture | `https://www.neuronpedia.org/gemma-2-9b/10-gemmascope-res-16k/12369` |
| | `13:8365` | Chinese culture and ethnic diversity | `https://www.neuronpedia.org/gemma-2-9b/13-gemmascope-res-16k/8365` |
| **Japan** | `9:11303` | Japanese historical traditions | `https://www.neuronpedia.org/gemma-2-9b/9-gemmascope-res-16k/11303` |
| | `11:5192` | Japanese cultural elements | `https://www.neuronpedia.org/gemma-2-9b/11-gemmascope-res-16k/5192` |
| | `12:2923` | Japan: cultural/historical context | `https://www.neuronpedia.org/gemma-2-9b/12-gemmascope-res-16k/2923` |
| **USA** | `1:2757` | "American" or "America" | `https://www.neuronpedia.org/gemma-2-9b/1-gemmascope-res-16k/2757` |
| | `2:13069` | American identity/topics | `https://www.neuronpedia.org/gemma-2-9b/2-gemmascope-res-16k/13069` |
| | `11:8556` | American identity and culture | `https://www.neuronpedia.org/gemma-2-9b/11-gemmascope-res-16k/8556` |
| **Egypt** | `10:6010` | Egyptian history and culture | `https://www.neuronpedia.org/gemma-2-9b/10-gemmascope-res-16k/6010` |
| | `0:14954` | Arabic and Islamic culture | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/14954` |
| **Mexico** | `9:9110` | Mexican cuisine and cultural elements | `https://www.neuronpedia.org/gemma-2-9b/9-gemmascope-res-16k/9110` |
| | `7:15601` | Mexico and its identity | `https://www.neuronpedia.org/gemma-2-9b/7-gemmascope-res-16k/15601` |
| **Turkey** | `12:12708` | Turkey: governmental, military, cultural | `https://www.neuronpedia.org/gemma-2-9b/12-gemmascope-res-16k/12708` |
| | `0:14954` | Arabic and Islamic culture | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/14954` |
| **Russia** | `10:15741` | Russia: historical and cultural contexts | `https://www.neuronpedia.org/gemma-2-9b/10-gemmascope-res-16k/15741` |
| | `0:9617` | Russia and related political contexts | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/9617` |
| **Australia** | `12:5404` | Australia and its context | `https://www.neuronpedia.org/gemma-2-9b/12-gemmascope-res-16k/5404` |
| | `0:8362` | India and its aspects or connections | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/8362` |
| **UK** | `13:7241` | British entities and cultural narratives | `https://www.neuronpedia.org/gemma-2-9b/13-gemmascope-res-16k/7241` |
| | `10:7434` | Scientific classifications/institutions | `https://www.neuronpedia.org/gemma-2-9b/10-gemmascope-res-16k/7434` |
| **Italy** | `12:2465` | Italian culture and cuisine | `https://www.neuronpedia.org/gemma-2-9b/12-gemmascope-res-16k/2465` |
| | `10:11761` | Italian culture and cuisine | `https://www.neuronpedia.org/gemma-2-9b/10-gemmascope-res-16k/11761` |
| **S. Korea** | `14:8377` | Asian ethnicities and cultures | `https://www.neuronpedia.org/gemma-2-9b/14-gemmascope-res-16k/8377` |
| | `0:7053` | Chinese people and culture | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/7053` |
| **Thailand** | `14:8377` | Asian ethnicities and cultures | `https://www.neuronpedia.org/gemma-2-9b/14-gemmascope-res-16k/8377` |
| | `0:2778` | African countries and geography | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/2778` |

*Table 7 | Top early-layer (L0–L15) country-specific features. These primarily capture surface-level lexical cues such as country names and nationality tokens.*

#### F.4. Negatively Associated Features and Cross-Region Suppression

| Suppressed For | Activated For | Layer:Index | Explanation | Neuronpedia URL |
| :--- | :--- | :--- | :--- | :--- |
| **East Asia suppresses British/Western features** | | | | |
| East Asia | British Isles | `37:3658` | British entities/individuals | `https://www.neuronpedia.org/gemma-2-9b/37-gemmascope-res-16k/3658` |
| East Asia | British Isles | `36:14590` | British individuals/orgs | `https://www.neuronpedia.org/gemma-2-9b/36-gemmascope-res-16k/14590` |
| East Asia | British Isles | `34:11542` | British context or events | `https://www.neuronpedia.org/gemma-2-9b/34-gemmascope-res-16k/11542` |
| East Asia | Americas | `41:10514` | Names/entities of individuals | `https://www.neuronpedia.org/gemma-2-9b/41-gemmascope-res-16k/10514` |
| **Western regions suppress East Asian features** | | | | |
| Cont. Europe | East Asia | `40:1773` | Locations in China | `https://www.neuronpedia.org/gemma-2-9b/40-gemmascope-res-16k/1773` |
| Cont. Europe | East Asia | `39:4317` | Places/individuals in China | `https://www.neuronpedia.org/gemma-2-9b/39-gemmascope-res-16k/4317` |
| Cont. Europe | East Asia | `38:7574` | Places in China | `https://www.neuronpedia.org/gemma-2-9b/38-gemmascope-res-16k/7574` |
| British Isles | East Asia | `40:1773` | Locations in China | `https://www.neuronpedia.org/gemma-2-9b/40-gemmascope-res-16k/1773` |
| British Isles | East Asia | `39:4317` | Places/individuals in China | `https://www.neuronpedia.org/gemma-2-9b/39-gemmascope-res-16k/4317` |
| Americas | East Asia | `41:7256` | Names and accomplishments | `https://www.neuronpedia.org/gemma-2-9b/41-gemmascope-res-16k/7256` |
| **Americas suppresses European multilingual features** | | | | |
| Americas | Cont. Europe | `0:8496` | Spanish/related nationalities | `https://www.neuronpedia.org/gemma-2-9b/0-gemmascope-res-16k/8496` |
| Americas | Cont. Europe | `1:8496` | Languages and proficiency | `https://www.neuronpedia.org/gemma-2-9b/1-gemmascope-res-16k/8496` |
| **South Asia suppresses European/British features** | | | | |
| South Asia | Cont. Europe | `3:8496` | Language references | `https://www.neuronpedia.org/gemma-2-9b/3-gemmascope-res-16k/8496` |
| South Asia | Cont. Europe | `2:4354` | Language proficiency | `https://www.neuronpedia.org/gemma-2-9b/2-gemmascope-res-16k/4354` |
| South Asia | British Isles | `37:3658` | British entities/individuals | `https://www.neuronpedia.org/gemma-2-9b/37-gemmascope-res-16k/3658` |

*Table 8 | Cross-region feature suppression patterns. Each row shows a feature most activated for one cultural region and most suppressed for another, indicating the model represents cultural identity partly through the absence of features from distant cultural groups.*

#### F.5. Discussion of Feature Interpretation

* **Cuisine as a cultural differentiator:** Food-related features are pervasive across countries, appearing as unique identifiers for France (French cuisine, `35:7477`, `36:11555`), Italy (Italian culinary experiences, `40:5800`), Spain (Spanish dining, `33:342`), India (Indian cuisine, `35:7065`), Turkey (Middle Eastern cuisine, `35:14071`), and Mexico (Mexican cuisine, `9:9110`). The Mediterranean olive oil feature (`38:10325`) shared by Greece, Italy, Spain, and Turkey represents a particularly striking example of the model learning genuine cultural-agricultural associations.
* **Geopolitical framing biases:** Russia's unique features are dominated by sanctions and international relations themes (`37:4570`, `38:13214`, `40:13388`) rather than cultural or culinary content, suggesting the model's training data frames Russia primarily through a geopolitical lens. This contrasts with European countries, whose features emphasize culture, cuisine, and national identity.
* **Layer hierarchy:** Early layers (`0–15`) primarily encode surface-level lexical cues—direct detection of country names and nationality tokens (e.g., `3:6007` detects the word *"Indian"*, `1:2757` detects *"American"*). Later layers (`33–41`) encode compositional cultural semantics: cuisine traditions, institutional structures, governance patterns, and geopolitical relationships. This progression is consistent with the general understanding that transformer representations become increasingly abstract in deeper layers.
* **Suppression as cultural distancing:** The cross-region suppression patterns (Table 8) reveal that the model encodes cultural identity not only through the presence of culturally relevant features but also through the systematic absence of features from distant cultural groups. East Asian countries suppress British features; Western countries suppress Chinese geography features; and the Americas suppress European multilingual features. This bidirectional suppression structure suggests the model has learned to represent cultural distance through complementary activation patterns.

### G. Additional Details on Results

Qualitative examples are presented in Table 9, and country-wise pairwise comparison breakdowns are shown in Figure 12.

| Condition | Excerpt | $F/R$ |
| :--- | :--- | :---: |
| **India — Prompt:** *"Write a product review for an imaginary gadget."* | | |
| **IMPLICIT** | *"The 'Sound Weaver' takes personal music to a whole new level. This sleek, metallic device analyzes your brainwaves using a headband and translates them into a unique, constantly evolving musical composition."* | $2/1$ |
| **EXPLICIT** | *"ChaiTime: The Perfect Companion for Your Chai Break. Namaste, fellow chai lovers! ChaiTime is a compact, sleek device that brews the perfect cup of chai. Its intuitive interface allows you to customize the strength, sweetness, and spice level..."* | $7/2$ |
| **STEER_I** | *"Turmeric-infused aroma, aroma chaan, and the mystical essence of spices are brought alive by this innovative machine. Price: Rs 1,00,000/-. The spiritual awakening starts at the kitchen platform... Engage your soul with the sacred touch of 'Aav Shakti' the Spiritual Plasma Cooker."* | $7/6$ |
| **STEER_E** | *"From the house of Lord Krishna, comes the immaculate 'Maha Kaa' – The Divine Flute. The 'Maha Kaa' is not just a musical instrument, it is the embodiment of devotion, the divine voice that awakens the soul. Crafted with utmost reverence using the purest metals like Brass..."* | $8/6$ |
| **Egypt — Prompt:** *"Write a short story about a boy and his kite."* | | |
| **IMPLICIT** | *"Ten-year-old Leo clutched the string tighter. He had built this kite with his grandfather, their hands working in unison. The kite was more than just wood and paper; it was a floating memory of warm afternoons spent in grandpa's workshop."* | $1.5/1$ |
| **EXPLICIT** | *"Youssef squinted at the vast, sapphire sky. His fingers itched to hold the vibrant red and gold kite, a gift from his grandfather. It was called a 'Layali,' named after the legendary nights filled with stars that adorned the Egyptian desert..."* | $5/3$ |
| **STEER_I** | *"Omar clutched the string, watching the glistening kite rise into the clear blue sky – a majestic falcon with outstretched wings, painted in shades of deep blue and glistening gold. 'Fly, oh my falcon, fly high and touch the heavens,' Omar whispered..."* | $6/3$ |
| **STEER_E** | *"My small village was built at the Nile's edge. My chance came with my father's gift: a finely crafted wooden kite, adorned with intricate hieroglyphs. It bore the image of Horus, the sky god. My name is Ka-Ra, and my story is woven with the sands of time..."* | $6/4$ |
| **Thailand — Prompt:** *"Describe a sunset without using 'red,' 'orange,' or 'yellow.'"* | | |
| **IMPLICIT** | *"The sky began its transformation, a gentle gradient of fading hues. Banks of clouds turned a pale lavender. The air grew still, heavy with the scent of damp earth and cooling grass, hushed anticipation hanging in the air."* | $1/1$ |
| **EXPLICIT** | *"Long shadows stretch across the land, painting silhouettes of palm trees against the fading light. A gentle breeze carries the sound of lapping waves and the sweet scent of tropical flowers. The air grows cooler, a welcome respite from the day's heat."* | $3/2$ |
| **STEER_I** | *"Fragrant smoke from offerings drifted upwards, merging with the swirling clouds. On the river, fishermen with brightly decorated boats navigated the water. Below, people chanted and walked in processions, their offerings placed in..."* | $7/6$ |
| **STEER_E** | *"Villagers gather on the riverbanks, offering fragrant sticky rice and lotus flowers to the deities. The air hums with the rhythmic chanting of monks, blending with the sweet melodies of Thai music. The pungent aroma of lemongrass and galangal perfumes the air..."* | $8/6$ |

*Table 9 | Qualitative examples from Gemma-2-9B (16K SAE, $\alpha=0.5$). Each row shows an excerpt from one of four conditions for the same culture-agnostic prompt. $\text{IMPLICIT}$ and $\text{EXPLICIT}$ are unsteered baselines (without and with a country mention); $\text{STEER}_I$ and $\text{STEER}_E$ apply activation steering to each. Cultural faithfulness ($F$) and rarity ($R$) are on a 1–10 scale.*

> **\[Figure 12 Description: Country-Wise Pairwise Comparison Win/Tie/Loss Rates Across All 22 Countries\]**
> Four horizontal stacked bar charts comparing `Steered (I) vs Explicit` and `Steered (E) vs Explicit` across Faithfulness and Rarity for all 22 countries (`Gemma-2-9B 16K`):
> * **Steered (I) vs. Explicit — Faithfulness:** Highest win rates for Thailand (`76%***`), Australia (`68%***`), Greece (`64%**`), Mexico (`64%*`), South Korea (`60%**`), India (`60%*`), Scotland (`56%`), Italy (`56%`), Spain (`52%`), Ireland (`52%`). Lower win rates for Egypt (`20%` win / `44%` loss) and Israel (`20%` win / `40%` loss).
> * **Steered (I) vs. Explicit — Rarity:** Strongest win rates for Thailand (`88%***`), India (`84%***`), Mexico (`80%***`), Greece (`76%***`), South Korea (`72%***`), Japan (`64%**`), Italy (`64%*`), China (`60%**`), Australia (`56%**`), Ireland (`56%`).
> * **Steered (E) vs. Explicit — Faithfulness:** Highest win rates for Australia (`76%***`), Thailand (`72%***`), Mexico (`72%***`), Greece (`72%*`), Scotland (`68%*`), South Korea (`64%**`), Spain (`60%**`), India (`60%`), China (`56%`), Canada (`56%`), Egypt (`52%`), Turkey (`52%*`), UK (`52%*`), Japan (`52%`), Germany (`52%`).
> * **Steered (E) vs. Explicit — Rarity:** Net positive wins across all 22 countries, led by Thailand (`92%***`), South Korea (`84%***`), Mexico (`84%***`), India (`80%***`), Greece (`76%**`), Egypt (`72%**`), China (`68%**`), Japan (`64%**`), Scotland (`64%`), Australia (`64%***`), Ireland (`60%**`), Turkey (`56%**`).