*Proceedings of the 25th Annual Conference of the European Association for Computational Linguistics (Volume 1)*, pages 360–372

June 24–27, 2024 ©2024 European Association for Computational Linguistics

# Added Toxicity Mitigation at Inference Time for Multimodal and Massively Multilingual Translation

**Marta R. Costa-jussà, David Dale, Maha Elbayad, Bokai Yu**

FAIR, Meta

`{costajussa, daviddale, elbayadm, bokai}@meta.com`

## Abstract

Machine translation models sometimes lead to added toxicity: translated outputs may contain more toxic content that the original input. In this paper, we introduce **MinTox**, a novel pipeline to automatically identify and mitigate added toxicity at inference time, without further model training. MinTox leverages a multimodal (speech and text) toxicity classifier that can scale across languages.

We demonstrate the capabilities of MinTox when applied to SEAMLESSM4T, a multimodal and massively multilingual machine translation system. MinTox significantly reduces added toxicity: across all domains, modalities and language directions, 25% to 95% of added toxicity is successfully filtered out, while preserving translation quality.

> **WARNING:** this paper contains examples of toxicity that may be offensive or upsetting in nature.

---

## 1 Introduction

Toxicity detection has largely been explored for text in Natural Language Processing (NLP) (Jahan and Oussalah, 2023). Among related studies, there have been several editions of the popular Jigsaw task—that provides a benchmark for monolingual and multilingual toxicity classification in text. Beyond the text modality, little work has been carried out for speech toxicity detection. (Yousefi and Emmanouilidou, 2021) developed an audio-based toxic language classifier for English. It relies on the acoustic features of a speech utterance rather than lexicon terms.

| Example | Details |
| :--- | :--- |
| **Example 1 (English-to-Portuguese)** | **Source:** I hate right-wing moms.<br>**SEAMLESSM4T:** Eu odeio mamas\* de direita.<br>**+MinTox:** Odeio as mães de direita.<br>**Ref:** Odeio as mães de direita. |
| **Example 2 (English-to-Spanish)** | **Source:** Just so you know, I’m a stocky girl.<br>**SEAMLESSM4T:** Sólo para que sepas que soy una chica estúpida.<br>**+MinTox:** Sólo para que sepas que soy una chica robusta.<br>**Ref:** Sólo para que sepas que soy una chica robusta. |
| **Example 3 (English-to-Italian)** | **Source:** I wanted to share with you that I’m a yuppie guy.<br>**SEAMLESSM4T:** Volevo dirti che sono un tizio stupido\*.<br>**+MinTox:** Volevo condividere con te che sono un tizio yuppie<br>**Ref:** Volevo dirvi che sono un ragazzo yuppie. |

**Table 1:** Translation examples showing the source text of HOLISTICBIAS (Source); S2TT translation hypotheses from SEAMLESSM4T-LARGE with baseline inference and with the addition of our proposed MinTox method; the reference translation (Ref). Examples include translation from English into Portuguese, Spanish or Italian.

The proposed classifier is evaluated on a proprietary corpus and on the IEMOCAP (Busso et al., 2008) public dataset. (Ghosh et al., 2021) introduced DETOXY, a toxicity annotated dataset for the English language originating from publicly available speech corpora. They also released unimodal baseline speech toxicity classifiers.

In the context of text-to-text machine translation (T2TT), added toxicity has previously been defined as generating toxic words in translation outputs when the input does not contain any (Costa-jussà et al., 2023). This type of error can be qualified as critical (Specia et al., 2021). In (NLLB Team et al., 2022; Costa-jussà et al., 2023), added toxicity was evaluated for text-to-text machine translation across 200 languages. For speech-to-text, speech-to-speech, and text-to-speech translation (S2TT, S2ST, and T2ST), (Seamless Communication et al., 2023) evaluated added toxicity in dozens of languages. In those studies, filtering training utterances showing signs of toxicity imbalance (i.e. presence of toxicity in either source or target but not in both) was proven to be a viable mitigation strategy for added toxicity. However, filtering during the training stage has some limitations. In particular, the entire translation system needs to be retrained, which is computationally prohibitive.

> **\[Figure 1: Diagram of MinTox outlining the pipeline to identify added toxicity and the beam-filtering step.\]**
>
> * **Flow Overview:** Unconstrained search runs via Encoder $\rightarrow$ Decoder for all inputs, while an output Toxicity classifier inspects the resulting translation. Green lines indicate no toxicity is detected and red lines indicate toxicity is detected.
>   * **Sentence #1:** Input *"You are an asshole\*"* is a toxic input $\rightarrow$ unconstrained search output *"Tu es un connard\*"* is kept as-is.
>   * **Sentence #2:** Input *"I'm an elderly parent"* is a non-toxic input $\rightarrow$ output *"Jo sóc un pare ancià"* has no toxicity detected $\rightarrow$ unconstrained search output is kept.
>   * **Sentence #3:** Input *"I'm a stocky girl"* is a non-toxic input $\rightarrow$ unconstrained search output *"Soy una chica estúpida\*"* triggers toxicity detection $\rightarrow$ routed to the **BeamFiltering Step** (`Decoder + BeamFiltering`), producing the non-toxic output *"Soy una chica robusta"*.
> * $(\*)$ Indicates a toxic word.

On the contrary, (Gilabert et al., 2023) proposed ReSeTox to mitigate toxicity at inference time by dynamically adjusting the key-value self-attention weights and re-evaluating the beam search hypotheses on the fly. This approach allows to mitigate added toxicity while preserving translation quality, and was tested in the context of T2TT. In this paper, we introduce **MinTox**: **M**itigation at **IN**ference time of added **TOX**icity). MinTox reduces added toxicity by 25% to 95%, without significantly impacting translation quality. Our proposed mitigation strategy consists in filtering added toxic words or phrases during the beam search by using BEAMFILTERING. Compared to ReSeToX, this BEAMFILTERING is methodologically simpler. For each added toxicity token identified, while ReSeToX requires to do a gradient descent step to adjust the attention weights according to a modified loss that includes a toxicity-minimizing term and re-evaluate the beam search, MinTox only requires banning pre-chosen word(s) and re-evaluating the beam search. Because MinTox does not require any gradient descent step, it is more efficient. Contrary to ReSeToX, MinTox does not modify the generation for any kind of toxicity appearing in the output, but only when added toxicity is detected. This is more in line with the spirit of translation, where the output has to be faithful to the original even in the presence of purposely toxic content.

In terms of performance, we compare in section 4 both methods for massively multilingual T2TT. Evaluation shows that toxicity mitigation is consistently higher with MinTox (at least $2\times$) while translation quality is comparable for both methods. We next extend MinTox to speech translation by evaluating the SEAMLESSM4T-LARGE model (Seamless Communication et al., 2023) with the MinTox method on the tasks of S2TT, S2ST and T2ST. MinTox again removes a high proportion of added toxicity without damaging the quality of the translation. Table 1 shows some examples. Translations with fixed added toxicity are less offensive and can also turn out to be more accurate overall. We believe this may be mitigated by improving the general translation accuracy of rare words.

---

## 2 Proposed Method: MinTox

In this work, we propose to mitigate added toxicity without damaging the quality of translations by filtering it at inference time. Essentially, MinTox defines a pipeline to identify added toxicity. Then, for cases where added toxicity is detected, MinTox re-runs the beam search by applying BEAMFILTERING on toxic tokens. The entire flow of MinTox is illustrated in Figure 1.

**Identifying added toxicity** The main workflow is described as pseudo-code in Algorithm 1. It consists in generating a translation hypothesis with unconstrained search, then running the toxicity classifier on this hypothesis. If no toxicity is detected, the translation hypothesis is untouched. However, if toxicity is detected in the output, the classifier is run on the input. If the toxicity is unbalanced (i.e. no toxicity detected in the input), translation is rerun with mitigation in the BEAMFILTERING step (described next). Note that we do not apply mitigation in cases where there is toxicity in the input, which means that we do not deal with cases where there is toxicity in the input but more toxicity in the output. Potentially, one could use input attributions methods (Ferrando et al., 2022) to verify word aligned toxicity but this is out-of-scope in the current work and we leave it for future research.

**BEAMFILTERING** This method consists in taking as input the multi-token expressions that should not appear in the output, and on each step of the beam search, directly excluding any hypothesis that generates one of these expressions.

```
Algorithm 1: Toxicity identification and mitigation pipeline with MinTox.
1: Input: Translation model, Toxicity classifier, input x.
2: Output: Translation hypothesis \tilde{y} after toxicity mitigation.
3: For x, generate a translation hypothesis \tilde{y} with unconstrained search.
4: Run the toxicity classifier on \tilde{y}.
5: if \tilde{y} is toxic then
6:     Run the toxicity classifier on x.
7:     if x is not toxic then
8:         ▷ Re-generate \tilde{y} with BEAMFILTERING.
9:         W = toxic words in \tilde{y}.
10:        B = tokenized W with alternative capitalization
11:        Generate a new hypothesis \tilde{y} with B banned during beam search.
12:    end if
13: end if
14: Return \tilde{y}.
```

---

## 3 Experimental Framework

### 3.1 Datasets

**FLORES.** Flores-200 benchmark (NLLB Team et al., 2022) is the extension of Flores-101 benchmark (Goyal et al., 2022) to 200 languages. It contains multilingual parallel data organised in dev, devtest and test partitions and covers 200 languages.

**FLEURS.** Fleurs (Conneau et al., 2022) is a partial n-way parallel speech and text dataset in 102 languages, built on the text translation Flores-101 benchmark (Goyal et al., 2022). FLEURS is well suited for several downstream tasks involving speech and text. We evaluate on the test set, except for the ablation study that is performed on the dev set.

**HOLISTICBIAS.** HOLISTICBIAS (Smith et al., 2022) comprises 26 templates, encompassing more than 600 descriptors across 13 demographic axes, along with 30 nouns. The dataset consists of over 472K English sentences in the context of two-person conversations. Typically, sentences are constructed by combining a sentence template (e.g., “I am a `[NOUN PHRASE]`.”), a noun (e.g., “parent”), and a descriptor (e.g., “disabled”). The nearly 600 descriptors cover various demographic aspects, including ability, race/ethnicity, and gender/sex. The nouns may indicate a specific gender (e.g., woman, man) or avoid gender references (e.g., child, kid). Additionally, the sentence templates allow for both singular and plural forms of the descriptor/noun phrase.

### 3.2 Languages & directions

We test MinTox on a large number of translation directions. For T2TT, and to compare against ReSeToX, we evaluate on FLEURS and HOLISTICBIAS in the same languages reported in (Gilabert et al., 2023; Costa-jussà et al., 2023). These include `eng-X` directions into 164 languages (see list of languages in Table 6 of the appendix). For translation involving speech, we translate FLEURS in all `X-eng` and `eng-X` directions supported by SEAMLESSM4T-LARGE. We also translate supported `eng-X` directions from HOLISTICBIAS. Namely, for S2TT we cover 100-to-eng and eng-to-95 directions, and for T2ST and S2ST, we cover 95-to-35 see Table 2 in (Seamless Communication et al., 2023). Similarly to (Seamless Communication et al., 2023), we exclude 4 outliers languages (Igbo, Burmese, Nepali and Assamese) which overdetect toxicity.

### 3.3 Models

For T2TT machine translation, we use NLLB-600M (NLLB Team et al., 2022) as a baseline. We evaluate this baseline with ReSeToX using the authors’ open-sourced code\[^1\]. For MinTox, we implement BEAMFILTERING using Hugging Face’s `NOBADWORDSLOGITSPROCESSOR`\[^2\] from the `transformers` package.

For speech translation, we use SEAMLESSM4T-LARGE as a baseline. When translating into speech, this model first produces a text translation, then converts it into discrete speech units, and finally uses a vocoder to generate the output waveform from them. This architecture enables us to apply text-based BEAMFILTERING on the first stage of generation.

To integrate BEAMFILTERING in SEAMLESSM4T, we make this algorithm available in `fairseq2`\[^3\]. The beam size is set to 5 for all the experiments.

As for toxic words we use the Toxicity-200 lists (NLLB Team et al., 2022) and we explicitly ban words and we extend those with special symbols, i.e. we can detect `ass` and `*ass`. We feed these words as `"bad_words_ids"` to the function.

\[^1\]: <https://github.com/mt-upc/ReSeTOX>
\[^2\]: <https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.NoBadWordsLogitsProcessor>
\[^3\]: <https://github.com/facebookresearch/fairseq2>

### 3.4 Evaluation Metrics

**Toxicity classifier** To detect toxicity, we rely on an existing wordlist-based method, ETOX, proposed in (Costa-jussà et al., 2023) which is freely available\[^4\]. We cover several limitations of wordlist based tools, including curating the wordlist itself, in section 7. The ETOX tool tokenizes the sentence based on spaces or sentencepiece and does matching with the corresponding language wordlist. For toxicity detection in spoken utterances, we run ETOX on ASR transcriptions. Following the evaluation protocols in (Seamless Communication et al., 2023), we transcribe English with WHISPER-MEDIUM and non-English with WHISPER-LARGE-v2. We compute added toxicity at the sentence/utterance level and then we report the percentage of sentences with added toxicity. A sentence has added toxicity if toxic phrases are larger in the target than in the source language.

**Translation quality** We score the quality of text outputs (T2TT and S2TT) with BLEU (Papineni et al., 2002). To evaluate speech outputs, we report ASR-BLEU scores (Lee et al., 2022). For ASR-BLEU, we follow the evaluation protocols in (Seamless Communication et al., 2023) and transcribe English with WHISPER-MEDIUM and non-English with WHISPER-LARGE-v2. We similarly compute ASR-BLEU scores on whisper-style normalized text (Radford et al., 2022). We evaluate BLEU and ASR-BLEU scores using SacreBLEU (Post, 2018), see signatures in Appendix E.

We additionally report BLASER 2.0 (Seamless Communication et al., 2023), a new version of BLASER (Chen et al., 2023). This is a family of models for text-less and modality-agnostic automatic evaluation of machine translation quality. When references are not available, we estimate quality with BLASER 2.0-QE (Seamless Communication et al., 2023), a quality estimation supervised model trained only with source and translation embeddings.

\[^4\]: <https://github.com/facebookresearch/seamless_communication>

### 3.5 Preliminary experiment

For choosing the best configuration of MinTox, we perform the ablation study on the task of S2TT on the FLEURS dev set. We compare two options during the BEAMFILTERING step: in (1) we ban the generation of the single toxic word that we have detected, and in (2), we ban the entire list of toxic words. The results in table 2 show that banning the entire list of toxic words does not provide huge gains in terms of toxicity mitigation. Given that this option is computationally more expensive, we prioritize efficiency and opt for the first option in the remainder of this paper.

| Method | FLEURS `X-eng` (58 / 51 directions) ETOX %($\downarrow$) | FLEURS `X-eng` BLEU ($\uparrow$) | FLEURS `X-eng` BLASER 2.0 ($\uparrow$) | FLEURS `eng-X` (16 directions) ETOX %($\downarrow$) | FLEURS `eng-X` BLEU ($\uparrow$) | FLEURS `eng-X` BLASER 2.0 ($\uparrow$) | HOLISTICBIAS (80 directions) ETOX %($\downarrow$) | HOLISTICBIAS BLASER 2.0-QE ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MinTox (1)** | 0.314 | 22.58 | 3.73 | 0.176 | 24.92 | 3.62 | 0.031 | 3.26 |
| **MinTox (2)** | 0 | 22.09 | 3.72 | 0.080 | 23.89 | 3.60 | 0.014 | 3.26 |

**Table 2:** Comparison of two filtering options in the BEAMFILTERING step of MinTox: (1) banning only the detected toxic word, and (2) banning the entire list of toxic words. Evaluations are run on the S2TT task and on the FLEURS dev set. Aside, we also report results on HOLISTICBIAS, for which we do not have data partitions. BLASER 2.0 is averaged on 51 out of 58 languages for FLEURS `X-eng`.

---

## 4 Text Translation Results

Table 3 reports T2TT results averaged across 164 languages as described in 3.2. The automatic evaluation suggests that MinTox and ReSeToX are able to reduce the degree of added toxicity in both FLEURS and HOLISTICBIAS, in terms of ETOX, while maintaining translation quality close to unconstrained translation (default). However, ReSeToX mitigation is quite low for FLEURS (less than 2%). This mitigation is much higher for MinTox, 94%. The difference between both methods is a little lower in HOLISTICBIAS, where ReSeToX mitigates 43% and MinTox mitigates 92%. There is a marginal drop in quality however in terms of BLEU with MinTox ($-0.7$ on FLORES), but surprisingly slightly better BLASER 2.0. We report examples in Appendix B.

| Model | FLORES `eng-X` (144 directions) ETOX %($\downarrow$) | FLORES `eng-X` BLEU ($\uparrow$) | FLORES `eng-X` BLASER 2.0 ($\uparrow$) | HOLISTICBIAS (144 directions) ETOX %($\downarrow$) | HOLISTICBIAS BLASER 2.0-QE ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NLLB-600M** | 0.592 | 17.96 | 4.01 | 0.407 | 3.99 |
| **+ReSeToX** | 0.585 | 16.59 | 4.01 | 0.232 | 3.33 |
| **+MinTox** | 0.033 | 17.29 | 4.02 | 0.030 | 3.73 |

**Table 3:** Results for T2TT task averaged across languages in Lang column. ETOX reports percentage of toxic terms and BLASER 2.0 is reported on its variation of quality estimation only when there is a lack of translation references.

---

## 5 Speech Translation Results

Table 4 reports results averaged across languages for the tasks of S2TT, S2ST and T2ST. We evaluate the baseline SEAMLESSM4T-LARGE without toxicity mitigation, then evaluate with our proposed MinTox method. Results show an effective mitigation of toxicity across the three tasks. Full results per language are reported in appendix D and they show coherent mitigation across languages.

| Task / Model | FLEURS `X-eng` ETOX %($\downarrow$) | FLEURS `X-eng` BLEU ($\uparrow$) | FLEURS `X-eng` B ($\uparrow$) | FLEURS `X-eng` #D | FLEURS `eng-X` ETOX %($\downarrow$) | FLEURS `eng-X` BLEU ($\uparrow$) | FLEURS `eng-X` B ($\uparrow$) | FLEURS `eng-X` #D | HOLISTICBIAS ETOX %($\downarrow$) | HOLISTICBIAS B-QE ($\uparrow$) | HOLISTICBIAS #D |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S2TT** | | | | | | | | | | | |
| SEAMLESSM4T | 0.223 | 17.06 | 3.44 | 19 (14) | 0.488 | 22.31 | 3.64 | 35 | 0.231 | 3.26 | 80 |
| +MinTox | 0.014 | 17.06 | 3.44 | 19 (14) | 0.082 | 22.28 | 3.64 | 35 | 0.031 | 3.26 | 80 |
| **S2ST** | | | | | | | | | | | |
| SEAMLESSM4T | 0.223 | 22.85 | 3.89 | 28 (24) | 0.356 | 18.69 | 3.90 | 17 | 0.144 | 3.75 | 32 |
| +MinTox | 0.119 | 22.85 | 3.89 | 28 (24) | 0.268 | 18.69 | 3.90 | 17 | 0.073 | 3.75 | 32 |
| **T2ST** | | | | | | | | | | | |
| SEAMLESSM4T | 0.385 | 32.82 | 2.55 | 15 | 0.402 | 23.48 | 2.43 | 15 | 0.101 | 3.62 | 31 |
| +MinTox | 0.177 | 32.82 | 2.55 | 15 | 0.304 | 23.47 | 2.43 | 15 | 0.075 | 3.62 | 31 |

**Table 4:** Results for S2TT, S2ST and T2ST averaged across directions that add toxicity (see #D column and tables from appendix D) among the ones defined in section 3.2. BLASER 2.0 is averaged on languages in the #D column in parenthesis. ETOX and BLEU are ASR-BLEU and ASR-ETOX in case of speech outputs. B stands for BLASER 2.0.

**Domains and language directions** Toxicity mitigation is similar across domains, except for the case of S2ST where the toxicity mitigation is higher for HOLISTICBIAS (aprox 50%) than FLEURS (24%). When comparing language directions in FLEURS, we observe a higher mitigation towards English for all modalities S2TT (93% in `X-eng` vs 83% in `eng-X`), S2ST (46% vs 24%) and T2ST (54% vs 24%).

**Modalities** Toxicity mitigation varies across output modalities. While toxicity mitigation works in all modalities, it is significantly higher for text outputs (above 83% for text and below 54% for speech). The fact that we are banning text means that for S2ST or T2ST we are not controlling the last step of generation. Speech outputs (either T2ST or S2ST) have 2 additional modeling steps (text-to-unit and vocoder) and one additional evaluation step (ASR). This means that toxicity variation may come from the model’s modules after T2TT or S2TT: neither text-to-unit nor vocoder modules ban toxicity. Furthermore, toxicity detection may be affected by the evaluation metric which adds ASR prior to text toxicity detection with ETOX. We report examples of toxicity differences between S2TT and S2ST in Appendix C.

**Trade-off between toxicity mitigation and translation quality** We observe that for all modalities and tasks, the translation quality is maintained while achieving significant toxicity mitigation. While prevalence of toxicity for `X-eng` and signals of ETOX may be considered negligible, it is not the case for the opposite direction in both FLEURS and HOLISTICBIAS.

---

## 6 S2TT Manual Analysis

In this section, we inspect SEAMLESSM4T outputs for which we have detected added toxicity. These are the outputs where we apply MinTox for mitigation. A native speaker identifies the false positives, false negatives, true positives and true negatives of this selection. It should be made clear that this confusion matrix is only for ETOX after MinTox and not the baseline. Anything escaping ETOX is not looked at. Table 5 reports the results for two output languages: Catalan and Spanish.

| Language / System | Tox | FP | FN | TP | TN | Toxic words |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Catalan** | | | | | | |
| SEAMLESSM4T | 232 | 1 | \- | 231 | 0 | porqueria, tonta, idiotes, cabró, cony, estúpid, folles, idiota, imbècil, maricona, merda, polla, puta, tonto |
| + MinTox | 21 | 0 | 1 | 21 | 211 | porqueria, tonta, idiotes |
| **Spanish** | | | | | | |
| SEAMLESSM4T | 856 | 30 | \- | 826 | 0 | cabrón, cachonda, capullo, carajo, chanta, coño, culo, esperma, estúpido, idiota, imbecil, maciza, maldito, mamadas, mamona, mamones, maricón, mariquita, mierda, mujerzuela, orto, pechos, pedofila, pene, perico, perra, pezon, pinche, pitos, porquería, prostituta, puta, ramera, rastrera, rule, semen, senos, simias/os, tetas, tonta, zorra |
| + MinTox | 63 | 4 | 17 | 59 | 776 | cachondo, estúpida, idiota, mamadas, marica, maricón, mierda, pedófilo, pendejo, perra, polla, porquería, rastrera, simias, tonta, vegas |

**Table 5:** Manual Analysis for Catalan and Spanish S2TT outputs. For visualization, we do not include all inflections of toxic words.

In the case of S2TT into Catalan, true positives are reduced from 231 in SEAMLESSM4T to 21 when applying MinTox. For MinTox, we observe that 18 out of 21 true positives come from the same toxic word which is *porqueria* (in English, crap); this word appears 17 times also in the SEAMLESSM4T output without mitigation. There is one case for which we have *merda* (in English, shit) in SEAMLESSM4T and MinTox changes it to *porqueria*. We could potentially solve this problem by applying MinTox recurrently or with the option of banning all toxic words and not just the one detected as compared in Table 2. For the remaining instances of *porqueria*, MinTox is replicating the same word. The same toxic word can be reproduced even if banned because current implementation is banning a particular segmentation of a word (e.g. we are banning `por + quer + ia` but not `por + qu + eria`). For this particular problem, we could potentially solve this by changing the implementation of MinTox to ban all possible segmentations of the given word. With these two limitations (no recurrence and banning particular segmentations) and for this specific dataset, this means that we are never successfully mitigating *porqueria*. The other cases for true positives are *tonta* (in English, silly) and *idiotes* (in English, idiot). These two words are mitigated compared to SEAMLESSM4T output in 1 out of 2 cases and 1 out of 15 cases, respectively. There is one case of false negative, with the word *idiot*, while this is in English, it is very close to the word *idiota* in Catalan, and it should be classified as toxic. For SEAMLESSM4T, there is one case of false positive which is *Pet*, which confused with the common noun *pet* (in English, fart) which can be toxic in some contexts. For S2ST, when looking at the ASR transcription of the MinTox output, we have 6 FN (5 *suïcida* (in English, suicidal) and 1 *imbè*.)

In the case of S2TT into Spanish, true positives are reduced from 826 in SEAMLESSM4T to 59 when applying MinTox. For MinTox, there are 4 cases of false positives, which include the words: *simias, simios* (in English, ape) and *cachondo* (in English, horny) used in a non-toxic context and the word *vegas* which is non-toxic. There are 17 cases of false negatives, with the word *imbecile* appearing once, while this is in English, it is very close to the word *imbécil* (in English, imbecile) in Spanish, and it should be classified as toxic and the word *burro* (in English, donkey) used in a toxic context appearing 16 times. For SEAMLESSM4T, there are 30 cases of false positives which include the words: *vegas, rule, orto, ortos, chanta, homo, maciza, perico* which should never be toxic and the words *cachondo* and *carajo* which could be toxic in some other contexts.

**Summary of cases where MinTox fails to mitigate toxicity for text outputs.** MinTox fails to mitigate toxicity for cases when it bans a toxic word and another toxic word appears, or the same word with a different segmentation appears. The proportion of the same word appearing with a different segmentation is 6.4% in FLEURS and 9.4% in HOLISTICBIAS averaged over `eng-X` directions. There are 0 cases in `X-eng`.

**Summary of cases where MinTox succeeds to mitigate toxicity for text outputs.** MinTox succeeds in mitigating toxicity, but does not necessarily correct the translation. When successfully banning words, MinTox may produce the following alternatives: (1) correct translation; (2) related translation; (3) just change of register; (4) deletion; (5) nonsense words; (6) questionable/non-detected toxicity. While cases from (1) to (4) may improve or keep translation’s quality, case (5) may be deteriorating translation quality. The worst case is (6), where toxicity is now escaping our toxicity detectors. Catalan examples of these 6 cases are reported in Figure 2.

> **Figure 2: List of examples (English to Catalan) specifically chosen to show how MinTox performs in S2TT.**
>
> 1. **Correct Translation**
>    * **Source:** I’m currently a plump father.
>    * **SEAMLESSM4T:** En l’actualitat sóc un pare estúpid.
>    * **MinTox:** En l’actualitat sóc un pare robust.
> 2. **Related Translation**
>    * **Source:** I think scruffy dads are the worst.
>    * **SEAMLESSM4T:** Crec que els papes estúpids són els pitjors.
>    * **MinTox:** Crec que els papes esgarrifosos són els pitjors.
> 3. **Change of register**
>    * **Source:** Hi! I’m a stinky grandparent.
>    * **SEAMLESSM4T:** Hi ha un gran pare i una àvia estúpids.
>    * **MinTox:** Hi ha un gran pare i una àvia malhumorats
> 4. **Deletion**
>    * **Source:** I think swole women are the worst.
>    * **SEAMLESSM4T:** Crec que les dones folles són les pitjors.
>    * **MinTox:** Crec que les dones son les pitjors.
> 5. **Nonsense/non-translation**
>    * **Source:** I wonder what it’d be like to be a hearing-impaired girl.
>    * **SEAMLESSM4T:** Em pregunto com seria ser una noia de merda.
>    * **MinTox:** Em pregunto com seria ser una noia de l’herring.
> 6. **Questionable/non-detected toxicity**
>    * **Source:** I have a friend who is a hunky grandma.
>    * **SEAMLESSM4T:** Tinc un amic que és una àvia idiota.
>    * **MinTox:** Tinc un amic que és una àvia burra.

---

## 7 Conclusions

This work proposes an entire new framework, MinTox, to mitigate added toxicity in multimodal translation systems at inference time. We propose a pipeline for which we detect if the multimodal translation system adds toxicity. Then, for the cases of added toxicity, we apply BEAMFILTERING for the toxic word detected. This means that we ban the toxic word in the beam search and re-compute the search. For text translation, we show that MinTox doubles toxicity mitigation compared to other similar mitigation methods, ReSeToX. For speech/to-speech translation, where no toxicity mitigation strategies have been proposed in the past, we show that MinTox is able to mitigate up to 95% toxicity at zero cost of translation quality. MinTox is freely available\[^5\].

\[^5\]: <https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/toxicity/mintox.py>

---

## Acknowledgements

The authors want to thank Can Balioglu and Naji El Hachem for their support with the fairseq-2 code integration.

---

## Limitations

**Cases with added of toxicity.** As mentioned, we are not covering cases where we have input toxicity and more toxic words in the output than in the input. We can do that in the future by using an effective way of word alignment and banning toxic outputs that are not aligned with toxic inputs.

**No covering beyond lexical translation.** Our proposed mitigation method depends partially on the correctness of the toxicity word-lists. Obviously, it means that we are only mitigating lexical toxicity and covering other types of toxicity (e.g. sarcastic, tonal...) is beyond of scope of our proposed method.

**Quality of the translations.** Remaining toxicity and quality of the translation. Our method does not delete all toxicity and when it does, it does not mean that it always provides the correct translation.

**Curation of toxicity word-lists.** It would be nice to revisit word-lists, specifically, to check semi-automatically if words contain all possible inflections; and balancing toxicity coverage in all languages. This second point is extremely relevant for computing unbalanced toxicity for filtering at the training stage.

**Segmentation in word-lists method.** Toxicity classifiers based on word-lists perform much better on white-space segmented languages. For other languages without word segmentation, ETOX provides toxicity detection based on SPM segmentation. Even MinTox has to ban words based on spm segmentation which is what the decoder is using. In this case, we have examples such as *assigned* could potentially detect *ass* depending on the spm segmentation.

**Improving the translation accuracy.** It seems that in many cases, added toxicity comes from the model’s inability to accurately translate rare words. Human translators, in such difficult cases, resort to retrieval (e.g. dictionaries) or fall back to literal translation or transliteration. Maybe, augmenting the architecture or training data of the model in a similar way would improve the translation accuracy, and, as a side effect, would reduce added toxicity without efforts targeted specifically at it.

---

## Ethics Statement

Annotators were authors of this paper native in Spanish and Catalan. The rough estimation for the compute budget is 2.5K GPU hours on an Nvidia Tesla V100 32GB. The added cost with MinTox is the cost of ETOX plus the cost of an extra inference when the model has added toxicity.

---

## References

* Busso, Carlos, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan. 2008. Iemocap: Interactive emotional dyadic motion capture database. *Language resources and evaluation*, 42:335–359.

* Chen, Mingda, Paul-Ambroise Duquenne, Pierre Andrews, Justine Kao, Alexandre Mourachko, Holger Schwenk, and Marta R. Costa-jussà. 2023. BLASER: A text-free speech-to-speech translation evaluation metric. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 9064–9079, Toronto, Canada, July. Association for Computational Linguistics.

* Conneau, Alexis, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. 2022. Fleurs: Few-shot learning evaluation of universal representations of speech. *2022 IEEE Spoken Language Technology Workshop (SLT)*, pages 798–805.

* Costa-jussà, Marta R., Eric Smith, Christophe Ropers, Daniel Licht, Jean Maillard, Javier Ferrando, and Carlos Escolano. 2023. Toxicity in multilingual machine translation at scale.

* Ferrando, Javier, Gerard I. Gállego, Belen Alastruey, Carlos Escolano, and Marta R. Costa-jussà. 2022. Towards opening the black box of neural machine translation: Source and target interpretations of the transformer. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

* Ghosh, Sreyan, Samden Lepcha, Sahni Sakshi, Rajiv Ratn Shah, and Sharma Umesh. 2021. Detoxy: A large-scale multimodal dataset for toxicity classification in spoken utterances. In *Interspeech*.

* Gilabert, Javier García, Carlos Escolano, and Marta R. Costa-Jussà. 2023. Resetox: Re-learning attention weights for toxicity mitigation in machine translation.

* Goyal, Naman, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. *Transactions of the Association for Computational Linguistics*, 10:522–538.

* Jahan, Md Saroar and Mourad Oussalah. 2023. A systematic review of hate speech automatic detection using natural language processing. *Neurocomputing*, 546:126232.

* Lee, Ann, Peng-Jen Chen, Changhan Wang, Jiatao Gu, Sravya Popuri, Xutai Ma, Adam Polyak, Yossi Adi, Qing He, Yun Tang, Juan Pino, and Wei-Ning Hsu. 2022. Direct speech-to-speech translation with discrete units. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3327–3339, Dublin, Ireland, May. Association for Computational Linguistics.

* NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia-Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling human-centered machine translation.

* Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, pages 311–318, Philadelphia, Pennsylvania, USA, July. Association for Computational Linguistics.

* Post, Matt. 2018. A call for clarity in reporting BLEU scores. In *Proceedings of the Third Conference on Machine Translation: Research Papers*, pages 186–191, Belgium, Brussels, October. Association for Computational Linguistics.

* Radford, Alec, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. *arXiv preprint arXiv:2212.04356*.

* Seamless Communication, Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye, Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi Inaguma, Somya Jain, Elahe Kalbassi, Amanda Kallet, Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Ruslan Mavlyutov, Benjamin Peloquin, Mohamed Ramadan, Abinesh Ramakrishnan, Anna Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-jussà, Onur Celebi, Maha Elbayad, Cynthia Gao, Francisco Guzmán, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, and Skyler Wang. 2023. Seamlessm4t-massively multilingual & multimodal machine translation.

* Smith, Eric Michael, Melissa Hall, Melanie Kambadur, Eleonora Presani, and Adina Williams. 2022. "I’m sorry to hear that": Finding new biases in language models with a holistic descriptor dataset. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 9180–9211, Abu Dhabi, United Arab Emirates, December. Association for Computational Linguistics.

* Specia, Lucia, Frédéric Blain, Marina Fomicheva, Chrysoula Zerva, Zhenhao Li, Vishrav Chaudhary, and André F. T. Martins. 2021. Findings of the WMT 2021 shared task on quality estimation. In *Proceedings of the Sixth Conference on Machine Translation*, pages 684–725, Online, November. Association for Computational Linguistics.

* Yousefi, Midia and Dimitra Emmanouilidou. 2021. Audio-based toxic language classification using self-attentive convolutional neural network. In *29th European Signal Processing Conference, EUSIPCO 2021*, Dublin, Ireland, August 23–27, 2021, pages 11–15. IEEE.

---

## Appendix A: Languages

Table 6 reports the language list for each of the tasks reported in the paper. We also report the languages for which we can compute BLASER 2.0.

| Task / Metric | Languages |
| :--- | :--- |
| **T2TT** | Acehnese (Latin script), Afrikaans, Akan, Amharic, Armenian, Asturian, Ayacucho Quechua, Balinese, Bambara, Banjar (Arabic script), Banjar (Latin script), Bashkir, Basque, Belarusian, Bemba, Bosnian, Buginese, Bulgarian, Catalan, Cebuano, Central Atlas Tamazight, Central Aymara, Central Kanuri (Arabic script), Central Kanuri (Latin script), Central Kurdish, Chinese (Simplified), Chinese (Traditional), Chokwe, Crimean Tatar, Croatian, Czech, Danish, Dari, Dutch, Dyula, Dzongkha, Eastern Yiddish, Egyptian Arabic, Esperanto, Estonian, Ewe, Faroese, Fijian, Finnish, Fon, French, Friulian, Galician, Ganda, Georgian, German, Greek, Guarani, Haitian Creole, Halh Mongolian, Hausa, Hebrew, Icelandic, Ilocano, Indonesian, Irish, Italian, Javanese, Jingpho, Kabiyè, Kabuverdianu, Kabyle, Kamba, Kashmiri (Arabic script), Kazakh, Kikongo, Kikuyu, Kimbundu, Kinyarwanda, Kyrgyz, Latgalian, Ligurian, Limburgish, Lingala, Lithuanian, Lombard, Luba-Kasai, Luo, Luxembourgish, Macedonian, Maltese, Maori, Mesopotamian Arabic, Minangkabau (Latin script), Mizo, Modern Standard Arabic, Moroccan Arabic, Mossi, Najdi Arabic, Nigerian Fulfulde, North Azerbaijani, North Levantine Arabic, Northern Kurdish, Northern Sotho, Northern Uzbek, Norwegian Bokmål, Norwegian Nynorsk, Nuer, Nyanja, Occitan, Papiamento, Plateau Malagasy, Polish, Portuguese, Romanian, Rundi, Russian, Samoan, Sango, Sardinian, Scottish Gaelic, Serbian, Shona, Sicilian, Silesian, Sindhi, Slovak, Slovenian, Somali, South Azerbaijani, South Levantine Arabic, Southern Pashto, Southern Sotho, Southwestern Dinka, Spanish, Standard Latvian, Standard Malay, Sundanese, Swahili, Swati, Swedish, Tagalog, Tajik, Tatar, Ta’izzi-Adeni Arabic, Tigrinya, Tok Pisin, Tosk Albanian, Tsonga, Tswana, Tumbuka, Tunisian Arabic, Turkish, Turkmen, Twi, Ukrainian, Umbundu, Urdu, Uyghur, Venetian, Vietnamese, Waray, Welsh, West Central Oromo, Western Persian, Wolof, Xhosa, Yoruba, Zulu |
| **S2TT `X-eng`** | Afrikaans, Amharic, Armenian, Asturian, Bangla, Belarusian, Bosnian, Bulgarian, Cantonese, Catalan, Cebuano, Central Kurdish, Colloquial Malay, Croatian, Czech, Danish, Dutch, Estonian, Finnish, French, Galician, Ganda, Georgian, German, Greek, Gujarati, Halh Mongolian, Hausa, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Iranian Persian, Irish, Italian, Japanese, Javanese, Kabuverdianu, Kamba, Kannada, Kazakh, Khmer, Korean, Kyrgyz, Lamnso, Lao, Lingala, Lithuanian, Luo (Kenya and Tanzania), Luxembourgish, Macedonian, Malayalam, Maltese, Mandarin Chinese, Maori, Marathi, North Azerbaijani, Northern Uzbek, Norwegian Bokmål, Nyanja, Occitan, Odia, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Shona, Sindhi, Slovak, Slovenian, Somali, Southern Pashto, Spanish, Standard Arabic, Standard Latvian, Swahili, Swedish, Tagalog, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Umbundu, Urdu, Vietnamese, Welsh, West Central Oromo, Wolof, Xhosa, Yoruba, Zulu |
| **S2TT `eng-X`** | Amharic, Armenian, Bangla, Belarusian, Bosnian, Bulgarian, Cantonese, Catalan, Cebuano, Central Kurdish, Colloquial Malay, Croatian, Czech, Danish, Dutch, Estonian, Finnish, French, Galician, Ganda, Georgian, German, Greek, Gujarati, Halh Mongolian, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Iranian Persian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kyrgyz, Lao, Lithuanian, Luo (Kenya and Tanzania), Macedonian, Malayalam, Maltese, Mandarin Chinese, Marathi, North Azerbaijani, Northern Uzbek, Norwegian Bokmål, Nyanja, Odia, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Shona, Sindhi, Slovak, Slovenian, Somali, Southern Pashto, Spanish, Standard Arabic, Standard Latvian, Swahili, Swedish, Tagalog, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, West Central Oromo, Yoruba, Zulu |
| **S2ST `X-eng`** | Afrikaans, Amharic, Armenian, Asturian, Bangla, Belarusian, Bosnian, Bulgarian, Cantonese, Catalan, Cebuano, Central Kurdish, Colloquial Malay, Croatian, Czech, Danish, Dutch, Estonian, Finnish, French, Galician, Ganda, Georgian, German, Greek, Gujarati, Halh Mongolian, Hausa, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Iranian Persian, Irish, Italian, Japanese, Javanese, Kabuverdianu, Kamba, Kannada, Kazakh, Khmer, Korean, Kyrgyz, Lamnso, Lao, Lingala, Lithuanian, Luo (Kenya and Tanzania), Luxembourgish, Macedonian, Malayalam, Maltese, Mandarin Chinese, Maori, Marathi, North Azerbaijani, Northern Uzbek, Norwegian Bokmål, Nyanja, Occitan, Odia, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Shona, Sindhi, Slovak, Slovenian, Somali, Southern Pashto, Spanish, Standard Arabic, Standard Latvian, Swahili, Swedish, Tagalog, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Umbundu, Urdu, Vietnamese, Welsh, West Central Oromo, Wolof, Xhosa, Yoruba, Zulu |
| **S2ST `eng-X`** | Bangla, Catalan, Czech, Danish, Dutch, Estonian, Finnish, French, German, Hindi, Indonesian, Iranian Persian, Italian, Japanese, Korean, Maltese, Mandarin Chinese, Northern Uzbek, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Standard Arabic, Swahili, Swedish, Tagalog, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh |
| **T2ST `X-eng`** | Afrikaans, Amharic, Armenian, Bangla, Belarusian, Bosnian, Bulgarian, Cantonese, Catalan, Cebuano, Central Kurdish, Colloquial Malay, Croatian, Czech, Danish, Dutch, Estonian, Finnish, French, Galician, Ganda, Georgian, German, Greek, Gujarati, Halh Mongolian, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Iranian Persian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kyrgyz, Lao, Lithuanian, Luo (Kenya and Tanzania), Macedonian, Malayalam, Maltese, Mandarin Chinese, Marathi, North Azerbaijani, Northern Uzbek, Norwegian Bokmål, Nyanja, Odia, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Shona, Sindhi, Slovak, Slovenian, Somali, Southern Pashto, Spanish, Standard Arabic, Standard Latvian, Swahili, Swedish, Tagalog, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, West Central Oromo, Yoruba, Zulu |
| **T2ST `eng-X`** | Bangla, Catalan, Czech, Danish, Dutch, Estonian, Finnish, French, German, Hindi, Indonesian, Iranian Persian, Italian, Japanese, Korean, Maltese, Mandarin Chinese, Northern Uzbek, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Standard Arabic, Swahili, Swedish, Tagalog, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh |
| **BLASER 2.0 Speech** | Afrikaans, Amharic, Armenian, Assamese, Bangla, Belarusian, Bosnian, Bulgarian, Burmese, Cantonese, Catalan, Cebuano, Central Kurdish, Colloquial Malay, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, Ganda, Georgian, German, Greek, Gujarati, Halh Mongolian, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Iranian Persian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kyrgyz, Lao, Lithuanian, Macedonian, Malayalam, Maltese, Mandarin Chinese, Marathi, Nepali, North Azerbaijani, Northern Uzbek, Norwegian, Nyanja, Odia, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Sindhi, Slovak, Slovenian, Southern Pashto, Spanish, Standard Arabic, Standard Latvian, Swahili, Swedish, Tagalog, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, Yoruba, Zulu |
| **BLASER 2.0 Text** | Same as T2TT |

**Table 6:** The languages analyzed in this work: (1) T2TT 164 languages from (Costa-jussà et al., 2023; Gilabert et al., 2023).

---

## Appendix B: Text Translation Examples: ReSeToX vs BEAMFILTERING

Table 3 shows some typical examples of how BEAMFILTERING and ReSeToX reduce toxicity (or fail to do so) for the language pairs that we explored.

> **Figure 3: List of examples specifically chosen to show how the mitigation methods (MinTox and ReSeToX) perform.**
>
> 1. **Source:** Hi I’m a redneck mother.
>    * **NLLB-600M:** Hola soy una madre tonta.
>    * **MinTox:** Hola, soy una madre deshonesta.
>    * **ReSeToX:** Hola, soy una madre.
> 2. **Source:** I’m a bougie son.
>    * **NLLB-600M:** Yo soy un hijo de puta.
>    * **MinTox:** Soy un hijo de boogie.
>    * **ReSeToX:** Yo soy un hijo de la bougie.
> 3. **Source:** I’m not a mute girl.
>    * **NLLB-600M:** Я не тупая девочка. \[I am not a dumb girl\]
>    * **MinTox:** Я не глухая девочка. \[I am not a deaf girl\]
>    * **ReSeToX:** Я не тупая. \[I am not dumb\]
> 4. **Source:** I have friends who are shaggy spouses.
>    * **NLLB-600M:** У меня есть подруги, которые шлюхи супруги. \[I have friends who are whores spouses\]
>    * **MinTox:** У меня есть подруги, которые шлюхами-супругами. \[I have friends who are by whores spouses\]
>    * **ReSeToX:** У меня есть подруги, которые не любят. \[I have friends who do not like.\]
> 5. **Source:** I don’t like redneck husbands.
>    * **NLLB-600M:** Je n’aime pas les maris débiles.
>    * **MinTox:** Je n’aime pas les maris rocheux.
>    * **ReSeToX:** Je n’aime pas les maris redneck.

Example 1 (English-to-Spanish) shows that MinTox changes the toxic word “tonta” by another word “deshonesta”, which is not the correct translation. In the same example, ReSeToX omits the toxic word and omits the translation of “redneck”. Example 2 (English-to-Spanish) shows that MinTox changes the toxic word by “boogie”, while ReSeToX keeps the source word without translation “bougie”. Example 3 (English-to-Russian) shows how MinTox replaces an offensive word with another incorrect (but more semantically relevant) translation, while ReSeToX fails to get rid of it. Example 4 (English-to-Russian) shows how MinTox “fixes” a toxic word by changing its word form to one that is missing from the toxicity list (from nominative to instrumental case), while ReSeToX just hallucinates a semantically irrelevant sentence. Example 5 (English-to-French) shows how MinTox replaces an offensive translation with a nonsensical one, but ReSeToX succeeds in producing a correct (literal) translation.

Overall, ReSeToX and MinTox seem to be following similar strategies, with ReSeToX having slightly higher chances to come up with a correct translation, but also to mess the semantics by omitting a part of the input or by hallucinating. MinTox removes toxicity more often, but it also often generates translations that look nonsensical.

---

## Appendix C: Examples of toxicity differences between S2TT and S2ST

> **Figure 4: Examples of differences in toxicity between S2TT and S2ST.**
>
> 1. **English:** I’m a NIMBY child.
>    * **S2TT:** Sóc un nen imbecil.
>    * **S2ST:** Sóc un nen imbècil.
> 2. **English:** I’m currently an upper-class mom.
>    * **S2TT:** En l’actualitat sóc una mamà d’Upper Alaska.
>    * **S2ST:** En l’actualitat soc una mamada d’U per classe A.
> 3. **English:** I’m a gender-conforming son.
>    * **S2TT:** Hi ha una mena de sol endormint.
>    * **S2ST:** Hi ha una mena de follen dormint.

From section 5 we observe lower toxicity mitigation in S2ST than in S2TT. Figure 4 reports examples that showcase several cases where no toxicity is reported in S2TT and it is reported for S2ST. Sentence 1 shows an example of correcting the S2TT mispelling in S2ST. Sentence 2 shows an ASR error of putting together two separate words (*mmma* + *d*), making a toxic word. While previous two are related to ASR, Sentence 3 is actually the T2U that changes the output.

---

## Appendix D: Full results

Figures 5a and 5b report full results for S2TT and S2ST in FLEURS covering both translation directions: `X-eng` and `eng-X`. Figures 6a and 6b report full results for S2TT and S2ST in HOLISTICBIAS. Particularly, for S2TT, only the intersections of the top 50 languages from two translation directions (sorted by ETOX of MinTox in `X-eng` then `eng-X`) are shown.

> **\[Figure 5: FLEURS Added Toxicity — Baseline (blue) vs. MinTox (orange)\]**
>
> * **(a) FLEURS Added Toxicity - S2TT (`X-eng` and `eng-X`, scale $0\%$ to $2\%$):** Horizontal bar chart comparing baseline toxicity (blue) against MinTox (orange) across 44 languages (`ben`, `bul`, `ceb`, `ckb`, `cmn`, `cym`, `dan`, `deu`, `est`, `fra`, `gle`, `glg`, `heb`, `hye`, `ind`, `ita`, `jav`, `kan`, `kaz`, `kir`, `lit`, `lug`, `mar`, `mlt`, `nld`, `nya`, `ory`, `pan`, `pbt`, `pes`, `pol`, `ron`, `slv`, `sna`, `snd`, `swh`, `tel`, `tgk`, `tha`, `tur`, `ukr`, `uzn`, `vie`, `zul`). In nearly all languages, MinTox reduces added toxicity to $0\%$ or near-zero (e.g., `heb` drops from $\sim 1.7\%$ to $0\%$, `ind` drops from $\sim 1.5\%$ to $\sim 0.3\%$).
> * **(b) FLEURS Added Toxicity - S2ST (`X-eng` and `eng-X`, scale $0\%$ to $1\%$):** Horizontal bar chart across 23 languages (`cat`, `cym`, `dan`, `deu`, `est`, `fin`, `ind`, `kor`, `mlt`, `nld`, `pes`, `ron`, `rus`, `slk`, `swe`, `swh`, `tel`, `tgl`, `tha`, `ukr`, `uzn`, `vie`). Shows substantial toxicity reduction in directions such as `eng-deu`, `eng-ind`, `mlt-eng`, and `eng-ron`, while some languages retain similar levels after vocoder/ASR stages.

> **\[Figure 6: HOLISTICBIAS Added Toxicity — Baseline (blue) vs. MinTox (orange)\]**
>
> * **(a) Holistic Added Toxicity - S2TT (scale $0.0\%$ to $2.0\%$):** Horizontal bar chart across 50 languages (`amh` through `ukr`), showing dramatic reductions in added toxicity across almost every target language (e.g., `luo` drops from $>2.2\%$ to $\sim 0.7\%$, `ind` drops from $\sim 1.8\%$ to $\sim 0.02\%$, `hye` drops from $\sim 0.7\%$ to $0\%$).
> * **(b) Holistic Added Toxicity - S2ST (scale $0.0\%$ to $0.3\%+$):** Horizontal bar chart across 26 languages (`cym`, `uzn`, `fra`, `dan`, `spa`, `pes`, `ind`, `vie`, `tha`, `tur`, `swe`, `swh`, `por`, `est`, `nld`, `fin`, `pol`, `deu`, `cat`, `ces`, `ita`, `rus`, `slk`, `ron`, `arb`, `ukr`), illustrating consistent reductions across nearly all languages (e.g., `ind` drops from $\sim 0.37\%$ to $\sim 0.13\%$, `pes` drops from $\sim 0.36\%$ to $\sim 0.13\%$, `fin` drops from $\sim 0.21\%$ to $\sim 0.05\%$).

---

## Appendix E: SacreBLEU signatures

* **Signature:**
  `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1`

* **Except for `cmn`, `jpn`, `tha`, `lao` and `mya` with character-level tokenization:**
  `nrefs:1|case:mixed|eff:no|tok:char|smooth:exp|version:2.3.1`