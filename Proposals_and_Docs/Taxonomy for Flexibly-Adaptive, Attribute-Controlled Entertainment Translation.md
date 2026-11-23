# **Taxonomy for Flexibly-Adaptive, Attribute-Controlled Entertainment Translation**

## **Core idea** 

Entertainment translation currently:

1. **Faithful transfer** of a source text/audio/video into a target language, and  
2. **A single, fixed editorial judgment** about what content is "appropriate" for *some* imagined audience (usually decided once, upstream, by a broadcaster, censor board, or dubbing studio).  
3. 

We want to **decouple** these. Translation should produce a *space* of valid outputs constrained by 

(a) meaning-preservation and 

(b) a set of consumer- or context-set parameters

| Axis | What it controls | Nature of the control | Example |
| ----- | ----- | ----- | ----- |
| **A : Content Intensity Knobs** | How strongly a sensitive attribute already present in the source is expressed | A **dial** (0→max), symmetric, can go up or down, ideally reversible | Turn profanity from "graphic" to "mild" |
| **B : Cultural Localization** | How a culture-bound reference/expression is rendered so it lands naturally for the target audience | Substitution, Omission, Addition or Simplification | "NFL Superbowl" →the Huge American Football League |
| **C : Audience/Context Profile** | Which settings Axis A and B should take, for whom, right now | A **preference/policy layer** (metadata), set by the viewer, a guardian, a platform default, or law | "Family viewing, kids present, rural, Hindi" |

## **1\. Define "culture"** 

Anchored on **Singh, Patidar & Vig (2024), "Translating Across Cultures: LLMs for Intralingual Cultural Adaptation"** (CoNLL 2024\) 

* **Hall's Iceberg / Triad model of culture** (technical / formal / informal layers — visible, semi-visible, invisible), via Katan (2014). Used to decide *what kind* of content is even localizable:  
  * **Technical layer** (food, dress, institutions, visible behaviour) → learnable, explicitly translatable, our main CSI target.  
  * **Formal layer** (rituals, customs, appropriacy, Skopos-oriented) → translatable but requires functional/communicative equivalence, not literal mapping.  
  * **Informal layer** (values, orientations \- power distance, individualism, thinking style) → "out-of-awareness," this is closer to the register/worldview of the whole piece and is where over-adaptation risks erasing authorial intent   
* **Culture-Specific Items (CSI)**, following Newmark (1988) as used in Singh-CoNLL: Ecology; Material Culture; Social Culture; Institutions/Organizations/Ideas; Gestures & Habits, plus Singh-CoNLL's four entertainment-specific additions: **Slang/Figure of Speech, Offensive Content, Socially Sensitive/Taboo Topics, Humour.**  
* **Foreignness levels (1/2/3)** from Singh-CoNLL:   
  * (1) foreign-origin but locally assimilated (pizza, cricket, coffee \- don't touch),   
  * (2) recognized but not mainstream (sushi, k-pop, beer — *localize if audience profile calls for it*),   
  * (3) largely unfamiliar (kimono, rodeo, Thanksgiving — *default to localize/explain*).   
* **Davies (2003)'s seven CSI translation strategies**, as catalogued in Singh-CoNLL: Preservation, Addition, Omission, Globalisation, Localisation, Transformation, Creation. 

## **2\. Axis A : Content Intensity Knobs** 

### **2.1 Profanity / Verbal Aggression**

**Sub-categories:** 

- mild expletives   
- religious/blasphemous swearing   
- scatological language   
- sexual profanity   
- identity-directed slurs : **Slurs targeting a protected identity (race/caste/religion/gender/sexuality) cant be "dialled down"** They need a categorical policy (retain-with-context-warning, or replace-with-narratively-equivalent-non-slur insult), because "slightly less racist" is not a coherent output.  
- sarcasm/insult intensity (non-swear verbal aggression, e.g. "you absolute waste of oxygen").

**Modality:** 

- text (subtitles),   
- audio (voice delivery/loudness/tone of the dub actor where swearing can is softened lexically but still be shouted)  
- on-screen text (graffiti, signs).

|  | Example |
| ----- | ----- |
| **Original (EN, source)** | "What the hell is wrong with you, you absolute dumbass? Get the f\*\*\* out of my house\!" |
| **Ideal, tuned down (Hindi, family setting)** | "आख़िर तुम्हें हुआ क्या है? यहाँ से निकल जाओ\!" *(lit. "What on earth is wrong with you? Get out of here\!")* |
| **Problematic/ not true to source** | "तुम्हें क्या दिक्कत है, यार, यहाँ से जाओ।"  reads as mild annoyance between friends, losing the *rage* of the scene entirely |

### **2.2 Sexual Content / Obscenity**

**Sub-categories:** 

- romantic/suggestive dialogue  
- innuendo/double-entendre   
- non-explicit intimacy (kissing, implied)   
- explicit dialogue   
- nudity (visual only)   
- simulated sexual acts (visual/audio)   
- sexual violence (categorically different, generally should not be "tuned," only gated/warned).

|  | Example |
| ----- | ----- |
| **Original** | "I've been thinking about last night all day." *(scene continues into an explicit visual sequence)* |
| **Ideal, tuned down** | Dialogue kept as-is (it's mild) but the **visual layer** is handled by the video-knob: cut to a fade/ellipsis edit at the point |
| **Problematic** | Silently deleting the entire scene, including the *plot-relevant* conversation that follows it the next morning referencing what happened  |

### **2.3 Violence**

**Sub-categories:**

- slapstick/cartoon violence   
- physical fights (non-graphic)   
- weapons use  
-  graphic/gory violence   
- torture   
- war/mass-casualty depiction   
- self-harm depiction \-  
- Psychological/emotional abuse (non traditional coverage)

|  | Example |
| ----- | ----- |
| **Original** | Extended, graphically-depicted torture scene with detailed screams and visible injury, dialogue: "Tell me where it is, or I'll make sure you never walk again." |
| **Ideal, tuned down (kids' profile)** | Threat dialogue retained (villain is still menacing, plot stakes intact); visual cuts away before injury is shown; audio screams reduced/muted; narration/subtitle may add "\[he was tortured, off-screen\]" if the plot needs the audience to know it happened. |
| **Problematic** | Removing the entire interrogation scene,  |

### **2.4 Substance Use (alcohol, drugs, smoking, gambling)**

**Sub-categories:** 

- casual/social use (a toast, a cigarette as characterization)   
- abuse/addiction depiction   
- drug-dealing/procurement scenes   
- glamorization vs. consequence-shown depiction. Many jurisdictions (India included) require a **static disclaimer card** for this rather than content alteration .

### **2.5 Horror / Disturbing Imagery**

**Sub-categories:** 

- jump-scares   
- body horror   
-  dread/atmosphere (non-graphic psychological horror)   
- disturbing-but-bloodless imagery (this is the category kids' content most needs a knob for, since it's not "violence" or "gore" per se.

### **2.6 Discriminatory / Hateful Content** 

**Sub-categories:** 

- racist  
- casteist,   
- communal/religious-hatred,   
- misogynistic, homophobic content..

## **3\. Axis B : Cultural Localization Categories (the "substitutions")**

This axis is **not a dial**. A culturally foreign reference doesn't get "less foreign" by turning a knob down — it gets **replaced, explained, generalized, or kept**, using one of Davies' (2003) strategies as catalogued by Singh-CoNLL:

| Strategy | What it does | Example (Newmark/Singh-CoNLL-style, invented) |
| ----- | ----- | ----- |
| **Preservation** | Keep the foreign term as-is (no adaptation needed/wanted) | "Rotten Tomatoes" kept as "Rotten Tomatoes" — a global brand name best left untranslated |
| **Addition** | Keep term, add a short gloss | "Wendy's" → "Wendy's, an American fast-food chain" |
| **Omission** | Drop the term when no equivalent exists and it's non-essential | "getting a taco from Taco Bell" → "getting a taco" |
| **Globalisation** | Replace with a broader, culture-neutral term | "Kimono" → "traditional garment"; "Hamburger" → "burger" |
| **Localisation** | Replace with a local functional equivalent | "sausage" → "kebab"; "Wendy's" → "Haldiram's" |
| **Transformation** | Swap for a *different* but locally-resonant item (not an equivalent, an analogue) | "football game" → "local cricket match" |
| **Creation** | Introduce a new locally-flavoured item where the source had none (used cautiously — risk of over-localizing, per Singh-CoNLL's finding that this correlates with *lower* naturalness/content-preservation scores) | Adding a culturally specific idiom not present in the source at all |

**CSI category × modality matrix (our extension for entertainment beyond text):**

| CSI Category (Newmark/Singh-CoNLL) | Text example | Audio-only equivalent | Visual/video equivalent |
| ----- | ----- | ----- | ----- |
| Ecology (flora/fauna/geography) | "Grand Canyon" | Regional place-name pronunciation/accent cues | A shot of a canyon vs. a shot of a ghat/valley recognisable to target audience |
| Material Culture (food, clothes, transport) | "hamburger," "sweater" | — | On-screen food/clothing — video *cannot* be relexicalized; requires either subtitle-note or accepting a visual mismatch (a known hard limit of dubbing vs. text translation) |
| Social Culture (leisure, social occasions) | "Thanksgiving," "prom" | Background music genre carrying a "festive/solemn" cue that doesn't map 1:1 | A wedding scene's visual iconography (real limit — cannot relocalize footage) |
| Institutions (brands, agencies) | "FICA" → "Income Tax" | — | Signage/logos on screen (same hard limit as above) |
| Gestures & Habits | "honk honk," "jazz hands" | Non-verbal vocalizations, laughter type | A hand gesture that is innocuous in the source culture but obscene in the target (this is a *video-only* CSI category with no text analogue — needs its own handling, e.g. reframing/cropping or an on-screen note) |
| Slang / Figure of Speech | idioms, wordplay | Prosody/rhythm that carried a pun (often untranslatable — a known, accepted lossy case) | — |
| Humour | puns, culturally-specific jokes | Comic timing/pause structure | Visual gags (harder to "translate" at all — usually preserved as-is since humor is often visual/physical and language-independent) |
| Offensive Content / Taboo Topics | swear words, taboo references | Delivery tone | — (interacts directly with Axis A §2.1–2.2) |

**Key methodological point carried over from Singh-CoNLL's findings (RQ3/RQ4):** across all evaluated LLMs, *higher localization scores correlated with lower naturalness and content-preservation scores* — i.e., aggressive localization risks distorting meaning and introducing unintended stereotypes/generalizations about the target culture. This is direct evidence for why **Axis B needs a "how much to localize" throttle of its own** (tied to Axis C's familiarity/foreignness-level parameter, §4.3), not just an on/off switch. Singh-CoNLL explicitly recommend a *selective* adaptation approach rather than adapting every CSI.

**Multimodal note:** all four attached papers (Singh-CoNLL, Khanuja-CuE, CASAT/Singh-Zaki-Wasnik, and the ACT/PREF-FAME-MT paper) are **text-only**. Extending Axis B to audio/video is, as far as we can tell, an open contribution of your work — worth stating explicitly in your framing rather than assuming prior art covers it.

## **4\. Axis C :  Audience/Context Profile**

This is the **policy layer**: the set of parameters that decides where Axis A's knobs and Axis B's localization-throttle should sit.

### **4.1 Age bracket**

We recommend anchoring sub-brackets to a real-world, already-legitimized scheme rather than inventing our own from scratch: the **CBFC (Central Board of Film Certification, India)**'s 2023-amended categories are a useful, India-relevant reference point:

* U (Universal), U/A 7+, U/A 13+, U/A 16+, A (Adults only), S (restricted/professional audiences). This gives us **6 ordered tiers** rather than a binary kid/adult split — directly useful as a default preset for Axis A's knob positions (a "U/A 13+" preset ≠ "U/A 7+" preset on the violence knob, etc.). Comparable Western analogues (MPAA ratings G/PG/PG-13/R/NC-17; BBFC U/PG/12/15/18) can be cited for cross-cultural comparison of *where the lines get drawn differently* (e.g., US ratings are historically stricter on sexual content than violence, relative to European boards) — itself an argument for why *fixed national rating boards* are a blunt instrument compared to a personalizable system like ours.

### **4.2 Viewing companionship / social context**

Solo · with partner · family (mixed generations) · with kids specifically · with friends/peers · public/theatre setting. This is **orthogonal to age** — the "adult watching an explicit film but with family present" case in your prompt is exactly this: the *viewer's own* age-appropriateness is fine, but the *context* isn't, which is why this needs to be a separate parameter, not folded into "age bracket."

### **4.5 Culture \- Region / language / dialect / other demographics**

Which target language, which register conventions apply (e.g., Hindi tu/tum/aap pronoun-formality choices, addressed directly by the **CoCoA-MT / MT-GenEval / PREF-FAME-MT** line of work in the attached preference-optimization paper), which locally-known referents exist for Axis B substitution.

Moreover, not just this, but also cultural background should help us eventually tune our Axis A with th ehelp of our survey or synthetic user profiles \- where people come from or their background often dictates what they prefer to tune out or tune in while watching.

### **4.6 Genre expectation**

The "acceptable dosage" of every Axis A knob is **genre-relative**, not absolute: a horror film's baseline gore expectation is categorically higher than a family sitcom's, even within the *same* age-bracket audience. Genre should therefore act as a **baseline offset** that Axis A knobs are tuned *relative to*, not an independent knob of its own. Similarly for an awareness based movie like something raising awareness against racism or casteism.

### **4.7 Forewarnings**

I was also exploring tagging \- the movie/tv show could be tagged, along with the usual age ratings and sexual abuse etc. With more of our attributes, and every few seconds before something the user might want to rule out comes up, we can show a warning so they may skip or mute the content for that part. 

## **6\. Localization/Personalization vs. Censorship**

**Working definition of censorship (for this project):** an alteration or suppression of content, imposed by a third party (state, broadcaster, platform) rather than chosen by the consuming audience or their guardian, that is **opaque** (the audience is not told what was changed or why), **non-reversible** (no access to the unmodified original), and/or **erases rather than substitutes** meaning, frequently in ways that disproportionately target political, religious, or social critique rather than the sensory intensity of a scene.

**Ours:** a **transparent, reversible, consumer-set (or guardian-set) preference layer**, defaulting to the unmodified source, that **substitutes or dampens rather than deletes**, 

## **8\. Current pain points in entertainment translation (India)**

Synthesized from the attached papers plus general domain knowledge:

1. **Sentence-isolated translation, no session/scene context**   
2. **Static, one-shot style transfer**   
3. **Literal translation of idiom/culture-bound reference**   
4. **No sub-national register modeling**   
5. **Idiom/knowledge-base approaches don't cover Indian languages**   
6. **Inconsistent, opaque, all-or-nothing censorship**   
7. **No user-level personalization exists today**  
8. **Gender/formality attribute control is still an open, hard problem even at the text level** — the attached ACT/preference-optimization paper shows that even *just* two attributes (formality, gender) across 71 language pairs required dedicated contrastive datasets and preference-optimization (not just prompting) to get right in small-medium models, and even then, attribute accuracy gains routinely traded off against translation quality — a caution for how hard it will be to extend to a much larger knob set (Axis A) reliably.

*Sources cited: Singh, Patidar & Vig (CoNLL 2024\) — "Translating Across Cultures"; Khanuja et al. (2026) — "Steering LLMs for Culturally Localized Generation" (Google DeepMind); Singh, Zaki & Wasnik (AAAI-25) — "Enhancing Entertainment Translation for Indian Languages Using Adaptive Context, Style and LLMs" (CASAT); anonymous ACL submission — "Attribute-Controlled Translation with Preference Optimization" (PREF-FAME-MT); Adilazuarda et al. (2024); Newmark (1988); Davies (2003); Katan (2014); Logacheva et al. (2022, ParaDetox); Dementieva et al. (2024, MultiParaDetox); Dathathri et al. (2020, PPLM); Keskar et al. (2019, CTRL); Krause et al. (2021, GeDi); Rafailov et al. (2023, DPO); Central Board of Film Certification, India (Cinematograph Act 1952, as amended 2023).*

