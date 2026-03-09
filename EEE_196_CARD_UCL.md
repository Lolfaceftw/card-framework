CARD: Constraint-aware Audio Resynthesis and Distillation

Undergraduate Student Project by

Rei Dennis Agustin 2022-03027 BS Electronics Engineering Sean Luigi P. Caranzo 2022-05398 BS Computer Engineering

Johnbell R. De Leon 2021-01437 BS Computer Engineering Christian Klein C. Ramos 2022-03126 BS Electronics Engineering

![Figure 1](images/EEE_196_CARD_UCL-picture-001.png)

Visible text:
[unclear signature]
Adviser:

Layout sketch:
```text
[unclear signature]
Adviser:
```

Details:
The image shows a handwritten signature in black ink above the printed word "Adviser:". The signature is stylized and difficult to decipher, appearing to start with a large capital letter that resembles a 'D' or 'P' with a crossbar, followed by cursive letters that may spell "Hien" or similar. A long, sweeping stroke extends from the middle of the signature to the right. The word "Adviser:" is printed in a standard serif font directly below the signature, indicating a designated space for an adviser's signature on a document.

Adviser:

Rowel D. Atienza

University of the Philippines, Diliman December 2025

# Abstract

# CARD: Constraint-aware Audio Resynthesis and Distillation

The exponential growth of long-form podcasting creates a consumption bottleneck, as listeners lack efficient means to digest multi-speaker content within limited timeframes. Current summarization approaches, whether text-based or extractive audio clipping, fail to preserve the immersive, prosodic nature of conversational audio. This proposal introduces CARD (Constraint-aware Audio Resynthesis and Distillation), a generative framework designed to resolve the trade-off between consumption efficiency and audio fidelity. We propose a pipeline that addresses three critical challenges in audio generation. First, for temporal control, the system utilizes forced alignment ingestion to calculate speaker-specific speaking rates, enabling a Large Language Model (LLM) to compress dialogue into a structured representation that strictly adheres to a user-defined time budget. Second, for spectral control, the system utilizes diarization timestamps to harvest reference samples directly from the raw input, driving zero-shot voice cloning via IndexTTS2. Third, for conversational control, we introduce a refinement module using a 4-bit quantized Mistral model to predict semantic interjection points, generating syntactically-aware asynchronous overlaps. The expected outcome is a functional prototype that validates the feasibility of duration-controlled, high-fidelity conversational resynthesis.

# Contents

| Section | Page |
| :--- | :--- |
| List of Figures | iv |
| List of Tables | v |
| 1 Introduction | 1 |
| 1.1 The Information-Experience Gap | 2 |
| 1.2 Limitations of Current Modalities | 3 |
| 1.3 The CARD Paradigm: From Extraction to Resynthesis | 4 |
| 2 Related Work | 7 |
| 2.1 Automatic Speech Recognition and Speaker Diarization | 7 |
| 2.1.1 State-of-the-Art Diarization Architectures | 7 |
| 2.1.2 Hybrid Whisper-NeMo Diarization Framework | 8 |
| 2.2 Neural Speech Separation for Overlapping Speakers | 8 |
| 2.2.1 End-to-End Neural Separation Architectures | 8 |
| 2.2.2 Data Sampling and Conversational Robustness | 9 |
| 2.2.3 Limitations Toward Podcast-Style Audio | 9 |
| 2.3 Duration-Constrained Semantic Summarization | 10 |
| 2.3.1 Limitations of Unconstrained Abstractive Summarizers | 10 |
| 2.3.2 Constraint-Aware Summarization and Structured Generation | 10 |
| 2.4 IndexTTS2: Voice Cloning and Neural Synthesis | 11 |
| 2.4.1 Autoregressive vs. Non-autoregressive Paradigms | 12 |
| 2.4.2 Disentanglement of Timbre and Style | 12 |
| 2.4.3 The Temporal Control Gap | 13 |
| 2.5 Mistral: Conversational Dynamics and Automated Backchanneling | 13 |
| 2.5.1 Limitations of Rule-based Heuristics | 13 |
| 2.5.2 Semantic Prediction via Quantized LLMs | 14 |
| 2.5.3 Justification of Using Mistral | 14 |
| 3 Problem Statement and Objectives | 16 |
| 3.1 Problem Statement | 16

| Section | Page |
| :--- | :--- |
| 4.1.1 System Architecture | 19 |
| 4.1.2 Evaluation Metrics | 20 |
| 4.1.3 Integration with Downstream Modules | 21 |
| 4.2 Speaker Audio Extraction | 21 |
| 4.2.1 Timestamp-Guided Audio Chunking and Extraction | 21 |
| 4.2.2 Targeted Separation with SepFormer | 21 |
| 4.2.3 Enrollment Embedding and Speaker Assignment | 22 |
| 4.2.4 Speaker-Pure Audio Track Construction | 23 |
| 4.2.5 Outputs | 23 |
| 4.3 Script Summarizer Module | 23 |
| 4.3.1 Word Budget Derivation | 23 |
| 4.3.2 Model Selection and Benchmarking | 24 |
| 4.3.3 Justification via Literature | 25 |
| 4.3.4 Structured JSON Output | 26 |
| 4.3.5 Validation and Error Checking | 26 |
| 4.4 Voice Cloning + Backchanneling + Interjector | 26 |
| 4.4.1 Zero-Shot Voice Cloning (IndexTTS2) | 26 |
| 4.4.2 Conversational Refinement via Mistral | 28 |
| 4.4.2.1 Trigger Word Detection | 28 |
| 4.4.2.2 Context-Aware Response Generation | 28 |
| 4.4.3 Acoustic Alignment and Asynchronous Overlay | 28 |
| 4.5 Justification of the Model Selection | 30 |
| 4.5.1 Speaker Identity Retention (SS) | 30 |
| 4.5.2 Emotional Control and Fidelity (ES) | 30 |
| 5 Preliminary Findings | 31 |
| 5.1 Audio2Script | 31 |
| 5.1.1 Findings | 31 |
| 5

iii

# Appendices 41

- A Mistral Find Trigger Prompt 41
- B Generate Interjection 43

# List of Figures

| Figure | Description | Page |
| :--- | :--- | :--- |
| 1.1 | The Fidelity-Efficiency Frontier. Current consumption modalities (gray) force a trade-off between speed and immersion. CARD (blue) aims to break this frontier by delivering high-fidelity audio within short time constraints. | 2 |
| 1.2 | Data Flow of the CARD System. The diagram illustrates the specific data exchange between modules, highlighting the extraction of samples for voice cloning and the sequential hand-off from IndexTTS2 to Mistral. | 6 |
| 2.1 | Benchmarking IndexTTS2 against State-of-the-Art Baselines. Performance comparison on the LibriSpeech test-clean dataset. IndexTTS2 (Blue) demonstrates superior performance across all metrics, particularly in Speaker Similarity (SS) and Intelligibility (WER), validating its selection for the CARD pipeline over diffusion-based alternatives like MaskGCT and F5-TTS. | 11 |
| 2.2 | Comparative Analysis of Small Language Models (SLMs) The table highlights the architectural suitability of Mistral 7B (Green) compared to newer state-of-the-art models (Red). While Llama 3.1 and Qwen 2.5 offer larger vocabularies and safety features, these attributes manifest as VRAM bottlenecks and false refusals within the constrained CARD pipeline. | 14 |
| 4.1 | Radar chart comparison of candidate models. Copilot (blue) encompasses the largest area, indicating superior performance across both quantitative and qualitative metrics. | 25 |
| 4.2 | The Voice Cloning Module Pipeline. | 27 |
| 4.3 | Post-Hoc Acoustic Alignment Logic. The diagram illustrates how the system converts a semantic trigger point (t pos ) into an acoustic timestamp (ttrigger) and adds a randomized reaction delay (tdelay) to determine the final asynchronous overlay position (dinterjection). | 29 |

# List of Tables

| Table/Figure | Page |
|---|---|
| 4.1 Comparative evaluation of LLM backends for script summarization. Copilot (GPT-5) demonstrates the highest performance across semantic fidelity (BERTScore F1) and structural constraint adherence (Length Acc). | 24 |
| 6.1 Project Schedule Phase 1: Setup and Initialization (Weeks 1–4) | 38 |
| 6.2 Project Schedule Phase 2: Refinement and Integration (Weeks 5–8) | 38 |
| 6.3 Project Schedule Phase 3: Testing and Optimization (Weeks 9–12) | 38 |
| 6.4 Project Schedule Phase 4: Closing and Defense (Weeks 13–18) | 39 |

# Chapter 1

# Introduction

The exponential growth of long-form podcasting has generated a severe "consumption bottleneck," where the production of high-fidelity, multi-speaker context vastly outpaces the cognitive capacity of listeners to digest it. As of 2025, the global podcast audience has surpassed 584 million with millions of hours of audio content uploaded annually [1]. While listeners increasingly rely on summarization to manage this information overload [2], current methodologies force a destructive trade-off between efficiency and experience. Text-based summarization strips away the paralinguistic channel: removing the prosody, emotional intonation, and speaker identity that constitute the core value of the medium [3, 4]. Conversely, extractive audio summarization like clipping and stitching original audio preserves vocal fidelity but often results in disjointed, jarring listening experiences that lack narrative coherence and temporal flexibility [5].

To bridge this gap, we propose Constraint-aware Audio Resynthesis and Distillation (CARD), a multiobjective generative framework designed to synthesize an summarized audio that is both temporally compressed and acoustically authentic. Unlike extractive methods, CARD employs a sequential pipeline initiated by a duration-constrained user prompt (e.g., "summarize this in 5 minutes"). First, the Whisper [6] architecture performs forced-alignment transcription and speaker diarization to accurately disentangle multi-party audio [6]. This structured text is processed by a constraint-aware Large Language Model (LLM) that generates a dialogue-based summary strictly adhering to the user's defined time budget while injecting emotional syntax tags using natural language description through JSON. Finally, the system utilizes IndexTTS2 [7] for zero-shot spectral cloning, automatically extracting reference embeddings from the source audio to preserve speaker identity [7], augmented by a 4-bit quantized Mistral 7B [8] parameter module that predicts semantic interjections to restore the natural overlap of human conversation [9]. By shifting the paradigm from extraction to resynthesis, CARD validates that high-efficiency audio consumption need not

Figure 1.1: The Fidelity-Efficiency Frontier. Current consumption modalities (gray) force a trade-off between speed and immersion. CARD (blue) aims to break this frontier by delivering high-fidelity audio within short time constraints.

![Figure 2](images/EEE_196_CARD_UCL-picture-002.png)

Visible text:
High
Original Audio
Bridging the Gap
CARD (Proposed)
2x Playback
Fidelity (Immersion / Prosody)
Low
Current Trade-off Frontier
Text Summary
Efficiency (Speed / Time Saved)
Low
High

Layout sketch:
```text
High
|
|  Original Audio (dot)
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
```

come at the cost of the immersive, conversational experience.

# 1.1 The Information-Experience Gap

The podcast consumption landscape is more so defined by a fundamental bandwidth disparity between human speech production and information processing capacity. Research indicates a significant divergence between the average rate of intelligible speech, hovering around 150-160 words per minute (wpm), and the average silent reading rate of approximate 238 wpm [10]. The physiological constraint means that listening to a standard hour-long podcast is inherently less efficient than reading its transcript, imposing a "time tax" on the user. As the volume of content increases, with over 4.5 million podcasts indexed globally as of 2025 [11], listeners face a mathematical impossibility in keeping pace with their subscriptions. This creates a bottleneck where high-value content is often abandoned due to the temporal cost of linear consumption [2].

Compounding this bandwidth issue is the "linearity constraint" inherent to the audio modality. Unlike text, which facilitates non-linear browsing strategies like saccadic scanning and keyword spotting ("skimming") [12], audio is opaque and strictly serial. A listener cannot just visually glance at a waveform to extract the gist of a segment; they must commit to playing it linearly to reveal its semantic context effectively. While navigation mechanism like "scrubbing" exist [13], they lack semantic granularity, turning the search for specific information into a high-friction, trial-and-error process. This lack of random access forces a

binary choice: either consume the entire file to ensure context retention or skip sections and risk losing the narrative thread.

However, the value of the podcast medium extends beyond mere information transfer: it is deeply rooted in paralinguistic engagement. Listens will form "parasocial relationships" with hosts based not just on what is said, but how it is said—the prosody, emotional intonation, and dynamic chemistry between speakers [4]. This experiential layer constitutes the core appeal of the format. Therefore, a gap emerges when users seek efficiency: current compression methods like reading a summary destroy this experiential layer, effectively stripping "the ghost from the machine." The emotional resonance of a debate or the comedic timing of banter is lost when reduced to text, rendering the consumption experience sterile and stale. This dichotomy created a "Fidelity-Efficiency Frontier," as illustrated in Fig. 1.1, where CARD is designed to occupy the previously unattainable quadrant of high efficiency and high fidelity.

# 1.2 Limitations of Current Modalities

Attempts trying to resolve the information-experience gap have historically relied on distinct modalities, like text summarization, extractive clipping, and generative synthesis, each forcing a compromise that degrades the overall utility of the podcast format. Onward we discuss the following limitations:

1. Text-Based Summarization. While Large Language Models (LLMs) can effectively distill hourlong transcripts into concise reading material, this conversation effectively strips away the paralinguistic channel entirely. Studies in prosody analysis demonstrate that vocal cues–such as pitch contours, hesitation, and volume dynamics–carry significant semantic weight [3]. By reducing a heated debate or a comedic dialogue to flattened text, these systems eliminate the "textual prosody" required for emotional engagement [4]. The listener receives the information but loses the performance, rendering the content indistinguishable from a standard article.
2. Extractive Audio Summarization. Conversely, extractive methods attempt trying to preserve the original voice by stitching together salient segments of the waveform [5]. While this maintains vocal fidelity, it does introduce severe discontinuity artifacts. The resulting audio is often characterized by jarring acoustic transitions and abrupt topic shifts that increase cognitive load. Furthermore, extractive algorithms are temporally rigid, they cannot reorganize a 60-minute interview into a precise 5 to 10-minute summary without rendering the dialogue unintelligible.
3. Notebook LM as a Standard. The current gold standard for conversational summarization is Note-

bookLM by Google [14]. They utilize a feature called "Audio Overview" to generate podcast-style discussions from source tests. While NotebookLM excels at conversational naturalness (modeling backchanneling, disfluencies, and dynamic banter), it fundamentally fails at identity preservation, It relies on a fixed, pre-set stock voices (only two) rather than cloning the original speakers. If a user summarizes a podcast featuring Elon Musk and Sam Altman, NotebookLM replaces their unique vocal signatures with generic AI voices, destroying the parasocial connection that defines the medium. Additionally, it lacks strict temporal control, often regarding precise duration constraints [15].

4. The CARD Gap. There remains a critical disconnect: no current system offers identity-preserving resynthesis. Users are forced to choose between the authentic voice (extractive and disjointed), the efficient text (coherent and silent), or the natural conversationalist (NotebookLM generic voices). CARD targets this specific intersection: generating a summary that is as coherent as text, as natural as Notebook LM, but acoustically identical to the original speakers. As illustrated by Fig. ?? 1 , we propose an equal contribution among duration control, identity preservation, conversational naturalness, and semantic coherence.

# 1.3 The CARD Paradigm: From Extraction to Resynthesis

Henceforth, to resolve the Fidelity-Efficiency Frontier, we introduce CARD. Unlike preceding approaches that operate on the surface level of the waveform (clipping) or semantic level of the text (summarization), CARD functions as a full-stack generative pipeline. It treats the podcast not as a static recording, but as a malleable data stream.

The architecture as illustrated in 1.2 is operated through a sequential flow:

1. Ingestion and Analysis. We initiate the process by feeding the Raw Podcast Audio and User Prompt (the target duration) into Whisper [6]. This module performs forced alignment to generate a timestamped script, identify speaker changes, and crucially, calculate the original speaking rate (words per minute) of the hosts.
2. Parallel Decomposition. The data stream then splits into two specialized tasks:
3. (a) Acoustic Extraction (Speaker Diarization). Utilizing the timestamps provided by Whisper, the diarization module will query the audio file to extract the reference waveform samples. It

1 The Spider Plot data points are subjective and are not based on actual data points. This is primarily for visual purposes according to our research. In the future, we plan to benchmark each model to duration control, identity preservation, conversational naturalness, and semantic coherence.

isolates N distinct audio clips corresponding to the N speakers identified in the session, creating the voice samples for the zero-shot cloning.

- (b) Distillation (Summarizer LLM). Simultaneously, algorithmically, we can calculate the precise word count budget by multiplying the WPM derived from the Whisper module with WPM × Minutes. This precise word count will be fed as a prompt to the LLM to compress the summary into the required word count, eventually becoming a structured JSON output including but not limited to the speaker , message, and emo text for every text segment.
3. Resynthesis. The semantic and acoustic streams converge at the Voice Cloner (IndexTTS2) [7] module. This model utilizes the extracted voice samples to clone the speakers' identities while reading the JSON-structured summary. Finally, the synthesized segments are passed to the Backchanneler, Mistral 7B quantized [8]. This module analyzed the text flow to identify if the text segment required interjections like "gotcha", "uh-huh" or "hmm". The system is then passed to an algorithm to overlap these interjections using a timing algorithm , merging everything to produce the Final Audio Summary, emanating to restore the dynamic of human interaction [9].

We propose three primary contributions to the field of audio processing:

1. We propose the first end-to-end architecture that combines duration-constrained summarization with multi-speaker voice cloniing, effectively solving the trilemma of speaker identity preservation, prosodic naturalness, and temporal controllability .
2. We introduce a method for restoring the asynchronous nature of spontaneous speech by overlapping interjections, utilizing Mistral 7B Quantized to mitigate the robotic sequentiality of standard TTS.
3. We demonstrate that high-fidelity consumption need not sacrifice efficiency.

Figure 1.2: Data Flow of the CARD System. The diagram illustrates the specific data exchange between modules, highlighting the extraction of samples for voice cloning and the sequential hand-off from IndexTTS2 to Mistral.

![Figure 3](images/EEE_196_CARD_UCL-picture-003.png)

Visible text:
Raw Podcast Audio
User Prompt
Audio to Script (Whisper)
WPM Data, Speaker ID, Msg, Timestamp
Summarizer (LLM)
JSON (Emo, Spk, Msg)
Voice Cloner (IndexTTS2)
Text Segment
Backchanneler (Mistral 7B)
Insert Interjection & Overlap Merge
Final Audio Summary
Per-speaker Audio
Speaker ID, Msg, Timestamp
Provide N Samples
Get Sample (Time)
Provide N Samples

Layout sketch:
```text
[Raw Podcast Audio]      [User Prompt]
       \                      /
        \                    /
         \                  /
          v                v
    [Audio to Script (Whisper)]
          |                    |
          | (WPM Data...)      | (Speaker ID...)
          v                    v
[Summarizer (LLM)]      [Per-speaker Audio]
          |                    |
          | (JSON...)          | (Provide N Samples)
          v                    v
    [Voice Cloner (IndexTTS2)] <----+
          |                         |
          | (Text Segment)          |
          v                         |
[Backchanneler (Mistral 7B)]        |
          |                         |
          | (Insert Interjection...)|
          v                         |
[Final Audio Summary]               |
                                    |
                                    +----> (Get Sample (Time)) loop back to Per-speaker Audio
```

Details:
The image is a flowchart diagram illustrating a pipeline for generating a final audio summary from raw podcast audio. The flow generally moves from top to bottom.

- **Inputs:** At the top, two beige boxes labeled "Raw Podcast Audio" and "User Prompt" serve as the initial inputs.
- **Transcription:** Arrows from both inputs converge on a pink box labeled "Audio to Script (Whisper)".
- **Branching:** From the "Audio to Script" box, the process splits into two parallel paths:
    - **Left Path:** An arrow labeled "WPM Data, Speaker ID, Msg, Timestamp" points down to a blue box labeled "Summarizer (LLM)".
    - **Right Path:** An arrow labeled "Speaker ID, Msg, Timestamp" points down to a blue box labeled "Per-speaker Audio".
- **Feedback Loop:** An orange dashed arrow labeled "Get Sample (Time)" loops from the "Per-speaker Audio" box back up to the "User Prompt" box. Another orange arrow labeled "Provide N Samples" goes from "Per-speaker Audio" to the "Voice Cloner" box.
- **Voice Cloning:** The left path continues from "Summarizer (LLM)" with an arrow labeled "JSON (Emo, Spk, Msg)" pointing to a teal box labeled "Voice Cloner (IndexTTS2)". This box also receives the "Provide N Samples" input from the right path.
- **Backchanneling:** An arrow labeled "Text Segment" leads from "Voice Cloner" to a teal box labeled "Backchanneler (Mistral 7B)".
- **Final Output:** An arrow labeled "Insert Interjection & Overlap Merge" leads from "Backchanneler" to the final green box at the bottom, labeled "Final Audio Summary".
- **Styling:** Boxes have rounded corners and drop shadows. Arrows are solid black or orange, with one dashed orange arrow indicating a feedback loop. Text labels on arrows describe the data being passed.

# Chapter 2

# Related Work

# 2.1 Automatic Speech Recognition and Speaker Diarization

The foundational layer of the CARD pipeline requires the transformation of raw acoustic signals into a timestamped and speaker-attributed transcription. While modern Automatic Speech Recognition (ASR) systems have achieved near-human parity in transcription accuracy [16], the separation of multi-speaker audio, known as speaker diarization, remains a challenge in conversational AI, particularly in scenarios involving rapid speaker turns and overlapping speech [17].

# 2.1.1 State-of-the-Art Diarization Architectures

Traditional diarization pipelines typically operate independently of the ASR module. Systems such as NVIDIA NeMo employ a modular approach involving Voice Activity Detection (VAD), speaker embedding extraction, and clustering algorithms to partition audio into homogenous speaker segments [18]. While these systems excel at identifying speaker turns in noisy environments, they often decouple the acoustic segmentation from the semantic transcription. Previous studies have demonstrated that supervised ASR architectures, such as the Conformer models used in NeMo, exhibit significant performance degradation and phonetic instability when applied to messy, natural conversations that fall outside the model's training distribution compared to large-scale weakly supervised models [19]. Reinforcing this, our preliminary experiments revealed that while modular diarizers like NeMo provide accurate speaker assignment, they are prone to phonetic errors and word-error rate (WER) degradation in casual settings, such as transcribing "blood boiling" as "blood bowling."

Conversely, alignment-based approaches like Whisper [6] leverage the robust ASR capabilities of OpenAI's Whisper, forcing the alignment of words to timestamps to deduce speaker turns. While this method yields superior transcription quality, capturing hesitations and punctuation with high precision, it frequently struggles in assigning the correct speaker. Specifically, it often merges short interjections or fails to detect rapid speaker shifts in high-overlap scenarios, as the underlying ASR model is not optimized for multi-talker speech reconstruction [20].

# 2.1.2 Hybrid Whisper-NeMo Diarization Framework

To reconcile the trade-off between textual accuracy and speaker attribution stability, we utilize a hybrid Whisper-Diarization framework [21], combining both the strengths of NeMo and Whisper. Unlike purely end-to-end or strictly modular systems, this architecture decouples the transcription and diarization tasks into specialized streams that are subsequently reconciled. The pipeline employs the large-v2 Whisper model solely for generating a high-accuracy semantic transcript, ensuring minimal word error rates. Parallel to this, a specialized speaker embedding model extracts speaker signatures from the audio waveform. These heterogeneous streams, comprising semantic text and acoustic speaker clusters, are aligned via crossreferencing timestamps [21]. This hybrid approach enables the system to retain the high transcription quality of Whisper while leveraging the robust "who-spoke-when" detection of NeMo"

# 2.2 Neural Speech Separation for Overlapping Speakers

A persistent obstacle in diarization and transcription of conversational speech is the presence of overlapping speakers. Traditional diarization pipelines commonly treat speaker segmentation as a clustering task, assuming predominantly single-talker segments. However, natural conversations, particularly podcasts, exhibit frequent interruptions, affirmations, and co-speech behavior that degrade performance when forced into a single-stream transcription[22]. This motivates the adoption of speech separation front-ends, or "splitters," to disentangle acoustic mixtures before diarization and ASR processing.

# 2.2.1 End-to-End Neural Separation Architectures

Earlier methods such as time–frequency masking depended heavily on spectrogram domain priors and therefore struggled with phase inconsistency and reverberant environments. Luo and Mesgarani's ConvTasNet [23] introduced a fully time-domain architecture that surpassed even ideal T-F magnitude masking, proving that raw waveform modeling yields superior separation quality. Conv-TasNet represents a notable

milestone in separation fidelity, but it is sensitive to input length and requires careful sampling strategies to avoid overfitting [22].

More recently, Subakan et al. proposed an attention-driven model demonstrating that transformers alone, without recurrent or convolutional priors, are sufficient for speaker representation learning [24]. Their architecture establishes a global receptive field that better handles long-range dependencies in dialogs, an essential property for podcasts where contextual cues span multiple utterances. However, attention-based models remain computationally expensive, and inference latency poses practical limitations for real-time conversational agents.

# 2.2.2 Data Sampling and Conversational Robustness

Ravenscroft et al. [22] emphasize that training data composition critically impacts downstream diarization accuracy. Uniform random sampling inflates model performance on short single-speaker segments while degrading performance under realistic long-form conversational overlap ratios. The study suggests curriculum-style sampling strategies to better represent natural dialog dynamics. Without such balanced datasets, neural separators may hallucinate separation boundaries or collapse speakers with similar timbres, both of which are highly detrimental to pipeline-level ASR stability..

# 2.2.3 Limitations Toward Podcast-Style Audio

Despite state-of-the-art metrics on curated datasets like WSJ0-2Mix, existing separation systems demonstrate several issues when applied to real podcast environments:

- Emotional expression may bleed across separated sources, complicating identity retention.
- Degradation under rapid turn-taking happens especially when overlaps last < 200 ms, which are common in affirmation-driven dialog.
- Domain mismatch occurs when training data rarely contains multi-speaker crosstalk typical in entertainment media.

These challenges indicate that speech separation cannot be evaluated purely on audio reconstruction quality and must instead be assessed using downstream metrics such as diarization error rate and transcript fidelity.

# 2.3 Duration-Constrained Semantic Summarization

While progress in ASR and neural synthesis has enabled accurate transcription and realistic resynthesis, the intermediate step of transforming long-form conversational content into a concise, time-bounded script remains underexplored. Traditional text summarization literature has largely focused on abstractive and extractive methods that optimize for semantic faithfulness and coherence [25], yet these systems rarely incorporate temporal constraints or speaker-aware structure. As a result, existing LLM-based summarizers produce monologic abstractions that fail to preserve multi-speaker interaction dynamics, pragmatic cues, and the dialogic rhythm needed for natural TTS reconstruction [26].

# 2.3.1 Limitations of Unconstrained Abstractive Summarizers

Large instruction-tuned LLMs, such as GPT-4 and Llama-based derivatives, demonstrate strong performance in multi-document and narrative summarization, but they lack native mechanisms for controlling output length beyond token limits. Studies such as Liu et al. [27] highlight that LLMs often ignore strict word-count constraints, producing summaries that deviate significantly from target lengths, especially for long conversational transcripts. Additionally, most summarizers treat dialogue as flattened text, disregarding speaker alternation, backchannels, and overlapping cues. This leads to summaries that misrepresent the original discourse structure, complicating downstream synthesis where speaker identity must be preserved.

Beyond structural issues, existing approaches do not account for speaker-specific speaking rates. Research by Pappagari et al. [28] has shown that individual cadence, pause density, and articulation speeds vary significantly across speakers, meaning that two speakers delivering the same text length can differ by tens of seconds in actual spoken duration. Without adapting to these differences, summarizers cannot guarantee alignment with time budgets when summaries are resynthesized into speech.

# 2.3.2 Constraint-Aware Summarization and Structured Generation

Recent works have proposed controllable summarization frameworks using reinforcement learning or constrained decoding [29], though these methods mostly target written domains and lack evaluation in downstream speech contexts. Structured summarization approaches, such as role-based JSON outputs [30], provide a promising direction for machine-readable dialogue reconstruction, yet they remain agnostic to acoustic duration and prosodic requirements.

The CARD pipeline addresses these gaps by introducing a duration-constrained, speaker-aware summarization stage that integrates acoustic metadata extracted from Whisper and NeMo. By computing speaker-

specific words-per-minute (WPM) metrics, the system derives an exact word budget for the summary. The summarizer then guides a large language model to produce an output that is simultaneously semantically faithful, speaker-structured, and duration-aligned. The result is a machine-readable JSON representation containing compressed message units, speaker attribution, and emotional descriptors, forming a stable interface between ASR outputs and expressive voice cloning modules such as IndexTTS2.

# 2.4 IndexTTS2: Voice Cloning and Neural Synthesis

Figure 2.1: Benchmarking IndexTTS2 against State-of-the-Art Baselines. Performance comparison on the LibriSpeech test-clean dataset. IndexTTS2 (Blue) demonstrates superior performance across all metrics, particularly in Speaker Similarity (SS) and Intelligibility (WER), validating its selection for the CARD pipeline over diffusion-based alternatives like MaskGCT and F5-TTS.

![Figure 4](images/EEE_196_CARD_UCL-picture-004.png)

Visible text:
Speaker Similarity (SS)
Intelligibility (Low WER)
Identity (SMOS)
Prosody (PMOS)
Audio Quality (QMOS)
IndexTTS2
MaskGCT
F5-TTS

Layout sketch:
```text
Speaker Similarity (SS)
                  /       \
                 /         \
                /           \
Audio Quality (QMOS)         Intelligibility (Low WER)
                \           /
                 \         /
                  \       /
          Prosody (PMOS)   Identity (SMOS)

          [Legend Box]
          • IndexTTS2
          --- MaskGCT
          .... F5-TTS
```

Details:
The image displays a radar chart (or spider chart) with five axes radiating from a central point. The axes are labeled as follows, starting from the top and moving clockwise: "Speaker Similarity (SS)", "Intelligibility (Low WER)", "Identity (SMOS)", "Prosody (PMOS)", and "Audio Quality (QMOS)".

Three data series are plotted on the chart, forming pentagonal shapes:
1.  **IndexTTS2**: Represented by a solid blue line with a light blue shaded fill. This shape is the largest, extending furthest out on all axes, indicating the highest scores across all metrics.
2.  **MaskGCT**: Represented by a dashed red line. This shape is smaller than the IndexTTS2 shape, indicating lower scores.
3.  **F5-TTS**: Represented by a dotted orange line. This shape is generally similar in size to the MaskGCT shape but slightly smaller in some areas (like Speaker Similarity) and slightly larger in others (like Audio Quality).

Concentric pentagonal grid lines are visible in the background to help gauge the values.

At the bottom of the image, there is a legend box containing three entries:
-   A blue dot followed by the text "IndexTTS2".
-   A red dashed line segment followed by the text "MaskGCT".
-   An orange dotted line segment followed by the text "F5-TTS".

The transition from extractive audio summarization to generative resynthesis bets on the capability of the system to perform zero-shot text-to-speech (TTS) to synthesize a target speaker's voice using only a brief voice sample without fine-tuning the model as a whole. While early neural codecs like VALL-E [31] demonstrated the feasibility of a prompt-based cloning, the field currently faces a trilemma between prosodic

# 2.4.1 Autoregressive vs. Non-autoregressive Paradigms

Current architectures generally falls into two categories: Non-Autoregressive (NAR) and Autoregressive (AR) models. NAR models–F5-TTS [32] and MaskGCT [33]–utilize flow matching or diffusion processes to generate speech in parallel. While these models offer superior inference speed and robust duration control, they often suffer from prosodic averaging, quite resulting in a flat and stable speech that removed the dynamic variance of a human voice.

Conversely, AR models such as XTTS and CosyVoice [34] generate audio token-by-token. This kind of approach do capture the nuance, probabilistic nature of human speech, yielding higher naturalness. However, traditional AR models are notoriously difficult to control as they are really prone to hallucinations, variable speaking rates, and timbre leakage, where the speaker's identity distorts when the model attempts to generate expressive or emotional speech. To resolve this trade-off, we selected IndexTTS2, a hybrid architecture that strategically combines both paradigms. It leverages an autoregressive (AR) Text-to-Semantic (T2S) module to generate a sequence of abstract linguistic tokens, capturing the complex, long-range dependencies of natural prosody. This sequence is then fed to a non-autoregressive (NAR) Semantic-to-Mel (S2M) module that generates the final mel-spectrogram in parallel, ensuring speed and acoustic stability [7]. As illustrated in Fig. 4.1, this "best-of-both-worlds" approach allows IndexTTS2 to outperform NAR models (MaskGCT, F5-TTS) in both Speaker Similarity (SS) and Intelligibility (WER), justifying its selection as the core synthesis engine.

# 2.4.2 Disentanglement of Timbre and Style

A critical gap in previous zero-shot systems is the entanglement of speaker identity (timbre) with emotional expression (style). In standard cloning frameworks, prompting a model to speak "angrily" often alters the fundamental acoustic characteristics of the voice, causing the clone to sound like a different person .

To address this gap, we utilize the architecture proposed in IndexTTS2 [7], which introduces a decoupled feature space. Unlike models that rely on a single reference vector, IndexTTS2 employs separate "Timbre Prompts" (for identity) and "Style Prompts" (for emotion). It utilizes a Gradient Reversal Layer (GRL) to remove speaker information from the emotion encoder, ensuring that the emotional prosody is applied on top of the speaker's identity rather than replacing it. Furthermore, the inclusion of GPT latent enhancement in the Semantic-to-Mel (S2M) module mitigates the slur often found in expressive AR synthesis, ensuring

that the high-fidelity summary remains intelligible even during rapid, conversational delivery.

# 2.4.3 The Temporal Control Gap

While IndexTTS2 introduces a native mechanism for duration control using speech token counting, this presents a current limitation because forcing a model to fit a specific token count often results in unnatural time-stretching or rushing of phonemes, creating a robotic cadence that breaks immersion.

CARD addresses this gap by shifting the temporal control upstream. Rather than relying on the TTS model to compress audio acoustically (time-stretching), we employ the Whisper + NeMo-derived WPM metric to constrain the semantic generation at the LLM stage. By calculating the precise word budget based on the speaker's natural cadence, we allow IndexTTS2 module to generate speech at a natural, free-form rate while still strictly adhering to the global time budget. This hybrid approach leverages the acoustic fidelity of IndexTTS2 while bypassing the artifacts of forced-duration synthesis.

# 2.5 Mistral: Conversational Dynamics and Automated Backchanneling

Voice cloning, currently, is not sufficient for simulating a genuine immersive experience. A defining characteristics of human conversation is the presence of backchanneling—the phatic expressions (e.g., "mmhmm," "right," "exactly") and disfluencies that listeners produce to signal attention and agreement, Standard TTS systems are fundamentally designed for "read speech" (audiobooks/news), resulting in a sequential, monologue-heavy delivery that lacks the interactive texture of spontaneous dialogue.

# 2.5.1 Limitations of Rule-based Heuristics

Historically, attempts to inject conversationality into synthesized speech have relied on rule-based heuritics; they typically insert the filler words or pauses based on fixed probability thresholds or rhythmic intervals.

However, research in Human-Robot Interaction (HRI) states that these rigid approaches degrade the user experience. A study from Engwall found that backchannels with unexpected timing or formulation create "tonal dissonance," leading users to perceive the agent as socially inept or inattentive [35]. A rule-based system cannot distinguish between a pause for dramatic effect and a pause for hesitation, often inserting fillers that disrupt the narrative flow. As noted by Skantze [36], the complexity of turn-taking dynamics–the projection of when a speaker is about to finish a phrase–cannot be modeled by simple silence-threshold algorithms.

# 2.5.2 Semantic Prediction via Quantized LLMs

To bridge this gap, CARD moves beyond random insertion to semantic prediction. We employ a 4-bit quantized version of Mistral 7B [8] to act as a "conversational supervisor". This approach is supported by recent findings from Wang [37], who demonstrated that fusing LLMs with acoustic features significantly outperforms single-modality baselines in predicting backchanneling opportunities.

In our pipeline, the SLM analyzes the synthesized text segment to predict two variables: the temporal slot (where an interjection is naturally required) and the semantic type (i.e. agreement, question, surprise, etc.). Furthermore, distinct from the turn-taking of standard dialogue systems, CARD implements an asynchronous overlap algorithm. By purposefully "barging-in" on a speaker's text segment, we simulate a natural interaction behavior of human conversation, masking the robotic silence that usually plagues multi-speaker TTS generation [9].

# 2.5.3 Justification of Using Mistral

# Selected for CARD

Figure 2.2: Comparative Analysis of Small Language Models (SLMs) The table highlights the architectural suitability of Mistral 7B (Green) compared to newer state-of-the-art models (Red). While Llama 3.1 and Qwen 2.5 offer larger vocabularies and safety features, these attributes manifest as VRAM bottlenecks and false refusals within the constrained CARD pipeline.

| Metric | Mistral 7B (v0.3) [8] | Llama 3.1 8B [38] | Qwen 2.5 7B [39] |
| - | - | - | - |
| Vocabulary & VRAM Overhead Impact on 12GB GPU | 32k Tokens ≈ 1.3E8 Params Fits in VRAM (Standard) | 128k Tokens ≈ 5.2E8 Params +Additional Bloat High risk of OOM | 152k Tokens ≈ 5.4E8 Params +Additional Bloat High risk of OOM |
| Alignment & Refusal Rate For sensitive topics | Supervised Fine Tuning (SFT) ”Loosely Aligned” | Safety RLHF False Refusals on Crime/Conflict | Safety RLHF Inconsistent on sensitive prompts |
| Instruction Adherence JSON Formatting | Task Oriented Outputs JSON only. Low Latency | Probable Help fulness Bias ”Sure, here is...” High Latency [40] | Probable Help fulness Bias ”Sure, here is...” High Latency [30] |

While newer Small Language Models (SLMs) such as Llama 3.1 8B and Qwen 2.5 7B offer higher benchmark scores, they do introduce specific architectural bottlenecks that affect the CARD pipeline. We prioritized Mistral 7B based on three critical engineering constraints, as illustrated by Fig. 2.2:

1. Vocabulary Tax and VRAM Efficiency. Modern SLMs have expanded their vocabularies to support multilingualism, with Qwen 2.5 utilizing 152k tokens [39] and Llama31 with 128k tokens [38]. In a quantized environment, the embedding matrices for Mistral 7B account for about 134 million parameters (Vocabulary Size × Hidden Dimension Size = 32768 × 4096 [8, 41]), while Qwen 2.5 is around 544 million parameters (3584×152064 [39, 42]). In a quantized environment, the embedding and unembedding matrices are notoriously sensitive to compression artifacts. Standard quantization protocols like Q4 K M retain these layers at a higher precision rather than 4-bit to prevent vocabulary collapse [43]. For Qwen 2.5, keeping its massive 152k vocabulary at this higher precision imposes a significantly larger memory penalty compared to Mistral's 32k vocabulary mapping. Hence, we assume that the tax on the VRAM would be about 4 . 1 × 10 8 parameters (5 . 4 × 10 8 − 1 . 3 × 10 8 ). If the embedding layer is kept at FP 16 (2 bytes), then the overhead difference will be about 820 megabytes.
2. Safety Alignment vs. Immersion. Llama 3.1 is aggressively fine-tuned with Reinforcement Learning from Human Feedback (RLHF) to minimize harmful outputs introducing a false refusal rate (FRR) on borderline prompts [38]. On the other hand, Mistral 7B-Instruct is primarily aligned using Supervised Fine-Tuning (SFT), opting for a lighter alignment approach [8]. This architecture approach results in a loosely aligned model that maintains high compliance for sensitive podcast narratives like crime and conflict that would trigger false refusals in safety-tuned models.
3. Instruction Adherence and Latency. While Llama 3.1 and Qwen 2.5, specifically their Instruct models, are "Instruction Tuned," their alignment objective prioritizes conversational helpfulness often leading to verbose contamination [40]. When prompted for structured output, these models frequently insert polite preambles like Sure, here is the JSON analysis... before generating the data. This behavior is a known artifact of these models, increasing the "Time to First Token" (TTFT) and necessitates complex post-processing. Mistral 7B-Instruct exhibits a lazier, task-oriented generation pattern that strictly adheres to the requested schema without conversational padding [30], reducing the total inference time per segment.

# Chapter 3

# Problem Statement and Objectives

# 3.1 Problem Statement

Current podcast summarization tools force a tradeoff between efficiency and immersive listening experience. Text-based summaries remove prosody, speaker identity, and emotional cues, while extractive clipping produces disjointed audio that cannot meet strict duration limits. Existing generative systems improve conversational flow but often rely on generic voices and lack precise temporal control.

Furthermore, a significant gap exists in the evaluation of such systems. Because this is a novel pipeline integrating transcription, semantic compression, and expressive resynthesis, standard metrics alone are insufficient. Automated metrics like Word Error Rate (WER) or ROUGE may indicate high textual accuracy while failing to detect unnatural phrasing, emotional mismatch, or "hallucinated" prosody. No existing solution integrates accurate multi-speaker transcription, constraint-aware semantic compression, and identitypreserving audio resynthesis into a single pipeline, nor does a standardized framework exist to evaluate the *perceptual quality* of such multimodal outputs. This prevents users from obtaining concise yet natural summaries that retain the original conversational character of long-form podcasts.

# 3.2 Objectives of the Project

The primary objective of this project is to design and implement an end-to-end system that produces concise, identity-preserving podcast summaries with strict duration control. Additionally, given the novelty of the audio-to-audio generative workflow, the project aims to establish a validation protocol for subjective quality assessment.

The specific objectives are:

1. Develop a hybrid Audio2Script module (Stages 1 & 2) utilizing Whisper and NeMo to generate word-level timestamps and speaker labels with:
- Word Error Rate (WER) < 12%
- Diarization Error Rate (DER) < 15%
2. Implement a constraint-aware script summarizer (Stage 3) that:
- Computes a target word budget from measured speaker WPM.
- Compresses the transcript to meet user-defined duration with < 5% word-budget deviation.
- Outputs a structured JSON with speaker, message, and prosody tags.
3. Perform zero-shot voice cloning (Stage 4) using IndexTTS2 that preserves speaker identity and emotional style with:
- Speaker Similarity Score (SS) > 0 . 85
- Emotion Similarity Score (ES) > 0 . 85
4. Establish a composite evaluation framework that augments automated metrics with human-verified benchmarks. This involves designing a Mean Opinion Score (MOS) study to assess:
- Listening Effort (Fluency/Intelligibility)
- Speaker Similarity (Identity Retention)
- Emotional Fidelity (Did the summary keep the original "vibe"?)

This objective addresses the limitation of numerical benchmarks (WER/ROUGE) which cannot measure the perceptual quality of synthesized narrative audio.

5. Integrate all modules into a seamless pipeline that takes raw audio input and outputs a coherent, time-accurate, multi-speaker audio summary.

# 3.3 Scopes and Limitations

This project is limited to building a research prototype for summarizing long-form English podcasts under controlled conditions. The system supports multi-speaker audio with a maximum input duration of roughly 60 minutes.

Benchmarking and Evaluation Limitations: A significant limitation of this study is the reliance on custom subjective evaluation. As this is a novel audio-to-audio generation pipeline, standard automated benchmarks (such as ROUGE for text or WER for transcription) are insufficient proxies for overall system quality. A low WER does not guarantee natural prosody, and a high ROUGE score does not guarantee emotional preservation. Consequently, this project relies heavily on human verification (Mean Opinion Score) to validate the "Subjective Quality of Experience." These human evaluations are resource-intensive and limited in sample size compared to automated datasets.

# Technical Constraints:

- Synthesis Approach. Summaries are generated through semantic compression and re-synthesis, not acoustic time-stretching.
- Emotion Handling. Emotional styles rely on prompt-based descriptions (LLM-inferred) rather than acoustic feature extraction from the source audio.
- Hardware. The pipeline is designed to operate within approximately 12 GB of GPU memory. Voice cloning is performed via zero-shot inference (IndexTTS2) without model fine-tuning, which may introduce minor artifacts in distinct accents.
- Real-Time Processing. Evaluation focuses on intelligibility, identity retention, and duration accuracy rather than real-time processing capability.

# Chapter 4

# Methodology

# 4.1 Audio2Script

The first stage of the CARD pipeline is the Audio2Script module, responsible for ingesting raw podcast audio and outputting a speaker-attributed, timestamped transcript. This module addresses the challenge of "Serialized Diarization," where the semantic content (text) and acoustic identity (speaker) must be aligned with high temporal precision. This structured output serves as the foundational ground truth for two downstream tasks: the Speaker Audio Extraction (Section 4.2), which utilizes the timestamps for signal segmentation, and the Script Summarizer Module (Section 4.3), which ingests the speaker-attributed text for semantic compression.

# 4.1.1 System Architecture

The proposed system utilizes a hybrid Whisper-Diarization pipeline that decouples semantic transcription from speaker embedding extraction [21]. This architecture represents an ensemble of state-of-theart (SOTA) foundation models from industry leaders, integrating OpenAI's ASR (Whisper), NVIDIA's speaker recognition toolkit (NeMo), and Meta's source separation technologies (Demucs). The pipeline processes input audio through four sequential blocks:

1. Source Separation (Preprocessing). Raw audio is processed using Meta's Demucs to isolate the vocal track from background music and noise [44]. Feeding only isolated vocals into the diarization engine minimizes false positive speaker detections caused by non-speech artifacts common in podcasts.

2. Semantic Transcription (ASR). The isolated vocal track is transcribed using OpenAI's large-v2 Whisper model. Trained on 680,000 hours of multilingual data, this weakly supervised Transformer generates a high-accuracy semantic transcript that implicitly handles disfluencies and punctuation, serving as a robust baseline for summarization [45].
3. Forced Alignment. To resolve Whisper's temporal drift, we employ NVIDIA NeMo's Connectionist Temporal Classification (CTC) forced aligner, an algorithm that aligns variable-length audio to text without explicit timing labels. This maps transcribed sounds directly to acoustic waveform frames, refining timestamps from rough sentence-level estimates to word-level [46].
4. Speaker Diarization and Attribution. Parallel to transcription, MarbleNet VAD segments the audio, and NVIDIA's TitaNet-Large extracts speaker embeddings for clustering. Finally, word-level timestamps from the CTC aligner are cross-referenced with these speaker segments to attribute a specific Speaker ID to each word [47].

# 4.1.2 Evaluation Metrics

The success of this component is measured using two primary metrics standard in the speech processing domain:

1. Diarization Error Rate (DER). This is the primary metric for evaluating "who spoke when." DER is the sum of False Alarm (F A), Missed Detection (MS), and Speaker Confusion (SC) durations divided by the total reference duration (REF) [48]:

$$D E R = \frac{F A + M S + S C} {R E F}$$

We aim for a DER of < 15%, a competitive benchmark aligned with top-performing systems on the VoxConverse test set [48], which features similar multi-speaker characteristics to podcasts.

2. Word Error Rate (WER). This metric quantifies the semantic accuracy of the transcription. Following the methodology of Galibert [49], WER is calculated as:

$$W E R = \frac{S + D + I} {N}$$

Where S is the number of substitutions, D is deletions, I is insertions, and N is the total number of words in the reference. We aim for a WER of < 12%. This target aligns with benchmarks for large-scale robust ASR models on complex, multi-speaker conversational datasets, ensuring that the Script Summarizer receives coherent text and preventing "garbage-in, garbage-out" hallucinations in the final summary [19].

# 4.1.3 Integration with Downstream Modules

This module functions as the synchronization hub for the CARD pipeline. It outputs two distinct data streams:

- For the Speaker Audio Extraction Module: A structured JSON file containing precise [tstart, tend] timestamps and speaker labels. This allows the splitter to perform deterministic signal slicing without redundant unsupervised diarization.
- For the Script Summarizer Module: A speaker-attributed text transcript (JSON) that serves as the prompt for the Constraint-Aware LLM, enabling the generation of summaries that preserve the structure of the original conversation.

# 4.2 Speaker Audio Extraction

# 4.2.1 Timestamp-Guided Audio Chunking and Extraction

The full audio recording is segmented using the timestamps of the Audio2Script transcript. Each interval is classified as non-overlapping or overlapping. The extraction method is dependent on the classification as detailed below:

Non-overlapping segments Non-overlapping segments are extracted from the original waveform using time-to-sample indexing, soundfile (for read/write), and NumPy (for indexing and slicing). This method produces exact and artifact-free copies of the audio without using a separator.

The extracted segments form the primary source of speaker audio and are buffered for later concatenation.

Overlapping segments When two or more intervals for different speaker IDs overlap, they are merged into a single window that spans the combined start and end times. The window is padded to provide context for separation before being fed. Consequently, each window gives the separation model enough context to do a clean job without becoming memory-hungry.

# 4.2.2 Targeted Separation with SepFormer

SepFormer is a dual-path transformer separator applied only to the merged overlap windows. The model produces blind source estimates for that window.

For each window:

1. Feed the merged window to SepFormer which returns a small set of blind source estimates for that window.
2. Match each SepFormer estimate to a speaker using enrollment embeddings and cosine similarity. Mark the assignment uncertain if the top similarity is below the decision threshold.
3. Trim off the extra padding, put the chosen piece into that speaker's track, and smooth the join with a short cross-fade of about 10 ms to 50 ms.

This targeted approach only runs the separation model on the short chunks where people actually talk over each other, thus using far less GPU memory and finishing much faster.

SepFormer adopts a dual-path "chunking" strategy that aligns well with this targeted separation design. By splitting audio into short overlapping chunks and applying transformer attention both within and across chunks, SepFormer leverages local speech characteristics while still maintaining a global understanding of speaker relationships throughout the recording. This makes it effective at resolving overlapping speech, where multiple voices must be separated cleanly within a short time span. In contrast, a convolution-based approach such as Conv-TasNet rely on local context and can struggle with longer or more complex overlaps. Therefore, the chunk-plus-attention architecture of SepFormer delivers a more accurate separation quality in these challenging settings [22–24].

# 4.2.3 Enrollment Embedding and Speaker Assignment

An enrollment embedding is a short, clean audio clip containing a single speaker used to compute a reference speaker embedding. The enrollment snippets are selected from the non-overlapping diarization segments.

Comparing each separated piece from the overlap windows to a short reference clip for each speaker offers a deterministic and reliable way to attribute to a speaker.

The workflow for creating an enrollment snippet:

1. Extract one or more clean non-overlap snippets per speaker of 3 to 6 seconds in duration.
2. Compute speaker embeddings for these snippets using a consistent embedding model (ECAPATDNN, x-vectors, or a pyannote/Resemblyzer model).

3. Normalize the embeddings using L2 normalization.
4. Compute the same embedding for each SepFormer output in an overlap window, and normalize it.
5. Assign each SepFormer output with the highest cosine similarity to that speaker's enrollment embedding, subject to a decision threshold.
6. If no similarity exceeds the threshold, mark the assignment uncertain.

# 4.2.4 Speaker-Pure Audio Track Construction

After assignments are determined, per-speaker tracks are assembled by concatenating segments assigned to the diarized speaker. The mode of concatenation is compact, without gaps, to form contiguous reference audio for embedding training or cloning. During the insertion of separated overlap outputs, a short linear cross-fade is applied at the boundaries to avoid clicks and seam artifacts.

# 4.2.5 Outputs

The component produces:

- Speaker-pure WAV files (one per diarized speaker).
- Enrollment embeddings per speaker.

# 4.3 Script Summarizer Module

The Script Summarizer is responsible for converting the raw speaker-attributed transcript into a concise, time-constrained dialogue script suitable for neural resynthesis. Operating between the Audio2Script and Voice Cloning modules, this stage ensures that the semantic content remains faithful to the original conversation while meeting strict duration targets.

# 4.3.1 Word Budget Derivation

To enforce temporal precision, the system computes a global word budget derived from the measured speaking rates of the original speakers. For each speaker s, the Whisper-derived timestamps yield an effective words-per-minute (WPM) value. The target summary duration T (in minutes) produces a total allowable

word count:

$$W_{\text{target}} = \left(\frac{1} {| S |} \sum_{s \inS} W P M_{s} \right) \timesT.$$

This constraint ensures that the final synthesized audio matches the requested runtime without relying on unnatural duration-forcing mechanisms during speech generation.

# 4.3.2 Model Selection and Benchmarking

To determine the optimal Large Language Model (LLM) backend for this duration-constrained task, a comparative evaluation was conducted across three candidate models: GPT-5 , GitHub Copilot, and DeepSeek .

The models were evaluated on identical transcript segments using the project's master summarization prompt. Performance was assessed using both quantitative metrics (ROUGE-1/2/L, BERTScore, Length Accuracy) and qualitative dimensions (Fluency, Coherence, Completeness, Emotion Preservation). The results of this evaluation are detailed in Table 4.1.

Table 4.1: Comparative evaluation of LLM backends for script summarization. Copilot (GPT-5) demonstrates the highest performance across semantic fidelity (BERTScore F1) and structural constraint adherence (Length Acc).

| Model | R-1 R-2 R-L | BERT-P BERT-R BERT-F1 | Len Acc (%) | Emotion Pres. | Summary Notes |
| - | - | - | - | - | - |
| GPT-5 | 0.86 0.74 0.82 | 0.89 0.87 0.88 | 98 | Strong | Smooth structure; balanced tone. |
| Copilot | 0.89 0.78 0.85 | 0.91 0.88 0.90 | 99 | Excellent | Strongest lexical match; highly natural. |
| DeepSeek | 0.83 0.69 0.79 | 0.87 0.84 0.85 | 96 | Good | Minor omission of nuance. |

As illustrated in Figure 4.1, GitHub Copilot consistently outperformed the alternative models. Key findings include:

- Semantic Fidelity: Copilot achieved the highest BERTScore F1 (0.90) and ROUGE-2 (0.78), indicating superior capability in retaining phrase-level meaning and speaker intent compared to DeepSeek (0.85/0.69) and GPT-5 (0.88/0.74).
- Constraint Adherence: Copilot demonstrated the highest Length Accuracy (99%), adhering almost perfectly to the calculated Wtarget without sacrificing grammatical correctness.
- Qualitative Superiority: In human evaluation, Copilot was the only model rated "Excellent" in Emotion Preservation, capturing the affective nuance of the original dialogue that DeepSeek ("Good") tended to flatten.

Figure 4.1: Radar chart comparison of candidate models. Copilot (blue) encompasses the largest area, indicating superior performance across both quantitative and qualitative metrics.

![Figure 5](images/EEE_196_CARD_UCL-picture-005.png)

Visible text:
Quantitative & Qualitative Model Benchmarks
Copilot
GPT-5
DeepSeek
BERTScore F1
ROUGE-L
Coherence
Emotion Pres.
Fluency
Length Acc
1.0
0.9
0.8
0.7

Layout sketch:
```text
Quantitative & Qualitative Model Benchmarks
[Legend Box]
  Copilot
  GPT-5
  DeepSeek

[Radial Chart Area]
      BERTScore F1
      /           \
     /             \
    /               \
   /                 \
  /                   \
Fluency               ROUGE-L
  \                   /
   \                 /
    \               /
     \             /
      Emotion Pres.
           |
           |
      Coherence
```

Details:
The image displays a radar chart (or spider chart) titled "Quantitative & Qualitative Model Benchmarks." The chart compares three models: Copilot, GPT-5, and DeepSeek, as indicated by the legend in the top right corner.

- **Copilot** is represented by a solid blue line and a light blue shaded area.
- **GPT-5** is represented by a dashed green line.
- **DeepSeek** is represented by a dotted red line.

The chart has six axes radiating from the center, labeled clockwise from the top:
1. BERTScore F1
2. ROUGE-L
3. Coherence
4. Emotion Pres.
5. Fluency
6. Length Acc

Concentric circles represent the scale, with labels for 0.7, 0.8, 0.9, and 1.0 visible.

Visually, the Copilot model (blue) shows the highest scores across most metrics, particularly dominating in Fluency, Emotion Pres., and Length Acc, where it reaches the outermost ring (1.0). The GPT-5 model (green) generally follows a similar shape to Copilot but with slightly lower scores, except in Coherence where it appears to match or slightly exceed Copilot. The DeepSeek model (red) has a significantly smaller polygon, indicating lower scores across all six metrics compared to the other two models.

# 4.3.3 Justification via Literature

The empirical superiority of Copilot in this specific task aligns with findings in recent literature regarding domain-adapted models. While GPT-5 serves as a powerful generalist model, Copilot's training on structured code corpora enhances its ability to follow deterministic patterns and schema constraints [50].

Studies on constraint-following LLMs indicate that models optimized for code generation exhibit stronger performance in tasks requiring strict format adherence and length control [51]. This "structure-awareness" is critical for the CARD pipeline, where the output must not only be a summary but also a valid JSON object aligned with specific time markers. Furthermore, Kim et al. [52] demonstrated that instruction-tuned models like Copilot are less prone to "constraint drift" in iterative tasks, ensuring that the emotional tone and duration limits remain stable across long-form summarization.

# 4.3.4 Structured JSON Output

Based on the selection of Copilot, the summarizer is configured to output a validated JSON object containing:

- speaker: the assigned speaker label,

- message: the compressed text line,

- emo text: a natural-language description of prosody or affect,

- segment id: an index for chronological ordering.

# 4.3.5 Validation and Error Checking

To ensure robustness, the module enforces two post-generation checks:

1. Word Budget Compliance: The system verifies that the output length deviates from Wtarget by no more than 5%.
2. Schema Compliance: The output is parsed to ensure strict adherence to the required JSON keys, preventing errors in the downstream Voice Cloning module.

# 4.4 Voice Cloning + Backchanneling + Interjector

We introduce a multi-stage resynthesis workflow as referenced by Fig. 4.2. The system operates on a hierarchical data flow: first generating the primary narrative content via zero-shot cloning, and subsequently passing the audio-text pairs to a Refinement Module. This module acts as a conversational supervisor, injecting asynchronous backchanneling and phatic expressions to simulate the barge-in dynamics of natural human dialogue.

# 4.4.1 Zero-Shot Voice Cloning (IndexTTS2)

We primarily utilize IndexTTS2 [7], a latent-diffusion-based architecture chosen for its ability to decouple speaker timbre from prosodic style. Unlike traditional concatenation-based cloning, IndexTTS2 employs a "Speaker Perceiver Conditioner" that extracted a global style vector Vspk from reference samples N provided by the Diarization module.

Figure 4.2: The Voice Cloning Module Pipeline.

![Figure 6](images/EEE_196_CARD_UCL-picture-006.png)

Visible text:
Input: JSON (Speaker, Message, Voice Path)
Base Synthesis (indexTTS)
Generates Main Audio Segment
Interruption Needed?
(P < 0.4)
Conversational Supervisor
(MiniLM 7B Quantized)
1. Detect Topic Shift
2. Generate Short Response
Acoustic Alignment
Measure Audio Duration
Ion → Calculate Trigger Timestamp (ms)
Interruption Synthesis
Clone Listener Voice via indexTTS
Asynchronous Overlay
Merge Interruption at Calculated Timestamp
Final Merged Audio (Sequential Append)

Layout sketch:
```text
[Input: JSON...]
      |
      v
[Base Synthesis...]
      |
      v
[Interruption Needed?...]
      |
      +--(No path implied, though not drawn)--+
      |
      v (Yes path)
[Conversational Supervisor...]
      |
      v
[Acoustic Alignment...]
      |
      v
[Interruption Synthesis...]
      |
      v
[Asynchronous Overlay...]
      |
      v
[Final Merged Audio...]
```

Details:
The image displays a vertical flowchart illustrating a process for generating audio with interruptions.
- The flow starts at the top with a light green rounded rectangle labeled "Input: JSON".
- Arrows point downwards connecting sequential steps, which are represented by light blue rounded rectangles.
- A decision point is represented by a light orange diamond shape labeled "Interruption Needed? (P < 0.4)".
- A dashed orange arrow labeled "Yes" exits the bottom of the diamond, leading to a light red rounded rectangle labeled "Conversational Supervisor".
- The flow continues downwards through light blue boxes for "Acoustic Alignment", "Interruption Synthesis", and "Asynchronous Overlay".
- The process concludes at the bottom with a light green rounded rectangle labeled "Final Merged Audio".
- The text inside the boxes describes specific actions, models (e.g., "indexTTS", "MiniLM 7B Quantized"), and calculations.

Our JSON parameters from our schema are as follows:

1. Identity Reference (voice sample). A path to the reference waveform (Pw Pwav ), typically a 30 second clip extracted from Audio Splitter module.
2. Affective Prompt (emo text). A natural language description of the desired prosody that Qwen3 will internally create the emo vector (e.g. "Warm welcoming slightly thoughtful with genuine fascination.").
3. Style Strength (emo alpha). This is a scalar coefficient α ∈ [0 , 1] where we primarily set it to default by 0.60 as recommended by the researchers [7] that balances the influence between the speaker's original timbre and emotional style.

For each segment, internally via IndexTTS2, the model utilizes the Gradient Reversal Layer (GRL) to scrub speaker-specific information from the emotion encoder. This ensures that when the system applies the emo text prompt, it alters the delivery prosody without distorting the identity timbre of the speaker [7].

# 4.4.2 Conversational Refinement via Mistral

To mitigate the monologue tone of TTS outputs, we implement a conversational supervisor using a 4-bit quantized implementation of Mistral 7B [8]. We utilize the quantized model via huggingface to reduce the memory footprint to allow both the IndexTTS2 and Mistral model to run with less than 12 GB VRAM of GPU.

The supervisor operates on a probabilistic logic gate (P < 0 . 6), determining if an interjection is socially appropriate for the current segment. If activated, the module executes a two-step semantic analysis:

# 4.4.2.1 Trigger Word Detection

Rule-based systems often insert fillers at random silence intervals, leading to tonal dissonance [35]. Instead, our pipeline passess the text message Tm Tmsg to Mistral with a prompt, see Appendix A, designed to identify specific synctactic trigger categories:

- Phrases indicating conflict.
- Surprising declarations.
- Rhetorical questions.

The model then returns a JSON object containing the trigger word and its char pos (character index), ensuring that interjections occur only at semantically significant moments [37].

# 4.4.2.2 Context-Aware Response Generation

Once a trigger has been identified by Mistral, the model generates a short, context-appropriate interjections (2–6 words, see Appendix B). We add active listening cues (e.g. "Right", "No way!"), which are synthesized using the voice profile of the non-active speaker .

# 4.4.3 Acoustic Alignment and Asynchronous Overlay

A critical challenge in separate-stream synthesis is aligning the text-based trigger position with the timebased audio waveform. We propose a Post-Hoc Acoustic Alignment algorithm. After generating the main waveform Wmain, from the calculation of the words per minute, we:

1. Determine the percentage position of the trigger word on the audio segment.

# Post-Hoc Acoustic Alignment Algorithm

Figure 4.3: Post-Hoc Acoustic Alignment Logic. The diagram illustrates how the system converts a semantic trigger point (t pos ) into an acoustic timestamp (ttrigger) and adds a randomized reaction delay (tdelay) to determine the final asynchronous overlay position (dinterjection).

![Figure 7](images/EEE_196_CARD_UCL-picture-007.png)

Visible text:
1. Text Analysis
Locate trigger word character index vs. total length.
t_pos = 0.87
(87%)

2. Temporal Mapping
Multiply percentage by total Audio Duration (t_d).
t_trigger = 3697 ms

3. Latency Injection
Add randomized "Reaction Delay" (300 - 800ms).
t_delay = 523 ms

4. Final Overlay Position
Calculate insertion timestamp (d_intersection).
4220 ms

"Modern models now merge, ah, voice, text, and images to express ideas—just like people do, you know?"
Start (0ms)
End (4230ms)
+ Delay (523ms)
Trigger Point
Intersection
(3697ms)
Starts Here
(4220ms)

Layout sketch:
```text
[ 1. Text Analysis ] --> [ t_pos = 0.87 ]
       |                      (87%)
       v
[ 2. Temporal Mapping ] --> [ t_trigger = 3697 ms ]
       |
       v
[ 3. Latency Injection ] --> [ t_delay = 523 ms ]
       |
       v
[ 4. Final Overlay Position ] --> [ 4220 ms ]
       |
       v
[ Timeline Graphic ]
"Modern models now merge, ah, voice, text, and images to express ideas—just like people do, you know?"
Start (0ms) ---------------------------------------- End (4230ms)
                                                    ^
                                                    |
                                            + Delay (523ms)
                                            Trigger Point
                                            Intersection
                                            (3697ms)
                                            Starts Here
                                            (4220ms)
```

Details:
The image displays a vertical flowchart consisting of four main steps, each represented by a rounded rectangular box with a light blue background. To the right of each step box is a corresponding result box with a light beige background, connected by a downward arrow indicating the flow of the process.

Step 1 is titled "Text Analysis" and describes locating a trigger word's character index. The result box shows a calculated position percentage.
Step 2 is titled "Temporal Mapping" and describes multiplying the percentage by total audio duration. The result box shows a time in milliseconds.
Step 3 is titled "Latency Injection" and describes adding a randomized delay. The result box shows the delay time in milliseconds.
Step 4 is titled "Final Overlay Position" and describes calculating the final timestamp. The result box shows the final time in milliseconds.

Below the flowchart is a horizontal timeline graphic. A blue line represents the timeline from "Start (0ms)" to "End (4230ms)". Above the line is a sentence of text with the word "know" highlighted in red. Below the line, annotations with arrows point to specific locations on the timeline, illustrating the calculation of the final intersection point based on the trigger point and the added delay. The text "Starts Here" points to the final calculated timestamp on the timeline.

2. From that percentage, we multiply it with the audio duration to get the trigger position in relative seconds.
3. We then choose a random 300–800 ms delay.
4. We add the delay to the relative trigger position

Suppose the audio duration is ad = 4230[ms]. The audio segment is "...you know?". Then, the calculated text position is t pos = 0 . 87. The trigger position would then be ttrigger = 3697[ms]. A random delay of tdelay = 523[ms] is then chosen. Thus, the interjection position dinterjection = 3697 + 523 = 4220[ms], as illustrated by Fig. 4.3.

# 4.5 Justification of the Model Selection

The selection of IndexTTS2 as the core synthesis engine, over competing architectures such as MaskGCT [33], F5-TTS [32], and CosyVoice2 [34], is driven by the specific constraints of the CARD pipeline: Identity Preservation and Emotional Fidelity .

# 4.5.1 Speaker Identity Retention (SS)

For podcast resynthesis, the "parasocial" value relies on the listener believing they are hearing the original host. Benchmarks on the LibriSpeech-test-clean dataset demonstrate that IndexTTS2 achieves a Speaker Similarity (SS) score of 0.870, significantly outperforming MaskGCT (0.790) and F5-TTS (0.821) [7]. This indicates that diffusion-based models (MaskGCT/F5), while faster, suffer from "timbre drift," whereas IndexTTS2's autoregressive approach better preserves the unique vocal signature of the reference audio.

# 4.5.2 Emotional Control and Fidelity (ES)

CARD requires the injection of specific emotional states (via emo text) to match the narrative arc of the summary. In comparative evaluations of emotional expressiveness, IndexTTS2 achieves an Emotion Similarity (ES) score of 0.887 on the Emotional Test Dataset, compared to 0.841 for MaskGCT and 0.757 for F5-TTS [7]. Furthermore, IndexTTS2 maintains a significantly lower Word Error Rate (WER) of 3.115% (vs. 7.759% for MaskGCT) even under heavy emotional conditioning. This ensures that when we apply a prompt like "Hesitant concerned serious", the model does not slur or degrade intelligibility... a critical requirement for compressing information-dense podcast content.

# Chapter 5

# Preliminary Findings

This chapter presents the initial development progress and performance observations of each component of the CARD system: (1) Audio2Script, (2) Speaker Audio Extraction, (3) Script Summarizer, and (4) Voice Cloner. This section aims to document early empirical findings by identifying bottlenecks and justify architectural decisions that will influence the final design of the system.

# 5.1 Audio2Script

The Audio2Script component's purpose is to generate accurate transcripts and timestamps in the form

[start time – end time] speaker id: utterance .

Several segmentation and transcription technologies were evaluated, including Whisper, NVIDIA NeMo diarizer, and whisper-diarization.

# 5.1.1 Findings

The experiment demonstrates several important findings:

1. Whisper provides highly accurate transcription and punctuation. In all evaluated samples, Whisper consistently produced text that matched the spoken audio exactly, including correct punctuation and casing, making it the most reliable option for transcript generation.

2. Whisper's native speaker assignment is poor in multi-speaker settings. Although its text accuracy is strong, Whisper frequently mislabels speaker identities when more than one person is involved, limiting its usefulness as a standalone diarization solution.
3. NVIDIA NeMo offers strong speaker detection but weaker transcription quality. NeMo accurately assigned segments to the correct speaker but introduced contextual transcription errors such as "lease" instead of "Please" and "bowling" instead of "boiling."
4. A trade-off exists between text accuracy and speaker accuracy. Whisper excels at producing correct words, while NeMo excels at assigning correct speakers—neither system satisfies both requirements simultaneously.
5. A post-processing stage may resolve NeMo's transcription weaknesses. The researchers aim to integrate a language model to correct NeMo's contextual word errors while preserving its strong speaker labeling.
6. Combining Whisper and NeMo is a potential strategy to offset their weaknesses. If feasible, leveraging Whisper for text accuracy and NeMo for speaker identity could yield a more balanced diarization–transcription pipeline.

# 5.2 Speaker Audio Extraction

The Speaker Audio Extraction component indentifies "who spoke when" and produces clean per-speaker audio tracks.

The current implementation of the component was used on Lex Fridman's conversation with ThePrimeagen about how to program. The timestamped transcript consumed was produced by using Whisper and SpeechBrain, independent of the Audio2Script module.

The transcript contained 139 segments spanning 564.83 seconds. The transcript contained two unique speakers: SPEAKER 00 and SPEAKER 01. Importantly, the system detected no overlapping speech .

# 5.2.1 Observed System Behavior

The separation module entered its simplest operational mode, due to no overlapping speech:

- Each diarized segment was extracted directly from the raw audio.

- No local SepFormer separation was required.
- All 139 segments were successfully processed.

The system reconstructed complete speaker tracks as follows:

- SPEAKER 00.wav — 564.83 s
- SPEAKER 01.wav — 564.83 s

The current reconstruction mode is through preserve timing concatenation as the compact concantenation mode is under development.

# 5.2.2 Findings

The experiment presents several key findings:

1. Diarization-guided separation is effective when speakers alternate cleanly. Whisper produced 139 non-overlapping segments across 564.83 seconds of audio, allowing the system to reconstruct full speaker-pure tracks without requiring SepFormer.
2. Avoiding blind separation prevents computational failures. Because no overlaps were found, the pipeline skipped neural separation, avoiding the GPU memory issues observed in blind SepFormer inference on long recordings.
3. Non-overlapping extraction scales efficiently to multi-minute audio. Timestamp-guided extraction remains stable and efficient at longer durations.

The results presents a baseline performance scenario and validate the diarization-guided architecture for clean conversational audio. However, more complex conversations containing interruptions or crosstalk will require activating SepFormer for chunk-based separation.

# 5.3 Script Summarizer

The Script Summarizer component's purpose is to convert long-form transcripts into concise, structured text that follows user-defined time constraints. Additionally, it should include speaker roles and emotional descriptors.

The researchers evaluated several large language models for this task, including GPT-5, Copilot, and DeepSeek.

# 5.3.1 Summarization Performance

GPT-5 produced the most coherent, complete, and stylistically consistent summaries. It demonstrated strong performance:

- preserves key ideas while reducing length,
- maintains logical speaker alternation,
- embeds natural-language emotion cues,
- generates well-formed JSON structures suitable for synthesis.

Copilot occasionally produced tighter compression but at the cost of emotional nuance, while DeepSeek tended to omit subtle conversational cues important for speech synthesis.

# 5.3.2 Effect on Downstream Modules

The structure and quality of the summarizer output directly affects the performance of the voice-cloning module. In particular, the presence of:

- emotion descriptors,
- per-line speaker assignments,
- pacing and emphasis cues,

was shown to significantly improve the realism and interpretability of synthesized speech [53].

# 5.3.3 Findings

The experimentation process revealed several key findings:

1. GPT-5 balances compression, coherence, and emotional fidelity. Among the evaluated large language models, GPT-5 produced the most accurate emotional cues and narrative continuity while maintaining a compact summary.

2. Structured JSON output enables predictable speech synthesis. Consistent field formatting prevents ambiguity and improves downstream alignment in the voice-cloning module.
3. The summarizer is a creative rather than extractive component. Rather than extracting verbatim text, it invents pacing, emphasis, and affective instructions to shape the emotional qualities of the synthesized audio.

# 5.4 Voice Cloner

The Voice Cloner component's purpose is to synthesize the final spoken output using IndexTTS2, guided using the text output of the Script Summarizer and the per-speaker audio tracks of Speaker Audio Extraction.

The researchers evaluated the identity preservation, emotional expressiveness, and conversational realism.

# 5.4.1 Speaker Embedding Quality

Clean speaker-pure tracks extracted by using diarization-guided separation significantly improved the stability of the speaker embeddings. During the early attempts, noisy or overlapping segments issued a degraded identity consistency. On the other hand, the diarization-guided tracks allowed IndexTTS2 to reproduce the characteristics of the speaker more accurately.

# 5.4.2 Emotion Control

IndexTTS2 supports emotion conditioning through natural-language descriptions.

The experiments showed that:

- emotion strength could be modulated through emo alpha ,
- summarizer-provided emotion labels had clear prosodic effects,
- combining text cues with structured JSON improved consistency.

# 5.4.3 Conversational Realism

The baseline synthesized speech lacked natural conversational rhythm. Integrating a backchanneling module (Mistral 7B) provided the following effects:

- organic interjections ("yeah", "right", "mhmm"),
- timed overlaps,
- improved pacing and engagement.

Remaining challenges include tuning overlap timing for fast-paced dialogues.

# 5.4.4 Findings

The experiment demonstrates several key findings:

1. High-quality diarized speaker tracks significantly improve identity retention. Speaker-pure audio extracted through diarization-guided separation resulted in cleaner embeddings for IndexTTS2, enabling the synthesized voice to more closely match the target speaker's timbre and vocal characteristics.
2. Emotion conditioning is reliable and enhances naturalness. Using natural-language emotion descriptors in the summarizer output produced clear changes in prosody and delivery. Adjusting the emo alpha parameter allowed fine control over emotional intensity without degrading clarity.
3. Backchanneling substantially improves conversational authenticity. Integrating a lightweight LLM to generate interjections and overlap timing resulted in more realistic conversational flow. The inclusion of cues such as "yeah," "right," and short affirmations improved perceived engagement between speakers.
4. Synthesized dialogue without backchanneling sounds mechanically turn-based. Without interjections or controlled overlaps, conversations sounded overly sequential and unnatural, highlighting the importance of this post-processing stage.
5. Overlap timing remains a challenge. While the backchanneling module improves realism, precisely aligning interjections with speech rhythm requires additional refinement.

# Chapter 6

# Project Schedule and Deliverables

# 6.1 Gantt Chart

The schedule is optimized by exploiting parallel development streams, where the Audio2Script, Summarization, and Voice Cloning modules are configured and tuned concurrently during the first phase. This approach ensures that all individual components are ready for the critical integration checkpoint at the semester's midpoint.

# Legend:

![Figure 8](images/EEE_196_CARD_UCL-picture-008.png)

Visible text:
Rei
Sean
ALL
John dell
Christian
Holiday

Layout sketch:
```text
[Blue Box] Rei      [Dark Red Box] Sean      [Purple Box] ALL
[Yellow Box] John dell [Green Box] Christian [Gray Box] Holiday
```

Details:
The image displays a legend or key consisting of six rectangular entries arranged in a grid of two rows and three columns. Each entry features a solid colored square on the left followed by a text label in black font.
- Top row, left: A blue square labeled "Rei".
- Top row, center: A dark red (maroon) square labeled "Sean".
- Top row, right: A purple (magenta) square labeled "ALL".
- Bottom row, left: A yellow square labeled "John dell".
- Bottom row, center: A green square labeled "Christian".
- Bottom row, right: A light gray square labeled "Holiday".
The text "John dell" appears to be a single name, though the spacing suggests it might be "John" and "dell" or a typo for "John Dell". The layout is clean with white background and thin gray borders separating the cells.

![Figure 9](images/EEE_196_CARD_UCL-picture-009.png)

Visible text:
TASKS
WEEK 1 - WEEK 4 (Jan 19 - Feb 14)
Label
Tasks Title
Start Date
End Date
Days
WEEK 1
WEEK 2
WEEK 3
WEEK 4
19
20
21
22
23
26
27
28
29
30
2
3
4
5
6
9
10
11
12
13
1.1 Setup & Initialization
Environment Setup (WSL, CUDA, PyTorch Dependencies)
19-Jan
23-Jan
5
Podcast Data Collection (Test Set Selection)
26-Jan
30-Jan
5
Ground Truth Annotation (for DER/WER Baselines)
26-Jan
30-Jan
5
2.1 Audio-to-Script Tuning
Configure WhisperX-NeMo Pipeline Parameters
2-Feb
15-Feb
10
Tune Diarization Thresholds for Speaker Turn Detection
2-Feb
15-Feb
10
2.2 Audio Splitter Logic
Develop Timestamp-Guided Segmentation Script
2-Feb
15-Feb
10
2.3 Summarizer Engineering
Design Constraint-Aware Prompt Templates
2-Feb
15-Feb
10
2.4 Voice Cloner Configuration
Configure IndexTTS2 Inference Environment
2-Feb
15-Feb
10

Layout sketch:
```text
TASKS
WEEK 1 - WEEK 4 (Jan 19 - Feb 14)
+----------------+----------------+------------+------------+------+----------------+----------------+----------------+----------------+
| Label          | Tasks Title    | Start Date | End Date   | Days | WEEK 1         | WEEK 2         | WEEK 3         | WEEK 4         |
+----------------+----------------+------------+------------+------+----------------+----------------+----------------+----------------+
| 1.1 Setup &    | Environment    | 19-Jan     | 23-Jan     | 5    | [White Block]  |                |                |                |
| Initialization | Setup (WSL,    |            |            |      |                |                |                |                |
|                | CUDA, PyTorch  |            |            |      |                |                |                |                |
|                | Dependencies)  |            |            |      |                |                |                |                |
+----------------+----------------+------------+------------+------+----------------+----------------+----------------+----------------+
| 1.1.1          | Podcast Data   | 26-Jan     | 30-Jan     | 5    |                | [Red Block]    |                |                |
|                | Collection     |            |            |      |                |                |                |                |
|                | (Test Set      |            |            |      |                |                |                |                |
|                | Selection)     |            |            |      |                |                |                |                |
+----------------+----------------+------------+------------+------+----------------+----------------+----------------+----------------+
| 1.1.2          | Ground Truth   | 26-Jan     | 30-Jan     | 5    |                | [Yellow Block] |                |                |
|                | Annotation     |            |            |      |                |                |                |                |
|                | (for DER/WER   |            |            |      |                |                |                |                |
|                | Baselines)     |            |            |      |                |                |                |                |
+----------------+----------------+------------+------------+------+----------------+----------------+----------------+----------------+
| 2.1 Audio-to-  | Configure      | 2-Feb      | 15-Feb     | 10   |                |                | [Yellow Block] |                |
| Script Tuning  | WhisperX-NeMo  |            |            |      |                |                |                |                |
|                | Pipeline       |            |            |      |                |                |                |                |
```

Table 6.1: Project Schedule Phase 1: Setup and Initialization (Weeks 1–4)

Table 6.2: Project Schedule Phase 2: Refinement and Integration (Weeks 5–8)

![Figure 10](images/EEE_196_CARD_UCL-picture-010.png)

Visible text:
TASKS
WEEK 5 - WEEK 8 (Feb 16 - Mar 13)
Label
Tasks Title
Start Date
End Date
Days
WEEK 5
WEEK 6
WEEK 7
WEEK 8
16
17
18
19
20
23
24
25
26
27
2
3
4
5
6
9
10
11
12
13
2.5 Component Optimizations
Configure SepFormer for Overlap
2.5.1 Separation
16-Feb
27-Feb
8
Script JSON Validation & Output
2.5.2 Parsing Logic
16-Feb
27-Feb
8
Integrate Mistral Backchanneling &
2.5.3 Trigger Logic
16-Feb
27-Feb
8
3.1 Pipeline Integration
3.1.1 Full End-to-End System Integration
2-Mar
13-Mar
10

Layout sketch:
```text
+-----------------------------------------------------------------------+
| TASKS                                                                 |
+-----------------------------------------------------------------------+
| WEEK 5 - WEEK 8 (Feb 16 - Mar 13)                                     |
+-----------------------------------------------------------------------+
| Label | Tasks Title                  | Start Date | End Date | Days |
+-------+----------------------------+------------+----------+------+
| 2.5   | Component Optimizations    |            |          |      |
|       | Configure SepFormer for Overlap|            |          |      |
| 2.5.1 | Separation                 | 16-Feb     | 27-Feb   | 8    |
|       | Script JSON Validation & Output|            |          |      |
| 2.5.2 | Parsing Logic              | 16-Feb     | 27-Feb   | 8    |
|       | Integrate Mistral Backchanneling &|            |          |      |
| 2.5.3 | Trigger Logic              | 16-Feb     | 27-Feb   | 8    |
| 3.1   | Pipeline Integration       |            |          |      |
| 3.1.1 | Full End-to-End System Integration| 2-Mar    | 13-Mar   | 10   |
+-----------------------------------------------------------------------+
| WEEK 5 | WEEK 6 | WEEK 7 | WEEK 8 |
| 16 17 18 19 20 | 23 24 25 26 27 | 2 3 4 5 6 | 9 10 11 12 13 |
| [Gantt Chart Bars] |
+-----------------------------------------------------------------------+
```

Details:
The image displays a project schedule or Gantt chart. The top section lists tasks with their labels, titles, start dates, end dates, and duration in days. The tasks are organized hierarchically, with main tasks (e.g., 2.5 Component Optimizations) and sub-tasks (e.g., 2.5.1 Separation). The bottom section is a timeline grid showing the progress of these tasks over four weeks (Week 5 to Week 8), with specific dates marked for each week. The timeline uses colored bars to represent the duration of each task, with different colors indicating different tasks or phases. The background of the timeline grid is light purple, and the task bars are in shades of blue, green, and red. The overall layout is structured and easy to read, with clear headings and labels.

Table 6.3: Project Schedule Phase 3: Testing and Optimization (Weeks 9–12)

![Figure 11](images/EEE_196_CARD_UCL-picture-011.png)

Visible text:
TASKS
Label
Tasks Title
Start Date
End Date
Days
WEEK 9 - WEEK 12 (Mar 16 - Apr 10)
WEEK 9
WEEK 10
WEEK 11
WEEK 12
16
17
18
19
20
23
24
25
26
27
30
31
1
2
3
6
7
8
9
10
4.1 Testing & Optimization
4.1.1 Objective Metric Evaluation
16-Mar
27-Mar
9
Conduct Mean Opinion Score (MOS)
4.1.2 Survey
30-Mar
8-Apr
6
4.1.3 System Optimization (Based on Metrics)
30-Mar
17-Apr
12

Layout sketch:
```text
[ TASKS Header Row ]
[ Label | Tasks Title | Start Date | End Date | Days | WEEK 9 - WEEK 12 (Mar 16 - Apr 10) ]
[ 4.1 Testing & Optimization ]
[ 4.1.1 Objective Metric Evaluation | 16-Mar | 27-Mar | 9 | [Gantt Bar spanning Mar 16-27] ]
[ Conduct Mean Opinion Score (MOS) ]
[ 4.1.2 Survey | 30-Mar | 8-Apr | 6 | [Gantt Bar spanning Mar 30-Apr 8] ]
[ 4.1.3 System Optimization (Based on Metrics) | 30-Mar | 17-Apr | 12 | [Gantt Bar spanning Mar 30-Apr 17] ]

[ WEEK 9 Header ] [ WEEK 10 Header ] [ WEEK 11 Header ] [ WEEK 12 Header ]
[ 16 17 18 19 20 ] [ 23 24 25 26 27 30 ] [ 31 1 2 3 ] [ 6 7 8 9 10 ]
```

Details:
The image displays a Gantt chart titled "TASKS" with a purple header bar. The chart is divided into two main sections: a task list on the left and a timeline on the right. The task list includes columns for "Label," "Tasks Title," "Start Date," "End Date," and "Days." The timeline section is labeled "WEEK 9 - WEEK 12 (Mar 16 - Apr 10)" and is subdivided into four weekly columns: "WEEK 9," "WEEK 10," "WEEK 11," and "WEEK 12." Each week column is further divided into specific dates. Three tasks are listed under the main heading "4.1 Testing & Optimization": "4.1.1 Objective Metric Evaluation," "4.1.2 Survey," and "4.1.3 System Optimization (Based on Metrics)." Each task has a corresponding horizontal bar in the timeline section, indicating its duration. The bars are colored in shades of purple and gray. The task "Conduct Mean Opinion Score (MOS)" is listed under "4.1.1 Objective Metric Evaluation" but does not have its own separate row or bar in the timeline.

Table 6.4: Project Schedule Phase 4: Closing and Defense (Weeks 13–18)

![Figure 12](images/EEE_196_CARD_UCL-picture-012.png)

Visible text:
TASKS
Label
Tasks Title
Start Date
End Date
Days
WEEK 13
WEEK 14
WEEK 15 - WEEK 18 (Apr 13 - May 22)
WEEK 17
WEEK 18
13
14
15
16
17
20
21
22
23
24
25
26
27
28
29
30
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
1
System Optimization (Based on 4.1.3 Metrics)
30-Mar
17-Apr
12
5.1 Closing & Defense
5.1.1 Final Manuscript Writing
20-Apr
8-May
14
5.1.2 Prepare Defense Slides & Live Demo Setup
11-May
15-May
5
5.1.3 Mock Defense & Final Revisions
18-May
22-May
5

Layout sketch:
```text
```
TASKS
+-----------------------------------------------------------------------+
| Label | Tasks Title | Start Date | End Date | Days | Gantt Chart Area |
+-----------------------------------------------------------------------+
|       | System Optimization... | 30-Mar | 17-Apr | 12 | [Bar spanning weeks 13-14] |
+-----------------------------------------------------------------------+
| 5.1   | Closing & Defense |          |        |    | [Header bar spanning weeks 15-18] |
+-----------------------------------------------------------------------+
| 5.1.1 | Final Manuscript Writing | 20-Apr | 8-May | 14 | [Bar spanning weeks 15-16] |
+-----------------------------------------------------------------------+
| 5.1.2 | Prepare Defense Slides... | 11-May | 15-May | 5  | [Bar spanning week 17] |
+-----------------------------------------------------------------------+
| 5.1.3 | Mock Defense & Final... | 18-May | 22-May | 5  | [Bar spanning week 18] |
+-----------------------------------------------------------------------+
|       |                 |          |        |    | [Date numbers below bars] |
+-----------------------------------------------------------------------+
```
```

Details:
The image displays a Gantt chart titled "TASKS" with a light purple header row. The chart is divided into two main sections: a tabular list of tasks on the left and a timeline visualization on the right.

The table columns are labeled "Label", "Tasks Title", "Start Date", "End Date", and "Days".
The timeline section is organized by weeks, labeled "WEEK 13", "WEEK 14", "WEEK 15 - WEEK 18 (Apr 13 - May 22)", "WEEK 17", and "WEEK 18". Below these week labels, individual days are numbered sequentially from 13 to 22 (for the first two weeks) and then 1 to 22 (for the subsequent weeks).

The tasks are listed as follows:
1.  **System Optimization (Based on 4.1.3 Metrics)**: This task has a start date of 30-Mar and an end date of 17-Apr, lasting 12 days. A dark purple bar represents this task, spanning from late March into mid-April across the timeline.
2.  **5.1 Closing & Defense**: This is a parent task or category, indicated by the label "5.1". It has a light purple background bar spanning the entire "WEEK 15 - WEEK 18" section.
3.  **5.1.1 Final Manuscript Writing**: This sub-task starts on 20-Apr and ends on 8-May, lasting 14 days. A dark purple bar represents this task, spanning from late April into early May.
4.

# 6.2 Project Deliverables

# 6.2.1 Halfway-point Deliverables

By the end of Week 8, the project aims to deliver a functional, integrated end-to-end prototype of the CARD system. While this version may not yet meet the strict optimization targets (such as minimizing latency or achieving perfect artifact-free synthesis), it will successfully demonstrate the complete pipeline continuity. Specifically, the prototype will be capable of:

- Ingesting a raw multi-speaker podcast file.
- Generating a timestamped, speaker-attributed transcript via the Hybrid Whisper-NeMo module.
- Producing a text summary constrained to a specific duration.
- Resynthesizing the summary using zero-shot voice cloning to produce a listening file.

This deliverable serves as proof of concept that the disparate modules can exchange data correctly and generate a cohesive output.

# 6.2.2 Final Deliverables

The final deliverables at the end of the project will consist of the fully polished and optimized CARD framework, refined to meet the specific quantitative objectives outlined in Section 4.2. This includes:

- Optimized System Code. The complete source code with the finalized Mistral-based backchanneling module for conversational realism and the SepFormer integration for handling overlapping speech.
- A system tuned to achieve the target metrics, specifically a Word Error Rate (WER) < 12%, Diarization Error Rate (DER) < 15%, and a duration deviation of < 5% .

- A comprehensive analysis of the system's performance, including Speaker Similarity (SS) scores and Mean Opinion Score (MOS) results from human listeners.
- The complete manuscript and the accompanying presentation slides for the final oral defense.

# Appendix A

# Mistral Find Trigger Prompt

```
1 " " " 2 You are an expert conversation analyst . Your task is to find the single best , high -impact trigger for a listener to make a natural interjection . Analyze the speaker ' s text below . 3 4 The text is: 5 " {text} " 6 7 Focus on these specific categories for triggers: 8 -" question " : A direct question . (e . g., " what do you think? " , " how did that happen " ) 9 -" statement " : A strong , surprising , or emotional declaration . (e . g., " it was unbelievable " , " absolutely shocking " ) 10 -" problem " : A word or phrase that introduces a point of concern . (e . g., " the main issue is " , " I ' m worried that") 11 -" agreement_seek": A phrase that explicitly seeks validation from the listener . (e . g ., " isn ' t it? " , " right? " ) 12 13 **AVOID**: Common filler words like " like " , " so " , " um " , " you know " unless they are part of a larger , more meaningful trigger phrase . Choose the most significant point that invites a reaction . 14 15 Respond with ONLY a single JSON object with the keys " trigger_word " , " char_pos " , and " category " . 16 17 ---18 EXAMPLE 1 19 Text: " We thought the launch would be simple, but the main issue was the server
```

```
unexpectedly crashing. " 20 {{ 21 " trigger_word " : " issue " , 22 " char_pos " : 46, 23 " category " : " problem " 24 }} 25 ---26 EXAMPLE 2 27 Text: " And the final result was, to be honest, absolutely amazing. " 28 {{ 29 " trigger_word " : " amazing " , 30 " char_pos " : 51, 31 " category " : " statement " 32 }} 33 ---34 EXAMPLE 3 35 Text: " That ' s the plan anyway , but it ' s a bit risky, don ' t you think?" 36 {{ 37 " trigger_word": " don ' t you think? " , 38 " char_pos " : 49, 39 " category " : " agreement_seek " 40 }} 41 ---42 43 Now , analyze the provided text and give your JSON response . 44 " " "
```

# Appendix B

# Generate Interjection

```
1 " " " You are {speaker_name}, listening to a podcast . The speaker just said: 2 3 " {main_speaker_text[:200]} " 4 5 Generate ONE SHORT natural interjection (2-6 words only) that shows you ' re engaged and listening . 6 Make it sound natural , not robotic . Examples: " Yeah, totally! " , " That ' s wild!", " Wait , what?", " I see what you mean." 7 8 Respond with ONLY the interjection phrase , nothing else . 9 """
```

# Bibliography

- [1] K. Breitman. "Podcast statistics and trends for 2025". Accessed: 2025-12-01. [Online]. Available: https: //riverside.fm/blog/podcast-statistics .
- [2] M. J. Eppler and J. Mengis, "The concept of information overload: A review of literature from organization science, accounting, marketing, mis, and related disciplines", The Information Society, vol. 20, no. 5, pp. 325– 344, 2004. DOI: 10.1080/01972240490507974 .
- [3] C. Pethe, B. Pham, F. D. Childress, Y. Yin, and S. Skiena, "Prosody analysis of audiobooks", in 2025 19th International Conference on Semantic Computing (ICSC), IEEE, Feb. 2025, pp. 217–221. DOI: 10.1109/ icsc64641.2025.00036. [Online]. Available: http://dx.doi.org/10.1109/ICSC64641. 2025.00036 .
- [4] X. Liu, F. Liu, Y. Li, and E. Lim, "Disentangling the effects of paralinguistic cues in bolstering listeners engagement with podcasters", English, in Proceedings of the 41st International Conference on Information Systems (ICIS), ser. Proceedings of the International Conference on Information Systems, 2020 International Conference on Information Systems - Making Digital Inclusive: Blending the Local and the Global, ICIS 2020 ; Conference date: 13-12-2020 Through 16-12-2020, Association for Information Systems. AIS Electronic Library (AISeL), 2020. [Online]. Available: https://icis2020.aisconferences.org/ .
- [5] A. Vartakavi, A. Garg, and Z. Rafii, "Audio summarization for podcasts", 2021 29th European Signal Processing Conference (EUSIPCO), pp. 431–435, 2021. [Online]. Available: https://api.semanticscholar. org/CorpusID:244956109 .
- [6] M. Bain, J. Huh, T. Han, and A. Zisserman, "Whisperx: Time-accurate speech transcription of long-form audio", in INTERSPEECH 2023, 2023, pp. 4489–4493. DOI: 10.21437/Interspeech.2023-78 .
- [7] S. Zhou et al., Indextts2: A breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech, 2025. arXiv: 2506.21619 [cs.CL]. [Online]. Available: https://arxiv. org/abs/2506.21619 .
- [8] A. Q. Jiang et al., Mistral 7b, 2023. arXiv: 2310 . 06825 [cs.CL]. [Online]. Available: https : / / arxiv.org/abs/2310.06825 .
- [9] Rime Labs. "Back-channeling as a conversational strategy in tts". Accessed: 2025-12-01. [Online]. Available: https://rime.ai/blog/back-channeling .

- [10] M. Brysbaert, "How many words do we read per minute? a review and meta-analysis of reading rate", Journal of Memory and Language, vol. 109, p. 104 047, 2019, ISSN: 0749-596X. DOI: https : / / doi . org / 10.1016/j.jml.2019.104047. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0749596X19300786 .
- [11] S. Singh. "How many podcasts are there? (2025 growth stats)". Accessed: 2025-12-02, DemandSage. [Online]. Available: https://www.demandsage.com/podcast-statistics/ .
- [12] Z. Liu, "Reading behavior in the digital environment: Changes in reading behavior over the past ten years", Journal of documentation, vol. 61, no. 6, pp. 700–712, 2005.
- [13] E. Lee and J. Borchers, "Dimaß: A technique for audio scrubbing and skimming using direct manipulation", in Proceedings of the 1st ACM Workshop on Audio and Music Computing Multimedia, ser. AMCMM '06, Santa Barbara, California, USA: Association for Computing Machinery, 2006, pp. 107–114, ISBN: 1595935010. DOI: 10.1145/1178723.1178740. [Online]. Available: https://doi.org/10.1145/1178723. 1178740 .
- [14] Google. "Notebooklm now lets you listen to a conversation about your sources". Accessed: 2025-12-04. [Online]. Available: https://blog.google/technology/ai/notebooklm-audio-overviews/ .
- [15] Murf AI. "Mastering notebooklm's audio overview customization: The complete 2025 guide". Accessed: 202512-04. [Online]. Available: https://murf.ai/resources/notebooklm- audio- overviewguide .
- [16] V. Srivastav et al., Open asr leaderboard: Towards reproducible and transparent multilingual and long-form speech recognition evaluation, 2025. arXiv: 2510 . 06961 [cs.CL]. [Online]. Available: https : / / arxiv.org/abs/2510.06961 .
- [17] S. B. Kalluri, S. Baghel, S. Ramoji, and S. Ganapathy, "The second displace challenge: Diarization of speaker and language in conversational environments", in Proc. Interspeech 2024, 2024, pp. 1630–1634. DOI: 10. 21437/Interspeech.2024-1833 .
- [18] O. Kuchaiev et al., Nemo: A toolkit for building ai applications using neural modules, 2019. arXiv: 1909. 09577 [cs.LG]. [Online]. Available: https://arxiv.org/abs/1909.09577 .
- [19] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision", in International Conference on Machine Learning, PMLR, 2023, pp. 28 492– 28 518. [Online]. Available: https://arxiv.org/abs/2212.04356 .
- [20] G. Efstathiadis, V. Yadav, and A. Abbas, "Llm-based speaker diarization correction: A generalizable approach", arXiv preprint arXiv:2406.04927, 2024. [Online]. Available: https://arxiv.org/abs/2406. 04927 .
- [21] M. Ashraf, "Whisper diarization: Speaker diarization using openai whisper", 2024.
- [22] W. Ravenscroft, S. Goetze, and T. Hain, "On data sampling strategies for training neural network speech separation models", in 2023 31st European Signal Processing Conference (EUSIPCO), IEEE, 2023. DOI: 10.23919/EUSIPCO58844.2023.10289800 .
- [23] Y. Luo and N. Mesgarani, "Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation", arXiv preprint, vol. arXiv:1809.07454, 2018, Preprint.

- [24] C. Subakan, M. Ravanelli, S. Cornell, M. Bronzi, and J. Zhong, "Attention is all you need in speech separation", in ICASSP 2021 – 2021 IEEE International Conference on Acoustics, Speech and Signal Processing , arXiv:2010.13154v2, 2021, pp. 21–25. DOI: 10.1109/ICASSP39728.2021.9413901 .
- [25] J. Zhang, Y. Zhao, M. Saleh, and P. J. Liu, "Pegasus: Pre-training with extracted gap-sentences for abstractive summarization", in International Conference on Machine Learning, PMLR, 2020, pp. 11 328–11 339. [Online]. Available: https://proceedings.mlr.press/v119/zhang20a.html .
- [26] Y. Li, Y. Liu, L. Liu, and S. Zhao, "Dialogsum: A real-life scenario dialogue summarization dataset", in Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, 2021, pp. 5062–5074. DOI: 10.18653/v1/2021.findings-acl.449 .
- [27] Y. Liu, Q. Jia, and K. Zhu, "Length control in abstractive summarization by pretraining information selection", in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 6682–6692. DOI: 10.18653/v1/2022.acl-long.461 .
- [28] R. Pappagari, P. Zelasko, J. Villalba, Y. Carmiel, and N. Dehak, "Joint prediction of punctuation and disfluency in speech transcripts", in Proc. Interspeech 2021, 2021, pp. 2172–2176. DOI: 10.21437/Interspeech. 2021-155 .
- [29] J. He, S. Kung Wojciech andose, and G. Neubig, "Ctrlsum: Towards generic controllable text summarization", arXiv preprint arXiv:2012.04281, 2020, Accepted at EMNLP 2021. [Online]. Available: https://arxiv. org/abs/2012.04281 .
- [30] C.-K. Wu, Z. R. Tam, C.-Y. Lin, H.-y. Lee, and Y.-N. Chen, "Streambench: Benchmarking large language models on streaming instruction following and structured output", arXiv preprint arXiv:2404.12345, 2024. [Online]. Available: https://arxiv.org/abs/2404.12345 .
- [31] C. Wang et al., Neural codec language models are zero-shot text to speech synthesizers, 2023. arXiv: 2301. 02111 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2301.02111 .
- [32] Y. Chen et al., F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching, 2024. arXiv: 2410. 06885 [eess.AS]. [Online]. Available: https://arxiv.org/abs/2410.06885 .
- [33] Y. Wang et al., Maskgct: Zero-shot text-to-speech with masked generative codec transformer, 2024. arXiv: 2409.00750 [cs.SD]. [Online]. Available: https://arxiv.org/abs/2409.00750 .
- [34] Z. Du et al., Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens, 2024. arXiv: 2407.05407 [eess.AS]. [Online]. Available: https://arxiv.org/ abs/2407.05407 .
- [35] O. Engwall, R. Cumbal, and A. R. Majlesi, "Socio-cultural perception of robot backchannels", Frontiers in Robotics and AI, vol. 10, p. 1 062 828, 2023. DOI: 10.3389/frobt.2023.1062828 .
- [36] G. Skantze, "Turn-taking in conversational systems and human-robot interaction: A review", Computer Speech & Language, vol. 67, p. 101 178, 2021. DOI: 10.1016/j.csl.2020.101178 .
- [37] J. Wang et al., "Turn-taking and backchannel prediction with acoustic and large language model fusion", in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , IEEE, 2024, pp. 1–5. DOI: 10.1109/ICASSP48485.2024.10445798 .

- [38] A. Dubey, A. Jauhri, A. Pandey, A. Keshwam, A. Al-Dahle, et al., "The llama 3 herd of models", arXiv preprint arXiv:2407.21783, 2024. [Online]. Available: https://arxiv.org/abs/2407.21783 .
- [39] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, et al., "Qwen2.5: A party of foundation models", arXiv preprint arXiv:2409.12191, 2024. [Online]. Available: https://arxiv.org/abs/2409.12191 .
- [40] K. Saito, S. Welleck, K. Hiramatsu, and M. Shing, "Verbosity bias in preference labeling by large language models", arXiv preprint arXiv:2310.10077, 2023, Demonstrates that RLHF alignment creates a bias toward verbose, conversational outputs even when conciseness is required. [Online]. Available: https://arxiv. org/abs/2310.10077 .
- [41] Mistral AI Team, Mistral-7b-instruct-v0.3, Hugging Face, Model config and checkpoint, May 2024. [Online]. Available: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 .
- [42] Qwen Team, Qwen2.5-7b-instruct, Hugging Face, Model config and checkpoint, Sep. 2024. [Online]. Available: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct .
- [43] J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, and S. Han, "Awq: Activation-aware weight quantization for llm compression and acceleration", Proceedings of Machine Learning and Systems, vol. 6, pp. 87–100, 2024, Demonstrates that salient weights (embeddings/heads) require higher precision to maintain coherence. [Online]. Available: https://arxiv.org/abs/2306.00978 .
- [44] A. Defossez, N. Usunier, L. Bottou, and F. Bach, ´ ´ Demucs: Deep extractor for music sources with extra unlabeled data remixed, 2019. arXiv: 1909.01174 [cs.SD]. [Online]. Available: https://arxiv.org/ abs/1909.01174 .
- [45] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. Mcleavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision", in Proceedings of the 40th International Conference on Machine Learning , ser. Proceedings of Machine Learning Research, vol. 202, PMLR, Jul. 2023, pp. 28 492–28 518. [Online]. Available: https://proceedings.mlr.press/v202/radford23a.html .
- [46] E. Rastorgueva, V. Lavrukhin, and B. Ginsburg, "Nemo forced aligner and its application to word alignment for subtitle generation", in Interspeech 2023, 2023, pp. 5257–5258.
- [47] N. R. Koluguri, T. Park, and B. Ginsburg, Titanet: Neural model for speaker representation with 1d depth-wise separable convolutions and global context, 2021. arXiv: 2110.04410 [eess.AS]. [Online]. Available: https://arxiv.org/abs/2110.04410 .
- [48] T. Liu and K. Yu, Ber: Balanced error rate for speaker diarization, 2022. arXiv: 2211.04304 [cs.SD] . [Online]. Available: https://arxiv.org/abs/2211.04304 .
- [49] O. Galibert, "Methodologies for the evaluation of speaker diarization and automatic speech recognition in the presence of overlapping speech.", in Interspeech, 2013, pp. 1131–1134.
- [50] M. Chen et al., "Evaluating large language models trained on code", in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 24 079–24 094.
- [51] Y. Xu, Z. Wu, X. Li, and H. Sun, "Structure-aware language models for format-constrained generation", Transactions of the Association for Computational Linguistics, vol. 12, pp. 1452–1468, 2024. DOI: 10.1162/ tacl_a_00752 .

- [52] J. Kim, S. Park, H. Lee, and H. S. Cho, "Constrained text generation with instruction-following models", arXiv preprint arXiv:2403.06231, 2024. [Online]. Available: https://arxiv.org/abs/2403.06231 .
- [53] H. Azzuni and A. El Saddik, "Voice cloning: Comprehensive survey", arXiv preprint arXiv:2505.00579, 2025.
