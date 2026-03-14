# Milestone 1 Technical Report

## 1. Project Objective and Scope

This milestone focuses on preparing an Arabic question-answering dataset for downstream modeling. The work combines two sources:

- QA CSV files containing question and answer pairs.
- Transcript text files used to build context around candidate answers.

The main technical objective of this milestone was not yet to optimize final model accuracy, but to construct a reliable preprocessing and representation pipeline that can support sequence-based learning. The final output of this stage is a cleaned and normalized QA table, context-augmented inputs, tokenized integer sequences, and padded tensors suitable for model training.

## 1.1 Dataset Structure

The dataset is organized into two primary folders:

- `data/QA`: 13 CSV files containing structured question-answer examples and metadata (e.g., `video_id`, `video_title`, `question`, `answer`, difficulty fields).
- `data/Transcripts`: 13 UTF-8 text files containing full episode transcripts used to recover context around candidate answers.

Pipeline-level interpretation of structure:

1. QA files provide supervised targets (`answer`) and prompts (`question`).
2. Transcript files provide long-form context from which local answer windows are extracted.
3. The framework links both sources through title matching and normalized substring lookup.

## 1.2 Dataset Statistics

Computed from the current milestone data snapshot:

- Number of QA files: 13
- Total QA rows: 3,890
- QA rows per file: min 290, max 300, mean 299.23
- Number of transcript files: 13

These statistics indicate that the dataset is moderately uniform per QA file, while text length variation is wide enough to require explicit sequence-length controls before model training.

## 2. Design Choices and Rationale

### 2.1 Data integration strategy

All QA CSV files were concatenated into a single DataFrame. This design allows consistent preprocessing, frequency analysis, and tokenization over one unified corpus. It also simplifies quality checks such as missing values, text-length statistics, and vocabulary tracking.

Transcript files were loaded separately and later used to build contextual windows around answers. Keeping transcripts independent during early exploration reduced coupling and made debugging easier.

### 2.2 Exploratory-first pipeline

Before cleaning, the notebook performs multiple exploratory checks:

- Transcript token frequency inspection.
- QA table schema and missing-value inspection.
- Linguistic characteristic profiling (MSA markers, Egyptian dialect markers, and code-switching).
- Word-frequency visualization before cleaning.

This decision is important because preprocessing decisions in Arabic NLP are highly sensitive to writing conventions and dialect variation. An exploratory-first approach prevents over-cleaning and ensures that each transformation is justified by observed noise.

### 2.3 Cleaning and normalization strategy

The chosen text cleaning stack includes:

- Stopword removal using Arabic stopword lists.
- Punctuation stripping for Arabic and Latin punctuation symbols.
- Removal of tokens containing English letters.
- Removal of Arabic diacritics (tashkeel).
- Letter normalization for common orthographic variants such as hamza-alef forms, final alef maqsura, taa marbuta, and Persian kaf variant.

Reasoning behind this design:

1. The dataset contains mixed writing styles and informal variation.
2. Surface-form variability inflates vocabulary and reduces token consistency.
3. The task benefits from semantic token alignment more than orthographic fidelity.

In short, the cleaning pipeline is intentionally aggressive against noise but conservative on semantic words.

### 2.4 Context extraction design

The context column is constructed by matching each QA row title to the closest transcript title (fuzzy matching), locating the normalized answer span inside the matched transcript, and extracting a fixed window around it.

This approach was chosen because exact title matching is brittle in real Arabic data, especially with title formatting differences and punctuation variance. The fuzzy match plus normalized answer search offers a practical compromise between robustness and implementation complexity.

### 2.5 Tokenization approach

The pipeline uses a word-level Keras Tokenizer with:

- Vocabulary cap of 20,000 words.
- Out-of-vocabulary token support.
- Input fitting over question plus context.
- Separate conversion of answer text to output sequences.

This design is suitable for milestone scale because:

- It is lightweight and easy to audit.
- It produces interpretable integer sequences.
- It controls sparsity through vocabulary capping.
- It handles unseen tokens safely through OOV mapping.

Character-level tokenization was not chosen because the current objective is semantic sequence learning, and word-level tokens better preserve phrase-level meaning with lower sequence lengths.

### 2.6 Padding policy

Padding and truncation are both applied post sequence, with separate maximum lengths for input and output:

- Input max length: 55
- Output max length: 10

These values are grounded in the observed sequence-length distribution (max, min, and percentile checks). The percentile-based selection balances two competing goals: preserving enough context while avoiding excessive zero-padding.

## 3. Output Analysis and Observations

### 3.1 Linguistic characteristics

The marker-based analysis indicates that transcripts include a mixture of formal Arabic patterns, Egyptian colloquial markers, and occasional Arabic-English script transitions. This confirms that the corpus is linguistically heterogeneous.

Implication:

A single rigid normalization policy may remove useful sociolinguistic signals, but leaving all variation unprocessed would fragment the vocabulary. The current pipeline handles this by normalizing orthographic noise while keeping core lexical content.

### 3.2 Noise detection results

The dedicated noisy-data inspection cell surfaces examples of:

- English characters embedded in Arabic text.
- Heavy punctuation artifacts.
- Tashkeel occurrences.
- Orthographic inconsistency in Arabic letters.

Implication:

These observations directly validate the need for the cleaning pipeline. The notebook does not rely on assumptions; it demonstrates concrete noise patterns before transformation.

Representative noisy examples observed by the noise-inspection cells include:

- English fragments mixed into Arabic lines (code-switching tokens and embedded Latin words).
- Arabic and Latin punctuation artifacts attached to tokens.
- Tashkeel-annotated forms of words that also appear unvowelized.
- Orthographic variants of equivalent lexical forms (`إ/أ/آ/ا`, `ى/ي`, `گ/ك`, and `ة/ه` under the selected normalization policy).

### 3.3 Vocabulary behavior before and after normalization

The vocabulary comparison cell reports vocabulary size before normalization and after normalization. The expected and observed trend is a contraction in vocabulary size after cleaning.

Interpretation:

A reduced vocabulary indicates that multiple surface forms were merged into normalized representations. This is a strong signal that the pipeline is reducing sparsity and improving token consistency, which should support better generalization in downstream models.

### 3.3.1 Before/After normalization examples

Examples below illustrate how the normalization policy maps variant forms to a consistent token space:

| Before normalization                 | After normalization | Transformation type                            |
| ------------------------------------ | ------------------- | ---------------------------------------------- |
| `إجابة`                       | `اجابه`      | Hamza-alef normalization + taa marbuta mapping |
| `آثار`                         | `اثار`        | Alef variant normalization                     |
| `على`                           | `علي`          | Alef maqsura normalization                     |
| `موسيقى`                     | `موسيقي`    | Final letter normalization                     |
| `چ/گ` variants in borrowed words | `ك`-based form   | Script variant normalization                   |

These examples explain why multiple visually different forms collapse into fewer normalized tokens, improving token-frequency reliability.

### 3.4 Frequency plots before and after cleaning

Pre-cleaning frequency plots are typically dominated by high-frequency functional or noisy symbols/forms. After cleaning and normalization, frequent terms become more semantically informative and less orthographically redundant.

Interpretation:

The shift in top-word composition suggests the pipeline is moving representation capacity away from punctuation and writing variants and toward content words.

### 3.5 Sequence and padding diagnostics

The sequence-length histograms and padding-ratio visualizations provide operational validation:

- Most sequences are covered by selected max lengths.
- Padding does not fully overwhelm the tensors.
- Decoded samples from token IDs remain interpretable.

Interpretation:

The tokenization and padding stack is functioning correctly from a data-shape and semantic sanity perspective.

### 3.6 Handling unknown and rare tokens

The tokenizer uses `oov_token="<OOV>"` with `num_words=20000`.

Operational effect:

- Frequent words are preserved in the active vocabulary.
- Rare words outside the top-20k and unseen inference-time words are mapped to `<OOV>`.
- This avoids index errors, stabilizes embedding lookup, and limits dimensional growth.

Modeling trade-off:

- Positive: better robustness and controlled memory footprint.
- Negative: semantic detail for rare entities can be compressed into a shared unknown bucket.

## 4. Key Insights

1. Arabic QA preprocessing cannot be treated as generic whitespace tokenization. Orthographic normalization materially improves representation quality.
2. Building context from transcript windows is feasible with fuzzy title matching, but answer matching quality remains the key bottleneck.
3. Vocabulary control and OOV handling are essential early decisions because they influence all later model components.
4. Visualization-driven diagnostics are valuable not only for presentation but for detecting implementation issues (for example, over-padding, length misalignment, or weak normalization effects).

## 4.1 Final Cleaning and Preprocessing Output

At the end of preprocessing, the framework produces a training-ready table and sequence representation with the following finalized steps:

1. Concatenate all QA files into one DataFrame.
2. Detect and document noisy patterns (English fragments, punctuation, tashkeel, orthographic variants).
3. Clean and normalize `question` and `answer` fields.
4. Compare vocabulary before versus after normalization.
5. Build context windows from transcripts using fuzzy title matching and answer span lookup.
6. Remove short-answer rows (length threshold) and drop unrelated metadata columns.
7. Fit tokenizer on input text (`question + context`) and convert both input and output to integer sequences.
8. Pad input/output sequences with separate max lengths for model compatibility.

This is the final MS1 preprocessing artifact consumed by the next modeling stages.

## 5. Framework Limitations

Despite producing a usable training-ready representation, the framework has several limitations:

### 5.1 Rule-based normalization risk

The normalization rules may collapse distinctions that matter in some contexts (for example, dialect nuances, named entities, or stylistic emphasis). This can reduce expressive richness.

### 5.2 Marker heuristics are simplistic

MSA and dialect analysis uses static marker lists. This is useful for quick diagnostics but not robust for linguistic classification. It can undercount context-dependent forms or overcount ambiguous tokens.

### 5.3 Context extraction fragility

Context extraction relies on fuzzy title alignment and direct substring search after normalization. If transcript wording diverges from answer wording, context can be missing or misaligned.

### 5.4 Limited evaluation at this milestone

The milestone emphasizes preprocessing quality, not final QA performance metrics. Therefore, the current conclusions are about data and representation readiness rather than end-task accuracy.

## 6. Recommendations for Next Milestone

1. Add quantitative preprocessing metrics such as context-found ratio, average context length, and OOV rate on a held-out set.
2. Compare word-level tokenization against subword tokenization under identical train-validation splits.
3. Strengthen context retrieval using normalized token-span matching instead of plain substring search.
4. Try different modeling styles: no-cleaning vs cleaning vs cleaning-plus-normalization to measure impact on downstream model quality.
5. Introduce baseline model metrics.

## 6.1 Limitations and Planned Solutions for MS2 and MS3

### MS2 (Baseline Modeling and Evaluation)

Primary risk carried from MS1:

- Word-level tokenization may underperform on rare morphology and unseen forms.

Planned MS2 solutions:

1. Build baseline seq2seq or encoder-decoder model with current pipeline to establish reference metrics.
2. Add explicit OOV-rate reporting on train/validation/test splits.
3. Run controlled ablations: with and without aggressive normalization.
4. Introduce automated context-quality checks (answer-covered-by-context ratio).

### MS3 (Robustness and Advanced Representation)

Planned MS3 solutions:

1. Evaluate subword tokenization (WordPiece/BPE/SentencePiece) against word-level baseline.
2. Replace simple substring context matching with token-span alignment or semantic retrieval.
3. Add error-driven normalization refinements based on model failure cases.
4. Expand evaluation to robustness slices: dialect-heavy samples, code-switched samples, and rare-answer subsets.

## 7. Conclusion

The milestone successfully establishes a coherent Arabic NLP preprocessing framework that transforms noisy multi-source text into model-ready sequences. The design decisions are evidence-driven: exploratory diagnostics motivated cleaning, vocabulary tracking justified normalization, and sequence/padding visualizations validated tokenization behavior. While limitations remain in heuristic language analysis and context matching robustness, the current pipeline provides a solid foundation for reliable model experimentation in subsequent milestones.
