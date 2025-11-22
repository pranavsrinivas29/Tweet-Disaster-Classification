### Tweet Disaster Classification
#### This is Kaggle's Competition.

A complete multi-approach text classification project using:
✔ Traditional Machine Learning
✔ Deep Learning (LSTMs)
✔ Transformer-based NLP (DistilBERT)

This repo implements:
- Text preprocessing
- Tokenization
- Dataset + DataLoader
- DistilBERT fine-tuning
- Evaluation metrics
- Predictions on test set
- Submission CSV generation

| Approach                | Model Type              | Techniques Used                          | Performance |
| ----------------------- | ----------------------- | ---------------------------------------- | ----------- |
| **1. Machine Learning** | Classical ML            | TF-IDF, Logistic Regression, Naive Bayes | Good        |
| **2. Deep Learning**    | Neural Networks         | Tokenizer, Embedding, BiLSTM + GloVe     | Better      |
| **3. Transformers**     | Large Pretrained Models | DistilBERT Fine-Tuning                   | **Best**    |

### Preprocessing
´´´bash
def preprocess_text(text):
    import re
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9!? ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
´´´

### Machine Learning Approach (TF-IDF + Logistic Regression and Complement NB approach)
Why This Works:
- Fast
- Good baseline
- Captures word importance
- Works well for short texts

Steps: 
- Preprocess text
- Convert text → TF-IDF matrix
- Train ML models
- Evaluate on validation

  ### **1. Term Frequency-Inverse Document Frequency (TF-IDF)**

| Concept | Explanation |
| :--- | :--- |
| **What it is** | A **numerical statistic** that reflects how important a word is to a document in a collection or corpus. |
| **Purpose** | To **weigh terms** based on their relevance, giving higher scores to terms that appear **frequently** in a specific document but **rarely** across the entire corpus. |
| **Calculation** | The product of two components: **Term Frequency (TF)** $\times$ **Inverse Document Frequency (IDF)**. |
| **Key Takeaway** | It helps **filter out common words** (like "a," "the," "is") that appear everywhere and focuses on the unique, discriminative terms. |

**Mathematical Formulation:**

* **TF (Term Frequency):** $\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$
* **IDF (Inverse Document Frequency):** $\text{IDF}(t, D) = \log\left(\frac{\text{Total number of documents in corpus } D}{\text{Number of documents } d \text{ containing term } t}\right)$
* **TF-IDF:** $\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$ 

---

### **2. Logistic Regression**

| Concept | Explanation |
| :--- | :--- |
| **What it is** | A **linear model** used for **classification** (not regression, despite the name). |
| **Underlying Math** | It uses the **Sigmoid function** (or **logistic function**) to map any real-valued number into a value between 0 and 1. |
| **Purpose** | To estimate the **probability** that an instance belongs to a particular class (e.g., probability of an email being spam). |
| **Decision Rule** | If the calculated probability is above a certain **threshold** (usually 0.5), the model predicts class 1; otherwise, it predicts class 0. |
| **Key Takeaway** | Despite being simple and fast, it is a powerful **baseline classifier**, especially for **binary classification** tasks. |

**Sigmoid Function ($\sigma$):**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the linear combination of input features and weights: $z = w_0 + w_1x_1 + w_2x_2 + \dots$ 

---

### **3. Complement Naive Bayes (Complement NB)**

| Concept | Explanation |
| :--- | :--- |
| **What it is** | An adaptation of the standard Naive Bayes classifier, primarily designed for **imbalanced datasets** in text classification. |
| **Standard NB Issue**| Standard Naive Bayes can be heavily biased towards the **majority class** when the class distribution is skewed. |
| **How it Works** | Instead of modeling the probability of a word belonging to a class $P(\text{word} \mid \text{class})$, Complement NB models the probability of a word *not* belonging to a class, $P(\text{word} \mid \neg \text{class})$. |
| **Key Benefit** | It is known to perform particularly well on **text classification** tasks where document length and frequency of terms can vary widely, and it **mitigates the bias** of the standard Naive Bayes on imbalanced data. |
| **Main Assumption** | Like all Naive Bayes models, it assumes **conditional independence** between features (terms) given the class, which simplifies the calculation. |

### Deep Learning Approach (BiLSTM + GloVe Embeddings)
Why BiLSTM + GloVe?
- BiLSTM → Learns context in both directions
- GloVe embeddings → Inject semantic meaning
- Better than ML, but lighter than BERT

Steps: 
- Tokenize text
- Pad sequences
- Load GloVe pretrained word vectors
- Build embedding matrix
- Train a Bidirectional LSTM model
- Evaluate

### **1. GloVe (Global Vectors for Word Representation)**

| Concept | Explanation |
| :--- | :--- |
| **What it is** | A **word embedding** model that learns vector representations of words by analyzing **global word-word co-occurrence statistics** from an entire corpus. |
| **Philosophy** | It merges the benefits of **local context window** methods (like Word2Vec's predictive approach) with **global matrix factorization** techniques (like counting methods). |
| **Core Idea** | The **ratio of co-occurrence probabilities** between words (how often words $A$ and $B$ appear together) is what carries the meaning, encoding relationships like analogy. |
| **Training Objective**| Minimize a cost function such that the **dot product** of any two word vectors (e.g., $w_i \cdot w_j$) approximates the **logarithm of their co-occurrence count** in the corpus. |
| **Key Takeaway** | Produces **static** (context-independent) word vectors that effectively capture **semantic** and **syntactic** relationships in a continuous vector space, often demonstrated by vector arithmetic (e.g., "king" - "man" + "woman" $\approx$ "queen"). |

**GloVe Objective Function (Simplified):**

$$\mathcal{J} = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where $X_{ij}$ is the co-occurrence count of word $i$ and word $j$, $w$ and $\tilde{w}$ are the word and context vectors, and $f(X_{ij})$ is a weighting function. 

---

### **2. Bidirectional Long Short-Term Memory (Bi-LSTM)**

| Concept | Explanation |
| :--- | :--- |
| **What it is** | An advanced type of **Recurrent Neural Network (RNN)**, specifically an extension of the **Long Short-Term Memory (LSTM)**, designed to process sequential data. |
| **Architecture** | It consists of **two parallel LSTM layers** applied to the input sequence: <br> 1. A **Forward LSTM** that processes the sequence from beginning to end ($t=1 \to N$). <br> 2. A **Backward LSTM** that processes the sequence from end to beginning ($t=N \to 1$). |
| **Core Benefit** | At any given time step $t$, the output is a combination (usually concatenation) of the forward and backward hidden states. This allows the model to access **context from both the past and the future** simultaneously. |
| **Application** | Crucial for tasks in **Natural Language Processing (NLP)** where the full context of a word is needed for prediction, such as named entity recognition (NER), machine translation, and text classification. |
| **Key Takeaway** | Overcomes the limitation of standard (unidirectional) LSTMs, which only see the **past** context, leading to a much richer and more **contextually aware** representation of the sequence. |

**Output Combination (Concatenation):**

$$h_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}]$$

Where $\overrightarrow{h_t}$ is the hidden state from the forward layer, $\overleftarrow{h_t}$ is the hidden state from the backward layer, and $h_t$ is the final output vector at time $t$.

### Transformer Approach (DistilBERT Fine-Tuning)
Why Transformers?
- State-of-the-art NLP architecture
- Uses self-attention
- Learns context extremely well
- Pretrained on billions of words

Steps:
- Use DistilBERT tokenizer
- Convert text → input IDs + attention masks
- Build PyTorch Dataset/DataLoader
- Load pretrained DistilBERT
- Fine-tune on your training data
- Evaluate
- Predict and export results

### **1. Transformer Architecture**

| Concept | Explanation |
| :--- | :--- |
| **What it is** | A neural network architecture introduced in 2017 ("Attention is All You Need") that powers most modern Large Language Models (LLMs). |
| **Core Innovation**| Replaced **Recurrent** and **Convolutional** layers with the **Self-Attention** mechanism. |
| **Self-Attention** | Allows the model to **weigh the importance** of every word in the input sequence relative to every other word, capturing long-range dependencies efficiently. |
| **Architecture** | Typically consists of a stack of **Encoders** (for understanding input, like in BERT) and/or a stack of **Decoders** (for generating output, like in GPT). |
| **Key Advantage** | The self-attention mechanism is **highly parallelizable**, which enables much faster training on hardware like GPUs compared to sequential models like RNNs/LSTMs. |

**The Attention Mechanism (Scaled Dot-Product Attention):**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

---

### **2. DistilBERT**

| Concept | Explanation |
| :--- | :--- |
| **What it is** | A smaller, faster, and lighter version of the foundational **BERT** (Bidirectional Encoder Representations from Transformers) model. |
| **Compression Method** | Uses **Knowledge Distillation**, where a smaller model (the **Student**, DistilBERT) is trained to mimic the behavior and outputs of a larger, pre-trained model (the **Teacher**, BERT). |
| **Architectural Change** | Reduces the number of Transformer encoder layers by half (e.g., from 12 to 6 in the base version) and removes token-type embeddings and the pooler. |
| **Performance Gain** | Achieves a **40% reduction in size** and up to **60% faster inference speed** compared to BERT-base, while retaining approximately **97% of BERT's language understanding performance**. |
| **Key Benefit** | Enables the deployment of powerful Transformer-based models in **resource-constrained environments** (e.g., mobile devices or real-time inference). |

---

### **3. Tokenization**

| Concept | Explanation |
| :--- | :--- |
| **What it is** | The process of converting raw text into numerical inputs that a machine learning model can understand. |
| **Mechanism** | The text is broken down into small units called **tokens**, which are then mapped to unique integers (IDs) in the model's fixed **vocabulary**. |
| **Type (for Transformers)** | **Subword Tokenization** (e.g., **WordPiece** or **Byte-Pair Encoding, BPE**). This balances word-level meaning with character-level flexibility. |
| **How Subword Works**| Splits rare or unknown words into smaller, frequently occurring meaningful subwords. E.g., "tokenization" $\to$ \["token", "##ization"\]. |
| **Why it Matters** | 1. **Handles Out-of-Vocabulary (OOV) words** (by breaking them down). 2. **Manages vocabulary size** (smaller vocabulary than pure word-level). 3. **Reduces sequence length** (more efficient than character-level). |

**Subword Example (WordPiece/BERT):**

| Input Word | Tokenized Output |
| :--- | :--- |
| `unbelievable` | `un`, `##believe`, `##able` |
| `running` | `runn`, `##ing` |
| `is` | `is` |
---

### Performance Comparison
| Model                            | Accuracy      | F1 Score      | Notes                    |
| -------------------------------- | ------------- | ------------- | ------------------------ |
| **TF-IDF + Logistic Regression** | 0.80          | 0.75          | Strong baseline          |
| **TF-IDF + Complement NB**       | 0.80          | 0.75          | Strong baseline          |
| **BiLSTM + GloVe**               | 0.80          | 0.77          | Good deep learning model |
| **DistilBERT**                   | 0.82          | 0.79          | Best overall             |
