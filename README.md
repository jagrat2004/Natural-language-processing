# NLP (Natural Language Processing) Learning Repository

A comprehensive collection of Jupyter notebooks and projects demonstrating Natural Language Processing concepts, techniques, and real-world applications using Python and NLTK/scikit-learn libraries.

---

## 📚 Repository Structure

### Fundamental NLP Concepts & Preprocessing

#### 1. **4-Tokenization Example using NLTK.ipynb**
   - **Objective:** Learn text tokenization techniques
   - **Topics Covered:**
     - Sentence tokenization (text → sentences)
     - Word tokenization (text → words)
     - WordPunct tokenization
     - TreeBank word tokenizer
   - **Key Libraries:** NLTK
   - **Use Case:** Breaking down text into manageable units for analysis

#### 2. **5-Stemming And Its Types- Text Preprocessing.ipynb**
   - **Objective:** Understand stemming for text normalization
   - **Topics Covered:**
     - Different stemming algorithms
     - Porter Stemmer implementation
     - Reducing words to their root forms
   - **Key Libraries:** NLTK
   - **Use Case:** Reducing vocabulary size and improving text matching

#### 3. **6-Lemmatization- Text Preprocessing.ipynb**
   - **Objective:** Learn lemmatization as an alternative to stemming
   - **Topics Covered:**
     - Lemmatization vs. Stemming
     - WordNet lemmatizer
     - Morphological analysis
   - **Key Libraries:** NLTK
   - **Use Case:** Linguistically meaningful text normalization

#### 4. **7-Text Preprocessing-Stopwords With NLTK.ipynb**
   - **Objective:** Remove stopwords from text
   - **Topics Covered:**
     - Identifying stopwords
     - Removing common English words
     - Custom stopword lists
   - **Key Libraries:** NLTK
   - **Use Case:** Cleaning text by removing non-informative words

#### 5. **8-Parts Of Speech Tagging.ipynb**
   - **Objective:** Identify and tag grammatical roles of words
   - **Topics Covered:**
     - POS tagging with NLTK
     - Understanding grammatical categories
     - Tagged corpora analysis
   - **Key Libraries:** NLTK
   - **Use Case:** Linguistic analysis and feature extraction

#### 6. **9-Named Entity Recognition.ipynb**
   - **Objective:** Extract named entities from text
   - **Topics Covered:**
     - Person, Organization, Location identification
     - Entity extraction techniques
     - NER applications
   - **Key Libraries:** NLTK
   - **Use Case:** Information extraction and content analysis

---

### Text Vectorization Techniques

#### 7. **15-Bag Of Words Practical's.ipynb**
   - **Objective:** Convert text into numerical features using Bag of Words
   - **Topics Covered:**
     - CountVectorizer implementation
     - Binary vs. frequency-based representations
     - Feature extraction from SMS spam dataset
   - **Key Libraries:** scikit-learn, NLTK
   - **Dataset Used:** Spam Classifier/SMSSpamCollection.txt (5,574 SMS messages)
   - **Output:** Sparse matrix representation of text features

#### 8. **16-TF-IDF Practical.ipynb**
   - **Objective:** Learn Term Frequency-Inverse Document Frequency
   - **Topics Covered:**
     - TF-IDF theory and implementation
     - Weighted term importance
     - Vectorization for better model performance
   - **Key Libraries:** scikit-learn
   - **Use Case:** Creating more informative text representations

#### 9. **26-Word2vec_Practical_Implementation.ipynb**
   - **Objective:** Deep learning-based word embeddings
   - **Topics Covered:**
     - Word2Vec architecture (Skip-gram and CBOW)
     - Vector representations of words
     - Semantic similarity between words
   - **Key Libraries:** gensim
   - **Use Case:** Capturing semantic meaning in vector space

---

### Real-World Projects

#### **PROJECT 1: Spam/Ham Classification**

Multiple implementations of SMS spam detection:

##### 10. **27-Spam Ham Classification Project Using TF-IDF And ML.ipynb**
   - **Objective:** Classify SMS messages as spam or legitimate (ham)
   - **Techniques:**
     - Data loading and preprocessing
     - TF-IDF vectorization with n-grams
     - Machine Learning classifier training
   - **Dataset:** SMSSpamCollection.txt
     - Total Messages: 5,574
     - Legitimate (Ham): 4,827 (86.6%)
     - Spam: 747 (13.4%)
   - **Workflow:**
     1. Load SMS messages from text file
     2. Data cleaning: regex substitution, lowercase conversion
     3. Tokenization and stopword removal
     4. Porter Stemming
     5. TF-IDF feature extraction with 2,500 features and 1-2 gram range
     6. Train ML classifiers
     7. Model evaluation and prediction

##### 11. **27.2-Spam Ham Classification Project Using BOW And TFIDF And ML.ipynb**
   - **Objective:** Compare Bag of Words vs TF-IDF approaches
   - **Key Differences:**
     - Side-by-side implementation of both vectorization methods
     - Performance comparison
     - Binary vs. weighted representations
   - **Use Case:** Understanding trade-offs between simple and advanced techniques

##### 12. **28 And 29 -Spam Ham Projects Using Word2vec,AvgWord2vec.ipynb**
   - **Objective:** Advanced spam classification using embeddings
   - **Techniques:**
     - Word2Vec representation
     - Average Word2Vec pooling
     - Deep learning embeddings for spam detection
   - **Advantage:** Better semantic understanding vs traditional BOW/TF-IDF

#### **PROJECT 2: Kindle Review Sentiment Analysis**

##### 13. **30 and 31-Project 2- Kindle Review Sentiment Analyis.ipynb**
   - **Objective:** Analyze sentiment of Amazon Kindle book reviews
   - **Dataset:** Kindle review/all_kindle_review.csv
   - **Expected Tasks:**
     - Review text preprocessing
     - Sentiment classification (positive/negative/neutral)
     - Sentiment distribution analysis
     - Feature importance identification
   - **Applications:**
     - Customer feedback analysis
     - Product quality assessment
     - Review trend analysis

---

## 📁 Dataset References

### 1. **Spam Classifier Directory**
   - **File:** `SMSSpamCollection.txt`
   - **Format:** Tab-separated file with 2 columns
     - Column 1: Label (ham/spam)
     - Column 2: Raw SMS text
   - **Statistics:**
     - Total Messages: 5,574
     - Ham Messages: 4,827 (86.6%)
     - Spam Messages: 747 (13.4%)
   - **Source:** Community-curated SMS spam corpus from:
     - Grumbletext forum (UK)
     - Caroline Tag's PhD research
     - NUS SMS Corpus (National University of Singapore)
     - SMS Spam Corpus v.0.1 Big

### 2. **Kindle Review Directory**
   - **File:** `all_kindle_review.csv`
   - **Content:** Amazon Kindle book reviews with ratings and text
   - **Use:** Sentiment analysis and review classification

---

## 🛠️ Technologies & Libraries

### Core Libraries
- **NLTK (Natural Language Toolkit):** Tokenization, stemming, lemmatization, POS tagging, NER
- **scikit-learn:** Machine learning algorithms, vectorization (CountVectorizer, TfidfVectorizer)
- **gensim:** Word2Vec and other advanced embeddings
- **pandas:** Data manipulation and CSV handling
- **numpy:** Numerical operations
- **regex (re):** Pattern matching for text cleaning

### Machine Learning Algorithms Used
- Naive Bayes
- Logistic Regression
- Support Vector Machines
- Random Forest
- Gradient Boosting
- Neural Networks (in Word2Vec projects)

---

## 🚀 Learning Path

### Beginner Level
1. Start with tokenization (Notebook 4)
2. Learn stopword removal (Notebook 7)
3. Explore stemming and lemmatization (Notebooks 5-6)

### Intermediate Level
4. POS tagging (Notebook 8)
5. Named Entity Recognition (Notebook 9)
6. Bag of Words vectorization (Notebook 15)
7. TF-IDF technique (Notebook 16)

### Advanced Level
8. Word2Vec embeddings (Notebook 26)
9. Complete projects with multiple techniques (Notebooks 27, 27.2, 28-29, 30-31)

---

## 💡 Key Concepts Covered

| Concept | Notebooks | Purpose |
|---------|-----------|---------|
| **Tokenization** | 4 | Break text into tokens |
| **Normalization** | 5, 6, 7 | Clean and standardize text |
| **Morphological Analysis** | 8, 9 | Understand grammatical structure |
| **Text Vectorization** | 15, 16, 26 | Convert text to numerical features |
| **Classification** | 27, 27.2, 28-29 | Predict categories (spam/ham) |
| **Sentiment Analysis** | 30-31 | Determine emotion/opinion |

---

## 📊 Project Specifications

### Spam/Ham Classification Pipeline
```
Raw SMS Text
    ↓
Regex Cleaning (remove special chars)
    ↓
Lowercase Conversion
    ↓
Tokenization & Stopword Removal
    ↓
Stemming (Porter Stemmer)
    ↓
Vectorization (BOW/TF-IDF/Word2Vec)
    ↓
Machine Learning Model
    ↓
Classification (Ham/Spam)
```

### Sentiment Analysis Pipeline
```
Review Text
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
Sentiment Classification
    ↓
Analysis & Insights
```

---

## 📈 Expected Outcomes

After completing this repository, you will:
- ✅ Understand fundamental NLP concepts and techniques
- ✅ Implement text preprocessing pipelines
- ✅ Convert text to numerical features using multiple methods
- ✅ Build and train text classification models
- ✅ Perform sentiment analysis on real datasets
- ✅ Compare different feature representation techniques
- ✅ Evaluate and optimize NLP models

---

## 🔧 Usage Instructions

### Prerequisites
```bash
pip install nltk scikit-learn pandas numpy gensim matplotlib seaborn
```

### Running Notebooks
1. Open any `.ipynb` file in Jupyter Notebook or VS Code
2. Install required NLTK datasets when prompted
3. Ensure data files are in correct relative paths
4. Run cells sequentially from top to bottom

### Data File Locations
- Spam data: `./Spam Classifier/SMSSpamCollection.txt`
- Kindle reviews: `./Kindle review/all_kindle_review.csv`

---

## 📝 Notes

- All notebooks use relative paths for data loading
- NLTK requires downloading language models (punkt, stopwords, etc.)
- Some notebooks may require one-time dataset downloads via `nltk.download()`
- The repository demonstrates both traditional (BOW, TF-IDF) and modern (Word2Vec) NLP approaches

---

## 🎓 Educational Value

This repository is ideal for:
- Students learning NLP fundamentals
- Data scientists exploring text mining techniques
- Professionals transitioning into NLP
- Those building text classification systems
- Practitioners in sentiment analysis and opinion mining

---

**Last Updated:** January 2026
**Repository Type:** Educational - NLP Learning & Implementation
