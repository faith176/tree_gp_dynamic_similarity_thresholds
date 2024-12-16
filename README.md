# **Genetic Programming for Duplicate Question Detection**

## **Overview**
This project applies **Genetic Programming (GP)** to generate dynamic similarity thresholds for identifying duplicate questions, using the Quora Duplicate Questions Dataset.

---

## **Features**
- **Custom Similarity Metrics**:
  - Parts of Speech Tags Similarity
  - Dependency Parsing Similarity
  - Sentiment Difference (using Vader)
  - Synonym Overlap
  - N Gram Similarity
  - Length Comparison
  - Unique Words Comparison
- **Genetic Programming**:
  - Multi-objective optimization for accuracy and complexity.
  - Feature engineering via GP-evolved expressions.
- **Visualization**:
  - Training Fitness Evolution
  - Pareto Front
  - Threshold statistics
  - Performance metrics.
- **Baseline Classifiers**: Performance comparison with traditional ML algorithms.

---

## **Pipeline**
1. **Data Preparation**:
   - Preprocess questions and extract features.
2. **Feature Engineering**:
   - Extract custom similarity features from question pairs.
3. **Genetic Programming**:
   - Optimize thresholds and prediction functions.
   - Evaluate solutions using k-fold cross-validation.
4. **Baseline Evaluation**:
   - Train Decision Tree, Random Forest, Linear Regression, and SVM for comparison.
5. **Visualization**:
   - Confusion matrices, Pareto solutions, threshold statistics, and model performance.

---

## **Technologies Used**
### **Libraries**:
- **Core Libraries**: `os`, `math`, `random`, `operator`
- **Data Processing**: `pandas`, `numpy`, `scipy`, `tqdm`
- **NLP**: `nltk`, `spacy`, `vaderSentiment`, `TextBlob`
- **Machine Learning**: `scikit-learn`, `sentence-transformers`
- **Visualization**: `matplotlib`, `seaborn`, `networkx`
- **Genetic Programming**: `deap`
