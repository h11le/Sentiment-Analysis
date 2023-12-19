# Sentiment-Analysis
Explore different approaches toward sentiment analysis through school reviews written by students

## 1. Review Collection:
- Source: https://www.ratemyprofessors.com/
- Schools: York University (Toronto, ON, Canada) and TMU - Toronto Metropolitan University (Toronto, ON, Canada)
- Labels including "positive", "negative", and "neutral" depend on the overall points given by the students

## 2. Approaches

The project explores various approaches to sentiment analysis, including:

1. **NLTK SentimentIntensityAnalyzer:**
   - Utilizes the NLTK library's SentimentIntensityAnalyzer for sentiment analysis based on pre-trained models.

2. **Bag-of-Words (BoW) Model:**
   - Implements a Bag-of-Words model to represent the text data and uses machine learning models, such as Naive Bayes, for sentiment classification.

## 3. Results
Switching from the Bag-of-Words (BoW) model to the SentimentIntensityAnalyzer significantly boosted sentiment analysis accuracy in this project. The BoW approach initially achieved 22% accuracy, while the SentimentIntensityAnalyzer, using a pre-trained model, elevated accuracy to 39%. This improvement underscores the effectiveness of leveraging pre-trained models for nuanced sentiment analysis in school reviews, showcasing the impact of method selection on overall performance.


## 4. Challenges and Approaches for Improvement

The current accuracy of 39% suggests that there is room for improvement in the sentiment analysis model. Several approaches can be considered to enhance accuracy:

1. **Feature Engineering:**
   - Experiment with different text preprocessing techniques, such as stemming, lemmatization, or handling stopwords differently. Adjusting these features may improve the model's understanding of the text.

2. **Advanced Models:**
   - Explore more sophisticated machine learning models, such as Support Vector Machines (SVM), Logistic Regression, or deep learning models like recurrent neural networks (RNNs) or transformers.

3. **Hyperparameter Tuning:**
   - Fine-tune hyperparameters for both the sentiment analysis method and the chosen machine learning model. Adjusting parameters like regularization strength, learning rates, or model complexity can impact performance.

4. **Ensemble Methods:**
   - Consider combining predictions from multiple models using ensemble methods (e.g., stacking or blending). Ensemble methods can sometimes outperform individual models by leveraging their complementary strengths.

5. **Data Augmentation:**
   - If the dataset is limited, consider augmenting it by generating variations of existing samples. This can help improve the model's generalization to diverse reviews.

6. **Addressing Class Imbalance:**
   - Check for class imbalances in the dataset. If certain sentiments are underrepresented, it might be beneficial to balance the classes through techniques like oversampling or undersampling.

7. **Error Analysis:**
   - Examine instances where the model makes mistakes. Understanding common misclassifications can provide insights into specific challenges the model faces, leading to targeted improvements.

8. **Domain-Specific Pre-training:**
   - If possible, pre-train the sentiment analysis model on a dataset specifically related to school reviews. This can help the model better capture the nuances and language used in educational contexts.
