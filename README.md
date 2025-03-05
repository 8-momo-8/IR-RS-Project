# Personalized Information Retrieval System

A comprehensive recommendation system for Q&A platforms that implements multiple retrieval and ranking approaches with personalization features.

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Components](#key-components)
4. [Usage](#usage)
5. [Evaluation](#evaluation)
6. [Future Work](#future-work)
7. [References](#references)

## Project Overview

This project implements a personalized recommendation system for Q&A platforms with the following key features:

- Multiple retrieval approaches (TF-IDF, BM25)
- Neural reranking using transformer models
- Query expansion with BERT and T5
- Personalization based on user history, tags, and metadata
- Comprehensive evaluation framework

## System Architecture

The system follows a modular architecture with four main layers:

### 1. Data Layer
- Handles data loading and preprocessing
- Supports CSV and JSON data sources
- Implements data cleaning and normalization

### 2. Retrieval Layer
- Base Retrieval: TF-IDF and BM25 algorithms
- Query Expansion: BERT and T5 based query enhancement
- Neural Reranking: BiEncoder architecture

### 3. Personalization Layer
- User profiling and history tracking
- Tag-based preferences
- Social and reputation features

### 4. Evaluation Layer
- Comprehensive metrics (Precision, Recall, MAP, nDCG)
- Cross-validation support
- Statistical significance testing

## Key Components

### DataPreprocessor
Handles data loading and preprocessing:
- Loads data from CSV and JSON files
- Cleans and preprocesses text data
- Creates lookup dictionaries
- Prepares text data for recommendation algorithms

### TFIDFRecommender
Implements TF-IDF based recommendation:
- Converts text to TF-IDF vectors
- Computes cosine similarities
- Generates recommendations based on similarity scores

### BM25Recommender
Implements BM25 ranking algorithm:
- More sophisticated than TF-IDF
- Considers document length normalization
- Implements Okapi BM25 scoring function

### QueryExpander
Implements query expansion using transformer models:
- Uses T5 and BERT for query expansion
- Generates multiple variations of input queries
- Includes cleaning and processing of expanded queries

### BiEncoder
Implements neural reranking:
- Uses sentence transformers for encoding
- Efficient batch processing
- Cosine similarity based reranking

### PersonalizedRecommender
Implements personalized recommendations:
- Combines content-based similarity with user history
- Maintains user profiles and preferences
- Implements weighted scoring system

### ComprehensiveEvaluator
Implements evaluation metrics:
- Precision, Recall, MAP, and nDCG
- Cross-validation support
- Statistical significance testing

## Usage

### Data Preprocessing
```python
preprocessor = DataPreprocessor()
answers, questions = preprocessor.load_data('data/answers.csv', 'data/questions.json')
```

### TF-IDF Recommendations
```python
tfidf_recommender = TFIDFRecommender()
tfidf_recommender.fit(questions, answers)
recommendations = tfidf_recommender.generate_recommendations()
```

### BM25 Recommendations
```python
bm25_recommender = BM25Recommender()
bm25_recommender.fit(questions, answers)
recommendations = bm25_recommender.generate_recommendations()
```

### Query Expansion
```python
query_expander = QueryExpander()
expanded_queries = query_expander.expand_query("How to implement a recommender system?")
```

### Neural Reranking
```python
bi_encoder = BiEncoder()
reranked_results = bi_encoder.rerank(query, candidate_answers)
```

### Personalized Recommendations
```python
personalized_recommender = PersonalizedRecommender()
personalized_recommender.fit(questions, answers, user_data, user_metadata)
recommendations = personalized_recommender.generate_recommendations(user_id, items)
```

### Evaluation
```python
evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate_all_models(recommendations_dict, relevant_df)
```

## Evaluation

The system implements comprehensive evaluation metrics:

- **Precision@k**: Fraction of relevant items in top-k recommendations
- **Recall@k**: Fraction of relevant items retrieved in top-k
- **MAP**: Mean Average Precision
- **nDCG**: Normalized Discounted Cumulative Gain

Evaluation is performed using:
- 5-fold cross-validation
- Statistical significance testing
- Early stopping based on nDCG

## Future Work

1. Integration with real-time recommendation systems
2. Support for additional data sources (e.g., MongoDB, Elasticsearch)
3. Implementation of advanced neural architectures (e.g., Transformer-based recommenders)
4. Deployment as a microservice
5. Integration with user feedback mechanisms

## References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
2. Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval.
3. Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems.
4. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
5. Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR.
