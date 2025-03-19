Essay Quality Assessment and Source Classification

The dataset, "DAIGT External Train Dataset," is sourced from Kaggle (DAIGT External Train Dataset). It contains essays labeled as human-written or AI-generated, suitable for training models in the "LLM: Detect AI Generated Text" competition. Given its origin, it is recommended not to include the dataset in the GitHub repository due to potential licensing restrictions and the large file size (195,603 entries, as seen in the code). Instead, the README should provide a link to the dataset and instruct users to download it, adjusting file paths in the code if running outside Kaggle.

Technologies and Dependencies

The project relies on several Python libraries, each serving a specific role in the analysis pipeline:
Pandas and NumPy for data manipulation and numerical operations.

Scikit-learn for machine learning utilities, such as train-test splits and feature scaling.

TensorFlow for building and training the neural network model.

SpaCy for natural language processing tasks, including tokenization and part-of-speech tagging.

TextBlob for sentiment analysis, enhancing the feature set.

Matplotlib and Seaborn for data visualization, aiding in result interpretation.


Model Architecture and Training
The classification model is a neural network designed for binary classification, with the following architecture:
Input layer accepting combined numerical and TF-IDF features.

Dense layer with 512 units, ReLU activation, followed by BatchNormalization and Dropout (0.3).

Dense layer with 256 units, ReLU activation, followed by BatchNormalization and Dropout (0.2).

Dense layer with 128 units, ReLU activation, followed by BatchNormalization and Dropout (0.1).

Output layer with 1 unit, Sigmoid activation for binary classification.

The model is compiled with the Adam optimizer (learning rate 0.001) and binary cross-entropy loss. Training involves 50 epochs, batch size of 32, and includes early stopping and learning rate reduction on plateau, as seen in the code with callbacks for monitoring validation loss. Class weights are used to handle potential class imbalance, calculated as:
Class 0 weight: len(y_train) / (2 * (y_train == 0).sum())

Class 1 weight: len(y_train) / (2 * (y_train == 1).sum())

Results and Performance
The model achieves impressive performance on the validation set, with metrics indicating high reliability:
Validation Accuracy: 0.9973

Validation AUC: 0.9992

Validation Precision: 0.9986

Validation Recall: 0.9972

These results suggest the model is highly effective at distinguishing between human-written and AI-generated essays, potentially surprising users given the complexity of natural language tasks. Visualizations, such as training metrics plots, are included to analyze accuracy, loss, precision, recall, and AUC over epochs, enhancing result interpretation.
Feature Extraction and Analysis Functions
The project includes comprehensive feature extraction, covering:
Text Preprocessing: Cleaning text, removing special characters, and lemmatizing using SpaCy.

Basic Statistics: Word count, sentence count, average word length.

Complexity Metrics: Text density, complex word ratio (words longer than 6 characters).

Linguistic Features: Part-of-speech ratios (nouns, verbs, adjectives, adverbs) and sentiment scores using TextBlob.

Advanced Features: Sentence length variation, lexical diversity, and syntactic complexity.

Functions like analyze_single_essay, visualize_essay_analysis, and generate_detailed_feedback allow for individual essay analysis, providing detailed reports and visualizations. For example, analyze_single_essay outputs metrics such as word count, sentence count, text density, and part-of-speech ratios, while visualize_essay_analysis creates bar plots for part-of-speech distribution and complexity metrics.
Future Work and Extensions
Potential future enhancements include:
Integrating transformer-based models like BERT for improved classification accuracy.

Developing a web application for user-friendly essay analysis, making it accessible to educators and students.

Expanding the feature set with additional NLP techniques, such as coherence analysis or readability scores.

Exploring unsupervised methods for essay quality assessment, potentially identifying patterns without labeled data.


