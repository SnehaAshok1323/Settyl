# Settyl
Settyl Data Science And Machine Learning Engineer Task
Introduction
In today's fast-paced business environment, the ability to accurately and efficiently predict internal statuses based on external descriptions is paramount. This project aims to leverage the power of machine learning to develop a solution that can automate this process, thereby enhancing operational efficiency and decision-making.
The project involves several key stages, including data preprocessing, model development, training and evaluation, API development, testing and validation, and documentation. By utilizing TensorFlow for model development and FastAPI for API implementation, we aim to create a robust and scalable solution.
This report outlines the methodologies employed, the challenges faced, and the results achieved throughout the project. It serves as a comprehensive documentation of the entire development process, providing insights into the technical aspects as well as the practical implications of the implemented solution.
Data Preprocessing
Objective
The goal of the data preprocessing stage was to prepare the dataset for model training by cleaning and formatting the external status descriptions and internal status labels. This involved handling any missing values and splitting the data into features and labels for model input.
Methodology
The preprocessing was performed using the following steps:
1.	Data Loading: The dataset was loaded into a Pandas DataFrame from a JSON file. This provided a structured format for easy manipulation and analysis.
2.	Missing Value Check: We checked for any missing values in the dataset to ensure data integrity. Fortunately, no missing values were found in either the external status or internal status columns.
3.	Data Splitting: The dataset was split into features (X) and labels (y). The features consisted of the external status descriptions, and the labels were the internal status labels.
Results
The data preprocessing stage resulted in a clean and structured dataset, with features and labels properly separated for model training. No missing values were detected, ensuring that the dataset was complete and ready for the next stages of model development and training.
 

Model Development and Training
Model Development
For this project, we utilized TensorFlow to develop a neural network model capable of predicting internal status labels based on external status descriptions. The model architecture comprised the following layers:

1.	Embedding Layer: Transforms the input text into dense vectors of fixed size (128 dimensions in this case).
2.	Global Average Pooling 1D Layer: Averages the embeddings across the sequence length, reducing the dimensionality of the data.
3.	Dense Layer: A fully connected layer with 128 neurons and ReLU activation.
4.	Dropout Layer: A dropout layer with a rate of 0.5 to prevent overfitting.
5.	Dense Layer: Another fully connected layer with 64 neurons and ReLU activation.
6.	Output Layer: The final dense layer with softmax activation to produce probability distributions over the internal status labels.
The model was compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the metric.
Text Vectorization
Before feeding the text data into the model, it was necessary to vectorize it. We used TensorFlow's TextVectorization layer to convert the external status descriptions into fixed-length integer sequences. The layer was adapted to the training data, ensuring that it could handle the vocabulary present in our dataset.
Model Training
The training process involved the following steps:
•	Label Encoding: The internal status labels were encoded into numerical format using scikit-learn's LabelEncoder.
•	Model Fitting: The model was trained on the vectorized text data and encoded labels for 20 epochs with a validation split of 20%. This allowed us to monitor the model's performance on unseen data during training.
Training Results
The model showed promising results, with a steady increase in accuracy over the epochs. The final epoch achieved a training accuracy of 93.24% and a validation accuracy of 98.37%. This indicates that the model was able to learn effectively from the training data and generalize well to the validation data.
 
Model Evaluation
Initial Performance
Initially, the model exhibited an accuracy of 58.88% on the test set. This highlighted the need for further tuning and optimization to enhance the model's predictive capabilities.
Hyperparameter Tuning and Epoch Increase
To improve the model's performance, hyperparameter tuning was employed, and the number of epochs for training was increased. This resulted in a significant improvement in the model's accuracy.
Model Evaluation Metrics
The final model was evaluated on a separate test set to assess its generalization ability. The evaluation metrics used were:
•	Accuracy: The proportion of correctly predicted labels to the total number of predictions.
•	Confusion Matrix: A matrix that visualizes the performance of the classification model by comparing the true labels with the predicted labels.
•	Classification Report: A detailed report that includes precision, recall, and F1-score for each class.
Results
The final evaluation of the model on the test set yielded the following results:
•	Test Accuracy: The model achieved a test accuracy of 98.37%, indicating a high level of predictive performance.
•	Confusion Matrix: The heatmap visualization of the confusion matrix, with darker shades of blue indicating higher values, provides a clear visual representation of the model's performance across different classes.
•	Classification Report: The classification report revealed high precision, recall, and F1-scores for all classes, confirming the model's effectiveness in predicting internal status labels.
Cross-Validation
To further validate the model's performance, k-fold cross-validation was performed. The average cross-validation accuracy was reported along with the standard deviation, providing a more robust assessment of the model's predictive ability.
Training and Validation Curves
The training and validation accuracy and loss curves were plotted to visualize the model's learning progress over the epochs. These plots helped in identifying any signs of overfitting or underfitting and ensured that the model was learning effectively.

 
API Development and Deployment
API Implementation
After achieving satisfactory performance with the machine learning model, the next step was to implement an API to expose the model for real-world usage. The API was developed using the FastAPI framework, known for its high performance and ease of use. The main objectives of the API were:
•	To accept external status descriptions as input.
•	To return the predicted internal status labels based on the input.
The implementation involved the following steps:
•	Model Saving: The trained model was saved to a file using TensorFlow's save method to ensure that it could be easily loaded and used in the API.
•	API Development: A main.py file was created to define the API endpoints and logic. The API was designed to receive external status descriptions, preprocess the input, use the saved model to make predictions, and return the predicted internal status labels.
•	Model Loading: The saved model was loaded into the API using TensorFlow's load_model function. This allowed the API to utilize the trained model for making predictions.
•	Local Testing: The API was tested locally to ensure its functionality and accuracy. This involved sending requests with example external status descriptions and verifying that the API returned correct predictions.
Deployment
The API was fully implemented and tested locally with the intention of deploying it on AWS for public access. However, due to technical issues with the AWS access key, the deployment on AWS was not completed. As a result, the API is currently not accessible via a public URL.
Despite this setback, the local implementation of the API successfully demonstrated its functionality and accuracy. The API is ready for deployment once the technical issues are resolved, and it can be made publicly accessible for users to interact with the machine learning model.
Conclusion
The development of the API, along with the local testing, marked a significant milestone in the project. While the deployment on AWS is pending due to technical challenges, the API serves as a valuable tool for automating the prediction of internal status labels based on external status descriptions. Once deployed, it will enhance operational efficiency in real-world scenarios
