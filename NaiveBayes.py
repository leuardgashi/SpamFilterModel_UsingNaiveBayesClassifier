# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """ Load the dataset from the given CSV file. """
    try:
        data = pd.read_csv('YouTube-Spam-Collection-v1/Youtube01-Psy.csv')
        logging.info("Data successfully loaded.")
        return data
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """ Preprocess the data by splitting it into features and labels, and encoding them. """
    try:
        X = data['CONTENT']
        y = data['CLASS']
        # Encode labels if they're categorical
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        logging.info("Data preprocessing successful.")
        return X, y_encoded
    except KeyError as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

def train_model(X_train, y_train):
    """ Train the Naive Bayes model using the training data. """
    try:
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
        pipeline.fit(X_train, y_train)
        logging.info("Model training successful.")
        return pipeline
    except Exception as e:
        logging.error(f"Error in training: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """ Evaluate the model using the test data and return accuracy and confusion matrix. """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        logging.info("Model evaluation successful.")
        return accuracy, conf_matrix
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        raise

def main():
    # Load data
    file_path = 'YouTube-Spam-Collection-v1/Youtube01-Psy.csv'
    data = load_data(file_path)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, conf_matrix = evaluate_model(model, X_test, y_test)
    
    # Output results
    print("Accuracy of the model: ", accuracy)
    print("Confusion matrix: \n", conf_matrix)

if __name__ == "__main__":
    main()
