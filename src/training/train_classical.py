#imported all the dependencies 
import pandas as pd
import numpy as np
from src.features.cleaner import BasicTextCleaner
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
#2nd dependencies
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import itertools
import os
import joblib

#created a class named training
class Training:
    
#created a constructor with object parameter self
    def __init__(self):
        self.X = None
        self.y = None
        self.skf = None
#created a method with self and file_path as parameters & applied the clenaer on entire dataset
#stored the cleaned data inside the new column cleaned_text 
#mapped the fake values in the label column as 1 and real as 0
#stripped them to remove any nan values and missing values
#stored them in X and y
    def load_and_prepare_data(self,file_path):
        df = pd.read_csv(file_path)
        df = df.dropna(subset = ['text', 'label']).reset_index(drop = True)
        cleaner = BasicTextCleaner()
        df["cleaned_text"] = df["text"].apply(cleaner.clean)
        df["label"] = df["label"].map({'real' : 0, 'fake' : 1})
        df = df[df['cleaned_text'].str.strip() != ""].reset_index(drop = True)
        self.X = df["cleaned_text"]
        self.y = df["label"]
    
        return self.X, self.y
#created a method with self and models as parameters
#initialized the skf
#created an empty list for storing the result values
#initialized the ngrams
#then created a closed loop for model ecaluation
#created indexes for testing and training sets of x and y
#initialized the tfidf vectorizer with max_features = 5000
#then fitted it on data
    def run_evaluation(self, models):
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []
        ngram_options = [(1,1),(1,2)]
        class_weight = [None, "balanced"]
        #check if model supports class weights
         #if not runs
        
        # OUTER LOOP: Iterate through each model
        for model_name, model_object in models.items():
            for ngram in ngram_options:
                for weight in class_weight:
                    if weight == 'balanced' and not hasattr(model_object, 'class_weight'):
                        continue
                        
                    print(f"\nModel: {model_name} | ngram: {ngram} | class_weight: {weight}")
                    fold_f1_scores = []
                    print(f"Evaluating {model_name}...")
            
            # INNER LOOP: Iterate through the 10 folds
            for train_index, test_index in self.skf.split(self.X, self.y):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                
                tfidf = TfidfVectorizer(max_features=10000, stop_words = "english", ngram_range = ngram)
                
                X_train_tfidf = tfidf.fit_transform(X_train)
                X_test_tfidf = tfidf.transform(X_test)
                model_object.set_params(class_weight = weight)
                model = clone(model_object)
                model_object.fit(X_train_tfidf, y_train)
                
                # Predict on TEST data
                y_pred = model_object.predict(X_test_tfidf)
                
                score = f1_score(y_test, y_pred, average='binary')
                fold_f1_scores.append(score)
                
                #classification report and confusion matrix
                print("\nConfusion Matrix:")
                print(confusion_matrix(y_test,y_pred))
                
                print("\nClassification Report:")
                print(classification_report(y_test,y_pred))
            
            # Calculate metrics for THIS model after all 10 folds are done
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)
            
            results.append({
                'Model': model_name,
                'ngram_range' : ngram,
                'class_weight' : weight,
                'Mean_F1': mean_f1,
                'Std_F1': std_f1,
                'Scores' : fold_f1_scores
            })
            print(f"\nModel: {model_name} | ngram: {ngram} | class_weight: {weight}")
        # --- THE FIX: This block is OUTSIDE all loops ---
        # It only executes once all models and all folds are finished
        results_df = pd.DataFrame(results).sort_values(by='Mean_F1', ascending=False)
        
        print("\n--- Final Baseline Results ---")
        print(results_df.to_string(index=False))
        
        os.makedirs("experiments", exist_ok=True)
        results_df.to_csv("experiments/baseline_results.csv", index=False)
        
        return results_df # The method ends here, after everything is done
    
    def tune_hyperparameters(self):
        print("\nStarting Hyperparameter Tuning...\n")
        
        pipeline = Pipeline({
            ("tfidf", TfidfVectorizer(stop_words = "english")),
            ("model", LogisticRegression(max_iter = 1000))
        })
        
        param_grid = {
            "tfidf__max_features": [5000,10000],
            "tfidf__ngram_range" : [(1,1),(1,2)],
            "model__c" : [0.1,1,10],
            "model__class_weight" : [None, "balanced"]
        }
        
        grid = GridSearchCV(
            pipeline,param_grid,cv = 5, scoring = "f1", n_jobs=-1, verbose = 2 
        )
        
        grid.fit(self.X, self.y)
        print("\nBest Parameters Found:")
        print(grid.best_params_)
        
        print("\nBest F1 Score:")
        print(grid.best_score_)
        
        os.makedirs("models", exist_ok = True)
        
        joblib.dump(grid.best_estimator_, "models/best_tuned_pipeline.pkl")
        
        print("\nBest tuned model saved to models/best_tuned_pipeline.pkl")
        
        return grid.best_estimator_
    
        
            
    def save_best_model(self, model_name, model_object):
        """
        Retrains the winner on ALL data and saves the artifacts.
        """
        print(f"\n--- Retraining and Saving Best Model: {model_name} ---")
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        
        # Fit on the entire dataset
        X_tfidf = tfidf.fit_transform(self.X)
        model_object.fit(X_tfidf, self.y)
        y_pred = model_object.predict(X_tfidf)

        false_positives = []
        false_negatives = []

        for text, true, pred in zip(self.X, self.y, y_pred):

            if true == 1 and pred == 0:
                false_positives.append(text)
            elif true == 0 and pred == 1:
                false_negatives.append(text)
            #TP predicted as FN
            print("\nSample False Positives (Real predicted Fake):")
            for t in false_positives[:5]:
                print(t[:200])
                print()
            #TN predicted as FP
            print("\nSample False Negatives (Fake predicted Real):")
            for t in false_negatives[:5]:
                print(t[:200])
                print()
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_object, "models/baseline_model.pkl")
        joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
        print("Model and Vectorizer saved successfully.")
    
                
                
#created main method
if __name__ == "__main__":
    file_path = "data/raw/fake_news_dataset.csv"
    classifier = Training()
    X, y = classifier.load_and_prepare_data(file_path)
    models_to_test = {
        "LogisticRegression": LogisticRegression(),
        "Linear svm": LinearSVC(),
        "naive bayes": MultinomialNB()
    }
    
    print("data loaded successfully")
    print("Number of samples", len(X))
    print("class distribution:")
    print(y.value_counts())    
    
    results = classifier.run_evaluation(models_to_test)
    
    best_pipeline = classifier.tune_hyperparameters()
    
    best_model_name = results.iloc[0]['Model']
    best_model_obj = models_to_test[best_model_name]
    
    classifier.save_best_model(best_model_name, best_model_obj)
    
    print("evaluation finished")