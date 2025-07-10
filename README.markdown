# Amazon Product Review Sentiment Analyzer

This project performs sentiment analysis on Amazon product reviews using a machine learning approach. It processes review text, classifies sentiments as positive, neutral, or negative, and provides visualizations of sentiment distribution. A Gradio interface is included for interactive sentiment prediction.

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn`, `wordcloud`, `gradio`
- NLTK data: `stopwords`

Install dependencies:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud gradio
```

Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Dataset
- The dataset should be a CSV file (`amazon product.csv`) with columns:
  - `reviews.text`: The text of the review.
  - `reviews.rating`: The rating (1-5).

## Usage
1. **Load and Preprocess Data**:
   - Reads the CSV file and selects `reviews.text` and `reviews.rating`.
   - Drops missing values.
   - Defines sentiment based on rating:
     - Positive: ≥4
     - Neutral: 3
     - Negative: ≤2
   - Cleans text by:
     - Converting to lowercase.
     - Removing HTML tags, non-alphabetic characters, and stopwords (including custom stopwords).
   - Balances classes using oversampling to ensure equal representation of positive, neutral, and negative sentiments.

2. **Model Training**:
   - Uses `TfidfVectorizer` with unigrams and bigrams for feature extraction.
   - Splits data into 80% training and 20% testing sets.
   - Trains a Multinomial Naive Bayes model.

3. **Evaluation**:
   - Outputs accuracy, confusion matrix, and classification report.
   - Visualizes sentiment distribution with bar and pie charts.

4. **Prediction**:
   - Provides functions to predict sentiment for new reviews with confidence scores.
   - Example predictions:
     ```python
     print(predict_sentiment("The product was really good", model, vectorizer))  # Output: positive
     print(predict_sentiment("The product was very bad.", model, vectorizer))    # Output: negative
     print(predict_sentiment("It's okay, not too good.", model, vectorizer))    # Output: neutral
     ```

5. **Gradio Interface**:
   - Launches an interactive web interface for sentiment prediction.
   - Run the script and access the interface via the provided URL.

## Running the Gradio Interface
```python
interface.launch(share=True)
```
- Enter a review in the text box.
- View the predicted sentiment and confidence scores.

## Notes
- The dataset path is set to `/content/amazon product.csv`. Update the path as needed.
- Custom stopwords include common words and domain-specific terms like 'product', 'amazon', etc.
- The model uses balanced classes (600 samples each) to avoid bias toward majority classes.
- Visualizations require a display environment (e.g., Jupyter Notebook or local Python with a GUI backend).

## Example Visualizations
- **Bar Plot**: Shows the count of each sentiment class.
- **Pie Chart**: Displays the proportion of sentiments.