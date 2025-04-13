import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import torch

def get_sentiment_score(model, heading):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = nlp(heading)
    
    if result[0]['label'] == "positive":
        return result[0]['score']
    elif result[0]['label'] == "neutral":
        return 0
    else:
        return -result[0]['score']

def get_vader_sentiment_score(heading):
    analyzer = SentimentIntensityAnalyzer()
    result = analyzer.polarity_scores(heading)
    
    if result['pos'] == max(result['neg'], result['neu'], result['pos']):
        return result['pos']
    if result['neg'] == max(result['neg'], result['neu'], result['pos']):
        return -result['neg']
    else:
        return 0

def FinBERT_sentiment_score(heading, max_length=512):
    """
    compute sentiment score using pretrained FinBERT on -1 to 1 scale. -1 being negative and 1 being positive
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        
        # If heading is a list, join it into a single string
        if isinstance(heading, list):
            heading = ' '.join(heading)
            
        # Truncate text if it's too long
        tokens = tokenizer(heading, truncation=True, max_length=max_length, return_tensors="pt")
        
        # Get prediction
        with torch.no_grad():
            outputs = finbert(**tokens)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get the highest probability class
        predicted_class = torch.argmax(predictions).item()
        score = predictions[0][predicted_class].item()
        
        # Map the prediction to -1 to 1 scale
        if predicted_class == 0:  # negative
            return -score
        elif predicted_class == 1:  # neutral
            return 0
        else:  # positive
            return score
            
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return 0  # Return neutral sentiment in case of error

def VADER_sentiment_score(heading):
    """
    compute sentiment score using pretrained VADER on -1 to 1 scale. -1 being negative and 1 being positive
    """
    try:
        nltk.download('vader_lexicon', quiet=True)
        analyzer = SentimentIntensityAnalyzer()
        
        # If heading is a list, join it into a single string
        if isinstance(heading, list):
            heading = ' '.join(heading)
            
        result = analyzer.polarity_scores(heading)
        if result['pos'] == max(result['neg'], result['neu'], result['pos']):
            return result['pos']
        if result['neg'] == max(result['neg'], result['neu'], result['pos']):
            return (0 - result['neg'])
        else:
            return 0
    except Exception as e:
        print(f"Error processing text with VADER: {str(e)}")
        return 0

# Process news data in batches
def process_news_batch(news_df, batch_size=100):
    BERT_sentiment = []
    VADER_sentiment = []
    
    for i in range(0, len(news_df), batch_size):
        batch = news_df.iloc[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(news_df)-1)//batch_size + 1}")
        
        for idx in range(len(batch)):
            news_list = batch.iloc[idx, 1:].tolist()
            news_list = [str(i) for i in news_list if i != '0' and pd.notna(i)]
            
            # Get sentiment scores
            score_BERT = FinBERT_sentiment_score(news_list)
            score_VADER = VADER_sentiment_score(news_list)
            
            BERT_sentiment.append(score_BERT)
            VADER_sentiment.append(score_VADER)
            
            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return BERT_sentiment, VADER_sentiment

# Load and process data
news_df = pd.read_csv("data/news.csv")
BERT_sentiment, VADER_sentiment = process_news_batch(news_df)

# Add sentiment scores to dataframe
news_df['FinBERT score'] = BERT_sentiment
news_df['VADER score'] = VADER_sentiment
news_df['combined_sentiment'] = (news_df['FinBERT score'] + news_df['VADER score']) / 2

# Save results
news_df.to_csv("data/news_w_sentiment2.csv", index=False)