from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Initialize FastAPI app
app = FastAPI()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = TFAutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# Define request body schema
class TextRequest(BaseModel):
    text: str

@app.post("/predict/")
async def predict(request: list[TextRequest]):
    
    results = []
    
    for item in request:
        
        text = item.text

        # Tokenize the text
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)

        # Perform inference with the model
        outputs = model(inputs)

        # Get the logits (raw scores) from the model output
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits, axis=-1)

        # Get the predicted class index
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]

        if predicted_class == 0:
            prediction = "Negative"
        elif predicted_class == 1:
            prediction = "Neutral"
        else: 
            prediction = "Positive"
            

        # Convert probabilities to a list
        probabilities_list = probabilities.numpy().tolist()[0]
        
        result = {
            "text": text,
            "predicted_class": prediction}
        
        results.append(result)
        # Return results
    return results
