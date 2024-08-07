# Financial-News-Sentiment-Classifier-Using-DistilRoberta
API for financial news classifier using Finetuned DistilRoberta model from HuggingFace.

Usecase - Financial News Sentiment Analysis Using DistilRoberta Model Used - DistilRoberta(Hugging Face)

Dataset Used - Phrase Bank Dataset (sentence_75agree - 75% data points validated by the experts)

Preprocessing - For preprocessing we used famous yogawicaksan helper github.

Metric Used : Because the dataset is imbalanced , I would go for confusin matrix and F1 Score. Both will provide an detailed view how the model is working on diff labels.Sometime we might get high accuracy to the biased imbalanced dataset but the F1 Score will the tell you the actual scenario. So my go to option will be F1 Score and Confusion Matrix. While training Layoutlmv3 model for information extraction documents I had used F1 score as I had alost 30 NER to extract , and F1 score gave me an good idea how robust my model is for all the diff labels.

HOW TO RUN THE CODE: The last cell of this notebook consists API code. Copy the code in a separate file named main.py. Install the dependencies - requirements.txt Simply run the command - uvicorn main:app --reload

The API can take multiple nested inputs. Tested the API on 10 unseen data points.Below is the output:

A LOT CAN BE DONE , THIS IS JUST LIKE A QUICK POC , WHICH NEEDS TO PRESENTED TO THE CLIENT :)