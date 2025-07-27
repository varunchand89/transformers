import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('path_to_saved_model')  # local folder or HF hub
model.eval()

def predict_conversion(text, reason):
    inputs = tokenizer(text, reason, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, axis=1).item()
    return "Match (1)" if prediction == 1 else "Mismatch (0)"

gr.Interface(
    fn=predict_conversion,
    inputs=["text", "text"],
    outputs="text",
    title="Text–Reason Match Predictor"
).launch()
