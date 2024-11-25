from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import BartForSequenceClassification, BartTokenizer
import torch
from typing import List, Optional

# Load the BART model and tokenizer
model_name = "facebook/bart-large-mnli"
model = BartForSequenceClassification.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Create FastAPI app
app = FastAPI(title="BART MNLI Model API", description="API for Natural Language Inference using BART model", version="1.0")

class NLIRequest(BaseModel):
    """
    Request model for Natural Language Inference (NLI).
    """
    premise: str = Field(..., description="The initial statement or context for inference.")
    hypothesis: Optional[str] = Field(None, description="The statement that is evaluated in relation to the premise. If omitted, labels must be provided.")
    labels: Optional[List[str]] = Field(None, description="A list of labels to classify the premise against if hypothesis is not provided.")

class NLIResponse(BaseModel):
    """
    Response model for Natural Language Inference (NLI).
    """
    label: str = Field(None, description="The predicted label for the premise.")
    scores: Optional[List[float]] = Field(None, description="The scores for each label when using candidate labels.")
    labels: Optional[List[str]] = Field(None, description="The list of labels sorted by their probability.")

@app.post("/predict", response_model=NLIResponse)
async def predict(nli_request: NLIRequest):
    """
    Predict the relationship between a premise and a hypothesis or classify the premise using candidate labels.
    
    If a hypothesis is provided, it returns the predicted label. If labels are provided, it classifies the premise 
    against those labels.
    """
    
    if nli_request.hypothesis:
        # If a hypothesis is provided, proceed with NLI prediction
        inputs = tokenizer(
            nli_request.premise,
            nli_request.hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Perform inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get predicted class index (0: entailment, 1: contradiction, 2: neutral)
        predicted_class = int(torch.argmax(logits, dim=1).item())
        labels = ["entailment", "contradiction", "neutral"]

        # Create response
        response = NLIResponse(label=labels[predicted_class], labels=labels, scores=None)
        return response

    elif nli_request.labels:
        # If labels are provided, classify the premise using them as hypotheses
        premise = nli_request.premise
        candidate_labels = nli_request.labels

        # Store the probabilities for each label
        label_scores = []

        for label in candidate_labels:
            hypothesis = f"This example is {label}."
            inputs = tokenizer.encode(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)

            # Perform inference
            with torch.no_grad():
                logits = model(inputs).logits

            # Get logits for entailment and contradiction
            entail_contradiction_logits = logits[:, [0, 2]]

            # Calculate probability of entailment (label being true)
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1].item()
            label_scores.append(prob_label_is_true)

        # Sort labels by their score in descending order
        sorted_indices = torch.tensor(label_scores).argsort(descending=True)
        sorted_labels = [candidate_labels[i] for i in sorted_indices]
        sorted_scores = [label_scores[i] for i in sorted_indices]

        # Create response with scores for each label
        response = NLIResponse(labels=sorted_labels, scores=sorted_scores, label=sorted_labels[0])
        return response

    else:
        raise HTTPException(status_code=422, detail="Either a hypothesis or a list of labels must be provided.")
