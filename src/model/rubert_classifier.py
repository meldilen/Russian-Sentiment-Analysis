"""
RuBERT-based 3-class sentiment classifier.
Uses DeepPavlov/rubert-base-cased for Russian text.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Optional, Union
import numpy as np


class RuBERTClassifier(nn.Module):
    """
    RuBERT model fine-tuned for 3-class sentiment classification.
    """
    
    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class RuBERTPipeline:
    """
    End-to-end pipeline for RuBERT sentiment classification.
    Handles tokenization and inference.
    """
    
    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        model_path: Optional[str] = None,
        num_labels: int = 3,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = RuBERTClassifier(
            model_name=model_name,
            num_labels=num_labels,
        )
        
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Predict sentiment for input text(s).
        
        Args:
            texts: Single text or list of texts
            return_probs: If True, also return probability distributions
            
        Returns:
            Predicted class indices, or (indices, probabilities) if return_probs
        """
        if isinstance(texts, str):
            texts = [texts]
            
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            logits = self.model(**encodings)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        if return_probs:
            return preds_np, probs_np
        return preds_np
    
    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Return probability distributions for each class."""
        _, probs = self.predict(texts, return_probs=True)
        return probs
