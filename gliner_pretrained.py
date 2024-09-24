from gliner import GLiNER

model = GLiNER.from_pretrained("/content/gliner_pii")
text="""
4.1.1.This involves Bill Gates reliance on specific linguistic structures to infer entities.
4.1.1.1.Checking
REQ_DRD_020: A capitalized word in the middle of a sentence might hint at a proper noun.
4.1.1.2.Security
REQ_DRD_030: In the realm of traditional machine learning methods for NER
4.1.1.3.Validation
REQ_DRD_040: The rule-based approach works best when you're dealing with specific
4.1.2.The surrounding words, whether preceding or succeeding, that offer clues
REQ_DRD_050: This method heavily relies on feature engineering
4.1.2.1.Word characteristics
REQ_DRD_060: information derived from the root form of a word or its morphological nuances.
4.1.3.After the features are prepared, the model is trained on this enriched data.
REQ_DRD_070: The machine learning-based approach is the way to go
"""
labels = ["requirement id"]
entities = model.predict_entities(text, labels)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
