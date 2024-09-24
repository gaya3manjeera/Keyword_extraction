import spacy
from gliner_spacy.pipeline import GlinerSpacy

# Configuration for GLiNER integration
custom_spacy_config = {
    "gliner_model": "urchade/gliner_multi_pii-v1",
    "chunk_size": 1024,
    "labels": ["requirement id"],
    "style": "ent",
    "threshold": 0.2,
    "map_location": "cpu" # only available in v.0.0.7
}

# Initialize a blank English spaCy pipeline and add GLiNER
nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

# Example text for entity detection
text = """
4.1.1.This involves Bill Gates reliance on specific linguistic structures to infer entities.
4.1.1.1.Checking
eDH_Pool_66: A capitalized word in the middle of a sentence might hint at a proper noun.
4.1.1.2.Security
REQ_DRD_030: In the realm of traditional machine learning methods for NER
4.1.1.3.Validation
REQ_DRD_040: The rule-based approach works best when you're dealing with specific
4.1.2.The surrounding words, whether preceding or succeeding, that offer clues
RNDS-B-00029   This method heavily relies on feature engineering
4.1.2.1.Word characteristics
GEN-GMV-PSP-PERF-0009(0): information derived from the root form of a word or its morphological nuances.
4.1.3.After the features are prepared, the model is trained on this enriched data.
[A: BT-LAH-219] The machine learning-based approach is the way to go
[A_BP-4.2.2.2_RQE765]
An offeror found to have a conflict of interest shall be disqualified. An offeror may be considered Where the conditions
[A_BP-4.2.2.2_RQE768]
If permitted,  If multiple/alternative quotes are being submitted, they must be clearly marked as “Main Quote” and “Alternative Quote”.
[A_BP-4.2.2.2_RQE819]
This RFQ is conducted in accordance with Policies and Procedures of IOM which can be accessed at IOM website IOM registration.
REQ_DRD_020: A capitalized word in the middle of a sentence might hint at a proper noun.
4.1.1.2. Security
"""

# Process the text with the pipeline
doc = nlp(text)

# Output detected entities
for ent in doc.ents:
    print(ent.text, ent.label_, ent._.score) # ent._.score only available in v. 0.0.7
