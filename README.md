# GLiNER Integration for Entity Detection
## Overview
This repository provides code examples for using the GLiNER (Generalized Linguistic Named Entity Recognition) model to detect specific entities, such as requirement IDs, from structured text. The repository showcases two different approaches for integrating GLiNER:

- Using GLiNER's pre-trained model directly.
- Integrating GLiNER with the spaCy pipeline for enhanced linguistic processing.

## Introduction to GLiNER
GLiNER is useful for tasks such as Named Entity Recognition (NER) where you want to extract custom entities based on specific labels like requirement IDs, person names, or organization names. 


Here I have used **"urchade/gliner_multi_pii-v1"** a fine-tuned model (trained by fine-tuning urchade/gliner_multi-v2.1 on the urchade/synthetic-pii-ner-mistral-v1 dataset) capable of recognizing various types of personally identifiable information (PII) like, Phone number, PAN, Aadhar Number, CVV, Train ticket number, blood type, licence plate number, tax identification number, medical condition, identity card number, national id number, ip address, email address, iban, credit card expiration date.

## GLiNER Integrated with spaCy

This example demonstrates integrating GLiNER into a spaCy pipeline, allowing for flexible entity detection with custom labels. It uses the same model, "urchade/gliner_multi_pii-v1", but adds spaCy's powerful NLP capabilities.

## Pre-Requisites

- gliner-spacy -0.0.8
- gliner - 0.2.7

## Acknowledgements

- **GLiNER**: A lightweight model for Named Entity Recognition.
- **spaCy**: A powerful NLP library used for linguistic processing and entity detection.
