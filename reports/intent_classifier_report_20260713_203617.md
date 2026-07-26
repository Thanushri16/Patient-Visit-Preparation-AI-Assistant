# Intent Classifier Evaluation Report

- Mode: rule
- Total samples: 30
- Correct predictions: 28
- Accuracy: 0.9333
- Macro F1: 0.9397
- Weighted F1: 0.9343
- Unknown prediction rate: 0.2
- Generated at: 2026-07-13T20:36:17.326876

## Per-intent Metrics
- show_menu: support=4, precision=1.0, recall=0.75, f1=0.8571
- appointment_preparation: support=3, precision=1.0, recall=0.6667, f1=0.8
- report_new_symptoms: support=4, precision=1.0, recall=1.0, f1=1.0
- review_health_notes: support=3, precision=1.0, recall=1.0, f1=1.0
- report_allergy: support=3, precision=1.0, recall=1.0, f1=1.0
- medication_question: support=3, precision=1.0, recall=1.0, f1=1.0
- emergency_support: support=3, precision=1.0, recall=1.0, f1=1.0
- view_summary: support=3, precision=1.0, recall=1.0, f1=1.0
- unknown: support=4, precision=0.6667, recall=1.0, f1=0.8

## Misclassifications
- expected=show_menu predicted=unknown confidence=0.0 status=unknown text='What can you do for me?'
- expected=appointment_preparation predicted=unknown confidence=0.0 status=unknown text='Please prepare me for my visit.'

## Confusion Matrix
| expected \ predicted | show_menu | appointment_preparation | report_new_symptoms | review_health_notes | report_allergy | medication_question | emergency_support | view_summary | unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| show_menu | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| appointment_preparation | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| report_new_symptoms | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| review_health_notes | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| report_allergy | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| medication_question | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| emergency_support | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| view_summary | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |