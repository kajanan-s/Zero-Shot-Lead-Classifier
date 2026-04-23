# Zero-Shot-Lead-Classifier
Zero shot text classifier used to determine where businesses' leads originate from

Classify sales leads into custom marketing channels **without any training data**. Built with Hugging Face's `facebook/bart-large-mnli` zero-shot model.

## Features
- No labeled training data required – works out of the box
- Define your own categories at runtime (e.g., `["LinkedIn Ad", "Podcast", "Billboard"]`)
- Builds a 100‑text evaluation dataset from live news headlines (NewsAPI)
- Compares accuracy against a majority‑class baseline
- Outputs: CSV of predictions, confusion matrix plot, evaluation summary
- **Interactive mode** – after evaluation, you can type your own lead messages and custom labels to see live predictions

## Setup Steps

### 1. Clone the repository
```bash
**git clone https://github.com/kajanan-s/Zero-Shot-Lead-Classifier/tree/main

cd zero-shot-lead-classifier
2. Install dependencies
bash
pip install -r requirements.txt

3. Get a free NewsAPI key
Register at newsapi.org
Copy your API key

4. Add your key to the script
Open zero_shot.py and replace:

python
NEWSAPI_KEY = "YOUR_API_KEY_HERE"
with your actual key.

How to Run
Run the script from the terminal:

bash
python zero_shot.py
What happens:

The script downloads the BART model (first run only, ~1.6 GB)

Builds a balanced dataset of 100 lead messages (20 per category) using NewsAPI + fallback

Classifies each message with the zero‑shot model

Prints evaluation metrics (accuracy, per‑class precision/recall, baseline comparison)

Saves classified_leads.csv, confusion_matrix.png, and evaluation_results.txt

Starts interactive mode – you can type your own texts and custom labels

Interactive mode (for live demo)
After evaluation, you see:

INTERACTIVE MODE - Try your own inputs
Enter lead message: 
Type a lead message, then enter comma‑separated labels. The model predicts the best label.

Example:

Enter lead message: I saw your TikTok ad about project management software.
Enter labels (comma separated): TikTok Ad, Instagram Ad, YouTube Ad
Prediction: TikTok Ad
Type quit to exit.

Sample Output
Below is the output from an actual run:

Building dataset (100 texts, 20 per category)...
  Facebook Ad: 20 examples
  Google Search: 20 examples
  Referral: 20 examples
  Email Campaign: 20 examples
  Event/Webinar: 20 examples

Dataset ready: 100 texts, 5 categories

Loading zero-shot classifier (facebook/bart-large-mnli)...
 Model loaded.

Running zero-shot classification on 100 texts...
100%|████████████████| 100/100 [00:45<00:00,  2.21it/s]

==================================================
EVALUATION RESULTS
==================================================
Majority Class Baseline: always predict 'Email Campaign'
  Baseline Accuracy: 20.00%
Zero-Shot Accuracy: 73.00%
Improvement over baseline: 53.00%

Zero-Shot Classification Report (per-class):
              precision    recall  f1-score   support
 Facebook Ad       0.76      0.65      0.70        20
Google Search       1.00      1.00      1.00        20
     Referral       0.42      1.00      0.59        20
Email Campaign       1.00      0.40      0.57        20
Event/Webinar       1.00      0.55      0.71        20

Confusion matrix saved as 'confusion_matrix.png'
Dataset with predictions saved as 'classified_leads.csv'

==================================================
INTERACTIVE MODE - Try your own inputs
==================================================
Enter lead message: 
Generated Files
File	Description
classified_leads.csv	100 texts with true and predicted labels
confusion_matrix.png	Heatmap of correct vs incorrect predictions
evaluation_results.txt	Accuracy, baseline, improvement
How to Use Your Own Labels
In the script: change the CANDIDATE_LABELS list before running.

In interactive mode: type any comma‑separated labels at runtime – no retraining required.

Limitations
English only
