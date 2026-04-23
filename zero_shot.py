"""
Zero-Shot Text Classifier for Lead Source Detection
Project submission for AI Village internship.

Requirements:
- Build dataset of 100 texts across 5 categories using NewsAPI
- Classify with zero-shot model (no training data)
- Evaluate accuracy & per-class metrics
- Compare against majority-class baseline
- Output: CSV with labels, confusion matrix plot
- Interactive mode: user can type their own text and custom labels
"""

import random
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from transformers import pipeline
from newsapi import NewsApiClient
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
NEWSAPI_KEY = "535f9a00b3194053bdb38bdcb79a8406"  # <-- YOUR KEY

# The 5 categories 
CANDIDATE_LABELS = [
    "Facebook Ad",
    "Google Search",
    "Referral",
    "Email Campaign",
    "Event/Webinar"
]

# Templates to convert news headlines into realistic lead messages
LEAD_TEMPLATES = {
    "Facebook Ad": [
        "Saw your Facebook ad about '{headline}'. Can you send me pricing?",
        "Your Instagram ad on '{headline}' caught my attention - interested!",
        "Clicked your Facebook carousel ad for '{headline}'. Tell me more."
    ],
    "Google Search": [
        "Found you on Google when searching '{headline}'.",
        "I Googled '{headline}' and your site came up first.",
        "Top search result for '{headline}' - exactly what I needed."
    ],
    "Referral": [
        "My colleague recommended you for '{headline}'.",
        "Sarah from your other client told me about '{headline}'.",
        "Your name came up regarding '{headline}'. My friend spoke highly of you."
    ],
    "Email Campaign": [
        "Just read your email about '{headline}'. Very interested!",
        "Your newsletter mentioned '{headline}' - can you share more details?",
        "Responding to your cold email about '{headline}'. Let's talk."
    ],
    "Event/Webinar": [
        "Attended your webinar on '{headline}'. When's the next session?",
        "Met you at the conference about '{headline}'. Following up as promised.",
        "Your presentation on '{headline}' was fantastic. Want to implement this."
    ]
}

# Fallback topics if NewsAPI fails
FALLBACK_TOPICS = [
    "small business growth strategies",
    "customer relationship management",
    "digital transformation",
    "business automation tools",
    "sales pipeline optimization"
]

# ========== 1. BUILD DATASET (100 texts, 20 per category) ==========
def build_dataset(api_key, samples_per_category=20):
    """Use NewsAPI + fallback to create 100 texts with true labels."""
    newsapi = NewsApiClient(api_key=api_key)
    
    category_keywords = {
        "Facebook Ad": ["facebook advertising", "instagram ads", "social media marketing"],
        "Google Search": ["google search", "seo optimization", "google ads"],
        "Referral": ["word of mouth marketing", "customer referral", "referral program"],
        "Email Campaign": ["email marketing", "newsletter strategy", "cold email outreach"],
        "Event/Webinar": ["business conference", "industry webinar", "trade show"]
    }
    
    all_data = []
    print("\n📊 Building dataset (100 texts, 20 per category)...")
    
    for category, keywords in category_keywords.items():
        examples = []
        for kw in keywords[:2]:
            if len(examples) >= samples_per_category:
                break
            try:
                articles = newsapi.get_everything(
                    q=kw, language='en', sort_by='relevancy',
                    page_size=5,
                    from_param=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                )
                if articles['status'] == 'ok' and articles['articles']:
                    for art in articles['articles'][:3]:
                        if len(examples) >= samples_per_category:
                            break
                        headline = art.get('title', '')
                        if not headline or len(headline) < 10:
                            continue
                        headline = headline[:80].replace('BREAKING:', '').strip()
                        template = random.choice(LEAD_TEMPLATES[category])
                        lead_text = template.format(headline=headline)
                        examples.append({'text': lead_text, 'true_label': category})
            except Exception as e:
                print(f"  Warning: NewsAPI error for {category}: {e}")
            time.sleep(0.3)
        
        # Fallback synthetic examples if needed
        if len(examples) < samples_per_category:
            needed = samples_per_category - len(examples)
            for _ in range(needed):
                topic = random.choice(FALLBACK_TOPICS)
                template = random.choice(LEAD_TEMPLATES[category])
                lead_text = template.format(headline=topic)
                examples.append({'text': lead_text, 'true_label': category})
        
        all_data.extend(examples[:samples_per_category])
        print(f"  {category}: {len(examples[:samples_per_category])} examples")
    
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ========== 2. LOAD ZERO-SHOT CLASSIFIER ==========
def load_classifier():
    print("\n🤖 Loading zero-shot classifier (facebook/bart-large-mnli)...")
    classifier = pipeline(
        'zero-shot-classification',
        model='facebook/bart-large-mnli',
        device=-1  # CPU
    )
    print(" Model loaded.\n")
    return classifier

def classify_text(classifier, text, candidate_labels):
    """Return the top predicted label for a single text."""
    result = classifier(text, candidate_labels, hypothesis_template="This lead came from {}")
    return result['labels'][0]

# ========== 3. EVALUATION WITH BASELINE ==========
def majority_class_baseline(true_labels):
    """Baseline: always predict the most common class."""
    most_common = pd.Series(true_labels).mode()[0]
    baseline_preds = [most_common] * len(true_labels)
    acc = accuracy_score(true_labels, baseline_preds)
    return most_common, baseline_preds, acc

def evaluate_and_save(classifier, df, candidate_labels):
    """Run classification, compute metrics, compare with baseline, save outputs."""
    texts = df['text'].tolist()
    true_labels = df['true_label'].tolist()
    
    # Zero-shot predictions
    print("🔄 Running zero-shot classification on 100 texts...")
    zero_shot_preds = []
    for t in tqdm(texts):
        zero_shot_preds.append(classify_text(classifier, t, candidate_labels))
    
    # Zero-shot accuracy
    zs_acc = accuracy_score(true_labels, zero_shot_preds)
    
    # Baseline
    most_common, baseline_preds, baseline_acc = majority_class_baseline(true_labels)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Majority Class Baseline: always predict '{most_common}'")
    print(f"  Baseline Accuracy: {baseline_acc:.2%}")
    print(f"Zero-Shot Accuracy: {zs_acc:.2%}")
    print(f"Improvement over baseline: {(zs_acc - baseline_acc):.2%}")

    # Save baseline comparison to a text file
    with open('evaluation_results.txt', 'w') as f:
        f.write("Zero-Shot Classifier Evaluation Results\n")
        f.write("======================================\n")
        f.write(f"Zero-Shot Accuracy: {zs_acc:.2%}\n")
        f.write(f"Baseline Accuracy:  {baseline_acc:.2%} (always predicts '{most_common}')\n")
        f.write(f"Improvement:        {zs_acc - baseline_acc:.2%}\n")

    print("\n📈 Zero-Shot Classification Report (per-class):")
    print(classification_report(true_labels, zero_shot_preds, target_names=candidate_labels))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, zero_shot_preds, labels=candidate_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=candidate_labels, yticklabels=candidate_labels)
    plt.title(f'Confusion Matrix - Zero-Shot Classifier (Acc: {zs_acc:.1%})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Save CSV with true and predicted labels
    df['predicted_label'] = zero_shot_preds
    df.to_csv('classified_leads.csv', index=False)
    print(" Dataset with predictions saved as 'classified_leads.csv'")
    
    return zs_acc

# ========== 4. INTERACTIVE MODE (for demo) ==========
def interactive_mode(classifier):
    """Allow user to type their own text and custom labels."""
    print("\n" + "="*50)
    print("INTERACTIVE MODE - Try your own inputs")
    print("="*50)
    print("Type your lead message and provide comma‑separated labels.")
    print("Type 'quit' to exit.\n")
    
    while True:
        text = input("Enter lead message: ").strip()
        if text.lower() == 'quit':
            break
        if not text:
            print("Please enter some text.\n")
            continue
        
        labels_input = input("Enter labels (comma separated, e.g., Facebook Ad, Google Search, Referral): ").strip()
        if not labels_input:
            print("At least one label required.\n")
            continue
        
        custom_labels = [label.strip() for label in labels_input.split(',') if label.strip()]
        if len(custom_labels) < 2:
            print("Please provide at least two labels.\n")
            continue
        
        try:
            pred = classify_text(classifier, text, custom_labels)
            print(f"\nPrediction: {pred}\n")
        except Exception as e:
            print(f"Error: {e}\n")

# ========== MAIN ==========
def main():
    # 1. Build dataset
    df = build_dataset(NEWSAPI_KEY, samples_per_category=20)
    print(f"\n✅ Dataset ready: {len(df)} texts, {df['true_label'].nunique()} categories")
    
    # 2. Load classifier
    classifier = load_classifier()
    
    # 3. Evaluate and save outputs
    evaluate_and_save(classifier, df, CANDIDATE_LABELS)
    
    # 4. Interactive mode for live demo (user inputs)
    interactive_mode(classifier)
    
    print("\n Project complete. Files created:")
    print("   - classified_leads.csv")
    print("   - confusion_matrix.png")
    print("   - evaluation_results.txt")

if __name__ == "__main__":
    main()