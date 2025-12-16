"""
ML Model Training Script for Med4Me
Run this script to create ml_model.pkl, vectorizer.pkl, and treatment_db.json

Usage:
    python train_model.py
"""

import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

# Training data - medical symptoms and conditions
training_data = [
    # Fever cases
    ("fever high temperature chills body ache", "fever", 28, 0),
    ("running fever since yesterday headache", "fever", 35, 1),
    ("high fever 102F with body pain", "fever", 42, 0),
    ("temperature 103 shivering cold", "fever", 25, 1),
    ("fever not coming down paracetamol", "fever", 33, 0),
    
    # Diabetes cases
    ("high blood sugar frequent urination thirst", "diabetes", 45, 0),
    ("diabetes type 2 sugar levels high", "diabetes", 52, 1),
    ("hyperglycemia increased thirst weight loss", "diabetes", 48, 0),
    ("blood glucose 250 fatigue blurred vision", "diabetes", 55, 1),
    ("diabetic symptoms excessive hunger", "diabetes", 50, 0),
    
    # Cold/URTI cases
    ("cough cold runny nose sneezing", "cold", 30, 1),
    ("nasal congestion sore throat", "cold", 22, 0),
    ("upper respiratory infection coughing", "cold", 28, 1),
    ("common cold symptoms stuffy nose", "cold", 35, 0),
    ("sneezing watery eyes throat irritation", "cold", 27, 1),
    
    # Headache cases
    ("severe headache pain temples", "headache", 32, 1),
    ("migraine with aura sensitivity light", "headache", 29, 1),
    ("tension headache stress related", "headache", 38, 0),
    ("throbbing headache one side", "headache", 31, 1),
    ("chronic headaches daily occurrence", "headache", 40, 0),
    
    # Hypertension cases
    ("high blood pressure 160/100", "hypertension", 55, 0),
    ("hypertension diagnosis elevated bp", "hypertension", 48, 1),
    ("blood pressure consistently high", "hypertension", 52, 0),
    ("hbp headache dizziness", "hypertension", 58, 1),
    ("stage 2 hypertension readings", "hypertension", 60, 0),
    
    # Asthma cases
    ("asthma attack wheezing breathing difficulty", "asthma", 25, 1),
    ("shortness of breath chest tightness", "asthma", 32, 0),
    ("reactive airway disease coughing night", "asthma", 28, 1),
    ("bronchial asthma inhaler needed", "asthma", 35, 0),
    ("wheezing episodes seasonal", "asthma", 30, 1),
    
    # Gastritis cases
    ("stomach pain acidity heartburn", "gastric", 40, 0),
    ("acid reflux burning sensation chest", "gastric", 38, 1),
    ("gastritis epigastric pain nausea", "gastric", 45, 0),
    ("indigestion bloating gas", "gastric", 35, 1),
    ("gerd symptoms frequent", "gastric", 42, 0),
    
    # Allergy cases
    ("allergic reaction rash itching", "allergy", 27, 1),
    ("skin allergy hives red patches", "allergy", 32, 0),
    ("urticaria severe itching", "allergy", 29, 1),
    ("food allergy symptoms swelling", "allergy", 25, 0),
    ("seasonal allergies sneezing eyes", "allergy", 30, 1),
    
    # Arthritis cases
    ("joint pain knee swelling", "arthritis", 58, 1),
    ("osteoarthritis both knees stiffness", "arthritis", 62, 0),
    ("arthritis pain morning stiffness", "arthritis", 55, 1),
    ("degenerative joint disease back pain", "arthritis", 60, 0),
    ("rheumatoid arthritis multiple joints", "arthritis", 52, 1),
    
    # Mental health cases
    ("anxiety panic attacks stress", "mental_health", 28, 1),
    ("depression sad mood sleeping problems", "mental_health", 35, 0),
    ("severe anxiety work related", "mental_health", 32, 1),
    ("depressive symptoms loss interest", "mental_health", 40, 1),
    ("panic disorder frequent episodes", "mental_health", 30, 0),
    
    # General cases
    ("general checkup routine health", "general", 35, 0),
    ("mild symptoms unclear diagnosis", "general", 28, 1),
    ("preventive care health maintenance", "general", 40, 0),
    ("wellness check no specific complaints", "general", 32, 1),
]

# Treatment database
treatment_db = {
    "fever": {
        "Medicine": "- Paracetamol 500 mg: Every 6 hours for fever\n- Maintain hydration with ORS",
        "Alternative": "- Ibuprofen 400 mg: Every 8 hours if no contraindications\n- Cold compress",
        "Lifestyle": "Fluids (2-3 liters/day), rest, monitor temperature",
        "Red Flags": "Fever >39°C for >3 days, severe headache, rash, breathing difficulty",
        "Follow-Up": "Review in 48 hours if fever persists"
    },
    "diabetes": {
        "Medicine": "- Metformin 500 mg: BD with meals\n- Monitor blood glucose regularly",
        "Alternative": "- Glimepiride 1-2 mg OD\n- DPP-4 inhibitors (Sitagliptin 100 mg OD)",
        "Lifestyle": "Low glycemic diet, 150 min exercise/week, weight management",
        "Red Flags": "Glucose >400 mg/dL, confusion, chest pain, fruity breath odor",
        "Follow-Up": "HbA1c every 3 months, annual eye/foot examination"
    },
    "cold": {
        "Medicine": "- Cetirizine 10 mg: Once daily at bedtime\n- Dextromethorphan cough syrup: 10 mL TDS",
        "Alternative": "- Loratadine 10 mg OD\n- Steam inhalation 2-3 times daily",
        "Lifestyle": "Rest, warm fluids, avoid cold beverages, humidify room",
        "Red Flags": "High fever >38.5°C, chest pain, difficulty breathing",
        "Follow-Up": "Review if symptoms persist beyond 5-7 days"
    },
    "headache": {
        "Medicine": "- Paracetamol 500 mg: Every 6-8 hours\n- Sumatriptan 50 mg for migraine",
        "Alternative": "- Ibuprofen 400 mg TDS\n- Rest in dark, quiet room",
        "Lifestyle": "Stress management, regular sleep, hydration, avoid triggers",
        "Red Flags": "Sudden severe headache, vision changes, neck stiffness",
        "Follow-Up": "Review if frequency/severity increases"
    },
    "hypertension": {
        "Medicine": "- Amlodipine 5 mg: Once daily\n- Monitor BP regularly",
        "Alternative": "- Losartan 50 mg OD\n- Enalapril 5 mg OD",
        "Lifestyle": "Low sodium diet, DASH diet, exercise, weight reduction",
        "Red Flags": "BP >180/120, chest pain, severe headache, vision changes",
        "Follow-Up": "BP monitoring weekly initially, review medications every 3 months"
    },
    "asthma": {
        "Medicine": "- Salbutamol inhaler: 2 puffs PRN\n- Budesonide 200 mcg: BD",
        "Alternative": "- Montelukast 10 mg once daily\n- Formoterol + Budesonide combination",
        "Lifestyle": "Avoid triggers, breathing exercises, healthy weight, flu vaccination",
        "Red Flags": "Severe breathing difficulty, blue lips, unable to speak sentences",
        "Follow-Up": "Review in 2 weeks, peak flow monitoring"
    },
    "gastric": {
        "Medicine": "- Omeprazole 20 mg: Once daily before breakfast\n- Antacid syrup after meals",
        "Alternative": "- Pantoprazole 40 mg OD\n- Ranitidine 150 mg BD",
        "Lifestyle": "Small frequent meals, avoid spicy/fatty foods, elevate head while sleeping",
        "Red Flags": "Vomiting blood, black stools, weight loss, difficulty swallowing",
        "Follow-Up": "Review in 4 weeks, consider endoscopy if persistent"
    },
    "allergy": {
        "Medicine": "- Cetirizine 10 mg once daily\n- Hydrocortisone cream 1%: Apply BD",
        "Alternative": "- Loratadine 10 mg OD\n- Calamine lotion for relief",
        "Lifestyle": "Identify triggers, loose cotton clothing, keep skin moisturized",
        "Red Flags": "Difficulty breathing, facial swelling, loss of consciousness",
        "Follow-Up": "Review in 1 week, allergy testing if recurrent"
    },
    "arthritis": {
        "Medicine": "- Ibuprofen 400 mg TDS after meals\n- Glucosamine + Chondroitin daily",
        "Alternative": "- Naproxen 250 mg BD\n- Topical diclofenac gel",
        "Lifestyle": "Weight reduction, low-impact exercises, physical therapy",
        "Red Flags": "Severe pain, joint swelling/warmth, inability to bear weight",
        "Follow-Up": "Review in 2 weeks, X-rays if severe"
    },
    "mental_health": {
        "Medicine": "- Escitalopram 10 mg once daily (after psychiatric evaluation)\n- Consider counseling first",
        "Alternative": "- Sertraline 50 mg OD\n- Cognitive Behavioral Therapy",
        "Lifestyle": "Regular exercise, adequate sleep, social support, meditation",
        "Red Flags": "Suicidal thoughts, self-harm, severe panic attacks",
        "Follow-Up": "Psychiatric referral recommended, review in 1 week"
    },
    "general": {
        "Medicine": "- Symptomatic treatment as needed\n- Maintain healthy lifestyle",
        "Alternative": "- Specific treatment based on detailed examination",
        "Lifestyle": "Balanced diet, regular exercise, adequate sleep",
        "Red Flags": "Any persistent or worsening symptoms",
        "Follow-Up": "As clinically indicated"
    }
}

def train_model():
    """Train the ML model and save artifacts"""
    print("Starting model training...")
    
    # Prepare data
    symptoms = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]
    ages = [item[2] for item in training_data]
    genders = [item[3] for item in training_data]
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(symptoms)
    
    # Combine features
    print("Combining features...")
    X = np.hstack([
        X_text.toarray(),
        np.array(ages).reshape(-1, 1),
        np.array(genders).reshape(-1, 1)
    ])
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.2%}")
    print(f"Testing accuracy: {test_score:.2%}")
    
    # Save artifacts
    print("\nSaving model artifacts...")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Save model
    model_path = script_dir / 'ml_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = script_dir / 'vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Saved vectorizer to: {vectorizer_path}")
    
    # Save treatment database
    treatment_path = script_dir / 'treatment_db.json'
    with open(treatment_path, 'w') as f:
        json.dump(treatment_db, f, indent=2)
    print(f"✓ Saved treatment database to: {treatment_path}")
    
    print("\n✅ Model training complete!")
    print("\nYou can now run your Streamlit app:")
    print("streamlit run app.py")
    
    return model, vectorizer, treatment_db

if __name__ == "__main__":
    train_model()
