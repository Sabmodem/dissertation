#!/usr/bin/env python3
"""
Quick preprocessing to test if trained models work on new large dataset
WITHOUT retraining. This creates a small test sample.

This script does NOT modify any existing code or models!
"""
import json
import pandas as pd
import sys

print("="*70)
print("QUICK TEST DATA PREPROCESSING")
print("="*70)

# Paths
fraud_file = '/mnt/1/dissertation/firm_years_labels.json'
all_file = '/mnt/1/dissertation/firm_years.json'
output_file = 'Quick_Test_Data.csv'

try:
    # Step 1: Load fraud cases (pre-labeled)
    print("\
[1/5] Loading fraud cases from firm_years_labels.json...")
    with open(fraud_file, 'r') as f:
        fraud_data = json.load(f)
    
    print(f"   Total fraud records available: {len(fraud_data)}")
    
    # Take first 500 fraud cases
    fraud_sample = fraud_data[:500]
    print(f"   Selected: 500 fraud cases")
    
    # Step 2: Load all filings
    print("\
[2/5] Loading all filings from firm_years.json...")
    print("   (This may take 30-60 seconds for 4.6 GB file...)")
    with open(all_file, 'r') as f:
        all_data = json.load(f)
    
    print(f"   Total records loaded: {len(all_data)}")
    
    # Step 3: Get fraud CIKs to exclude them from non-fraud sample
    print("\
[3/5] Identifying non-fraud companies...")
    fraud_ciks = set(f['cik'] for f in fraud_sample)
    print(f"   Fraud companies in sample: {len(fraud_ciks)}")
    
    # Get non-fraud records (companies NOT in fraud list)
    non_fraud_candidates = [f for f in all_data if f['cik'] not in fraud_ciks]
    print(f"   Non-fraud candidates found: {len(non_fraud_candidates)}")
    
    # Take first 500 non-fraud cases
    non_fraud_sample = non_fraud_candidates[:500]
    print(f"   Selected: 500 non-fraud cases")
    
    # Step 4: Format data for models
    print("\
[4/5] Formatting data to match Final_Dataset.csv structure...")
    # Extract MDA text and labels
    # X_fraud = [f['mda'] for f in fraud_sample]
    # X_non_fraud = [f['mda'] for f in non_fraud_sample]
    X_fraud = []
    for f in fraud_sample:
        try:
            X_fraud.append(f['mda'])
        except KeyError as e:
            print(f"   Warning: {e} not found in {f}")
    X_non_fraud = []
    for f in non_fraud_sample:
        try:
            X_non_fraud.append(f['mda'])
        except KeyError as e:
            print(f"   Warning: {e} not found in {f}")
    
    # Combine
    X_test = X_fraud + X_non_fraud
    y_test = ['yes'] * len(X_fraud) + ['no'] * len(X_non_fraud)
    
    print(f"   Total test samples: {len(X_test)}")
    print(f"   Fraud samples: {y_test.count('yes')}")
    print(f"   Non-fraud samples: {y_test.count('no')}")
    
    # Check for missing/empty MDA texts
    empty_count = sum(1 for text in X_test if not text or text.strip() == '')
    if empty_count > 0:
        print(f"   Warning: {empty_count} records have empty MDA text")
    
    # Step 5: Create DataFrame and save
    print("\
[5/5] Saving to CSV...")
    df = pd.DataFrame({
        'Fillings': X_test,
        'Fraud': y_test
    })
    
    df.to_csv(output_file, index=False)
    
    print(f"   ✅ Saved to: {output_file}")
    print(f"   File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Show statistics
    print("\
" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total records: {len(df)}")
    print(f"Fraud distribution:")
    print(df['Fraud'].value_counts())
    print(f"\
Average text length: {df['Fillings'].str.len().mean():.0f} characters")
    print(f"Min text length: {df['Fillings'].str.len().min():.0f} characters")
    print(f"Max text length: {df['Fillings'].str.len().max():.0f} characters")
    
    print("\
" + "="*70)
    print("✅ SUCCESS! Quick test data is ready.")
    print("="*70)
    print(f"\
You can now test your trained models on: {output_file}")
    print("\
This file has the SAME format as Final_Dataset.csv:")
    print("  - 'Fillings' column: MDA text")
    print("  - 'Fraud' column: 'yes' or 'no'")
    print("\
Next step: Run your trained models on this data to see if they transfer!")
    
except FileNotFoundError as e:
    print(f"\
❌ ERROR: File not found - {e}")
    print("Please check that the symlink /mnt/1/dissertation is accessible")
    sys.exit(1)
except Exception as e:
    print(f"\
❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
