#!/usr/bin/env python3
"""
Quick preprocessing v2 - Fixed to handle missing 'mda' fields properly
"""
import json
import pandas as pd
import sys

print("="*70)
print("QUICK TEST DATA PREPROCESSING V2 (Fixed)")
print("="*70)

fraud_file = '/mnt/1/dissertation/firm_years_labels.json'
all_file = '/mnt/1/dissertation/firm_years.json'
output_file = 'Large_Test_Data.csv'

try:
    # Step 1: Load fraud cases and FILTER out ones without 'mda'
    print("\n[1/5] Loading fraud cases...")
    with open(fraud_file, 'r') as f:
        fraud_data = json.load(f)
    
    print(f"   Total fraud records: {len(fraud_data)}")
    
    # Filter to only records WITH mda field
    fraud_with_mda = [f for f in fraud_data if 'mda' in f and f['mda'] and f['mda'].strip()]
    print(f"   Records with MDA text: {len(fraud_with_mda)}")
    print(f"   Records without MDA: {len(fraud_data) - len(fraud_with_mda)}")
    
    # Take 500 fraud cases
    fraud_sample = fraud_with_mda
    print(f"   Selected: {len(fraud_sample)} fraud cases for testing")
    
    # Step 2: Load all filings
    print("\n[2/5] Loading all filings...")
    print("   (This takes 30-60 seconds...)")
    with open(all_file, 'r') as f:
        all_data = json.load(f)
    
    print(f"   Total records: {len(all_data)}")
    
    # Filter to records WITH mda
    all_with_mda = [f for f in all_data if 'mda' in f and f['mda'] and f['mda'].strip()]
    print(f"   Records with MDA text: {len(all_with_mda)}")
    
    # Step 3: Get non-fraud records
    print("\n[3/5] Identifying non-fraud companies...")
    fraud_ciks = set(f['cik'] for f in fraud_sample)
    print(f"   Fraud companies in sample: {len(fraud_ciks)}")
    
    # Get non-fraud records
    non_fraud_candidates = [f for f in all_with_mda if f['cik'] not in fraud_ciks]
    print(f"   Non-fraud candidates: {len(non_fraud_candidates)}")
    
    # Take 500 non-fraud
    non_fraud_sample = non_fraud_candidates[:len(fraud_sample)]
    print(f"   Selected: {len(non_fraud_sample)} non-fraud cases")
    
    # Step 4: Format data
    print("\n[4/5] Formatting data...")
    
    X_fraud = [f['mda'] for f in fraud_sample]
    X_non_fraud = [f['mda'] for f in non_fraud_sample]
    
    X_test = X_fraud + X_non_fraud
    y_test = ['yes'] * len(X_fraud) + ['no'] * len(X_non_fraud)
    
    print(f"   Total samples: {len(X_test)}")
    print(f"   Fraud: {y_test.count('yes')}")
    print(f"   Non-fraud: {y_test.count('no')}")
    
    # Step 5: Save
    print("\n[5/5] Saving to CSV...")
    df = pd.DataFrame({
        'Fillings': X_test,
        'Fraud': y_test
    })
    
    df.to_csv(output_file, index=False)
    
    print(f"   ✅ Saved to: {output_file}")
    print(f"   File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total records: {len(df)}")
    print(f"\nFraud distribution:")
    print(df['Fraud'].value_counts())
    print(f"\nText length statistics:")
    print(f"  Average: {df['Fillings'].str.len().mean():.0f} characters")
    print(f"  Min: {df['Fillings'].str.len().min():.0f} characters")
    print(f"  Max: {df['Fillings'].str.len().max():.0f} characters")
    print(f"  Median: {df['Fillings'].str.len().median():.0f} characters")
    
    print("\n" + "="*70)
    print("✅ SUCCESS! Test data ready.")
    print("="*70)
    print(f"\nFile: {output_file}")
    print("Format: Same as Final_Dataset.csv")
    print("\nNext: Test your trained models on this data!")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

