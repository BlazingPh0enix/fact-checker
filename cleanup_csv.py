# CSV Cleanup and Standardization Script
import pandas as pd
import re
import os

def clean_and_standardize_csv(input_file, output_file):
    """
    Clean up CSV by:
    1. Removing duplicates
    2. Standardizing text format
    3. Removing date columns
    4. Adding index
    """
    print(f"\n🧹 Cleaning up {input_file}...")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ File {input_file} not found!")
        return None
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Original data: {len(df)} rows")
        
        # Extract only the headline column
        headlines = df['headline'].copy()
        
        # Clean and standardize the text
        def standardize_text(text):
            if pd.isna(text):
                return ""
            
            # Convert to string and strip whitespace
            text = str(text).strip()
            
            # Remove extra whitespace and normalize spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Standardize quotes
            text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
            
            # Remove any trailing/leading punctuation artifacts
            text = text.strip('.,;:')
            
            # Ensure proper sentence case (first letter capitalized)
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            
            return text
        
        # Apply standardization
        headlines_cleaned = headlines.apply(standardize_text)
        
        # Remove empty headlines
        headlines_cleaned = headlines_cleaned[headlines_cleaned != ""]
        
        # Remove duplicates (case-insensitive)
        print(f"Before removing duplicates: {len(headlines_cleaned)} headlines")
        headlines_unique = headlines_cleaned.drop_duplicates()
        print(f"After removing exact duplicates: {len(headlines_unique)} headlines")
        
        # Remove case-insensitive duplicates
        headlines_df = pd.DataFrame({'headline': headlines_unique})
        headlines_df['headline_lower'] = headlines_df['headline'].str.lower()
        headlines_df = headlines_df.drop_duplicates(subset=['headline_lower'])
        headlines_final = headlines_df['headline'].reset_index(drop=True)
        
        print(f"After removing case-insensitive duplicates: {len(headlines_final)} headlines")
        
        # Create final DataFrame with index
        final_df = pd.DataFrame({
            'index': range(1, len(headlines_final) + 1),
            'headline': headlines_final
        })
        
        # Save to new CSV
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Cleaned data saved to {output_file}")
        print(f"📊 Final count: {len(final_df)} unique headlines")
        
        return final_df
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {str(e)}")
        return None

def analyze_headlines(df, name):
    """Analyze and display statistics about the headlines"""
    if df is None or len(df) == 0:
        print(f"\n❌ No data available for {name}")
        return
    
    print(f"\n📈 {name} Analysis:")
    print(f"   Total headlines: {len(df):,}")
    
    # Length statistics
    lengths = df['headline'].str.len()
    print(f"   Average length: {lengths.mean():.1f} characters")
    print(f"   Shortest: {lengths.min()} characters")
    print(f"   Longest: {lengths.max()} characters")
    
    # Find shortest and longest headlines
    shortest_idx = lengths.idxmin()
    longest_idx = lengths.idxmax()
    
    print(f"\n   Shortest headline: {df.iloc[shortest_idx]['headline']}")
    print(f"   Longest headline: {df.iloc[longest_idx]['headline'][:100]}...")
    
    # Common words analysis
    all_text = ' '.join(df['headline']).lower()
    words = re.findall(r'\b\w+\b', all_text)
    
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    from collections import Counter
    word_counts = Counter(filtered_words)
    
    print(f"\n   Most common words:")
    for word, count in word_counts.most_common(8):
        print(f"     {word}: {count} times")

def create_consolidated_file(sample_cleaned, full_year_cleaned):
    """Merge both cleaned files into one consolidated file"""
    print("\n🔗 CREATING CONSOLIDATED FILE")
    print("="*35)
    
    all_headlines = []
    
    # Add headlines from sample file
    if sample_cleaned is not None:
        all_headlines.extend(sample_cleaned['headline'].tolist())
        print(f"Added {len(sample_cleaned)} headlines from sample file")
    
    # Add headlines from full year file
    if full_year_cleaned is not None:
        all_headlines.extend(full_year_cleaned['headline'].tolist())
        print(f"Added {len(full_year_cleaned)} headlines from full year file")
    
    if not all_headlines:
        print("❌ No headlines to consolidate")
        return None
    
    # Remove duplicates again (in case there's overlap between files)
    print(f"Total before deduplication: {len(all_headlines)}")
    
    # Convert to DataFrame for easier deduplication
    consolidated_df = pd.DataFrame({'headline': all_headlines})
    consolidated_df['headline_lower'] = consolidated_df['headline'].str.lower()
    consolidated_df = consolidated_df.drop_duplicates(subset=['headline_lower'])
    
    # Create final consolidated DataFrame
    final_consolidated = pd.DataFrame({
        'index': range(1, len(consolidated_df) + 1),
        'headline': consolidated_df['headline'].reset_index(drop=True)
    })
    
    # Save consolidated file
    final_consolidated.to_csv('pib_headlines_final_consolidated.csv', index=False, encoding='utf-8')
    
    print(f"✅ Consolidated file created with {len(final_consolidated)} unique headlines")
    print(f"📁 Saved as: pib_headlines_final_consolidated.csv")
    
    return final_consolidated

def main():
    """Main function to run the cleanup process"""
    print("🚀 Starting CSV cleanup process...")
    
    # Clean the sample file
    sample_cleaned = clean_and_standardize_csv('pib_headlines_sample.csv', 'pib_headlines_cleaned_sample.csv')
    
    # Clean the full year file
    full_year_cleaned = clean_and_standardize_csv('pib_headlines_2025.csv', 'pib_headlines_cleaned_2025.csv')
    
    print("\n" + "="*50)
    print("🎉 CLEANUP COMPLETED")
    print("="*50)
    
    # Preview the cleaned data
    print("\n📋 PREVIEW OF CLEANED DATA")
    print("="*40)
    
    if sample_cleaned is not None:
        print("\n🔍 Sample file (first 10 headlines):")
        for i, headline in enumerate(sample_cleaned['headline'].head(10), 1):
            print(f"{i:2d}. {headline}")
    
    if full_year_cleaned is not None:
        print(f"\n🔍 Full year file (showing 5 random headlines from {len(full_year_cleaned)} total):")
        random_sample = full_year_cleaned.sample(n=min(5, len(full_year_cleaned)))
        for _, row in random_sample.iterrows():
            print(f"{row['index']:4d}. {row['headline']}")
    
    # Generate statistics
    print("\n📊 DATA STATISTICS")
    print("="*30)
    
    if sample_cleaned is not None:
        analyze_headlines(sample_cleaned, "Sample Data")
    
    if full_year_cleaned is not None:
        analyze_headlines(full_year_cleaned, "Full Year 2025 Data")
    
    # Create consolidated file
    consolidated = create_consolidated_file(sample_cleaned, full_year_cleaned)
    
    if consolidated is not None:
        print(f"\n🎯 FINAL RESULT: {len(consolidated):,} unique PIB headlines")
        
        # Show a few examples from the consolidated file
        print("\n📋 Sample from consolidated file:")
        sample_indices = [1, len(consolidated)//4, len(consolidated)//2, 3*len(consolidated)//4, len(consolidated)]
        for idx in sample_indices:
            if idx <= len(consolidated):
                headline = consolidated.iloc[idx-1]['headline']
                print(f"{idx:4d}. {headline}")
    
    print("\n" + "="*50)
    print("🎊 ALL CLEANUP OPERATIONS COMPLETED!")
    print("="*50)
    print("\n📁 Final files in your project folder:")
    print("✅ pib_headlines_cleaned_sample.csv - Cleaned sample data")
    print("✅ pib_headlines_cleaned_2025.csv - Cleaned full year data")
    print("✅ pib_headlines_final_consolidated.csv - All unique headlines merged")

if __name__ == "__main__":
    main()
