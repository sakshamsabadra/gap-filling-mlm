"""
Gap Filling with Masked Language Modeling
Basic Version - Easy to understand and run
"""

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class GapFillingMLM:
    def __init__(self):
        """Initialize the MLM model"""
        print("🔄 Loading BERT model (this takes 30-60 seconds first time)...")
        self.model = pipeline('fill-mask', model='bert-base-uncased')
        print("✅ Model loaded successfully!\n")
    
    def predict(self, sentence):
        """
        Predict the masked word in a sentence
        
        Args:
            sentence (str): Sentence with [MASK] token
            
        Returns:
            list: Top 5 predictions with confidence scores
        """
        if '[MASK]' not in sentence:
            raise ValueError("Sentence must contain [MASK] token")
        
        predictions = self.model(sentence)
        return predictions
    
    def display_results(self, sentence, predictions):
        """Display predictions in a formatted way"""
        print("="*70)
        print(f"📝 Input Sentence: {sentence}")
        print("="*70)
        print("\n🎯 TOP 5 PREDICTIONS:\n")
        
        for i, pred in enumerate(predictions, 1):
            word = pred['token_str']
            confidence = pred['score'] * 100
            bar = '█' * int(confidence / 2)
            
            print(f"{i}. {word.upper():<15} {confidence:>6.2f}% {bar}")
        
        # Show completed sentence with best prediction
        best_word = predictions[0]['token_str']
        completed = sentence.replace('[MASK]', f"**{best_word.upper()}**")
        
        print("\n" + "="*70)
        print(f"✅ Best Completion:\n   {completed}")
        print("="*70 + "\n")

def main():
    """Main function to run the program"""
    print("\n" + "="*70)
    print("🎓 GAP FILLING WITH MASKED LANGUAGE MODELING")
    print("="*70 + "\n")
    
    # Initialize model
    mlm = GapFillingMLM()
    
    # Example sentences
    examples = [
        "The cat sat on the [MASK].",
        "I love to eat [MASK] for breakfast.",
        "The [MASK] is shining brightly today.",
        "Python is a [MASK] programming language.",
        "Machine learning is a subset of artificial [MASK].",
        "The capital of France is [MASK].",
        "She went to the [MASK] to buy groceries.",
        "Water boils at 100 degrees [MASK]."
    ]
    
    while True:
        print("\n📋 MENU:")
        print("1. Try example sentences")
        print("2. Enter your own sentence")
        print("3. Batch process multiple sentences")
        print("4. Exit")
        
        choice = input("\n👉 Select option (1-4): ").strip()
        
        if choice == '1':
            # Show examples
            print("\n📚 Example Sentences:")
            for i, ex in enumerate(examples, 1):
                print(f"   {i}. {ex}")
            
            try:
                ex_num = int(input("\n👉 Select example (1-8): ").strip())
                if 1 <= ex_num <= len(examples):
                    sentence = examples[ex_num - 1]
                    predictions = mlm.predict(sentence)
                    mlm.display_results(sentence, predictions)
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Please enter a number!")
        
        elif choice == '2':
            # Custom sentence
            sentence = input("\n👉 Enter sentence with [MASK]: ").strip()
            try:
                predictions = mlm.predict(sentence)
                mlm.display_results(sentence, predictions)
            except ValueError as e:
                print(f"❌ Error: {e}")
        
        elif choice == '3':
            # Batch processing
            print("\n📦 Batch Processing Mode")
            print("Enter sentences one by one (empty line to finish):")
            
            sentences = []
            while True:
                sent = input(f"   Sentence {len(sentences)+1} (or press Enter to finish): ").strip()
                if not sent:
                    break
                sentences.append(sent)
            
            print(f"\n🔄 Processing {len(sentences)} sentences...\n")
            for sentence in sentences:
                try:
                    predictions = mlm.predict(sentence)
                    mlm.display_results(sentence, predictions)
                    input("Press Enter to continue...")
                except ValueError as e:
                    print(f"❌ Skipping '{sentence}': {e}\n")
        
        elif choice == '4':
            print("\n👋 Thank you for using Gap Filling MLM!")
            print("⭐ Don't forget to star this project on GitHub!\n")
            break
        
        else:
            print("❌ Invalid choice! Please select 1-4.")

if __name__ == "__main__":
    main()