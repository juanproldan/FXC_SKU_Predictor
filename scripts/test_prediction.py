import sys
from predict_with_hierarchical import HierarchicalSKUPredictor

def main():
    # Initialize the predictor
    predictor = HierarchicalSKUPredictor()
    
    # Get input from command line or prompt user
    if len(sys.argv) > 1:
        description = " ".join(sys.argv[1:])
    else:
        print("Enter a product description to predict its SKU:")
        description = input("> ")
    
    # Create input data
    input_data = {
        "maker": "",  # Leave empty to predict maker
        "series": "",
        "model": "",
        "descripcion": description
    }
    
    # Make prediction
    prediction = predictor.predict(input_data)
    
    # Print results
    print("\n--- Prediction Results ---")
    print(f"Description: {description}")
    print(f"Predicted Maker: {prediction['predicted_maker']} (confidence: {prediction['maker_confidence']:.3f})")
    print(f"Predicted SKU: {prediction['predicted_sku']} (confidence: {prediction['sku_confidence']:.3f})")
    print(f"Overall Confidence: {prediction['overall_confidence']:.3f}")

if __name__ == "__main__":
    main()
