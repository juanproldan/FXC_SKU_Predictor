# Product Requirements Document (PRD)

**Project Title**: SKU Prediction System for Car Parts Bids  
**Document Version**: 1.1  
**Date**: [Insert Date]  

---

## 1. Introduction

This Product Requirements Document (PRD) outlines the requirements for an automated system designed to predict Stock Keeping Units (SKUs) for car parts based on bid descriptions submitted by insurance companies. The current manual process of identifying SKUs is inefficient and error-prone due to misspellings, inconsistent terminology, and variations in part descriptions. This system aims to automate SKU prediction using historical bid data, improving operational efficiency and accuracy.

---

## 2. Objectives

The key objectives of the SKU Prediction System are:

- **Automation**: Automate SKU prediction from bid descriptions.
- **Efficiency**: Reduce manual effort required for SKU identification.
- **Accuracy**: Enhance the precision of SKU predictions despite description inconsistencies.
- **Scalability**: Ensure the system can handle increasing volumes of bid data.

---

## 3. Scope

The project will include:

- A batch processing system to analyze historical bid data and train a machine learning model for SKU prediction.
- Preprocessing to standardize descriptions and correct errors.
- An equivalency table to manage interchangeable SKUs.
- SKU predictions output in JSON format.

### Out of Scope
- Real-time processing of bids.
- Integration with external systems.
- User interface development.

---

## 4. Functional Requirements

The system must include the following functionalities, with testing considerations for each phase:

### 4.1 Data Acquisition
- **Description**: Load historical bid data from a JSON file located at a configurable path (initially set to "C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\Fuente_Json_Consolidado\Consolidado.json") into a structured format (e.g., pandas DataFrame), extracting fields like part descriptions and assigned SKUs.
- **Testing**:
  - Verify correct parsing and loading of JSON data from the specified path.
  - Ensure the system handles incorrect or missing file paths gracefully (e.g., with appropriate error messages).
  - Confirm that all relevant fields are extracted accurately.

### 4.2 Data Preprocessing
- **Description**: 
  - Correct misspellings in descriptions (e.g., "capot" → "capó").
  - Standardize terminology (e.g., "cofre" → "capó").
  - Maintain an equivalency table for interchangeable SKUs.
- **Testing**:
  - Test spelling correction with a sample of known errors.
  - Validate standardization across varied description inputs.
  - Confirm the equivalency table groups SKUs correctly.

### 4.3 Feature Engineering
- **Description**: Convert preprocessed descriptions into numerical features (e.g., using TF-IDF).
- **Testing**:
  - Ensure features accurately represent descriptions.
  - Verify compatibility with the machine learning model.

### 4.4 Model Training
- **Description**: Train a classification model (e.g., Logistic Regression or Random Forest) to predict SKUs or equivalency groups, addressing class imbalance if present.
- **Testing**:
  - Evaluate model performance (accuracy, precision, recall).
  - Test handling of imbalanced SKU distributions.

### 4.5 Prediction
- **Description**: Preprocess new bid descriptions and predict SKUs using the trained model, outputting results in JSON format.
- **Testing**:
  - Validate predictions against a known test set.
  - Confirm JSON output is correctly formatted.

### 4.6 Maintenance
- **Description**: Support periodic model retraining with updated data to maintain accuracy.
- **Testing**:
  - Test retraining with new data to ensure integration.
  - Verify performance stability or improvement post-retraining.

### 4.7 Web Application for SKU Prediction

#### **Description**  
Develop a simple web application that allows users to input information about car parts and receive SKU predictions. The application will include:

- **User Input**: A form with four rows:
  - **Box #1**: Dropdown for selecting the car maker (e.g., "Mazda", "Renault", "Chevrolet").
  - **Box #2**: Dropdown for selecting the series (e.g., "CHEVROLET/OPTRA ADVANCE/BASICO").
  - **Box #3**: Dropdown for selecting the model year (e.g., "2008", "2009", "2010").
  - **Box #4**: Text input for entering a description of the part.
- **Output**: A fifth box in each row that displays the predicted SKU based on the input.
- **Integration**: The web app will generate a JSON file with the user input, which the main program can process to predict the SKU. The predicted SKU will then be displayed in the last box of each row.
- **Design**:
  - The web app will have a **black background** for a sleek, modern look.
  - The **company logo** will be displayed prominently at the top of the page. The logo file is located at:
    `C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\webapp\static\LogoFixacar.png`.

#### **Testing**  
- Verify that dropdowns and text input fields accept valid inputs.
- Ensure the JSON file is correctly formatted and includes all user inputs.
- Test integration with the main program to confirm accurate SKU predictions.
- Validate that the predicted SKU is displayed in the correct box.
- Confirm the logo is displayed correctly and the background is black.

#### **Technical Requirements**  
- **Frontend**: HTML, CSS, and JavaScript (or a framework like React for better scalability).
- **Backend**: Python (Flask or FastAPI) to handle JSON generation and integration with the main program.
- **Browser Compatibility**: Ensure the app works on Chrome and other modern browsers.
- **Output Format**: JSON file for integration with the main program.

#### **Non-Functional Requirements**  
- **Usability**: The interface should be simple and intuitive for users.
- **Performance**: The app should respond quickly to user inputs and display results in real-time.
- **Scalability**: The app should support multiple rows of input for batch predictions.

---

## 5. Non-Functional Requirements

The system must meet these standards, with testing for each:

### 5.1 Performance
- **Description**: Efficiently process large datasets (e.g., 3,000+ bids monthly, 5+ years of data) within hours.
- **Testing**:
  - Measure processing time for large datasets.
  - Optimize to meet performance goals.

### 5.2 Scalability
- **Description**: Handle growing data volumes and new SKUs with minimal rework.
- **Testing**:
  - Simulate increased data loads to test scalability.
  - Verify performance with added SKUs.

### 5.3 Maintainability
- **Description**: Provide a modular, documented codebase and allow easy updates to preprocessing dictionaries.
- **Testing**:
  - Review code modularity and documentation.
  - Test dictionary update process.

### 5.4 Reliability
- **Description**: Achieve >90% SKU prediction accuracy (target to be refined post-testing).
- **Testing**:
  - Use cross-validation and holdout sets to measure accuracy.
  - Refine model to meet reliability targets.

---

## 6. Technical Requirements

- **Programming Language**: Python
- **Libraries**:
  - `pandas` for data handling
  - `scikit-learn` for machine learning
  - `json` for output processing
  - Optional: `spaCy` for Spanish NLP
- **Output Format**: JSON
- **Environment**: Batch processing system (server or cloud-deployable)
- The system must allow the JSON file path to be configurable (e.g., via a configuration file or environment variable).
- **Testing**:
  - Confirm compatibility with specified libraries.
  - Test deployment in the target environment.

---

## 7. Assumptions and Constraints

### 7.1 Assumptions
- The historical bid data is initially located at "C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\Fuente_Json_Consolidado\Consolidado.json".
- Descriptions are primarily in Spanish with potential errors.
- An initial SKU equivalency table is provided.

### 7.2 Constraints
- Batch processing only (no real-time).
- Python-based with JSON output.
- Timeline TBD based on resources.

---

## 8. Risks and Mitigations

| **Risk**                           | **Mitigation**                                                                 | **Testing**                            |
|------------------------------------|--------------------------------------------------------------------------------|----------------------------------------|
| Incorrect or inaccessible file path| Implement robust error handling for file loading                                | Test with invalid paths and ensure errors are handled correctly |
| Poor data quality                  | Robust preprocessing for errors and standardization                            | Test preprocessing with bad data       |
| Low accuracy due to ambiguity      | Use equivalency groups to simplify predictions                                 | Validate accuracy with groups          |
| Class imbalance                    | Apply weighted loss or oversampling                                            | Test model on imbalanced data          |
| New SKUs or format changes         | Design for easy SKU and equivalency updates                                    | Test update process                    |

---

## 9. Timeline and Milestones

| **Milestone**                 | **Estimated Completion** | **Testing Phase**            |
|-------------------------------|--------------------------|------------------------------|
| Data Acquisition and Analysis | [Insert Date]            | Data validation              |
| Preprocessing and Features    | [Insert Date]            | Preprocessing accuracy       |
| Model Development             | [Insert Date]            | Model performance            |
| Testing and Validation        | [Insert Date]            | End-to-end system tests      |
| Deployment                    | [Insert Date]            | Deployment and maintenance   |

*Note: Dates to be finalized upon project start.*

---

## 10. Stakeholders

- **Data Team**: Supplies historical data and equivalency tables.
- **Developers**: Build and maintain the system.
- **End-Users**: Use predictions for bidding.
- **Project Manager**: Oversees execution.

---

## 11. Conclusion

This PRD defines a scalable, automated SKU Prediction System to improve efficiency and accuracy in car parts bidding. Each phase includes specific testing to ensure quality and reliability, from data acquisition to deployment.

---

**Approval**  
[Insert stakeholder names and signatures upon approval]  

---