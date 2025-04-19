# Product Requirements Document (PRD)

**Project Title**: FXC SKU Predictor - SKU Prediction System for Car Parts Bids
**Document Version**: 2.0
**Date**: April 19, 2025

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
- A web application for making predictions through a user interface.
- A feedback mechanism to improve predictions over time.
- Monitoring and backup utilities for production deployment.

### Out of Scope
- Real-time processing of bids (predictions are made on-demand but not in real-time streaming).
- Integration with external systems beyond the web application.
- Mobile application development.

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
- **Description**: Train a neural network model to predict SKUs or equivalency groups, addressing class imbalance if present. The system will support multiple model types including:
  - Neural Network (MLP Classifier)
  - Random Forest
  - Logistic Regression
- **Testing**:
  - Evaluate model performance (accuracy, precision, recall).
  - Test handling of imbalanced SKU distributions.
  - Compare performance across different model types.

### 4.5 Prediction
- **Description**: Preprocess new bid descriptions and predict SKUs using the trained model, outputting results in JSON format. The system will provide:
  - The most likely SKU prediction
  - A confidence score for the prediction
  - Alternative SKU suggestions with their confidence scores
- **Testing**:
  - Validate predictions against a known test set.
  - Confirm JSON output is correctly formatted.
  - Verify confidence scores are properly calculated and normalized.

### 4.6 Feedback and Continuous Learning
- **Description**: Implement a feedback mechanism to collect user corrections when predictions are incorrect. The system will:
  - Store user feedback in a database
  - Use feedback data for periodic model retraining
  - Track prediction accuracy over time
- **Testing**:
  - Test feedback collection and storage
  - Verify feedback data is properly incorporated during retraining
  - Confirm model accuracy improves with feedback incorporation

### 4.7 Maintenance and Monitoring
- **Description**: Provide utilities for system maintenance and monitoring, including:
  - Automated database backups
  - Log monitoring for error detection
  - Email notifications for critical errors
  - Performance metrics tracking
- **Testing**:
  - Test backup creation and restoration
  - Verify error detection in logs
  - Confirm notification system works correctly

### 4.8 Web Application for SKU Prediction

#### **Description**
Implemented a comprehensive web application that allows users to input information about car parts and receive SKU predictions. The application includes:

- **User Input**: A form with the following fields:
  - **Car Maker**: Dropdown for selecting the car maker (e.g., "Renault", "Chevrolet", "Ford").
  - **Series**: Dropdown for selecting the car series, dynamically populated based on the selected maker.
  - **Model Year**: Dropdown for selecting the model year, dynamically populated based on the selected series.
  - **Description**: Text input for entering a description of the part.
- **Output**:
  - The predicted SKU with confidence score
  - Alternative SKU suggestions with their confidence scores
  - Option to provide feedback on prediction accuracy
- **Integration**:
  - Direct API integration between frontend and backend
  - Real-time predictions without file generation
  - Asynchronous processing for better user experience
- **Design**:
  - Modern, responsive design with the company branding
  - The company logo displayed prominently at the top of the page
  - Intuitive user interface with clear feedback mechanisms

#### **Admin Dashboard**
- Statistics on prediction accuracy
- Feedback monitoring and analysis
- System performance metrics
- Option to trigger model retraining

#### **Testing**
- Verified that dropdowns and text input fields accept valid inputs
- Tested API integration to confirm accurate SKU predictions
- Validated that predictions and confidence scores are displayed correctly
- Confirmed the feedback mechanism works properly
- Tested responsive design across different screen sizes

#### **Technical Implementation**
- **Frontend**: HTML, CSS, and JavaScript with responsive design
- **Backend**: Python Flask application with RESTful API endpoints
- **Database**: SQLite for feedback storage with automated backups
- **Deployment**: Standalone application with unified entry point
- **Monitoring**: Integrated error logging and monitoring

#### **Non-Functional Achievements**
- **Usability**: Simple and intuitive interface with clear feedback mechanisms
- **Performance**: Fast response times with asynchronous processing
- **Reliability**: Error handling and logging for troubleshooting
- **Maintainability**: Modular code structure with comprehensive documentation

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
- **Description**: Provide a modular, well-organized package structure with comprehensive documentation. The system is organized as follows:
  - `fxc_sku_predictor/`: Main package
    - `core/`: Core functionality
    - `models/`: Model definitions
    - `utils/`: Utility functions
    - `web/`: Web application
  - Unified entry point (`run.py`) for all functionality
  - Comprehensive README with usage instructions
- **Testing**:
  - Review code modularity and documentation
  - Verify package structure follows best practices
  - Test unified entry point for all functionality

### 5.4 Reliability
- **Description**: Achieve >90% SKU prediction accuracy (target to be refined post-testing).
- **Testing**:
  - Use cross-validation and holdout sets to measure accuracy.
  - Refine model to meet reliability targets.

---

## 6. Technical Requirements

### 6.1 Development Stack
- **Programming Language**: Python 3.8+
- **Web Framework**: Flask for the web application
- **Libraries**:
  - `pandas` for data handling
  - `scikit-learn` for machine learning
  - `numpy` for numerical operations
  - `joblib` for model serialization
  - `flask` for web application
  - `sqlite3` for database operations

### 6.2 System Architecture
- **Package Structure**: Modular Python package (`fxc_sku_predictor`)
- **Web Application**: Flask-based web interface
- **Database**: SQLite for feedback storage
- **Monitoring**: Integrated logging and error monitoring
- **Backup**: Automated database backup system

### 6.3 Deployment
- **Environment**: Standalone application with unified entry point
- **Configuration**: Configurable paths and settings
- **Entry Point**: `run.py` script with command-line interface
- **Commands**:
  - `web`: Run the web application
  - `predict`: Make predictions from the command line
  - `backup`: Create database backups
  - `monitor`: Monitor logs for errors

### 6.4 Testing
- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **System Tests**: Test the entire system
- **Performance Tests**: Test system performance

---

## 7. Assumptions and Constraints

### 7.1 Assumptions
- The historical bid data is located at "C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\Fuente_Json_Consolidado\Consolidado.json".
- Descriptions are primarily in Spanish with potential errors.
- An initial SKU equivalency table is provided.
- Users have basic knowledge of car parts terminology.

### 7.2 Constraints
- On-demand processing (not continuous real-time streaming).
- Python-based implementation.
- Web application accessible via modern browsers.
- Single-server deployment (not distributed).

---

## 8. Risks and Mitigations

| **Risk**                           | **Mitigation**                                                                 | **Implementation**                     |
|------------------------------------|--------------------------------------------------------------------------------|----------------------------------------|
| Incorrect or inaccessible file path| Robust error handling for file loading with clear error messages                | Implemented comprehensive error handling with logging |
| Poor data quality                  | Advanced text preprocessing with spelling correction and standardization       | Created dedicated text preprocessing module with configurable dictionaries |
| Low accuracy due to ambiguity      | Neural network model with confidence scores and alternative suggestions        | Implemented neural network with top-5 predictions and confidence scores |
| Class imbalance                    | Filtering to ensure minimum examples per SKU and model selection               | Applied minimum example threshold and compared multiple model types |
| New SKUs or format changes         | Feedback mechanism and periodic retraining                                     | Implemented feedback database and retraining scheduler |
| System failures                    | Monitoring, logging, and automated backups                                     | Added monitoring utilities and backup system |

---

## 9. Timeline and Milestones

| **Milestone**                 | **Status**               | **Completion Date**         |
|-------------------------------|--------------------------|------------------------------|
| Data Acquisition and Analysis | Completed                | April 14, 2025               |
| Preprocessing and Features    | Completed                | April 15, 2025               |
| Model Development             | Completed                | April 18, 2025               |
| Web Application Development   | Completed                | April 19, 2025               |
| Feedback System Implementation| Completed                | April 19, 2025               |
| Project Reorganization        | Completed                | April 19, 2025               |
| Testing and Validation        | Completed                | April 19, 2025               |
| Documentation                 | Completed                | April 19, 2025               |

*Note: All milestones have been successfully completed.*

---

## 10. Stakeholders

- **Data Team**: Supplied historical data and equivalency tables.
- **Developers**: Built and maintain the system.
- **End-Users**: Use predictions for bidding through the web application.
- **Project Manager**: Oversaw execution and ensured quality standards.
- **Maintenance Team**: Monitors system performance and handles updates.

---

## 11. Conclusion

This PRD defines a scalable, automated SKU Prediction System that has been successfully implemented to improve efficiency and accuracy in car parts bidding. The system includes a web application with a user-friendly interface, a feedback mechanism for continuous improvement, and comprehensive monitoring and maintenance utilities.

The project has been completed with all milestones achieved. The system is now ready for production use, with a modular and maintainable codebase that can be easily extended with new features in the future.

---

**Approval**
This project has been approved and successfully implemented.

Juan P. Roldan - Project Lead
Fixacar Team - Stakeholders

---