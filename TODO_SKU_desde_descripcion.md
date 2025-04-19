# ToDo List for SKU Prediction System Development

This ToDo list outlines the steps to build, test, and deploy the SKU Prediction System, updated to reflect the configurable JSON file path requirement from the PRD.

---

## Phase 1: Data Acquisition and Analysis

### Tasks
1. **Obtain Historical Data**
   - [ ] Secure the JSON file with historical bid data from the company.
   - [ ] Confirm the data structure and key fields (e.g., descriptions, SKUs).

2. **Implement Configurable File Path**
   - [ ] Add a configuration option (e.g., config file, environment variable) to specify the JSON file path.
   - [ ] Set the default path to "C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\Fuente_Json_Consolidado\Consolidado.json".

3. **Load and Parse Data**
   - [ ] Write a Python script to load JSON data from the configurable path into a pandas DataFrame.
   - [ ] Extract relevant fields like part descriptions and assigned SKUs.
   - [ ] Implement error handling for invalid or missing file paths.

4. **Analyze Data**
   - [ ] Check for data quality issues (e.g., missing values, inconsistencies).
   - [ ] Identify patterns, such as common misspellings or terminology variations.

### Testing
- [ ] Verify that the system correctly loads data from the specified path.
- [ ] Test with an invalid path to ensure proper error handling.
- [ ] Confirm that all relevant fields are extracted accurately.
- [ ] Document initial findings on data quality and patterns.

---

## Phase 2: Data Preprocessing

### Tasks
1. **Spelling Correction**
   - [ ] Build a dictionary of common misspellings and their corrections (e.g., "capot" → "capó").
   - [ ] Write a function to apply these corrections to descriptions.

2. **Term Standardization**
   - [ ] Create a mapping to standardize terms (e.g., "cofre" → "capó").
   - [ ] Apply this mapping to all descriptions.

3. **Equivalency Table**
   - [ ] Obtain or develop an equivalency table for interchangeable SKUs.
   - [ ] Add logic to group SKUs based on equivalency.

### Testing
- [ ] Test spelling correction on a sample of descriptions with known errors.
- [ ] Validate term standardization across diverse description examples.
- [ ] Confirm SKU grouping matches the equivalency table.

---

## Phase 3: Feature Engineering

### Tasks
1. **Text Vectorization**
   - [ ] Convert preprocessed descriptions into numerical features using TF-IDF or similar methods.
   - [ ] Explore alternative vectorization techniques if needed.

2. **Prepare Labels**
   - [ ] Assign SKUs or equivalency groups as target labels for the model.

### Testing
- [ ] Check that vectorized features accurately reflect description content.
- [ ] Ensure labels are correctly aligned with each description.

---

## Phase 4: Model Development and Training

### Tasks
1. **Model Selection**
   - [ ] Select a classification model (e.g., Logistic Regression, Random Forest).
   - [ ] Divide data into training and testing sets.

2. **Model Training**
   - [ ] Train the model using the training set.
   - [ ] Address class imbalance with appropriate techniques if necessary.

3. **Model Evaluation**
   - [ ] Assess performance with metrics like accuracy, precision, and recall.

### Testing
- [ ] Evaluate model performance on the test set.
- [ ] Refine model parameters or switch algorithms to boost accuracy.
- [ ] Confirm proper handling of imbalanced classes.

---

## Phase 5: Prediction and Output

### Tasks
1. **Prediction Function**
   - [ ] Develop a function to preprocess new descriptions and predict SKUs with the trained model.

2. **Output Generation**
   - [ ] Produce predictions for new bids and format them in JSON.

### Testing
- [ ] Test predictions on new bids with known SKUs.
- [ ] Verify JSON output is correctly formatted and complete.

---

## Phase 6: Testing and Validation

### Tasks
1. **End-to-End Testing**
   - [ ] Run the full system on a subset of historical data to validate the workflow.
   - [ ] Include edge cases like short or vague descriptions.

2. **Performance Testing**
   - [ ] Measure processing time for large datasets.
   - [ ] Optimize code if performance goals aren’t met.

### Testing
- [ ] Ensure accurate predictions across different scenarios.
- [ ] Confirm processing times meet acceptable standards.

---

## Phase 7: Deployment and Maintenance

### Tasks
1. **Deployment**
   - [ ] Deploy the system in the target environment (e.g., server or cloud).
   - [ ] Set up a process for periodic retraining with fresh data.

2. **Documentation**
   - [ ] Document the codebase, including steps to update preprocessing dictionaries and the equivalency table.
   - [ ] Include instructions for configuring the JSON file path.

3. **Monitoring**
   - [ ] Create a process to monitor prediction accuracy over time.

### Testing
- [ ] Test the system in the production environment.
- [ ] Validate retraining with a small dataset.
- [ ] Ensure documentation is clear and actionable.

---

## Final Checks

- [ ] Review the system against project requirements to confirm all goals are met.
- [ ] Perform a final test round with stakeholders.
- [ ] Compile a report on system performance and suggestions for future enhancements.

---

This updated TODO list ensures alignment with the PRD by incorporating tasks for implementing and testing the configurable JSON file path, maintaining flexibility and reliability in the system.