
<img src="https://res.cloudinary.com/airzone/image/upload/v1707308010/images/airzone.svg" />

# MyZone Project

The MyZone Project leverages advanced data science techniques to address non-quality issues, reduce costs, and minimize customer returns due to product defects within the warranty period. By combining cutting-edge Natural Language Processing (NLP) and machine learning methodologies, this project aims to improve product quality and enhance customer satisfaction in a scalable and adaptable manner.

## Objectives

### Short-Term Goals

The immediate focus of the MyZone Project is on classifying and clustering after-sales complaints based on defect types and product families. This involves:
- Translating all multilingual textual fields into Spanish for uniformity.
- Applying state-of-the-art NLP techniques, including:
  - Preprocessing (cleaning, normalization, and tokenization).
  - Generating embeddings using transformer-based models like BERT.
  - Clustering textual data to identify recurring patterns and novel defect types.

### Long-Term Vision

The long-term goal of the project is to implement an end-to-end predictive system that:
- Detects potential product defects before they impact customers.
- Dynamically adapts to new types of defects using open-world learning techniques.
- Automates the analysis of after-sales data to streamline decision-making processes.
By achieving these goals, the project seeks to enhance the overall customer experience and drive operational excellence.

## Key Features

- **NLP Pipeline:** Processes multilingual textual data using advanced transformer-based models, ensuring semantic-rich embeddings for downstream tasks.
- **Dynamic Clustering:** Identifies both known and novel defect categories, leveraging unsupervised learning techniques and domain expertise.
- **Incremental Learning:** Incorporates new defect types into the system without retraining from scratch, using conformal prediction and other open-world learning methods.
- **Predictive Analytics:** Develops a scalable system to predict product quality issues and reduce customer complaints proactively.

## Getting Started

To set up the MyZone Project locally for development and testing, follow these steps:

### Installation

1. **Clone the repository:**
   ```bash
   git clone http://gitlab2.airzonesl.es:30080/data-science/myzone
   cd MyZone
   ```
   
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
   
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
4. **Set up environment variables:** 
   Ensure the following environment variables are properly configured:
   - `SQL_PASSWORD`
   - `MYSQL_PASSWORD`
   - `ORACLE_PASSWORD`

### Additional Setup (Optional)
- Ensure access to the `MyZone` database for extracting after-sales complaint data.
- Configure external APIs (e.g., translation services) for preprocessing if multilingual data is present.

## Repository Structure

The repository is organized as follows:

1. **`notebooks/`:** 
   - Contains Jupyter notebooks for exploratory data analysis (EDA), model training, and clustering visualizations.
   
2. **`src/`:**
   - Core scripts for preprocessing, embedding generation, clustering, and classification tasks.
   
3. **`data/`:**
   - Processed datasets used for training, validation, and testing.

4. **`models/`:**
   - Saved transformer models, clustering configurations, and evaluation metrics.

5. **`reports/`:**
   - Generated reports, including clustering visualizations, classification metrics, and error analyses.

## Contribution Guidelines

We welcome contributions to improve the MyZone Project. To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes with clear messages.
4. Push to your fork and submit a pull request.

## Contact

For questions or support, reach out to:
- **Project Lead:** Vitor Oliveira de Souza
- **Email:** vitor.odesouza@gmail.com
- **Advisor:** Javier Del Ser, PhD, Chief Artificial Intelligence Scientist, TECNALIA

---

This project represents a step forward in combining NLP, machine learning, and real-world business needs to improve operational efficiency and customer satisfaction. Join us in shaping the future of data-driven after-sales analysis!
