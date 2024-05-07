<img src="https://res.cloudinary.com/airzone/image/upload/v1707308010/images/airzone.svg" />

# MyZone Project

The project MyZone is an initiative aimed at leveraging data science to reduce non-quality issues, minimize costs, and decrease the volume of customer returns due to product defects within the warranty period. Our primary objective is to utilize advanced analytics to enhance product quality and improve customer satisfaction.

## Short-Term Goals

In the short term, our team is focused on classifying and clustering devolution orders by type of defect and product family. This involves translating all textual fields in the after-sale data to Spanish and performing comprehensive NLP (Natural Language Processing) analyses. These analyses include pre-processing, tokenization, vectorization, and clustering, among other techniques.

## Long-Term Vision

The long-term vision of Project DevQuality is to establish a robust system that predicts potential defects and intercepts them before they reach the customer, thus reducing the incidence of devolution and enhancing the overall product experience.

## Getting Started

To get started with Project MyZone and understand the analysis made by the team, follow these steps to set up the project on your local machine for development and testing purposes.

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
3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Set up the environment variables:**
   * SQL_PASSWORD
   * MYSQL_PASSWORD
   * ORACLE_PASSWORD
   
## Structure of the Repository

The repository is structured as follows:

1. **Notebooks:** Contains the Jupyter notebooks with the analysis and visualizations.
2. **Data:** Contains the data that was extracted and pre-processed by the codes.