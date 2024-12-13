import subprocess


def run_dvc_command(command):
    """Runs a DVC command and prints its output or error."""
    try:
        result = subprocess.run(
            command,
            check=True,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}\n{e.stderr}")
        raise


def create_dvc_pipeline(env="dev"):
    """Creates a DVC pipeline based on the specified environment."""

    print(f"Running DVC pipeline for environment: {env}")

    # Step 1: FETCH RAW DATA
    run_dvc_command(
        f"dvc stage add --force -n fetch_data "
        f"-d src/data_injection/fetch_data.py "
        f"-o raw_data/{env}/incidencias.csv "
        f"-o raw_data/{env}/piezas.csv "
        f"-o raw_data/{env}/estados.csv "
        f"-o raw_data/{env}/incidencias_tipo.csv "
        f"-o raw_data/{env}/articulos.csv "
        f"python -m src.data_injection.fetch_data --env {env}"
    )

    # Step 2: TRANSLATE RAW DATA TO SPANISH
    run_dvc_command(
        f"dvc stage add --force -n translate_data "
        f"-d raw_data/{env}/incidencias.csv "
        f"-d raw_data/{env}/piezas.csv "
        f"-d raw_data/{env}/estados.csv "
        f"-d raw_data/{env}/incidencias_tipo.csv "
        f"-d src/preprocessing/translation.py "
        f"-o output_data/{env}/desc_problema_translated.csv "
        f"-o output_data/{env}/problema_translated.csv "
        f"-o output_data/{env}/descripcion_translated.csv "
        f"python -m src.preprocessing.translation --env {env} "
        f"--input-incidencias raw_data/{env}/incidencias.csv "
        f"--input-piezas raw_data/{env}/piezas.csv "
        f"--input-estados raw_data/{env}/estados.csv "
        f"--input-incidencias-tipo raw_data/{env}/incidencias_tipo.csv "
        f"--output-path output_data/{env}"
    )

    # Step 3: FIND BEST MATCH
    run_dvc_command(
        f"dvc stage add --force -n find_best_matches "
        f"-d raw_data/{env}/piezas.csv "
        f"-d raw_data/{env}/articulos.csv "
        f"-o output_data/{env}/fuzzy_matches_w_scores.csv "
        f"python -m src.preprocessing.find_best_matches --env {env} "
        f"--input-piezas raw_data/{env}/piezas.csv "
        f"--input-articulos raw_data/{env}/articulos.csv "
        f"--output-best-matches output_data/{env}/fuzzy_matches_w_scores.csv"
    )

    # Step 4: PREPROCESSING
    run_dvc_command(
        f"dvc stage add --force -n preprocessing "
        f"-d output_data/{env}/desc_problema_translated.csv "
        f"-d output_data/{env}/problema_translated.csv "
        f"-d output_data/{env}/descripcion_translated.csv "
        f"-d output_data/{env}/fuzzy_matches_w_scores.csv "
        f"-d raw_data/{env}/incidencias.csv "
        f"-d raw_data/{env}/piezas.csv "
        f"-d raw_data/{env}/estados.csv "
        f"-d raw_data/{env}/incidencias_tipo.csv "
        f"-d raw_data/{env}/articulos.csv "
        f"-d src/preprocessing/preprocessing.py "
        f"-o output_data/{env}/preprocessed_data.csv "
        f"python -m src.preprocessing.preprocessing --env {env} "
        f"--translation-data-folder output_data/{env} "
        f"--raw-data-folder raw_data/{env} "
        f"--input-best-matches output_data/{env}/fuzzy_matches_w_scores.csv "
        f"--input-articulos raw_data/{env}/articulos.csv "
        f"--output-path output_data/{env}/preprocessed_data.csv"
    )

    # Step 5: CREATING CORPUS
    run_dvc_command(
        f"dvc stage add --force -n generate_corpus "
        f"-d output_data/{env}/preprocessed_data.csv "
        f"-d src/preprocessing/generate_corpus.py "
        f"-o output_data/{env}/corpus.csv "
        f"python -m src.preprocessing.generate_corpus --env {env} "
        f"--input-preprocessed-data output_data/{env}/preprocessed_data.csv "
        f'--input-documentation-path "\\\\central4\\Publica\\Product_technical_documentation-Documentación_técnica_producto" '
        f'--input-training-data-path "C:/Users/voliveira/OneDrive - Corporacion Empresarial Altra SL/00-Proyectos/Datos - Myzone/TrainningData" '
        f"--output-corpus output_data/{env}/corpus.csv"
    )

    # Step 6: TF-IDF encoding
    run_dvc_command(
        f"dvc stage add --force -n tfidf_encoding "
        f"-d output_data/{env}/preprocessed_data.csv "
        f"-d src/encoding/tfidf_encoding.py "
        f"-o output_data/{env}/tfidf_encoded_data.csv "
        f"python -m src.encoding.tfidf_encoding --env {env} "
        f"--input-data output_data/{env}/preprocessed_data.csv "
        f"--output-tfidf-encoded-data output_data/{env}/tfidf_encoded_data.csv "
        f"--output-tfidf-model encode_models/{env}/tfidf_model.joblib"
    )

    # Step 7: Doc2Vec encoding
    run_dvc_command(
        f"dvc stage add --force -n doc2vec_encoding "
        f"-d output_data/{env}/corpus.csv "
        f"-d src/encoding/doc2vec_encoding.py "
        f"-o output_data/{env}/doc2vec_encoded_data.csv "
        f"python -m src.encoding.doc2vec_encoding --env {env} "
        f"--input-data output_data/{env}/preprocessed_data.csv "
        f"--input-corpus output_data/{env}/corpus.csv "
        f"--output-doc2vec-encoded-data output_data/{env}/doc2vec_encoded_data.csv "
        f"--output-doc2vec-model encode_models/{env}/doc2vec_model.joblib"
    )

    # Step 8: SentenceTransformer encoding
    run_dvc_command(
        f"dvc stage add --force -n sentence_transformer_encoding "
        f"-d output_data/{env}/preprocessed_data.csv "
        f"-d src/encoding/sentence_transformer_encoding.py "
        f"-o output_data/{env}/sentence_transformer_encoded_data.csv "
        f"python -m src.encoding.sentence_transformer_encoding --env {env} "
        f"--input-data output_data/{env}/preprocessed_data.csv "
        f"--output-sentence-transformer-encoded-data output_data/{env}/sentence_transformer_encoded_data.csv"
    )

    # Step 9: Generate unsupervised dataset
    run_dvc_command(
        f"dvc stage add --force -n generate_unsupervised_dataset "
        f"-d output_data/{env}/tfidf_encoded_data.csv "
        f"-d output_data/{env}/doc2vec_encoded_data.csv "
        f"-d output_data/{env}/sentence_transformer_encoded_data.csv "
        f"-d output_data/{env}/preprocessed_data.csv "
        f"-d encode_models/{env}/tfidf_model.joblib "
        f"-d encode_models/{env}/doc2vec_model.joblib "
        f"-d src/encoding/generate_unsupervised_dataset.py "
        f"-d raw_data/{env}/TablaTipoErrorPostventa.csv "
        f"-o output_data/{env}/unsupervised_tfidf_dataset.csv "
        f"-o output_data/{env}/unsupervised_doc2vec_dataset.csv "
        f"-o output_data/{env}/unsupervised_sentence_transformer_dataset.csv "
        f"python -m src.encoding.generate_unsupervised_dataset --env {env} "
        f"--input-tfidf-encoded-data output_data/{env}/tfidf_encoded_data.csv "
        f"--input-doc2vec-encoded-data output_data/{env}/doc2vec_encoded_data.csv "
        f"--input-sentence-transformer-encoded-data output_data/{env}/sentence_transformer_encoded_data.csv "
        f"--input-preprocessed-data output_data/{env}/preprocessed_data.csv "
        f"--output-unsupervised-tfidf-dataset output_data/{env}/unsupervised_tfidf_dataset.csv "
        f"--output-unsupervised-doc2vec-dataset output_data/{env}/unsupervised_doc2vec_dataset.csv "
        f"--output-unsupervised-sentence-transformer-dataset output_data/{env}/unsupervised_sentence_transformer_dataset.csv "
        f"--input-tfidf-model encode_models/{env}/tfidf_model.joblib "
        f"--input-doc2vec-model encode_models/{env}/doc2vec_model.joblib "
        f"--input-tipo-error raw_data/{env}/TablaTipoErrorPostventa.csv"
    )

    # Step 10: Generate supervised dataset
    run_dvc_command(
        f"dvc stage add --force -n generate_supervised_dataset "
        f"-d output_data/{env}/unsupervised_sentence_transformer_dataset.csv "
        f"-d raw_data/{env}/reviewed_dataset.parquet "
        f"-d src/encoding/generate_supervised_dataset.py "
        f"-o output_data/{env}/supervised_dataset.parquet "
        f"python -m src.encoding.generate_supervised_dataset --env {env} "
        f"--input-reviewed-data raw_data/{env}/reviewed_dataset.parquet "
        f"--output-supervised-dataset output_data/{env}/supervised_dataset.parquet"
    )

    # Step 11: Train model
    run_dvc_command(
        f"dvc stage add --force -n training "
        f"-d output_data/{env}/supervised_dataset.parquet "
        f"-d src/training/training.py "
        f"-o output_models/{env}/model.joblib "
        f"python -m src.training.training --env {env} "
        f"--input-dataset output_data/{env}/supervised_dataset.parquet "
        f"--output-model output_models/{env}/model.joblib"
    )

    # Step 12: Train conformal prediction model
    run_dvc_command(f"dvc stage add --force -n conformal_prediction ")

    # Step 13: Evaluate model


if __name__ == "__main__":
    # Set the environment (e.g., 'dev' or 'prod')
    environment = input("Enter the environment (dev/prod): ").strip() or "dev"
    create_dvc_pipeline(environment)
