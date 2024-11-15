import subprocess


def run_dvc_command(command):
    """Runs a DVC command and prints its output or error."""
    try:
        result = subprocess.run(command, check=True, shell=True, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
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
        f"--input-documentation-path \"\\\\central4\\Publica\\Product_technical_documentation-Documentación_técnica_producto\" "
        f"--input-training-data-path \"C:/Users/voliveira/OneDrive - Corporacion Empresarial Altra SL/00-Proyectos/Datos - Myzone/TrainningData\" "
        f"--output-corpus output_data/{env}/corpus.csv"
    )


if __name__ == "__main__":
    # Set the environment (e.g., 'dev' or 'prod')
    environment = input("Enter the environment (dev/prod): ").strip() or "dev"
    create_dvc_pipeline(environment)
