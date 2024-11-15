@echo off
setlocal
set ENV=%1
if "%ENV%"=="" set ENV=dev

echo Running DVC pipeline for environment: %ENV%

REM Step 1: FETCH RAW DATA
dvc stage add --force -n fetch_data ^
    -d src/data_injection/fetch_data.py ^
    -o raw_data/%ENV%/incidencias.csv ^
    -o raw_data/%ENV%/piezas.csv ^
    -o raw_data/%ENV%/estados.csv ^
    -o raw_data/%ENV%/incidencias_tipo.csv ^
    -o raw_data/%ENV%/articulos.csv ^
    python -m src.data_injection.fetch_data --env %ENV%

REM Step 2: TRANSLATE RAW DATA TO SPANISH
dvc stage add --force -n translate_data ^
    -d raw_data/%ENV%/incidencias.csv ^
    -d raw_data/%ENV%/piezas.csv ^
    -d raw_data/%ENV%/estados.csv ^
    -d raw_data/%ENV%/incidencias_tipo.csv ^
    -d src/preprocessing/translation.py ^
    -o output_data/%ENV%/desc_problema_translated.csv ^
    -o output_data/%ENV%/problema_translated.csv ^
    -o output_data/%ENV%/descripcion_translated.csv ^
    python -m src.preprocessing.translation --env %ENV% ^
        --input-incidencias raw_data/%ENV%/incidencias.csv ^
        --input-piezas raw_data/%ENV%/piezas.csv ^
        --input-estados raw_data/%ENV%/estados.csv ^
        --input-incidencias-tipo raw_data/%ENV%/incidencias_tipo.csv ^
        --output-path output_data/%ENV%

REM Step 3: FIND BEST MATCH
dvc stage add --force -n find_best_matches ^
    -d raw_data/%ENV%/piezas.csv ^
    -d raw_data/%ENV%/articulos.csv ^
    -o output_data/%ENV%/fuzzy_matches_w_scores.csv ^
    python -m src.preprocessing.find_best_matches --env %ENV% ^
        --input-piezas raw_data/%ENV%/piezas.csv ^
        --input-articulos raw_data/%ENV%/articulos.csv ^
        --output-best-matches output_data/%ENV%/fuzzy_matches_w_scores.csv


REM Step 4: PREPROCESSING
dvc stage add --force -n preprocessing ^
    -d output_data/%ENV%/desc_problema_translated.csv ^
    -d output_data/%ENV%/problema_translated.csv ^
    -d output_data/%ENV%/descripcion_translated.csv ^
    -d output_data/%ENV%/fuzzy_matches_w_scores.csv ^
    -d raw_data/%ENV%/incidencias.csv ^
    -d raw_data/%ENV%/piezas.csv ^
    -d raw_data/%ENV%/estados.csv ^
    -d raw_data/%ENV%/incidencias_tipo.csv ^
    -d raw_data/%ENV%/articulos.csv ^
    -d src/preprocessing/preprocessing.py ^
    -o output_data/%ENV%/preprocessed_data.csv ^
    python -m src.preprocessing.preprocessing --env %ENV% ^
        --translation-data-folder output_data/%ENV% ^
        --raw-data-folder raw_data/%ENV% ^
        --input-best-matches output_data/%ENV%/fuzzy_matches_w_scores.csv ^
        --input-articulos raw_data/%ENV%/articulos.csv ^
        --output-path output_data/%ENV%/preprocessed_data.csv


REM Step 5: CREATING CORPUS
dvc stage add --force -n generate_corpus ^
    -d output_data/%ENV%/preprocessed_data.csv ^
    -d src/preprocessing/generate_corpus.py ^
    -o output_data/%ENV%/corpus.csv ^
    python -m src.preprocessing.generate_corpus --env %ENV% ^
        --input-preprocessed-data output_data/%ENV%/preprocessed_data.csv ^
        --input-documentation-path "\\central4\Publica\Product_technical_documentation-Documentación_técnica_producto" ^
        --input-training-data-path "C:/Users/voliveira/OneDrive - Corporacion Empresarial Altra SL/00-Proyectos/Datos - Myzone/TrainningData" ^
        --output-corpus output_data/%ENV%/corpus.csv


REM Step 6: TF-IDF encoding
dvc stage add --force -n tfidf_encoding ^
    -d output_data/%ENV%/corpus.csv ^
    -d src/preprocessing/tfidf_encoding.py ^
    -o output_data/%ENV%/tfidf_encoded_data.csv ^
    python -m src.preprocessing.tfidf_encoding --env %ENV% ^
        --input-corpus output_data/%ENV%/corpus.csv ^
        --output-tfidf-encoded-data output_data/%ENV%/tfidf_encoded_data.csv

REM Step 7: Doc2Vec encoding
dvc stage add --force -n doc2vec_encoding ^
    -d output_data/%ENV%/corpus.csv ^
    -d src/preprocessing/doc2vec_encoding.py ^
    -o output_data/%ENV%/doc2vec_encoded_data.csv ^
    python -m src.preprocessing.doc2vec_encoding --env %ENV% ^
        --input-corpus output_data/%ENV%/corpus.csv ^
        --output-doc2vec-encoded-data output_data/%ENV%/doc2vec_encoded_data.csv

REM Step 8: SentenceTransformer encoding
dvc stage add --force -n sentence_transformer_encoding ^
    -d output_data/%ENV%/corpus.csv ^
    -d src/preprocessing/sentence_transformer_encoding.py ^
    -o output_data/%ENV%/sentence_transformer_encoded_data.csv ^
    python -m src.preprocessing.sentence_transformer_encoding --env %ENV% ^
        --input-corpus output_data/%ENV%/corpus.csv ^
        --output-sentence-transformer-encoded-data output_data/%ENV%/sentence_transformer_encoded_data.csv