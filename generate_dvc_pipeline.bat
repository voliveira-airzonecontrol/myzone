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