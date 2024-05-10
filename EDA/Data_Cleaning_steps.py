"""

    Data Cleaning Steps:
    1. Translation Pipeline
        1.1 Load sav_incidencias, sav_piezas, sav_estados, sav_incidencias_tipo
        1.2 Merge all the dataframes
        1.3 Filter out incidencias_tipo different from 1 and sav_estados different from 2 and 6
        1.4 Get Unique Strings to translate
        1.5 Detect Language (Replace with Error if not translatable and Too short if length < 5)
        1.6 Filter out text already in spanish, Errors and Too short strings
        1.7 Translate the strings
        1.8 Save the translated strings in a csv file **Optional**

    2. Data Cleaning Pipeline
        2.1 Load the translated strings from disk
        2.2 Load the sav_incidencias, sav_piezas, sav_estados, sav_incidencias_tipo from DB again
        2.3 Merge the translated strings with the original dataframe
        2.4 Drop the columns that are not needed
        2.5 Fill NA with empty strings
        2.6 Create a new column "text_to_analyze" with the text to analyze (desc_problema_translated, descripcion_translated, problema_translated, articulo)
        2.7 Load Articulos and Caracteristicas from A3ERP DB
        2.8 Merging the characteristics with the articles
        2.9 Get best match for each codigo articulo with fuzzy matching
        2.10 Filter out those that could not be matched (vaule 0 or NaN)
        2.11 Filter out those texts with less than 25 characters
        2.12 Filter out those that has nothing much more than "NO FUNCIONA" in it
        2.13 Merge the text_to_analyse with articulos

    3. Data Preprocessing Pipeline
        3.1 Create model to predict the type of incidencia
        3.2 Feed the model with the text_to_analyze for vocabulary construction
        3.3 Train the model

"""