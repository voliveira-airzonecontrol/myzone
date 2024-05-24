import pandas as pd

# Define the families to remove
SPECIAL_FAMILIES = ["REEMPLAZOS", "PACKS"]
MEANLESS_FAMILIES = [
    "DOCUMENTACION",
    "EXPOSITORES",
    "TOBERAS",
    "MP_CONSUMIBLES",
    "OBS_CENTRALIZADO",
    "MP_ELECTRONICA",
    "MP_TERMINADOS/SEMI",
    "MP_MECANICOS",
    "OBS_REJILLA MOTORIZADA",
    "OBS_COMUNES",
    "OBS_DISTRIBUIDO",
    "DOCUMENTACION",
    "OBS_DIFUSOR SIN REGULACIÓN",
    "EXPOSITORES",
    "OBS_OTROS",
    "MP_SERVICIOS",
    "MP_I+D",
    "OBS_REJILLA SIN REGULACIÓN",
    "OBS_PANTALLAS GRAFICAS",
    "OBS_IB PRO USA",
    "Soporte técnico",
    "PUESTA EN MARCHA",
    "MP_DIFUSION",
    "OBS_ALARMAS TÉCNICAS",
    "MP_CHAPAS",
    "MP_CABLE",
    "OBS_CABLE",
    "MP_MOTORES",
    "MERCHANDISING",
    "OBS_ZONEPAD",
    "OBS_COMUNICACIONES",
    "OBS_DIFUSOR MOTORIZADO",
    "OBS_CAJA DE MEZCLA",
    "OBS_ANTREE",
]

class Dataset:

    def __init__(
            self,
            incidencias: pd.DataFrame,
            articulos: pd.DataFrame
    ):
        self.data = None
        self.incidencias = incidencias
        self.articulos = articulos


    def generate_dataset(self, threshold: float = 85):

        self.data = self.incidencias.merge(
            self.articulos,
            left_on="CODART_A3",
            right_on="CODART",
            how="left"
        )

        # Generate the text to analyse
        self.data['text_to_analyse'] = self.data[[
            "desc_problema_translated",
            "descripcion_translated",
            "problema_translated"
        ]].apply(lambda x: " ".join(x), axis=1)

        # Remove the meanless families
        self.data = self.data[~self.data["DESCCAR3"].isin(MEANLESS_FAMILIES)]
        # Remove the special families
        self.data = self.data[~self.data["DESCCAR3"].isin(SPECIAL_FAMILIES)]

        # Clean low similariy scores
        self.data = self.data[
            self.data["Fuzzy_Score"] >= threshold
        ]

        return self
