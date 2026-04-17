# MachineLearning
# RF-XGB-ArbolDecision_ARGEDUCATIVO.ipynb

🤖 Modelado de Machine Learning: Sistema de Alerta Temprana

Este proyecto implementa modelos de clasificación para identificar regiones-año en situación de Riesgo / No Riesgo educativo, basándose en la evolución de la promoción, repitencia, abandono y el acceso a internet.

Modelos Implementados: Se entrenaron y compararon algoritmos de Random Forest (RF), XGBoost (XGB) y Árboles de Decisión.

Estrategia de Evaluación: Se utilizó una metodología de Holdout 2022 (entrenamiento con datos 2019-2021 y testeo con 2022) y validación cruzada para garantizar la robustez geográfica y temporal.

Performance: En las pruebas de 2022, los modelos alcanzaron un F1-score de 0.50 para la clase de riesgo y una precisión general (accuracy) del 67%.

Variables de Entrada: La unidad de análisis integra 24 casos (6 regiones x 4 años) donde se cruzan las trayectorias escolares con el contexto del Ingreso Per Cápita Familiar de la EPH.

💡 Reflexión sobre el uso de ML en Sociología

La implementación de estos modelos demuestra el potencial de las estrategias computacionales para escalar como sistemas de alerta temprana. Esto permite a los organismos públicos focalizar intervenciones en aquellas regiones donde la probabilidad de fragmentación de las trayectorias escolares es mayor, pasando de un análisis reactivo a uno predictivo
