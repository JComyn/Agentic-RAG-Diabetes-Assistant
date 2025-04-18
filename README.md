# Agentic-RAG-Diabetes-Assistant

## Descripción del Proyecto
Este repositorio contiene el desarrollo del TFG titulado **"Sistema Multiagente basado en RAG para Asistencia en el Manejo de Diabetes"**. El objetivo del proyecto es diseñar e implementar un sistema multiagente de inteligencia artificial que utilice Retrieval-Augmented Generation (RAG) para asistir a pacientes con diabetes.

## Tecnologías Utilizadas
- **Lenguaje de programación**: 
- **Frameworks y librerías**: 
- **Modelos de lenguaje**: 
- **Interfaz de usuario**: 

## Estructura del Repositorio
```
TODO
```



Ground Truth Fuente (traducido al español):https://www.jmir.org/2024/1/e58041

## Instalación y Configuración
TODO 
virtual env

instalar requirements.txt


Configura el Entorno:
Crea un archivo .env en el mismo directorio del script con tu clave de API y el endpoint base si es necesario:
GITHUB_TOKEN ="tu_github_token"
GITHUB_MODELS_ENDPOINT="endpoint_base" # Opcional, si usas un endpoint diferente al de OpenAI
LLM_MODEL_NAME="modelo" # Cambia a otro modelo si lo deseas

TAVILY_API_KEY="tu_tavily_api_key" # Cambia a otro modelo si lo deseas

Prepara los Documentos:
Crea una carpeta llamada diabetes_docs en el mismo directorio. (no está en el repositorio por motivos de privacidad)
Coloca dentro todos los archivos PDF mencionados en tu TFG (guías clínicas, libros, etc.).



## Uso del Sistema
ejecitar main dos veces:
1. Primero para crear el índice de los documentos y guardarlo en un archivo pickle.
2. Luego, ejecuta el script para iniciar el asistente de diabetes.

## Contacto
Autor: **Javier Comyn Rodríguez**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/javier-comyn-rodriguez)

---
Este proyecto se desarrolla como TFG del Grado en Matemáticas e Informática en la ETSIINF de la UPM.

