# Transformers para resolver el problema TSP

Utilizamos una arquitectura Transformers de ML para resolver el problema de optimizacion TSP (*Travelling Salesman Problem*)

## 📋 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:
* **Python 3.8 o superior**
* **Git** (para clonar el repositorio y versionado)

## ⚙️ Instalación y Configuración

Sigue estos pasos para configurar tu entorno local y ejecutar el proyecto:

### 1. Clonar el repositorio
Puede ser en la terminal de VS Studio o en tu Power Shell en la carpeta que contendrá tu proyecto:

```bash
git clone https://github.com/felipe-astudillo-s/TransformerTSP.git
```
Despues ingresa a tu proyecto desde la terminal:

```
cd tu-repositorio
```

### 2. Crear un entorno virtual
En tu terminal corre el siguiente comando para crear tu entorno virtual dentro de tu carpeta de proyecto:

```
python -m venv venv
```

### 3. Activar el entorno virtual
Para activar tu entorno en Windows usa:

```
.\venv\Scripts\Activate
```

### 4. Instalar dependencias
Ya activado el entorno, instala todas las librerias necesarias especificadas en el requirements.txt:

```
pip install -r requirements.txt
```

### (Opcional) Desinstalar dependencias
Si deseas desactivar tu entorno virtual una vez terminado de trabajar, en la terminal de tu editor o Power Shell:

```
deactivate
```

### Como conectarlo con tu CUDA?

Al inicializar con tu Kernel es posible que no se conecte la GPU. Para ello hay que desintalar y volver a instalar una version de PyTorch:

```
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```