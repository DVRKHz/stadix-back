from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import ChiSquareRequest, ChiSquareResponse
from app.services.nonparametric import calculate_chi_square
import pandas as pd
import io
import numpy as np
from scipy.stats import chi2_contingency

router = APIRouter()

@router.post("/chi-square", response_model=ChiSquareResponse)
async def perform_chi_square(payload: ChiSquareRequest):
    result = calculate_chi_square(payload.observed_data)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # 1. Leer el archivo
        if file.filename.endswith(".csv"):
            # Probamos leer sin header primero para capturar todo
            df = pd.read_csv(io.BytesIO(contents), header=None)
        elif file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(contents), header=None, engine="openpyxl")
        else:
            raise HTTPException(status_code=400, detail="Formato no soportado. Usa CSV o Excel.")

        # 2. LIMPIEZA DE DATOS (Crucial)
        # Intentamos convertir todo a números. Lo que sea texto se vuelve NaN.
        df_clean = df.apply(pd.to_numeric, errors='coerce')
        
        # Eliminamos filas y columnas que sean puramente NaN (eran etiquetas o headers)
        df_clean = df_clean.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Rellenamos cualquier NaN restante con 0 (opcional, depende de tu lógica, o puedes lanzar error)
        df_clean = df_clean.fillna(0)

        matrix = df_clean.values.tolist()

        # 3. Validaciones finales
        if len(matrix) < 2 or len(matrix[0]) < 2:
            raise HTTPException(status_code=400, detail="La tabla resultante es muy pequeña. Asegúrate de incluir solo datos numéricos o que la limpieza no haya borrado todo.")

        # Verificar que no queden números negativos
        if (df_clean < 0).any().any():
             raise HTTPException(status_code=400, detail="La prueba Chi-Cuadrada no acepta números negativos.")

        # 4. Cálculo
        chi2, p, dof, expected = chi2_contingency(matrix)

        return {
            "statistic": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "expected_frequencies": expected.tolist(),
            "is_significant": bool(p < 0.05),
            "interpretation": "Dependencia significativa" if p < 0.05 else "Independencia"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error interno: {e}") # Imprimir en consola del servidor para debug
        raise HTTPException(status_code=500, detail=f"Error al procesar archivo: {str(e)}")