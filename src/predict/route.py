
import io
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd

from booking_model.prediction_model import PredictionModel


router = APIRouter(prefix="/trip")

@router.post('/predict', methods=['POST'])
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith('.csv.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Read the CSV file
    df = pd.read_csv(file.file)

    prediction = PredictionModel().predict_contries(test_data=df)

    output = io.StringIO()
    prediction.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment;filename=modified.csv"})