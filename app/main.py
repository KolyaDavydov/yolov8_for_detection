import json
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, Response, status, HTTPException, Form, UploadFile, Depends, File
from io import BytesIO
import symbol_predict as sp


app = FastAPI()

@app.post("/posts", status_code=200)
async def post_request(file: UploadFile = File()):

    # изображение в FastApi приходит в закодированном формате - декодируем его
    img_bytes = BytesIO(file.file.read()).read()
    img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Получаем результаты предсказания в виде изображения с нанесеной разметкой и словарь с информацией
    result, data = sp.predict(img)

    headers = {'data': json.dumps(data)}

    # кодируем изображение перед выведением в FastApi
    res, im_png = cv2.imencode(".png", result)

    return Response(content=im_png.tobytes(), status_code=200, media_type='image/png', headers=headers)


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8058, reload=True)
