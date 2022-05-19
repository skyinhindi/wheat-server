import uvicorn
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
from Prediction import predict
from fastapi.middleware.cors import CORSMiddleware

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    print(file)
    image = read_imagefile(await file.read())
    #print(image)
    prediction = predict(image)
    return prediction

#uvicorn server:app --reload
if __name__ == "__main__":
    uvicorn.run(app, debug=True)