from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse
from remove_bg import process_remove_background
import uvicorn

app = FastAPI()



@app.get("/", response_class=HTMLResponse)
async def form_page():
    return """
    <html>
        <head><title>Background Removal</title></head>
        <body style="font-family: sans-serif;">
            <h2>Upload an image to remove background</h2>
            <form action="/remove-bg" method="post" enctype="multipart/form-data">
                <label>Select Image:</label>
                <input type="file" name="file" accept="image/*" required><br><br>
                <button type="submit">Upload & Process</button>
            </form>
        </body>
    </html>
    """


@app.post("/remove-bg")
async def remove_bg(file: UploadFile, background_tasks: BackgroundTasks):
    return await process_remove_background(file, False, background_tasks)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,   # Start with 1 worker for file uploads
        log_level="info"
    )
