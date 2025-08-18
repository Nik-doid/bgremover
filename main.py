from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from remove_bg import process_remove_background

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

                <label>Transparent Background?</label>
                <input type="checkbox" name="transparent"><br><br>

                <button type="submit">Upload & Process</button>
            </form>
        </body>
    </html>
    """


@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    transparent: str = Form(None) 
):
    transparent_flag = transparent is not None
    return await process_remove_background(file, transparent_flag, background_tasks)
