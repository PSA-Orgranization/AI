# main.py
from fastapi import FastAPI, Form
from app.gpt_service import generate_response_with_context

app = FastAPI()

@app.post("/generate")
def generate_response(
    prompt: str = Form(...),
    session_id: int = Form(None),
    username: str = Form("default_user"),
    email: str = Form("default@example.com")
):
    response = generate_response_with_context(
        user_input=prompt,
        session_id=session_id,
        username=username,
        email=email
    )
    return {"response": response}
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)