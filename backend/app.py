# backend/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routes import generate_chart, analyze_trends
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Chart Builder")

# CORS configuration (adjust as needed for your frontend)
origins = [
    "http://localhost",
    "http://localhost:3000",  # React default port
    "*",
]  # Be cautious with wildcard in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Handles all unhandled exceptions."""
    error_id = logging.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal Server Error",
            "detail": f"An unexpected error occurred. Error ID: {error_id}",
        },
    )


@app.post("/generate-chart/")
async def create_chart(
    query: str = Form(...),
    chart_type: str = Form(default=None),
    file: UploadFile = File(default=None),
    json_data: str = Form(default=None),
    manual_data: str = Form(default=None),
):
    """Generates a chart based on user input."""
    try:
        content = None
        data_type = None
        filename = None

        if file:
            filename = file.filename
            if not file.filename.endswith((".csv", ".json")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only CSV and JSON files are supported",
                )
            content = await file.read()
            data_type = "file"

        elif json_data:
            content = json_data.encode("utf-8")
            data_type = "json"

        elif manual_data:
            content = manual_data.encode("utf-8")
            data_type = "manual"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No data provided"
            )

        response = generate_chart(query, content, chart_type, data_type, filename)
        return response

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions for FastAPI to handle
        raise http_exc
    except ValueError as value_exc:
        # Catch ValueErrors raised within the function
        logger.warning(f"Validation Error: {str(value_exc)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(value_exc)
        )
    except Exception as e:
        # Catch any other exceptions
        logger.error(f"Unexpected Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e # chain exceptions


@app.post("/analyze-trends/")
async def get_trends(query: str = Form(...), file: UploadFile = File(default=None)):
    """Analyzes trends in a CSV file."""
    try:
        if file and file.filename.endswith(".csv"):
            content = await file.read()
            response = analyze_trends(query, content)
            return response
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV file required for trend analysis",
            )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e # chain exceptions


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)