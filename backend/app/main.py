from fastapi import FastAPI
from api.department_routers import department_router
from api.chat_routes import chat_router
from api.employee_routers import employee_router
from api.document_routes import document_router
from api.system_routes import system_router
from fastapi.middleware.cors import CORSMiddleware
from db.database import create_db_and_tables


app = FastAPI()


@app.on_event("startup")
async def initialize_database() -> None:
    await create_db_and_tables()


# Cross-domain is allowed. For production environments, 
# please change * to a specific domain name
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(department_router)
app.include_router(chat_router)
app.include_router(employee_router)
app.include_router(document_router)
app.include_router(system_router)





