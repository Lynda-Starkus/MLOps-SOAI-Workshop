# **MLOps workshop with MLflow|Streamlit|FastAPI|python**

![](banner.png)

## **Outline**
- What's MLOps and why need CD/CI ops in AI projects ?
- Presenting MLflow for ML models lifecycle management
- Proof-of-concept of the robustness of MLflow+FastAPI+Steamlit deployement stack
- End-to-end implementation and Demo tutorial
- Application on managing a simple torch Linear model for digit recognition

## **How to launch the scripts**
1. cd to Code folder
2. In a DB Browser open sqlite database 'test.db'
3. Execute the following commands in seperate CLIs
    - "streamlit run frontend/streamlit_dashboard.py"
    - "uvicorn backend.main:app"
    - "mlflow ui --backend-store-uri sqlite:///test.db"

## **Proposed dev stack**
<img src="images/Stack.png" height="400" />
