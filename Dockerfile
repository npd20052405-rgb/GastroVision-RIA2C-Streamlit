FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY requirements.txt /app/requirements.txt
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118 && pip install -r /app/requirements.txt
COPY streamlit_app.py ria2c_app_manifest.json /app/
COPY .streamlit /app/.streamlit
RUN chown -R appuser:appuser /app
USER appuser
ENV RIA2C_MODEL_DIR=/models MPLCONFIGDIR=/tmp/matplotlib HOME=/home/appuser
EXPOSE 7860
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860"]
