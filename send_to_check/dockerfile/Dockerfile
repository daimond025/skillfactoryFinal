FROM python:3.8
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt
WORKDIR /workdir
COPY runner.py /workdir/runner.py
CMD ["python", "runner.py"]
