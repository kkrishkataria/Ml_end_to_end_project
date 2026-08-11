FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt 
# for installing purposes above line
CMD [ "python3","app.py" ]
# for Start the application purpose
