FROM registry.access.redhat.com/ubi9/python-312:1-20.1722518948
ENV PORT 8080
EXPOSE 8080
WORKDIR /usr/src/app

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"] 
CMD ["app.py"]
