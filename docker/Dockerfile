# Use the official Python image from the Docker Hub
FROM cnstark/pytorch:1.9.1-py3.9.12-cuda11.1.1-ubuntu20.04

# Set environment variables to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

COPY ./requirements_v2.txt .
COPY ./setup.sh /app/setup.sh
RUN pip install -r requirements_v2.txt

RUN pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install rational-activations==0.2.1
RUN pip install networkx==2.8.5

# For interactive use, you might want to start a bash shell
CMD ["/app/docker/setup.sh"]