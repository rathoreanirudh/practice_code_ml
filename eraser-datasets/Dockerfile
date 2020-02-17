FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv

RUN python3.6 -m pip install pip --upgrade

RUN apt-get update

RUN apt-get install -y \
    curl \
    git \
    pkg-config \
    python-dev \
    python-numpy \
    python-protobuf\
    software-properties-common \
    zip \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN pip3.6 install Flask==1.0.2 \
    Flask-RESTful==0.3.6 \
    Flask-CORS==3.0.6 \
    flask-restplus==0.11.0 \
    pymongo==3.7.2 \
    olefile

COPY . /root

RUN chmod -R 777 /root
CMD ["su","-","root","-c","/root/start_dataset_analysis.sh"]

EXPOSE 5000
