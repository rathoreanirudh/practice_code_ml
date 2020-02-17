from flask import Flask
from flask_restplus import Api, Resource, fields, reqparse
import random
import json
from random import randint
import logging.handlers

# an api is a application programming interface which helps one piece of software to talk to another
# REST APIs are representational state transfer APIs which use the HTTP protocol
# REST breaks down a transaction (which is a sequence of information exchange) to create a series of small modules
# they use GET to retrieve the state of a resource
# PUT to update the state of a resource
# POST to create a resource
# DELETE to delete a resource


# Create an object of the flask framework which helps to write easy to use and simple to setup web apis

app = Flask(__name__)

# wrap this around flask restplus framework which helps to provide automatic documentation with swagger UI
# which helps UI developers and other developers who have to use your apis
# as well as yourself when you have to use them at a later date

api = Api(app)

api.namespaces.clear()

# the main idea is to split your app into reusable namespaces
# namespaces are like modules which have various properties which could be potentially reused

dt = api.namespace('v1', description="Dataset analysis namespace")

ping = api.namespace('', description="Verify that the server is up")

dtv2 = api.namespace('v2', description='This is for the additional datasets for analyses')

# we create a model which helps in response marshalling as well as streamlining the input
# to set a particular structure of the input
# so the api can expect a particular format of the data and that has to be mentioned somewhere
# describing the model helps to do so

data_model = dt.model('boolq_data', {
    "annotation_id": fields.Integer,
    "classification": fields.Boolean,
    "evidence": fields.String,
    "query": fields.String,
    "text": fields.String
})

delete_model = dt.model('boolq_data_delete', {
    "annotation_id": fields.Integer
})

# create a parser for url parsing for arguments in the uris

data_parser = reqparse.RequestParser()
data_parser.add_argument('annotation_id', type=int, help='Instance to be queried')

# create log file with debug level to save all the errors and url requests and responses for debugging
# handler for backing up the log file to store the previous log files

LOGFILE = 'logs/eraser.log'
logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(
    LOGFILE, maxBytes=(1048576 * 5), backupCount=7
)
logFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(logFormatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())


with open('boolq/test.jsonl') as f:
    lines = f.readlines()


def generate_annotation_id():
    return randint(6000, 9000)


# in the ping namespace create a get method for verifying if the version 1 server is running or not
# this is to make sure if a particular version is running or not or if it has been deprecated or hasn't been released
# with full functionalities

# namespace documentation is done by the namespace description

# response descriptions with particular namespaces as namespace dot response

# also return the response code with the return statement because a lot of times
# developers just look at the response codes

# for every class we define the methods for that particular uri
# namespace dot route

@ping.route('/ping')
class HelloWorld(Resource):
    """Ping service to verify whether the service is up"""
    @ping.doc('which is the documentation')
    @ping.response(200, 'OK')
    @ping.response(500, 'INTERNAL SERVER ERROR')
    @ping.response(404, 'RESOURCE NOT FOUND')
    def get(self):
        return {
            "message": "Version 1 Dataset analysis service is running!"
        }, 200


# add the expected parser with namespace dot model name which adds the documentation
# note the logger dot debug statement which helps to look at the extracted annotation id from the parser
# get the parser using arg parser method which extracts it as a dictionary

# if there is no specific annotation id in the request we take a random value and return that


@dt.route('/boolq')
class BoolQ(Resource):
    @dt.doc("Returns a random example from the boolq dataset")
    @dt.response(200, 'OK')
    @dt.response(500, 'INTERNAL SERVER ERROR')
    @dt.expect(data_parser)
    def get(self):
        args = data_parser.parse_args()
        logger.debug(args)
        if args['annotation_id'] is not None:
            return json.loads(lines[int(args['annotation_id'])-6363]), 200
        else:
            return json.loads(random.choice(lines)), 200

    @dt.expect(data_model)
    @dt.response(200, 'OK')
    @dt.response(500, 'INTERNAL SERVER ERROR')
    @dt.response(400, 'BAD REQUEST')
    def post(self):
        if 'classification' not in dt.payload:
            return {
                'success': False,
                'message': 'class missing'
            }, 400
        if 'query' not in dt.payload:
            return {
                'success': False,
                'message': 'query missing'
            }, 400
        if 'text' not in dt.payload:
            return {
                'success': False,
                'message': 'text missing'
            }, 400
        if 'annotation_id' not in dt.payload:                  # if ann id is not in the payload we generate it
            ann_id = generate_annotation_id()

        return {
            'success': True,
            'message': 'Data appended'
        }, 200

    @dt.expect(data_model)
    @dt.response(200, 'OK')
    @dt.response(500, 'INTERNAL SERVER ERROR')
    @dt.response(400, 'BAD REQUEST')
    def put(self):                                          # we expect only the annotation id in the payload
        if 'annotation_id' not in dt.payload:               # whatever we get get apart from this we update in the db
            return {
                       'success': False,
                       'message': 'annotation id missing'
                   }, 400

        return {
                   'success': True,
                   'message': 'Data Updated'
               }, 200

    @dt.expect(delete_model)
    @dt.response(200, 'OK')
    @dt.response(500, 'INTERNAL SERVER ERROR')
    @dt.response(400, 'BAD REQUEST')
    def delete(self):
        if 'annotation_id' not in dt.payload:
            return {
                       'success': False,
                       'message': 'annotation id missing'            # delete method just needs the annotation id
                   }, 400

        return {
                   'success': True,
                   'message': 'Data Deleted'
               }, 200


@dtv2.route('/ping')
class NewData(Resource):
    def get(self):
        return {
            "message": "Version 2 Dataset analysis service is running!"     # version 2 of the dataset analysis
        }, 200


if __name__ == "__main__":
    app.run(debug=True, port='5000', host='0.0.0.0')          # remove debug = True in prod environment


# Let's talk a little about pytests framework which is used for unit testing and the best practice is to make your
# code as modular as possible and to write test case for every part of the code

# all the files with *_test.py and test_*.py get executes when you run pytests in a directory
# writing a test case for a method is very easy
# you just have to assert the output that you expect and try this for a few test cases
# what are fixtures -- fixtures are functions which run before each test case and provide input to the test function
# mocking -- mocks the output of a particular function


# containerization of python apps
# best for simplifying application deployment
# isolated virtual environment
# for example docker is an open source containerization
# docker image is an immutable file that contains source code, libraries, dependency and code
# containers are light-weight compared to vms because container provides an abstract os for a particular app to run
# images are read only and hence they are called snapshots
# container is a stateful instantiation of an image


