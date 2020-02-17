from flask import Flask
from flask_restplus import Api, Resource, fields
import random
import json
from random import randint

app = Flask(__name__)

api = Api(app)

dt = api.namespace('v1 eraser_datasets', description="Dataset analysis namespace")

ping = api.namespace('verifying namespace', description="Verify that the server is up")

dtv2 = api.namespace('v2 datasets', description='This is for the additional datasets for analyses')

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

with open('boolq/test.jsonl') as f:
    lines = f.readlines()


def generate_annotation_id():
    return randint(6000, 9000)


@ping.route('/ping')
class HelloWorld(Resource):
    '''Ping service to verify the service is up'''
    @ping.doc('which is the documentation')
    def get(self):
        return {
            "message": "Version 1 Dataset analysis service is running!"
        }, 200


@dt.route('/boolq')
class BoolQ(Resource):
    @dt.doc("Returns a random example from the boolq dataset")
    @dt.response(200, 'OK')
    @dt.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
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
                'message': 'class missing'
            }, 400
        if 'text' not in dt.payload:
            return {
                'success': False,
                'message': 'class missing'
            }, 400
        if 'annotation_id' not in dt.payload:
            ann_id = generate_annotation_id()

        return {
            'success': True,
            'message': 'Data appended'
        }, 200

    @dt.expect(data_model)
    @dt.response(200, 'OK')
    @dt.response(500, 'INTERNAL SERVER ERROR')
    @dt.response(400, 'BAD REQUEST')
    def put(self):
        if 'classification' not in dt.payload:
            return {
                       'success': False,
                       'message': 'class missing'
                   }, 400
        if 'query' not in dt.payload:
            return {
                       'success': False,
                       'message': 'class missing'
                   }, 400
        if 'text' not in dt.payload:
            return {
                       'success': False,
                       'message': 'class missing'
                   }, 400
        if 'annotation_id' not in dt.payload:
            return {
                       'success': False,
                       'message': 'class missing'
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
                       'message': 'class missing'
                   }, 400

        return {
                   'success': True,
                   'message': 'Data Deleted'
               }, 200


@dtv2.route('/ping')
class NewData(Resource):
    def get(self):
        return {
            "message": "Version 2 Dataset analysis service is running!"
        }, 200


if __name__ == "__main__":
    app.run(debug=True)
