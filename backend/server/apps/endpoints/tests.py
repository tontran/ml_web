from django.test import TestCase

from django.test import TestCase
from rest_framework.test import APIClient

class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
        input_data = {
            "age": 36,
            "workclass": "Private",
            "fnlwgt": 123011,
            "education": "HS-grad",
            "education-num": 9.0,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0.0,
            "capital-loss": 0.0,
            "hours-per-week": 40.0,
            "native-country": "United-States"
        }
        classifier_url = "/api/v1/income_classifier/predict"
        response = client.post(classifier_url, input_data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["label"], "<=50K")
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)