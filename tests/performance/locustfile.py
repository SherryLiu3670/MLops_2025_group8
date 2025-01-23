from locust import HttpUser, task, between
import os

# specify the path to the test image
test_image_path = "data/raw/fruit_vegetable/test/apple/Image_1.jpg"

class LoadTestAPI(HttpUser):
    '''Load test the API'''
    wait_time = between(1, 5)

    @task
    def test_label_endpoint(self):
                
        # Ensure the test image exists
        if not os.path.exists(test_image_path):
            print(f"Test image not found: {test_image_path}")
            return
        
        # Open the test image and send it as part of the request
        with open(test_image_path, "rb") as image_file:
            files = {"data": ("test_image.jpg", image_file, "image/jpeg")}
            response = self.client.post("/label/", files=files)
            
        # Check the response status code and output
        if response.status_code == 200:
            print(f"Prediction: {response.json()}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
