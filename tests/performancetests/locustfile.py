from locust import HttpUser, task, between
import os

class LoadTestAPI(HttpUser):
    wait_time = between(1, 5)
    host ="http://127.0.0.1:8000/"  


    @task
    def test_label_endpoint(self):
        # Path to a test image
        test_image_path = r"data\raw\fruit_vegetable\test\apple\Image_1.jpg"  # Replace with a valid image file path
        
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
