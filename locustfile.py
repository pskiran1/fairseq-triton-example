from locust import HttpUser, task, events
import json

request_count = 0
MAX_REQUESTS = 1
wrong_outputs = 0

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    global wrong_outputs
    print(f"\n**************** Total wrong outputs: {wrong_outputs} ****************")

class TritonTranslationUser(HttpUser):
    @task
    def translate(self):
        global request_count
        global wrong_outputs

        body = {
            "inputs": [
                {
                    "name": "src_tokens",
                    "shape": [1, 4],
                    "datatype": "INT64",
                    "data": [[134, 16, 65, 2]]
                },
                {
                    "name": "src_lengths",
                    "shape": [1, 1],
                    "datatype": "INT64",
                    "data": [[4]]
                }
            ]
        }

        with self.client.post("/v2/models/bls/infer", json=body, catch_response=True) as response:
            if response.status_code != 200:
                if response.status_code == 500:
                    wrong_outputs += 1
                print(f"Status code {response.status_code} - request_count: {request_count} - {response.text}")
                response.failure(f"Status code {response.status_code}")
            else:
                try:
                    response_json = response.json()
                    expected_result = [134, 16, 65, 2]
                    if response_json["outputs"][0]["data"] != expected_result:
                        wrong_outputs += 1
                        print(f'**************** Output mismatched: {response_json["outputs"][0]["data"]} - wrong_outputs: {wrong_outputs} - request_count: {request_count} ****************')
                        response.failure(f'**************** Output mismatched ****************')
                except Exception as e:
                    print(e)
                    response.failure(e)
        
        request_count += 1

        # print("########### request_count:", request_count, "#### wrong_outputs:", wrong_outputs, "#########")
        # if request_count >= MAX_REQUESTS:
        #    self.environment.runner.quit()
        # self.stop()
