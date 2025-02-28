from locust import HttpUser, task
from locust.exception import LocustError

request_count = 0
MAX_REQUESTS = 1
wrong_outputs = 0

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

        try:
            response = self.client.post("/v2/models/bls/infer", json=body)

            # Check HTTP status first
            if response.status_code != 200:
                raise LocustError(
                    f"Request failed: {response.status_code} - {response.text}"
                )

            # Attempt JSON parsing
            try:
                response_json = response.json()
                expected_result = [134, 16, 65, 2]
                if response_json["outputs"][0]["data"] != expected_result:
                    wrong_outputs += 1
                    print(
                        "****************",
                        wrong_outputs,
                        "--",
                        "Output mismatched: ",
                        response_json["outputs"][0]["data"],
                        " -",
                        request_count,
                    )
                else:
                    pass

            except json.JSONDecodeError:
                raise LocustError(f"Invalid JSON response: {response.text[:200]}")

        except Exception as e:
            print(f"Error during request: {str(e)}")

        request_count += 1
        # print("########### request_count:", request_count, "#### wrong_outputs:", wrong_outputs, "#########")
        # if request_count >= MAX_REQUESTS:
        #    self.environment.runner.quit()
        # self.stop()
