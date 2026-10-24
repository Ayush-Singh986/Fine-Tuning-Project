import unittest

import httpx

from finetune.api import create_app


class StubExtractor:
    def __init__(self, output: str):
        self.output = output

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.output


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, extractor, path, *, json_body=None):
        app = create_app(extractor)
        app.state.extractor = extractor
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.post(path, json=json_body)

    async def test_extract_returns_validated_result(self):
        extractor = StubExtractor(
            '{"vendor":"Northwind","invoice_number":"INV-1","date":null,"total":12.5,"currency":"USD"}'
        )
        response = await self.request(
            extractor,
            "/extract",
            json_body={"text": "Invoice 1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["invoice_number"], "INV-1")
        self.assertIn("Invoice 1", extractor.messages[-1]["content"])

    async def test_extract_reports_invalid_model_output(self):
        response = await self.request(
            StubExtractor("not-json"),
            "/extract",
            json_body={"text": "Invoice 1"},
        )
        self.assertEqual(response.status_code, 502)
        self.assertIn("invalid output", response.json()["detail"])

    async def test_request_length_is_bounded(self):
        response = await self.request(
            StubExtractor("{}"),
            "/extract",
            json_body={"text": "x" * 20_001},
        )
        self.assertEqual(response.status_code, 422)

    async def test_schema_is_applied_to_response(self):
        response = await self.request(
            StubExtractor('{"name":"Ada","age":37}'),
            "/extract",
            json_body={
                "text": "Extract the person",
                "schema": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {"name": {"type": "string"}},
                },
            },
        )
        self.assertEqual(response.status_code, 200)

    async def test_invalid_schema_is_a_client_error(self):
        response = await self.request(
            StubExtractor("{}"),
            "/extract",
            json_body={"text": "Extract fields", "schema": {"type": "not-a-type"}},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
