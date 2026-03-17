import json
import uuid
import boto3
import os
import traceback

client = boto3.client("bedrock-agentcore", region_name="us-east-1")
AGENT_RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:973030239480:runtime/TravelAgent-OWS4r078E8"
EXPECTED_API_KEY = os.environ.get("AGENT_API_KEY")

def lambda_handler(event, context):
    try:
        # --- Read API key ---
        headers = event.get("headers", {})
        incoming_key = headers.get("x-api-key")

        if incoming_key != EXPECTED_API_KEY:
            return {
                "statusCode": 403,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({"error": "Forbidden"})
            }

        # --- Parse body ---
        try:
            body = json.loads(event.get("body", "{}"))
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Invalid JSON body"})
            }
        user_prompt = body.get("prompt")
        session_id = body.get("session_id", str(uuid.uuid4()))
        itinerary_id = body.get("itinerary_id")

        if not user_prompt:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({"error": "Missing prompt"})
            }

        print(f"[lambda_handler] session_id={session_id}, prompt={user_prompt[:100]}")

        # --- Call AgentCore ---
        payload = json.dumps({
            "prompt": user_prompt,
            "session_id": session_id,
            "itinerary_id": itinerary_id
        })

        print(f"[lambda_handler] payload={payload}")

        response = client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=session_id,
            payload=payload,
            contentType="application/json"
        )

        response_body = response["response"].read()
        response_data = json.loads(response_body)

        # --- Return result ---
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "output": response_data.get("output", "No response"),
                "session_id": session_id
            })
        }

    except Exception as e:
        print(f"[lambda_handler] Error: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "Internal server error"})
        }