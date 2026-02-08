from dotenv import load_dotenv
import boto3
import json
import uuid

def call_travel_agent(session_id: str, user_prompt: str):
    # Retrieve AWS user credentials from .env file
    load_dotenv()

    # Connect to AgentCore and create payload
    client = boto3.client('bedrock-agentcore', region_name='us-east-1')
    payload = json.dumps({"prompt": user_prompt, "session_id": session_id})

    # Call AgentCore endpoint
    response = client.invoke_agent_runtime(
        agentRuntimeArn='arn:aws:bedrock-agentcore:us-east-1:973030239480:runtime/TravelAgent-OWS4r078E8',
        runtimeSessionId=session_id,
        payload=payload,
        contentType="application/json"
    )

    response_body = response['response'].read()
    response_data = json.loads(response_body)
    return response_data.get("output", "No response from agent.")



if __name__ == "__main__":
    session_id = str(uuid.uuid4()) #"57076c76-ad4c-4124-8a80-f4c151366844"
    user_prompt = "Hello!"
    agent_response = call_travel_agent(session_id, user_prompt)
    print("Agent Response:", agent_response)