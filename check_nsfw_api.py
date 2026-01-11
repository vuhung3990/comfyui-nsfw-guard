import requests
import json

def get_latest_prompt_id(comfy_url="http://127.0.0.1:8188"):
    """Get the ID of the most recent prompt execution."""
    try:
        response = requests.get(f"{comfy_url}/history")
        history = response.json()
        if not history:
            return None
        return list(history.keys())[-1]
    except Exception:
        return None

def check_nsfw_status(prompt_id, comfy_url="http://127.0.0.1:8188"):
    """
    Checks if a ComfyUI prompt failed due to NSFW content.
    
    Args:
        prompt_id (str): The ID of the prompt to check.
        comfy_url (str): The base URL of the ComfyUI server.
        
    Returns:
        dict: {
            "is_nsfw": bool,
            "status": str ("completed", "error", "running", "not_found"),
            "details": dict (if nsfw detected, contains 'confidence', 'threshold', 'prediction')
        }
    """
    try:
        response = requests.get(f"{comfy_url}/history/{prompt_id}")
        history = response.json()
    except Exception as e:
        return {"is_nsfw": False, "status": "connection_error", "error": str(e)}

    if prompt_id not in history:
        return {"is_nsfw": False, "status": "not_found"}

    run_data = history[prompt_id]
    
    # Check for errors FIRST (before checking outputs)
    if "status" in run_data:
        status_info = run_data["status"]
        status_str = status_info.get("status_str", "")
        
        if status_str == "error":
            messages = status_info.get("messages", [])
            
            for msg in messages:
                # msg is typically [event_type, {details}]
                if len(msg) >= 2 and isinstance(msg[1], dict):
                    exception_msg = msg[1].get("exception_message", "")
                    
                    # Try to parse our custom JSON error
                    try:
                        error_data = json.loads(exception_msg)
                        if error_data.get("error", {}).get("type") == "nsfw_content_detected":
                            return {
                                "is_nsfw": True,
                                "status": "error",
                                "details": error_data["error"]["details"]
                            }
                    except json.JSONDecodeError:
                        # Fallback: check for text string
                        if "nsfw_content_detected" in exception_msg:
                            return {
                                "is_nsfw": True,
                                "status": "error",
                                "details": {"confidence": 0.0, "threshold": 0.0, "prediction": "unknown"}
                            }
            
            # Other error (not NSFW)
            return {"is_nsfw": False, "status": "error", "message": "Other error occurred"}
    
    # Check if completed successfully (has non-empty outputs)
    if "outputs" in run_data and run_data["outputs"]:
        return {"is_nsfw": False, "status": "completed"}
    
    # Still running or unknown state
    return {"is_nsfw": False, "status": "running"}

# --- Example Usage ---
if __name__ == "__main__":
    print("Checking latest prompt status...")
    latest_id = get_latest_prompt_id()
    
    if not latest_id:
        print("No history found.")
    else:
        print(f"Latest Prompt ID: {latest_id}")
        result = check_nsfw_status(latest_id)
        
        if result["is_nsfw"]:
            print(f"⚠️ NSFW DETECTED!")
            d = result['details']
            print(f"Confidence: {d['confidence']:.2%}")
            print(f"Threshold:  {d['threshold']:.2%}")
        elif result["status"] == "completed":
            print("✅ Image generated successfully (Safe)")
        elif result["status"] == "error":
            print(f"❌ Error: {result.get('message', 'Unknown error')}")
        else:
            print(f"Status: {result['status']}")
