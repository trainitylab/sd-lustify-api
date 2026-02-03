import runpod
import requests
import time
import json

API_URL = "http://127.0.0.1:7860"

def wait_for_service(timeout=360):
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{API_URL}/sdapi/v1/sd-models", timeout=5)
            if response.status_code == 200:
                print("✅ SD WebUI est prêt !")
                return True
        except Exception as e:
            print(f"⏳ Waiting... {e}")
        time.sleep(3)
    print("❌ SD WebUI timeout après 360s")
    return False

def handler(job):
    job_input = job['input']
    if not wait_for_service():
        return {"error": "SD WebUI non disponible après 360s"}

    payload = {
        "prompt": job_input.get("prompt", ""),
        "negative_prompt": job_input.get("negative_prompt", "worst quality, low quality, blurry, bad anatomy"),
        "steps": job_input.get("steps", 45),
        "cfg_scale": job_input.get("cfg_scale", 7.0),
        "width": job_input.get("width", 832),
        "height": job_input.get("height", 1216),
        "sampler_name": "DPM++ SDE Karras",
        "seed": job_input.get("seed", -1),
        "enable_hr": True,
        "hr_scale": 2,
        "hr_upscaler": "R-ESRGAN 4x+",
        "hr_second_pass_steps": 25,
        "denoising_strength": 0.30,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    True,  # enabled1
                    {  # conf1
                        "ad_model": "mediapipe_face_mesh_eyes_only",
                        "ad_confidence": 0.3,
                        "ad_denoising_strength": 0.4,
                        "ad_inpaint_only_masked": True,
                        "ad_mask_blur": 4,
                        "ad_inpaint_padding": 32
                    },
                    True,  # enabled2
                    {  # conf2
                        "ad_model": "yolov8x-worldv2.pt",
                        "ad_confidence": 0.3,
                        "ad_denoising_strength": 0.4,
                        "ad_inpaint_only_masked": True,
                        "ad_mask_blur": 4,
                        "ad_inpaint_padding": 32
                    }
                ]
            }
        }
    }
    
    if job_input.get("face_image"):
        payload["alwayson_scripts"]["ControlNet"] = {
            "args": [{
                "enabled": True,
                "module": "ip-adapter-faceid-plusv2_sdxl",
                "model": "ip-adapter-faceid_sdxl",
                "weight": 1.4,
                "resize_mode": "Crop and Resize",
                "processor_res": 512,
                "control_mode": 0,
                "pixel_perfect": True,
                "guidance_start": 0,
                "guidance_end": 1,
                "image": job_input["face_image"]
            }]
        }

    try:
        print("🚀 Lancement txt2img...")
        response = requests.post(f"{API_URL}/sdapi/v1/txt2img", json=payload, timeout=900)
        if response.status_code != 200:
            return {"error": f"Erreur API {response.status_code}: {response.text[:200]}"}
        result = response.json()
        print("✅ Génération réussie")
        return {
            "image": f"data:image/png;base64,{result['images'][0]}",
            "seed": result.get("seed", -1),
            "info": result.get("info", "")
        }
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

runpod.serverless.start({"handler": handler})
