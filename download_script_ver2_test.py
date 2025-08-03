import asyncio
import aiohttp
import aiofiles
import json
import os
import time

DOWNLOAD_DIR = "datasets/leg_xray_png"            
BASE_IMG_URL = "https://openi.nlm.nih.gov"  
RETRY_LIMIT = 5
INITIAL_RETRY_DELAY = 2 

async def download_image_from_json(session, json_obj, current_DOWNLOAD_DIR, retries=0):
    """
    Downloads the image using the 'imgLarge' URL from the Open-i API JSON metadata.
    """
    pmcid = json_obj.get("pmcid")
    figure_id = json_obj.get("image", {}).get("id")
    img_path = json_obj.get("imgLarge")

    if not pmcid or not figure_id or not img_path:
        return

    image_url = BASE_IMG_URL + img_path

    print(f"-> Downloading image for {pmcid} {figure_id} (Attempt {retries + 1}/{RETRY_LIMIT})...")

    try:
        async with session.get(image_url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
            if response.status == 429:
                if retries < RETRY_LIMIT:
                    delay = INITIAL_RETRY_DELAY * (2 ** retries)
                    print(f"   [!] 429 rate limit. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    return await download_image_from_json(session, json_obj, current_DOWNLOAD_DIR, retries + 1)
                else:
                    print(f"   [!] Failed to download {image_url} after retries.")
                    return

            response.raise_for_status()

            # Save file
            filename = f"{pmcid}_{figure_id}.png"
            filepath = os.path.join(current_DOWNLOAD_DIR, filename)

            async with aiofiles.open(filepath, mode='wb') as f:
                await f.write(await response.read())

            print(f"   [✔] Saved: {filepath}")

    except Exception as e:
        print(f"   [!] Error downloading {image_url}: {e}")


async def main(json_obj):
    json_obj_idx_range = str(json_obj.get('min')) + "-" + str(json_obj.get('max'))
    current_download_path = os.path.join(DOWNLOAD_DIR, json_obj_idx_range)

    os.makedirs(current_download_path, exist_ok=True)

    # Save metadata JSON
    json_filepath = os.path.join(current_download_path, f"{json_obj_idx_range}.json")
    with open(json_filepath, "w") as f:
        json.dump(json_obj, f, indent=4)

    async with aiohttp.ClientSession() as session:
        tasks = [download_image_from_json(session, obj, current_download_path) for obj in json_obj['list']]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    import requests

    def get_img_json(query, m=1, n=10, vid=0):
        #url = f"https://openi.nlm.nih.gov/api/search?m={m}&n={n}&query={query}&vid={vid}&it=xg"
        url = f"https://openi.nlm.nih.gov/api/search?m={m}&n={n}&query={query}&vid={vid}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {"list": []}

    api_request_interval = 20   
    total_images_to_process = 10333  # adjust as needed
    delay_between_api_calls = 5

    for i in range(0, total_images_to_process, api_request_interval):
        m = i + 1
        n = i + api_request_interval    

        # change your search query here
        list_of_json_objects_batch = get_img_json("leg xray", m=m, n=n, vid=0)

        if 'list' in list_of_json_objects_batch and list_of_json_objects_batch['list']:
            print(f"\nProcessing {len(list_of_json_objects_batch['list'])} images ({m}-{n})")
            asyncio.run(main(list_of_json_objects_batch))

        if i + api_request_interval < total_images_to_process:
            time.sleep(delay_between_api_calls)

