import asyncio
import aiohttp
import aiofiles
import json
import os
from bs4 import BeautifulSoup
import time

DOWNLOAD_DIR = "datasets/orbit_xray"
ARTICLE_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
RETRY_LIMIT = 5
INITIAL_RETRY_DELAY = 2

async def fetch_cdn_image_url(session, article_url, figure_id, retries=0):
    """
    Fetches the main article page and parses it to find the direct
    CDN URL for the image within the specified figure ID, with retry logic.
    """
    print(f"-> Parsing page: {article_url} for figure {figure_id} (Attempt {retries + 1}/{RETRY_LIMIT})")
    try:
        async with session.get(article_url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
            if response.status == 429:
                if retries < RETRY_LIMIT:
                    delay = INITIAL_RETRY_DELAY * (2 ** retries) # Exponential backoff
                    print(f"   [!] Received 429 for {article_url}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    return await fetch_cdn_image_url(session, article_url, figure_id, retries + 1)
                else:
                    return None
            response.raise_for_status()

            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'lxml')
            selector = f'figure#{figure_id} img.graphic'
            img_tag = soup.select_one(selector)
            
            if img_tag and img_tag.has_attr('src'):
                return img_tag['src']
            else:
                return None
    except aiohttp.ClientResponseError as e:
        if e.status == 429 and retries < RETRY_LIMIT:
            delay = INITIAL_RETRY_DELAY * (2 ** retries)
            print(f"   [!] Received 429 for {article_url}. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
            return await fetch_cdn_image_url(session, article_url, figure_id, retries + 1)
        else:
            return None
    except aiohttp.ClientError as e:
        return None
    except Exception as e:
        return None

async def download_image_from_json(session, json_obj, current_DOWNLOAD_DIR, retries=0):
    pmcid = json_obj.get("pmcid")
    figure_id = json_obj.get("image", {}).get("id")

    if not pmcid or not figure_id:
        return

    article_url = f"{ARTICLE_BASE_URL}{pmcid}/"
    image_url = await fetch_cdn_image_url(session, article_url, figure_id)

    if not image_url:
        return


    print(f"-> Downloading image for {pmcid} {figure_id} (Attempt {retries + 1}/{RETRY_LIMIT})...")
    try:
        async with session.get(image_url) as response:
            if response.status == 429:
                if retries < RETRY_LIMIT:
                    delay = INITIAL_RETRY_DELAY * (2 ** retries)
                    await asyncio.sleep(delay)
                    return await download_image_from_json(session, json_obj, current_DOWNLOAD_DIR, retries + 1)
                else:
                    return
            response.raise_for_status()

            file_extension = os.path.splitext(image_url)[1].split('?')[0]
            filename = f"{pmcid}_{figure_id}{file_extension}"
            filepath = os.path.join(current_DOWNLOAD_DIR, filename)
            
            async with aiofiles.open(filepath, mode='wb') as f:
                await f.write(await response.read())
            print(f"Saved: {filepath}")

    except aiohttp.ClientResponseError as e:
        if e.status == 429 and retries < RETRY_LIMIT:
            delay = INITIAL_RETRY_DELAY * (2 ** retries)
            print(f"   [!] Received 429 for image {image_url}. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
            return await download_image_from_json(session, json_obj, current_DOWNLOAD_DIR, retries + 1)
        else:
            print(f"Failed: {image_url}")
    except Exception as e:
        print(f"Error: {pmcid}_{figure_id}")

async def main(json_obj):
    json_obj_idx_range = str(json_obj.get('min'))+"-"+str(json_obj.get('max'))
    current_download_path = os.path.join(DOWNLOAD_DIR, json_obj_idx_range)

    os.makedirs(current_download_path, exist_ok=True)

    json_filepath = os.path.join(current_download_path, f"{json_obj_idx_range}.json")
    with open(json_filepath, "w") as f:
        json.dump(json_obj, f, indent=4)

    async with aiohttp.ClientSession() as session:
        tasks = [download_image_from_json(session, obj, current_download_path) for obj in json_obj['list']]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    import requests

    def get_img_json(query, m=1, n=10, vid=0):
        url = f"https://openi.nlm.nih.gov/api/search?m={m}&n={n}&query={query}&vid={vid}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"list": []}

    api_request_interval = 20
    total_images_to_process = 1471
    delay_between_api_calls = 5

    for i in range(0, total_images_to_process, api_request_interval):
        m = i + 1
        n = i + api_request_interval
        # TODO: change query here ...
        list_of_json_objects_batch = get_img_json("orbit xray", m=m, n=n, vid=0)

        if 'list' in list_of_json_objects_batch and list_of_json_objects_batch['list']:
            print(f"Processing {len(list_of_json_objects_batch['list'])} images ({m}-{n})")
            asyncio.run(main(list_of_json_objects_batch))

        # small delay to avoid rate limiting
        if i + api_request_interval < total_images_to_process:
            time.sleep(delay_between_api_calls)