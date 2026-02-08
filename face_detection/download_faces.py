import os
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from bing_image_downloader import downloader
from tqdm.asyncio import tqdm as async_tqdm
from functools import partial
import time

# --- CONFIG ---
OUTPUT_DIR = "face_dataset"
IMAGES_PER_ENTITY = 20
ENTITY_JSON = "datasets/processed/all_data_unique_entities.json"
MAX_CONCURRENT_DOWNLOADS = 10  # Parallel downloads
TIMEOUT = 15  # Reduced timeout for faster failure detection
MAX_RETRIES = 3  # Retry failed downloads
RETRY_DELAY = 2  # Seconds to wait between retries


def load_entities(json_path):
    """Load entity list from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def create_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def download_images_for_entity(entity, output_dir, count=5):
    """
    Download images per entity using Bing Image Downloader.
    Skips download if entity folder already exists with images.
    Returns tuple: (entity, success, status, folder_path)
    """
    safe_entity_name = entity.replace(" ", "_").replace("/", "_")
    dst_dir = os.path.join(output_dir, safe_entity_name)
    
    # Check if entity folder already exists with images
    if os.path.exists(dst_dir):
        existing_images = [f for f in os.listdir(dst_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        if len(existing_images) > 0:
            return (entity, True, f"SKIPPED ({len(existing_images)} images exist)", dst_dir)
    
    # Download if folder doesn't exist or is empty
    try:
        downloader.download(
            query=entity,
            limit=count,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=TIMEOUT,
            verbose=False  # Suppress verbose output
        )

        # Handle folder renaming
        src_dir = os.path.join(output_dir, entity)

        if os.path.exists(dst_dir):
            return (entity, True, "DOWNLOADED", dst_dir)

        if os.path.exists(src_dir):
            os.rename(src_dir, dst_dir)
            return (entity, True, "DOWNLOADED", dst_dir)
        
        return (entity, False, "FAILED (no images found)", None)
    
    except Exception as e:
        return (entity, False, f"FAILED ({str(e)[:30]})", None)


async def download_entities_parallel(entities, output_dir, count):
    """
    Download images for multiple entities in parallel using ThreadPoolExecutor.
    """
    loop = asyncio.get_event_loop()
    
    # Create partial function with fixed arguments
    download_func = partial(download_images_for_entity, 
                           output_dir=output_dir, 
                           count=count)
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        # Create tasks for all entities
        tasks = [
            loop.run_in_executor(executor, download_func, entity)
            for entity in entities
        ]
        
        # Run with progress bar
        results = []
        for coro in async_tqdm(asyncio.as_completed(tasks), 
                              total=len(entities),
                              desc="Downloading"):
            result = await coro
            results.append(result)
    
    return results


def print_summary(results):
    """Print download summary statistics."""
    successful = sum(1 for _, success, _, _ in results if success)
    failed = sum(1 for _, success, _, _ in results if not success)
    downloaded = sum(1 for _, _, status, _ in results if status == "DOWNLOADED")
    skipped = sum(1 for _, _, status, _ in results if "SKIPPED" in status)
    
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"  Total entities: {len(results)}")
    print(f"  ✓ Successful: {successful}")
    print(f"    - Downloaded: {downloaded}")
    print(f"    - Skipped (already exist): {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"{'='*60}")
    
    if failed > 0:
        print("\nFailed entities:")
        for entity, success, status, _ in results:
            if not success:
                print(f"  - {entity}: {status}")


async def main_async():
    """Async main function."""
    # Load entity names
    entities = load_entities(ENTITY_JSON)
    
    print(f"Downloading images for {len(entities)} entities...")
    print(f"Parallel workers: {MAX_CONCURRENT_DOWNLOADS}")
    print(f"Images per entity: {IMAGES_PER_ENTITY}\n")
    
    # Ensure main dataset directory exists
    create_dir(OUTPUT_DIR)
    
    # Download in parallel
    results = await download_entities_parallel(
        entities, 
        OUTPUT_DIR, 
        IMAGES_PER_ENTITY
    )
    
    # Print summary
    print_summary(results)
    print(f"\nFace dataset saved in: {OUTPUT_DIR}")


def main():
    """Entry point."""
    # Run async main
    asyncio.run(main_async())


if __name__ == "__main__":
    main()