"""
This code is to download all the images onto your computer with use of selenium module for the dataset.
"""
image_limit = 200 # Specify how many images to download


# Importing necessary libraries
from selenium import webdriver # Selenium allows automating web browsers
from selenium.webdriver.common.by import By # For location elements on a webpage
from selenium.webdriver.support.ui import WebDriverWait # For waiting for elements to load
from selenium.webdriver.support import expected_conditions as EC # specific conditions such as when an element is full loaded
                                                                 # Helps to avoid trying to download when element is not ready
import bs4 # BeautifulSoup, for parsing through html of page
import time # for adding delays in script
            # Used to wait for the page to load
import requests # for downloading images
import os # used for operating system actions and creating folders.


# Start Chrome and wait/run when page loads 
driver = webdriver.Chrome()
driver.implicitly_wait(10)

# list of search topics
search_strings = ["thanos", "iron man", "captain america", "spiderman", "thor", "doctor strange", "scarlet witch", "star-lord", "gamora", "loki"]


for search in search_strings:
    i = 0 # used in naming convention

    # create a folder in the absolute path with the search variable name (any spaces get '_')
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_directory = os.path.join(script_directory, search.replace(' ', '_'))
    os.makedirs(image_directory, exist_ok=True)
    
    # open up google images website
    driver.get("https://images.google.com/")

    # find search box and search for term
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(search + " endgame")
    search_box.submit()

    # Wait for all images to load - error if takes longer than 10 sec
    print(f"Waiting for images to load for: {search}")
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, "//img")))
    except Exception as e:
        print(f"Error waiting for images: {e}")


    # scroll up to the top so we can collect all images starting from the top
    driver.execute_script("window.scrollTo(0, 0);")

    # get pages html and parse with beautiful soup
    page_html = driver.page_source
    pageSoup = bs4.BeautifulSoup(page_html, 'html.parser')

    # find all images containers 
        # div containers with specified class have the image
    containers = pageSoup.findAll('div', {'class': 'eA0Zlc WghbWd FnEtTd mkpRId m3LIae RLdvSe qyKxnc ivg-i PZPZlf GMCzAd'})
    len_containers = len(containers)
    print(f"Found {len_containers} image containers")

    # loop through the containers
    while i < image_limit:  # limit to specified image count
        for ind in range(len(containers)):
            if i >= image_limit:  # stop when image limit is achieved
                break

            # every 25th img div is a list of relevant searchs, not actual image so skip
            if ind % 25 == 0: continue

            try:
                """
                Google works by giving you a small version of each image. 
                To avoid downloading the small images, to get the image url. 
                This image url can be the larger image url or small image url
                so we wait 10 seconds for the image url to change to the 
                bigger image.
                """

                # simulate click on the image container to open the larger image
                xPath = f"""//*[@id="rso"]/div/div/div[1]/div/div/div[{ind}]""" # path of actual image
                driver.find_element(By.XPATH, xPath).click()

                # wait for the image to load & get image url (can be smaller image url)
                origElem = driver.find_element(By.XPATH, """//*[@id="Sva75c"]/div[2]/div[2]/div/div[2]/c-wiz/div/div[3]/div[1]/a/img[1]""")
                orig_url = origElem.get_attribute('src')
                image_url = orig_url

                # timer to see if url changes to bigger image url,
                # if url doesnt change then open whatever one currently saved
                timeStarted = time.time()
                while orig_url == image_url: # continuously check url
                    downloadElem = driver.find_element(By.XPATH, """//*[@id="Sva75c"]/div[2]/div[2]/div/div[2]/c-wiz/div/div[3]/div[1]/a/img[1]""")
                    image_url = downloadElem.get_attribute('src')

                    if time.time() - timeStarted > 10: # wait 10 seconds
                        print("Timeout! Will download a lower resolution image.")
                        break

                # try to download image
                try:
                    if not image_url:  # if no image url, skip
                        print(f"Skipping image {i} due to missing image URL.")
                        continue
                    
                    # download image with request library and rename with image index
                    img_data = requests.get(image_url).content
                    image_filename = os.path.join(image_directory, f"{search.replace(' ', '_')}_{i}.jpg")

                    # save image to local folder
                    with open(image_filename, 'wb') as f:
                        f.write(img_data) # write image in binary
                    print(f"Image {i} saved as {image_filename}") # Print image name
                    i += 1
                
                # if error, print reason and continue to next image
                except Exception as e:
                    print(f"Error downloading image {i}: {e}")
                    continue

            # if no image found, print and continue
            except Exception as e:
                print(f"Error: Could not find or click on image container {ind} for {search}: {e}")
                continue
        
        # google loads x number of  images per page
        # if div containers reached, but image_limit not 
        # scroll down to load more images and continue
        if i < image_limit:
            driver.execute_script("window.scrollBy(0, 1000);")  # scroll down
            time.sleep(2)  # wait for images to load
            page_html = driver.page_source # get new html
            pageSoup = bs4.BeautifulSoup(page_html, 'html.parser') # parse new page
            # get now image div containers
            containers = pageSoup.findAll('div', {'class': 'eA0Zlc WghbWd FnEtTd mkpRId m3LIae RLdvSe qyKxnc ivg-i PZPZlf GMCzAd'})
            print(f"Found {len(containers)} image containers after scrolling.")

    # images downladed for current search term
    print(f"Downloaded {i} images for {search}")

# close browser when done
driver.quit()
