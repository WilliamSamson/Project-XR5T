from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import json
import time


def extract_chat_data(url):
    # Set up Chrome WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in background
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(url)
    time.sleep(5)  # Wait for page to load

    messages = []
    chat_elements = driver.find_elements(By.CLASS_NAME, 'prose')

    for chat in chat_elements:
        messages.append(chat.text)

    driver.quit()

    return {"conversation": messages}


share_url = "https://chatgpt.com/share/67c2e9fa-c198-8009-b7f3-adbdf4af9a73"
chat_data = extract_chat_data(share_url)

with open("chat_data.json", "w") as f:
    json.dump(chat_data, f, indent=4)

print("Chat data saved as chat_data.json")
