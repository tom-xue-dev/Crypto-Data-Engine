from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def start_driver(url):
    """
    初始化浏览器。访问网页
    :param url: 目标网址
    :return: none
    """
    driver = webdriver.Chrome()
    print("start driver")
    driver.get(url)
    print("url get")
    while True:
        ask,bid = get_orderbook(driver,"ask-light","bid-light")
        for index, element in enumerate(ask):
            print(f"卖{index + 1}:",element.text)
        for index, element in enumerate(bid):
            print(f"买{index + 1}:", element.text)
        time.sleep(1)
        print()
    driver.quit()
def get_orderbook(driver, ask_div_name, bid_div_name):
    """
    使用selenium获取实时订单簿数据，读取前端数据
    :param url: 目标网址
    :param ask_div_name:买盘的div class 名称
    :param bid_div_name: 卖盘div class名称
    :return: 整个订单簿
    """
    # 初始化网络浏览器

    try:
        ask_elements = driver.find_elements(By.CLASS_NAME, 'pk-item-left.green')
        bid_elements = driver.find_elements(By.CLASS_NAME, 'pk-item-center')
        return ask_elements, bid_elements
    except:
        print("Stopped by user.")
        return None,None
start_driver("https://www.coinw.com/zh_TW/futures/usdt/btcusdt")