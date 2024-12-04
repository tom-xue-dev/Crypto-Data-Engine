import time
from selenium import webdriver
from selenium.webdriver.common.by import By

# 启动 WebDriver
driver = webdriver.Chrome()

# 打开目标网页
driver.get("https://www.coinw.com/zh_TW/futures/usdt/btcusdt")

# 获取所有潜在的文本节点（常见标签）
elements = driver.find_elements(By.CSS_SELECTOR, "div, span, p, li, h1, h2, h3, h4, h5, h6")

# 记录初始值和类名
initial_values = {}
for index, element in enumerate(elements):
    initial_values[index] = {
        "value": element.text.strip(),
        "class": element.get_attribute("class").strip() if element.get_attribute("class") else "NoClass"
    }

# 定期检查变化
try:
    while True:
        print("Checking for changes...")
        for index, element in enumerate(elements):
            try:
                # 获取当前文本值和类名
                new_value = element.text.strip()
                new_class = element.get_attribute("class").strip() if element.get_attribute("class") else "NoClass"

                # 对比文本内容是否变化
                if initial_values[index]["value"] != new_value:
                    print(f"Element {index}: Value changed from '{initial_values[index]['value']}' to '{new_value}'")
                    print(f"Class: {initial_values[index]['class']}")
                    # 更新记录
                    initial_values[index]["value"] = new_value

                # 对比类名是否变化
                if initial_values[index]["class"] != new_class:
                    print(f"Element {index}: Class changed from '{initial_values[index]['class']}' to '{new_class}'")
                    # 更新记录
                    initial_values[index]["class"] = new_class
            except Exception as e:
                # 如果元素已不存在，跳过检查
                print(f"Element {index} is no longer available.")
                continue
        print("start sleep")
        time.sleep(1)  # 每隔 1 秒检查一次
        print("sleep over")
finally:
    driver.quit()
