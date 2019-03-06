from selenium import webdriver
import requests

username=input("Please input your username: ")
passwd=input("Please input your password: ")
option = webdriver.ChromeOptions()
option.add_argument("headless")
driver = webdriver.Chrome(chrome_options=option)
driver = webdriver.Chrome()
driver.get("http://jwes.hit.edu.cn/queryWsyyIndex")
driver.find_element_by_id("username").send_keys(username)
driver.find_element_by_id("password").send_keys(passwd)
driver.find_element_by_class_name("auth_login_btn primary full_width").click()
driver.find_element_by_class_name("yy sm_yy").click()

print(driver.title)
driver.quit()
