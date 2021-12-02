import cv2 # นำเข้า opencv

path_image = " " # พาทรูปภาพ
img = cv2.imread(path_image) # อ่านรูปภาพจากพาท
cv2.imwrite("name_image_new.jpg", img) # บันทึกรูปภาพชื่อ name_image_new นามสกุล jpg << img
cv2.imshow("window out", img) # แสดงรูปภาพในหน้าต่าง window out << img
cv2.waitKey() # รอการสั่งเพื่อยกเลิกการแสดงภาพ

