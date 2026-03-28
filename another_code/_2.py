import cv2
import os
import easyocr

# 初始化 EasyOCR 识别器
reader = easyocr.Reader(['en'])  # 支持的语言，这里使用英文，可以根据需要添加其他语言


def read_image(image_path):
    """读取图像并进行基础验证"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image


def preprocess_image(image):
    """图像多级预处理"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # 调整卷积核大小
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15,
                                   2)  # 调整自适应阈值参数
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 调整结构元素大小
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)  # 调整迭代次数

    # 显示预处理后的图像
    cv2.imshow("Preprocessed Image", morph)
    cv2.waitKey(0)
    return morph


def locate_license_plate(processed_image, original_image):
    """改进的车牌定位算法"""
    contours, _ = cv2.findContours(processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"检测到的轮廓数量: {len(contours)}")

    potential_plates = []
    height, width = original_image.shape[:2]
    min_area = 0.001 * height * width  # 放宽最小面积阈值
    max_area = 0.2 * height * width  # 放宽最大面积阈值

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h

        if 1.5 < aspect_ratio < 10 and min_area < area < max_area:  # 放宽宽高比范围
            potential_plates.append((x, y, w, h, area))

    print(f"筛选后的潜在车牌区域数量: {len(potential_plates)}")
    if potential_plates:
        potential_plates.sort(key=lambda x: x[4], reverse=True)
        return potential_plates[0][:4]
    return None


def recognize_characters(image, plate_coords):
    """使用 EasyOCR 进行字符识别"""
    x, y, w, h = plate_coords
    plate_img = image[y:y + h, x:x + w]

    # 显示裁剪后的车牌区域
    cv2.imshow("License Plate Region", plate_img)
    cv2.waitKey(0)

    # 使用 EasyOCR 识别车牌区域的文本
    results = reader.readtext(plate_img, detail=0)  # detail=0 返回纯文本结果
    if results:
        return results[0]  # 返回识别到的第一个结果
    else:
        print("EasyOCR 未识别到任何内容")
        return "识别失败"


def main(image_path):
    """主函数，整合整个车牌识别流程"""
    try:
        print(f"正在处理图像: {image_path}")
        image = read_image(image_path)
        processed_image = preprocess_image(image)
        plate_coords = locate_license_plate(processed_image, image)

        if plate_coords:
            license_number = recognize_characters(image, plate_coords)
            print(f"识别的车牌号: {license_number}")

            x, y, w, h = plate_coords
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, license_number, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            result_path = os.path.splitext(image_path)[0] + "_result.jpg"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            cv2.imwrite(result_path, image)
            print(f"结果已保存至: {result_path}")

            cv2.imshow("车牌识别结果", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("未找到车牌区域")

    except Exception as e:
        print(f"处理过程中出错: {e}")


if __name__ == "__main__":
    # 使用绝对路径
    image_path = r"D:\Application\Python\Python_Study\PythonProject2\.venv\7.jpg"
    main(image_path)
