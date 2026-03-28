import cv2
import pytesseract
import os

# 配置tesseract可执行文件路径，需要根据实际安装路径修改
pytesseract.pytesseract.tesseract_cmd = 'D:\\Application\\Tesseract-OCR\\3.02.02\\Tesseract-OCR\\tessdata'
#(导入依赖库，指定T O程序路径)

def read_image(image_path):
    """读取图像并进行基础验证"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    image = cv2.imread(image_path)#(加载图像)
    if image is None:#(失败异常的情况)
        raise ValueError(f"无法读取图像: {image_path}")
    return image
#(检测文件是否纯在，不存在则抛异常)

def preprocess_image(image):
    """图像多级预处理"""
    # 转为灰度图(先灰色化)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊减少噪点(用5*5卷积核模糊图像，弱化噪声)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 自适应阈值二值化(根据局部区域动态确定值，让车牌文字/背景黑白分明)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    # 形态学操作(填补车牌区域的小空洞，让轮廓更连贯，方便后续找轮廓)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morph


def locate_license_plate(processed_image, original_image):
    """改进的车牌定位算法"""#(查找轮廓，只找最外层轮廓)
    contours, _ = cv2.findContours(processed_image.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    potential_plates = []

    # 获取图像尺寸用于比例计算p(适配不同大小图片)
    height, width = original_image.shape[:2]
    min_area = 0.005 * height * width  # 降低最小面积阈值(过滤太小的干扰)
    max_area = 0.1 * height * width  # 最大面积阈值(过滤太大的干扰)

    for contour in contours:#计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)#宽高比
        area = w * h#面积

        # 综合筛选条件 - 调整宽高比范围(宽高比：2-6,面积在阈值内)
        if 2 < aspect_ratio < 6 and min_area < area < max_area:
            potential_plates.append((x, y, w, h, area))

    # 如果找到多个候选，选择面积最大的(更真实的车牌)
    if potential_plates:
        potential_plates.sort(key=lambda x: x[4], reverse=True)
        return potential_plates[0][:4]  # 返回坐标和宽高
    return None


def recognize_characters(image, plate_coords):
    """改进的字符识别，增加预处理和多种模式尝试"""
    x, y, w, h = plate_coords#裁剪车牌区域
    plate_img = image[y:y + h, x:x + w]

    # 车牌区域预处理(从原图中裁剪出车牌区域)
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    plate_blur = cv2.GaussianBlur(plate_gray, (3, 3), 0)
    _, plate_thresh = cv2.threshold(plate_blur, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)#尝试多种配置，适配不同场景

    # 尝试多种OCR配置
    configs = [
        '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # 字母数字(单行识别+仅字母数字白名单)
        '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # 单行文本(单行识别为的是更灵活，白名单)
        '--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # 稀疏文本(适应模糊/不规整)+白名单
    ]

    results = []
    for config in configs:#OCR识别
        text = pytesseract.image_to_string(plate_thresh, config=config)
        # 简单的结果过滤
        clean_text = ''.join(e for e in text if e.isalnum())
        if len(clean_text) > 5:  # 假设车牌至少有6个字符(只保留字母数字，且长度大于等于6)
            results.append(clean_text)

    # 返回最长的有效识别结果
    return max(results, key=len) if results else "识别失败"


def main(image_path):
    """主函数，整合整个车牌识别流程"""
    try:
        # 读取图像
        image = read_image(image_path)

        # 预处理
        processed_image = preprocess_image(image)

        # 定位车牌
        plate_coords = locate_license_plate(processed_image, image)
        #(这三步都是调佣，串联整个识别流程)
        if plate_coords:
            # 识别字符
            license_number = recognize_characters(image, plate_coords)

            # 绘制结果
            x, y, w, h = plate_coords
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, license_number, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 显示结果
            print(f"识别的车牌号: {license_number}")

            # 保存结果图像
            result_path = os.path.splitext(image_path)[0] + "_result.jpg"
            cv2.imwrite(result_path, image)
            print(f"结果已保存至: {result_path}")

            # 显示结果
            cv2.imshow("车牌识别结果", image)
            cv2.waitKey(0)
        else:
            print("未找到车牌区域")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")


if __name__ == "__main__":
    # 将此处的图片路径替换为实际车牌图片路径
    main("1.jpg")