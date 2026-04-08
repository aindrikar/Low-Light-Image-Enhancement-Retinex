# 🌙 Retinex-Based Low-Light Image Enhancement System

## 📌 Overview
This project focuses on enhancing low-light images using Retinex-based image processing techniques. The system improves visibility, contrast, and overall image quality in poorly illuminated conditions.

The implementation uses Single-Scale Retinex (SSR) and Multi-Scale Retinex (MSR) algorithms to separate illumination and reflectance components, enabling effective enhancement of dark images.

---

## ⚙️ Methodology

### 🔹 Retinex Theory
Retinex theory is based on the idea that an image can be decomposed into:
- Illumination (lighting conditions)
- Reflectance (true object colors)

Enhancing the reflectance component improves visibility while preserving natural appearance.

---

### 🔹 Techniques Used

#### 1. Single-Scale Retinex (SSR)
- Applies Gaussian filtering to estimate illumination  
- Enhances contrast by normalizing pixel intensities  

#### 2. Multi-Scale Retinex (MSR)
- Combines multiple SSR outputs at different scales  
- Provides better dynamic range and detail preservation  

#### 3. Adaptive Enhancement
- Improves brightness and contrast dynamically  
- Reduces noise and enhances image clarity  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** OpenCV, NumPy, Matplotlib  
- **Techniques:** Image processing, Gaussian filtering, Retinex algorithms  

---

## 🔄 Workflow

1. Input low-light image  
2. Apply Gaussian filtering for illumination estimation  
3. Compute reflectance using Retinex algorithms  
4. Combine multi-scale outputs  
5. Enhance contrast and brightness  
6. Output improved image  

---

## 📊 Results

- Improved visibility in low-light conditions  
- Enhanced contrast and detail  
- Reduced noise in darker regions  
- Better image usability for real-world applications  

---

```markdown
![Original Image](original.png)
![Enhanced Image](enhanced.png)
