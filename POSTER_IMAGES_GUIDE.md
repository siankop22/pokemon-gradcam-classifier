# Poster Images Guide
## Which Pictures to Include and Where to Place Them

### **IMAGES AVAILABLE:**
- ✅ **18 Correct Prediction Visualizations** (with Grad-CAM heatmaps)
- ✅ **4 Incorrect Prediction Visualizations** (from latest run)
- ✅ **1 Comparison Visualization** (correct vs incorrect side-by-side)

---

## **RECOMMENDED IMAGES FOR YOUR POSTER:**

### **1. MAIN VISUALIZATION SECTION (Center of Poster)**

#### **A. Correct Predictions Examples (3-4 images)**

**Best choices:**
1. `explainability_results/correct/Bulbasaur.png`
   - **Why**: Popular Pokémon, clear visualization
   - **Caption**: "Correct Prediction: Bulbasaur - Model focuses on distinctive features"

2. `explainability_results/correct/Pyroar.png`
   - **Why**: Good example of focused attention
   - **Caption**: "Correct Prediction: Pyroar - High attention intensity (0.39)"

3. `explainability_results/correct/starmie.png`
   - **Why**: Interesting shape, clear heatmap
   - **Caption**: "Correct Prediction: Starmie - Model identifies star-shaped features"

4. `explainability_results/correct/Zigzagoon.png`
   - **Why**: Shows quadruped shape recognition
   - **Caption**: "Correct Prediction: Zigzagoon - Attention on body shape"

**Alternative options:**
- `explainability_results/correct/Yamper.png`
- `explainability_results/correct/eevee.png`
- `explainability_results/correct/nosepass.png`

---

#### **B. Incorrect Predictions Examples (2-3 images)**

**Best choices:**
1. `explainability_results/incorrect/starmie.png`
   - **Why**: Shows error (starmie → Flabébé)
   - **Caption**: "Error: Starmie misclassified as Flabébé - Lower attention intensity (0.30)"

2. `explainability_results/incorrect/meltan.png`
   - **Why**: Shows error (meltan → Tynamo)
   - **Caption**: "Error: Meltan confused with Tynamo - Similar appearance causes confusion"

3. `explainability_results/incorrect/Venusaur.png`
   - **Why**: Shows error (Venusaur → stunky)
   - **Caption**: "Error: Venusaur misclassified - Model focuses on wrong features"

4. `explainability_results/incorrect/Litleo.png`
   - **Why**: Shows error (Litleo → pawmo)
   - **Caption**: "Error: Litleo confused with pawmo - Similar quadruped shapes"

---

#### **C. Comparison Visualization (1 large image)**

**Use:**
- `explainability_results/comparisons/correct_vs_incorrect.png`
- **Why**: Shows side-by-side comparison of correct vs incorrect
- **Caption**: "Comparison: Correct (top row) vs Incorrect (bottom row) predictions. Notice the difference in attention patterns."

---

## **POSTER LAYOUT WITH IMAGES:**

```
┌─────────────────────────────────────────────────────────┐
│                    TITLE SECTION                         │
├──────────────┬──────────────┬──────────────────────┤
│ INTRODUCTION │   METHODS    │      METHODS (cont.)      │
│              │              │                         │
│              │ [Arch Diagram]│                         │
├──────────────┼──────────────┼──────────────────────┤
│   RESULTS    │ GRAD-CAM     │   ERROR PATTERNS         │
│              │ VISUALIZATIONS│                         │
│ [Accuracy    │              │ [Error Examples]         │
│  Chart]      │ [Correct 1]  │ [meltan.png]            │
│              │ [Correct 2]  │ [starmie.png]           │
│              │ [Correct 3]  │                         │
│              │              │                         │
│              │ [Incorrect 1]│                         │
│              │ [Incorrect 2]│                         │
│              │              │                         │
│              │ [Comparison] │                         │
├──────────────┼──────────────┼──────────────────────┤
│ KEY INSIGHTS │              │  CONCLUSIONS &           │
│              │              │  FUTURE WORK            │
└──────────────┴──────────────┴──────────────────────┘
```

---

## **SPECIFIC IMAGE PLACEMENT:**

### **Section 1: Results (Left Side)**
- **Image**: Accuracy chart/graph (create from your data)
- **Size**: Medium (1/4 of poster width)
- **Caption**: "Model Performance: 80% Accuracy (16/20 correct)"

### **Section 2: Grad-CAM Visualizations (Center - LARGE)**
- **Layout**: 2x3 grid or 3x2 grid
- **Top Row (Correct Predictions):**
  1. `correct/Bulbasaur.png`
  2. `correct/Pyroar.png`
  3. `correct/starmie.png`
- **Bottom Row (Incorrect Predictions):**
  1. `incorrect/starmie.png`
  2. `incorrect/meltan.png`
  3. `incorrect/Venusaur.png`
- **Size**: Each image ~15-20% of poster width
- **Caption**: "Grad-CAM Visualizations: Red/yellow regions show where the model focuses"

### **Section 3: Comparison (Center - Below main grid)**
- **Image**: `comparisons/correct_vs_incorrect.png`
- **Size**: Large (30-40% of poster width)
- **Caption**: "Side-by-side comparison showing attention differences"

### **Section 4: Error Analysis (Right Side)**
- **Images**: 
  - `incorrect/meltan.png` (meltan → Tynamo)
  - `incorrect/Litleo.png` (Litleo → pawmo)
- **Size**: Medium (2 images stacked)
- **Caption**: "Error Examples: Similar appearances cause confusion"

---

## **IMAGE SPECIFICATIONS:**

### **For Printing:**
- **Resolution**: Minimum 300 DPI
- **Format**: PNG or high-quality JPG
- **Size**: 
  - Small images: 3-4 inches wide
  - Medium images: 5-6 inches wide
  - Large images: 8-10 inches wide

### **For Digital Presentation:**
- **Resolution**: 150-200 DPI is fine
- **Format**: PNG (better quality)
- **Size**: Can be larger since it's on screen

---

## **CAPTIONS FOR EACH IMAGE:**

### **Correct Predictions:**
1. **Bulbasaur**: "Correct: Bulbasaur - Model focuses on plant bulb and body shape (Intensity: 0.39)"
2. **Pyroar**: "Correct: Pyroar - Attention on mane and facial features"
3. **Starmie**: "Correct: Starmie - Star shape clearly identified"
4. **Zigzagoon**: "Correct: Zigzagoon - Body pattern and shape recognized"

### **Incorrect Predictions:**
1. **Starmie (error)**: "Error: Starmie → Flabébé - Scattered attention (Intensity: 0.30)"
2. **Meltan**: "Error: Meltan → Tynamo - Similar metallic appearance causes confusion"
3. **Venusaur**: "Error: Venusaur → stunky - Wrong features focused"
4. **Litleo**: "Error: Litleo → pawmo - Similar quadruped shapes confused"

### **Comparison:**
- "Comparison: Top row shows correct predictions with focused attention. Bottom row shows incorrect predictions with scattered attention."

---

## **QUICK REFERENCE - BEST IMAGES TO USE:**

### **Must Include (6-8 images):**
1. ✅ `correct/Bulbasaur.png` - Popular, clear
2. ✅ `correct/Pyroar.png` - Good example
3. ✅ `correct/starmie.png` - Interesting shape
4. ✅ `incorrect/starmie.png` - Shows error clearly
5. ✅ `incorrect/meltan.png` - Error example
6. ✅ `comparisons/correct_vs_incorrect.png` - Great comparison

### **Nice to Have (2-3 more):**
7. `correct/Zigzagoon.png`
8. `incorrect/Venusaur.png`
9. `incorrect/Litleo.png`

---

## **TIPS FOR POSTER DESIGN:**

1. **Image Quality**: 
   - Use the PNG files directly (they're already high quality)
   - Don't resize down too much

2. **Consistency**:
   - Make all images the same size in each section
   - Use consistent borders/spacing

3. **Labels**:
   - Add clear labels: "Correct" or "Incorrect"
   - Include Pokémon names
   - Add intensity values if space allows

4. **Color Scheme**:
   - The heatmaps use red/yellow (hot) colors
   - Make sure your poster background doesn't clash
   - Use white or light gray background

5. **Size Guidelines**:
   - Standard poster: 36" x 48" or 42" x 56"
   - Each visualization image: 4-6 inches wide
   - Comparison image: 8-10 inches wide

---

## **FILE PATHS (Copy-Paste Ready):**

```
Correct Predictions:
explainability_results/correct/Bulbasaur.png
explainability_results/correct/Pyroar.png
explainability_results/correct/starmie.png
explainability_results/correct/Zigzagoon.png
explainability_results/correct/Yamper.png

Incorrect Predictions:
explainability_results/incorrect/starmie.png
explainability_results/incorrect/meltan.png
explainability_results/incorrect/Venusaur.png
explainability_results/incorrect/Litleo.png

Comparison:
explainability_results/comparisons/correct_vs_incorrect.png
```

---

## **FINAL CHECKLIST:**

Before printing/presenting:
- [ ] Selected 6-8 best images
- [ ] All images are high resolution (300 DPI for print)
- [ ] Captions are clear and informative
- [ ] Images are properly labeled (Correct/Incorrect)
- [ ] Comparison image is prominently displayed
- [ ] Images are arranged logically on poster
- [ ] Text is readable (minimum 12pt font for captions)
- [ ] Color scheme is consistent

---

## **ALTERNATIVE: CREATE A COLLAGE**

If you want one large visualization section, create a collage with:
- Top: 3 correct predictions
- Middle: Comparison image (larger)
- Bottom: 3 incorrect predictions

This creates a strong visual impact and clearly shows the difference between correct and incorrect predictions.

