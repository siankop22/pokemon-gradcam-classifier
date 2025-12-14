# GitHub Upload Guide
## Step-by-Step Instructions to Upload Your Project

### **Prerequisites**
- GitHub account (create one at [github.com](https://github.com))
- Git installed on your computer
- Terminal/Command Prompt access

---

## **Step 1: Prepare Your Project**

### **A. Create/Update .gitignore**
✅ Already created! The `.gitignore` file is set up to exclude:
- Virtual environment (`venv/`)
- Model checkpoints (`checkpoints/`, `*.pth`)
- Python cache files (`__pycache__/`)
- IDE files (`.vscode/`, `.idea/`)
- Log files

### **B. Review Files to Upload**
Files that **WILL** be uploaded:
- ✅ All Python scripts (`.py` files)
- ✅ Documentation (`.md` files)
- ✅ Data images (in `data/pokemon_pics/`)
- ✅ README.md

Files that **WON'T** be uploaded (excluded by `.gitignore`):
- ❌ `venv/` folder
- ❌ `checkpoints/` folder
- ❌ `explainability_results/` folder
- ❌ `__pycache__/` folders

---

## **Step 2: Initialize Git Repository**

Open terminal in your project directory and run:

```bash
# Navigate to project directory
cd /Users/tksiankop/Downloads/pokemon_gradcam_starter

# Initialize git repository
git init

# Check status (see what files will be added)
git status
```

---

## **Step 3: Add Files to Git**

```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

You should see files like:
- `gradcam.py`
- `explainability_analysis.py`
- `train_pokemon_classifier.py`
- `README.md`
- `data/pokemon_pics/*.png`
- etc.

But NOT:
- `venv/`
- `checkpoints/`
- `explainability_results/`

---

## **Step 4: Create Initial Commit**

```bash
# Create first commit
git commit -m "Initial commit: Pokémon Classifier with Grad-CAM explainability"
```

---

## **Step 5: Create GitHub Repository**

### **Option A: Using GitHub Website**

1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** icon in top right → **"New repository"**
3. Fill in:
   - **Repository name**: `pokemon-gradcam-classifier` (or your choice)
   - **Description**: "Pokémon Classifier with Grad-CAM Explainability Analysis"
   - **Visibility**: Public or Private (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### **Option B: Using GitHub CLI** (if installed)

```bash
gh repo create pokemon-gradcam-classifier --public --source=. --remote=origin --push
```

---

## **Step 6: Connect Local Repository to GitHub**

After creating the repository on GitHub, you'll see instructions. Use these commands:

```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pokemon-gradcam-classifier.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Example:**
```bash
git remote add origin https://github.com/johndoe/pokemon-gradcam-classifier.git
git branch -M main
git push -u origin main
```

---

## **Step 7: Verify Upload**

1. Go to your GitHub repository page
2. You should see all your files
3. Check that:
   - ✅ Python files are there
   - ✅ README.md is there
   - ✅ Data images are there
   - ❌ `venv/` is NOT there
   - ❌ `checkpoints/` is NOT there

---

## **Step 8: Add Repository Description (Optional)**

On your GitHub repository page:
1. Click the **gear icon** next to "About"
2. Add description: "Pokémon Classifier with Grad-CAM Explainability"
3. Add topics: `python`, `pytorch`, `grad-cam`, `explainable-ai`, `computer-vision`, `pokemon`
4. Click **"Save"**

---

## **Troubleshooting**

### **Problem: "Permission denied" or authentication error**

**Solution**: Set up authentication

```bash
# Option 1: Use Personal Access Token
# 1. Go to GitHub → Settings → Developer settings → Personal access tokens
# 2. Generate new token with 'repo' permissions
# 3. Use token as password when pushing

# Option 2: Use SSH (recommended)
# 1. Generate SSH key: ssh-keygen -t ed25519 -C "your_email@example.com"
# 2. Add to GitHub: Settings → SSH and GPG keys → New SSH key
# 3. Change remote URL:
git remote set-url origin git@github.com:YOUR_USERNAME/pokemon-gradcam-classifier.git
```

### **Problem: "Repository not found"**

**Solution**: 
- Check that you created the repository on GitHub first
- Verify the repository name matches
- Check your GitHub username is correct

### **Problem: Files are too large**

**Solution**: 
- Check `.gitignore` is working (large files should be excluded)
- If data images are too large, you can exclude them:
  ```bash
  # Add to .gitignore
  data/pokemon_pics/*.png
  ```

### **Problem: Want to update later**

**Solution**: After making changes:
```bash
git add .
git commit -m "Description of changes"
git push
```

---

## **Quick Reference Commands**

```bash
# Initialize
git init
git add .
git commit -m "Initial commit"

# Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main

# Update later
git add .
git commit -m "Update description"
git push
```

---

## **What Gets Uploaded - Summary**

### ✅ **Will Upload:**
- All Python scripts
- Documentation files (.md)
- Data images (pokemon_pics/)
- Configuration files
- README.md

### ❌ **Won't Upload:**
- Virtual environment (venv/)
- Model checkpoints (checkpoints/, *.pth)
- Results folder (explainability_results/)
- Python cache (__pycache__/)
- IDE files (.vscode/, .idea/)
- Log files

---

## **After Uploading**

1. **Share your repository**: Copy the URL and share it
2. **Add a license**: Consider adding LICENSE file
3. **Create releases**: Tag important versions
4. **Add badges**: Show build status, Python version, etc.

---

## **Example Repository URL**

After uploading, your repository will be at:
```
https://github.com/YOUR_USERNAME/pokemon-gradcam-classifier
```

---

## **Next Steps**

1. ✅ Initialize git: `git init`
2. ✅ Add files: `git add .`
3. ✅ Commit: `git commit -m "Initial commit"`
4. ✅ Create GitHub repo (on website)
5. ✅ Connect: `git remote add origin ...`
6. ✅ Push: `git push -u origin main`

**You're all set!** 🎉

