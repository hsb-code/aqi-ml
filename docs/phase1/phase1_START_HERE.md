# Phase 1 - START HERE 🚀

## What is Phase 1?

Phase 1 is all about **understanding the science** behind your project. You need to learn:
1. **How AQI works** - What pollutants matter and how to calculate AQI
2. **Where to get satellite data** - Which satellites track air pollutants
3. **How others do it** - What methods exist for satellite-based AQI

**Duration**: 1-2 weeks  
**Goal**: Have enough knowledge to start building the system

---

## 📅 Day-by-Day Action Plan

### **Week 1: Understanding AQI & Finding Data**

#### **Day 1-2: Learn AQI Basics** ✅ START HERE

**What to do:**

1. **Read about AQI calculation**
   - Visit: https://www.airnow.gov/aqi/aqi-basics/
   - Understand the 6 key pollutants
   - Learn how concentrations convert to AQI values

2. **Watch this video** (15 mins):
   - Search YouTube: "What is AQI Air Quality Index"
   - Recommended: EPA's AQI explanation videos

3. **Create your first document**:
   ```
   File: h:\AQI\docs\phase1\aqi_notes.md
   
   Write down:
   - What is AQI?
   - Which 6 pollutants are tracked?
   - What are the AQI categories (Good, Moderate, etc.)?
   - How is AQI calculated? (general idea)
   ```

**Expected Time**: 2-3 hours

---

#### **Day 3-4: Deep Dive into Pollutants**

**What to do:**

1. **Research each pollutant**:
   
   For each of these 6 pollutants, answer:
   - **PM2.5** (Particulate Matter)
   - **PM10** (Larger particles)
   - **NO₂** (Nitrogen Dioxide)
   - **SO₂** (Sulfur Dioxide)
   - **CO** (Carbon Monoxide)
   - **O₃** (Ozone)
   
   Questions to answer:
   - What produces this pollutant? (cars, factories, fires?)
   - Why is it dangerous?
   - What are typical concentration levels?

2. **Find the AQI breakpoint tables**:
   - Visit: https://www.airnow.gov/publications/air-quality-index/technical-assistance-document-for-reporting-the-daily-aqi/
   - Download the PDF (it has the tables)
   - Save to: `h:\AQI\docs\references\`

3. **Test your understanding**:
   - Try an example: If PM2.5 = 35.5 μg/m³, what's the AQI?
   - Use online calculator: https://www.airnow.gov/aqi/aqi-calculator/
   - Verify your calculation

**Expected Time**: 3-4 hours

---

#### **Day 5-7: Explore Satellite Data Sources**

**What to do:**

1. **Register for satellite data access**:

   **A. Copernicus (Sentinel-5P)**
   - Go to: https://dataspace.copernicus.eu/ and click "Register" in top right
   - Create account (use your work email if available)
   - This gives you: NO₂, SO₂, CO, O₃ data
   - **Write down your credentials** in a safe place

   **B. NASA Earthdata**
   - Go to: https://urs.earthdata.nasa.gov/users/new
   - Create account
   - After registration, go to: https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ
   - Approve "NASA GESDISC DATA ARCHIVE"
   - This gives you: AOD data (for PM2.5/PM10 estimation)

2. **Explore the data portals**:
   
   **Sentinel-5P**:
   - Login to: https://dataspace.copernicus.eu/
   - Browse available products
   - Pick a date and location (e.g., your city)
   - See what data looks like (don't download yet, just browse)
   
   **NASA Worldview** (Visual exploration):
   - Visit: https://worldview.earthdata.nasa.gov/
   - Select layers: Aerosol Optical Depth, NO₂
   - Zoom to your region
   - See how pollution looks from space!
   - Take screenshots for your notes

3. **Document what you found**:
   ```
   File: h:\AQI\docs\phase1\data_sources_notes.md
   
   For each satellite source, note:
   - What pollutants does it provide?
   - How often does it pass over (daily, weekly)?
   - What's the image resolution?
   - How do I download data?
   - What format is the data? (NetCDF, HDF, etc.)
   ```

**Expected Time**: 4-5 hours (includes registration waiting time)

---

### **Week 2: Literature Review & Technical Approach**

#### **Day 8-10: Study Existing Methods**

**What to do:**

1. **Search for research papers**:
   
   **Google Scholar searches**:
   - "satellite based PM2.5 estimation deep learning"
   - "air quality prediction satellite imagery CNN"
   - "TROPOMI NO2 air quality monitoring"
   - "AOD PM2.5 relationship machine learning"

2. **Find 3-5 key papers**:
   - Focus on papers from 2020-2024 (recent is better)
   - Look for papers that use:
     - Sentinel-5P or MODIS data
     - Deep learning (CNN, ResNet, LSTM)
     - Real-time or near-real-time processing
   
3. **Read the papers** (focus on these sections):
   - **Abstract** - What did they do?
   - **Methodology** - What model/approach did they use?
   - **Results** - How accurate was it?
   - **Discussion** - What were limitations?

4. **Create literature review notes**:
   ```
   File: h:\AQI\docs\phase1\literature_review.md
   
   For each paper:
   - Paper title & authors
   - Key idea (1-2 sentences)
   - Dataset used
   - Model architecture
   - Accuracy achieved (R², RMSE, etc.)
   - Can we use this approach?
   ```

**Expected Time**: 6-8 hours

**Papers to Start With** (search these titles):
- "Estimation of PM2.5 Concentrations from Himawari-8 AOD Using Deep Learning"
- "Real-time Air Quality Index Prediction using Satellite Images and Deep Learning"
- "Estimating Ground-Level PM2.5 Using Satellite Data"

---

#### **Day 11-12: Plan Your Technical Approach**

**What to do:**

1. **Decide on your approach** based on what you learned:
   
   Answer these questions:
   - Which satellites will you use? (Recommend: Start with Sentinel-5P for NO₂)
   - Which pollutant will you start with? (Recommend: NO₂ - easiest to get)
   - What model architecture? (Recommend: Start simple with CNN)
   - How will you validate? (Compare with ground stations)

2. **Create a technical design document**:
   ```
   File: h:\AQI\docs\technical_approach.md
   
   Include:
   - Data sources you'll use
   - Which pollutants you'll predict
   - Proposed model architecture (diagram if possible)
   - How you'll train the model
   - How you'll validate accuracy
   - Challenges you anticipate
   ```

3. **Draw a simple flowchart**:
   ```
   Satellite Image → Preprocessing → DL Model → Pollutant Value → AQI Calculation → Output
   ```
   Use: draw.io, PowerPoint, or just paper sketches

**Expected Time**: 3-4 hours

---

#### **Day 13-14: Organize & Prepare for Phase 2**

**What to do:**

1. **Organize all your notes**:
   - Ensure all documents are in `h:\AQI\docs\`
   - Clean up your notes
   - Create a summary document

2. **Create a reference library**:
   ```
   h:\AQI\docs\references\
   ├── papers\          (PDF papers you found)
   ├── screenshots\     (Satellite data screenshots)
   └── links.md         (Important links)
   ```

3. **Update your progress**:
   - Mark Phase 1 tasks as complete in [task.md](file:///C:/Users/hasee/.gemini/antigravity/brain/69f2a25c-2813-4381-aee5-20262f72515a/task.md)

4. **Prepare for Phase 2**:
   - Review [implementation_plan.md](file:///C:/Users/hasee/.gemini/antigravity/brain/69f2a25c-2813-4381-aee5-20262f72515a/implementation_plan.md) Phase 2 section
   - Ensure you have satellite portal credentials ready
   - Plan your first data download

**Expected Time**: 2-3 hours

---

## 📋 Phase 1 Checklist

Use this to track your progress:

### Week 1
- [ ] Understand what AQI is and how it's calculated
- [ ] Know the 6 key pollutants and their sources
- [ ] Have AQI breakpoint tables saved
- [ ] Registered for Copernicus account
- [ ] Registered for NASA Earthdata account
- [ ] Explored satellite data portals
- [ ] Took screenshots of pollution from satellite view
- [ ] Documented satellite data sources

### Week 2
- [ ] Found 3-5 relevant research papers
- [ ] Read and summarized key papers
- [ ] Understand existing approaches
- [ ] Decided which satellites to use
- [ ] Decided which pollutant to start with
- [ ] Created technical approach document
- [ ] Drew system architecture diagram
- [ ] Organized all documentation

---

## 🎯 Success Criteria for Phase 1

By the end of Phase 1, you should be able to answer:

✅ **Science**:
- What is AQI and how is it calculated?
- What are the 6 key pollutants and why they matter?
- Given a pollutant concentration, can you calculate AQI?

✅ **Data**:
- Which satellites provide which pollutants?
- How do I access satellite data?
- What format is the data in?

✅ **Approach**:
- What method will I use for my project?
- Which pollutant will I start with?
- How will I validate my results?

---

## 🆘 Need Help?

### Stuck on AQI calculation?
- Use online calculators to check your work
- Focus on PM2.5 first (most important pollutant)

### Can't find research papers?
- Use Google Scholar (not regular Google)
- Add "PDF" to your search
- Check university library access if you have it

### Satellite portals confusing?
- Start with NASA Worldview (just for visualization)
- You'll learn actual downloads in Phase 2

### Not sure about technical approach?
- It's OK to keep it simple!
- Start with: Sentinel-5P → NO₂ → Simple CNN
- You can improve later

---

## 💡 Pro Tips

1. **Don't aim for perfection** - You'll learn more in later phases
2. **Take notes** - You'll reference them constantly
3. **Visual learning** - Screenshots and diagrams help
4. **Ask questions** - Use your lead or online forums
5. **Timebox** - Don't spend more than 2 weeks on Phase 1

---

## ✅ When You're Done

After completing Phase 1:

1. Update [task.md](file:///C:/Users/hasee/.gemini/antigravity/brain/69f2a25c-2813-4381-aee5-20262f72515a/task.md) - Mark Phase 1 complete
2. Review your documents - Do they make sense?
3. Schedule a check-in with your lead - Show what you learned
4. Move to **Phase 2**: Data Acquisition!

---

## 📚 Quick Links

- [Full Implementation Plan](file:///C:/Users/hasee/.gemini/antigravity/brain/69f2a25c-2813-4381-aee5-20262f72515a/implementation_plan.md)
- [Task Tracker](file:///C:/Users/hasee/.gemini/antigravity/brain/69f2a25c-2813-4381-aee5-20262f72515a/task.md)
- [Phase 1 Details](file:///h:/AQI/docs/phase1/phase1_research.md)

---

**Ready? Start with Day 1-2! Good luck! 🚀**
