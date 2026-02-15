# Market Analyzer Dashboard

⚠️ **Disclaimer**

This project is **NOT financial advice** and should **NOT be used as a professional trading or investment tool**.

It is an **educational / experimental project** created for learning, research, and technical exploration.

---

## 🇨🇦 Scope

This analyzer is designed **specifically for the Canadian market**.

- Tickers are **currently hardcoded**
- Focused primarily on `.TO`, `.V`, `.CN`, `.NE` symbols
- Intended for experimentation, observation, and model validation

Future versions may introduce dynamic ticker management.

---

## 🚀 Overview

A lightweight Python-based market analysis tool that:

✔ Fetches market data using **yfinance**  
✔ Processes analysis via a CLI engine  
✔ Serves results through a **FastAPI backend**  
✔ Displays data in a browser dashboard  

The project combines:

- CLI Market Analyzer
- FastAPI Web API
- Minimal Dashboard UI

---

## 🌐 Live Dashboard

If available, the dashboard can be accessed at:

👉 **https://stage-radar.onrender.com/**

⚠️ **Important behavior to understand:**

This application is hosted on a free-tier environment.

- The server may enter **sleep mode after inactivity**
- First request after sleep may take **10–30 seconds**
- This is **normal hosting behavior**

✔ Simply wait — the server wakes automatically  
✔ No manual action required  
✔ The dashboard will display appropriate loading/status feedback  

---
