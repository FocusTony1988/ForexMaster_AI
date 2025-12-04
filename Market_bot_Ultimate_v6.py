import argparse
import sys
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings

# =========================================
# SYSTEM SETUP
# =========================================
os.system("") 
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    print("CRITICAL: Fehlende Pakete. Installiere via requirements.txt")
    sys.exit(1)

# DATEINAME FÜR DAS GEDÄCHTNIS
HISTORY_FILE = "signal_history.csv"

# =========================================
# 1. KONFIGURATION
# =========================================
PRO_INDICES = {
    "^GDAXI": "DAX 40 (DE)",
    "^GSPC":  "S&P 500 (US)",
    "BTC-USD": "Bitcoin (Crypto)", 
    "GC=F":   "Gold Futures"
}

DAX40_TICKERS = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE",
    "CBK.DE", "CON.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DTE.DE", "DHL.DE", "EOAN.DE",
    "FRE.DE", "FME.DE", "G1A.DE", "HNR1.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE",
    "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE",
    "G24.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE",
]

CONTEXT_TICKERS = ["DX-Y.NYB", "^VIX"] 
FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", 
    "NZDUSD=X", "USDCAD=X", "USDCHF=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CHFJPY=X",
    "GC=F", "BTC-USD"
]

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

# ==============================================================================
# MODUL: TRACK MANAGER (DAS GEDÄCHTNIS)
# ==============================================================================
class TrackManager:
    def __init__(self, filename):
        self.filename = filename
        self.columns = ["Date", "Ticker", "Type", "Entry", "Target", "StopLoss", "Status", "CloseDate", "Result"]
        
    def load_history(self):
        if os.path.exists(self.filename):
            return pd.read_csv(self.filename)
        return pd.DataFrame(columns=self.columns)

    def save_history(self, df):
        df.to_csv(self.filename, index=False)

    def update_open_trades(self, current_prices):
        """Prüft offene Trades auf Zielerreichung"""
        df = self.load_history()
        if df.empty: return df, []

        updates_log = []
        today = datetime.now().strftime('%Y-%m-%d')

        for idx, row in df.iterrows():
            if row['Status'] == "OPEN":
                # Ticker Match prüfen (Manchmal mit =X, manchmal ohne)
                ticker = row['Ticker']
                curr_price = None
                
                # Versuche Preis zu finden
                if ticker in current_prices: curr_price = current_prices[ticker]
                elif ticker + "=X" in current_prices: curr_price = current_prices[ticker + "=X"]
                
                if curr_price:
                    # CHECK LOGIC
                    is_long = "LONG" in row['Type'] or "BUY" in row['Type']
                    
                    if is_long:
                        if curr_price >= row['Target']:
                            df.at[idx, 'Status'] = "CLOSED"
                            df.at[idx, 'Result'] = "WIN (TP)"
                            df.at[idx, 'CloseDate'] = today
                            updates_log.append(f"💰 WIN: {ticker} erreichte Target {row['Target']}")
                        elif curr_price <= row['StopLoss']:
                            df.at[idx, 'Status'] = "CLOSED"
                            df.at[idx, 'Result'] = "LOSS (SL)"
                            df.at[idx, 'CloseDate'] = today
                            updates_log.append(f"❌ LOSS: {ticker} stop out bei {row['StopLoss']}")
                    else: # SHORT
                        if curr_price <= row['Target']:
                            df.at[idx, 'Status'] = "CLOSED"
                            df.at[idx, 'Result'] = "WIN (TP)"
                            df.at[idx, 'CloseDate'] = today
                            updates_log.append(f"💰 WIN: {ticker} erreichte Target {row['Target']}")
                        elif curr_price >= row['StopLoss']:
                            df.at[idx, 'Status'] = "CLOSED"
                            df.at[idx, 'Result'] = "LOSS (SL)"
                            df.at[idx, 'CloseDate'] = today
                            updates_log.append(f"❌ LOSS: {ticker} stop out bei {row['StopLoss']}")
        
        self.save_history(df)
        return df, updates_log

    def add_new_signal(self, ticker, signal_type, price, target, stop_loss):
        """Fügt ein neues Signal hinzu, wenn es noch nicht existiert"""
        df = self.load_history()
        # Check ob bereits ein offener Trade für diesen Ticker existiert
        existing = df[(df['Ticker'] == ticker) & (df['Status'] == "OPEN")]
        if not existing.empty:
            return False # Schon offen

        new_row = {
            "Date": datetime.now().strftime('%Y-%m-%d'),
            "Ticker": ticker,
            "Type": signal_type,
            "Entry": round(price, 4),
            "Target": round(target, 4),
            "StopLoss": round(stop_loss, 4),
            "Status": "OPEN",
            "CloseDate": "-",
            "Result": "-"
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_history(df)
        return True

# ==============================================================================
# MODUL: ANALYSE ENGINE (Forex/Stocks/Indices)
# ==============================================================================

def create_link(ticker, text):
    base_url = f"https://finance.yahoo.com/chart/{ticker}?interval=1d"
    return f'<a href="{base_url}" target="_blank" style="text-decoration:none; color:#2980b9;">{text}</a>'

def fetch_data(tickers):
    try:
        data = yf.download(tickers, period="1y", group_by='ticker', progress=False, auto_adjust=False)
        return data
    except: return None

def analyze_forex(data, ticker):
    try:
        df = data[ticker].copy()
        if len(df) < 50: return None
        
        # Indicators
        close = df['Close']
        sma20 = close.rolling(20).mean()
        std = close.rolling(20).std()
        bbu = sma20 + 2*std
        bbl = sma20 - 2*std
        
        # Stoch
        low_min = df['Low'].rolling(10).min()
        high_max = df['High'].rolling(10).max()
        k = 100 * (close - low_min) / (high_max - low_min)
        k = k.rolling(3).mean()
        d = k.rolling(3).mean()
        
        # Logic
        c_price = float(close.iloc[-1])
        c_k = float(k.iloc[-1]); c_d = float(d.iloc[-1])
        p_k = float(k.iloc[-2]); p_d = float(d.iloc[-2])
        
        verdict = "-"; priority = 99; stop = 0.0
        
        # Buy Signal
        if c_price <= float(bbl.iloc[-1]) or (c_price - float(bbl.iloc[-1]))/c_price < 0.002:
            if p_k <= p_d and c_k > c_d and c_k < 30:
                verdict = "★ BUY SIGNAL"
                priority = 1
                stop = float(bbl.iloc[-1]) * 0.995 # SL etwas unter BB

        # Sell Signal
        if c_price >= float(bbu.iloc[-1]) or (float(bbu.iloc[-1]) - c_price)/c_price < 0.002:
            if p_k >= p_d and c_k < c_d and c_k > 70:
                verdict = "★ SELL SIGNAL"
                priority = 1
                stop = float(bbu.iloc[-1]) * 1.005 # SL etwas über BB

        return {
            "Ticker": ticker.replace("=X", ""),
            "RawTicker": ticker,
            "Price": c_price,
            "Target": float(sma20.iloc[-1]),
            "Stop": stop,
            "Verdict": verdict,
            "Priority": priority,
            "Stoch": f"{c_k:.0f}"
        }
    except: return None

def analyze_stock(data, ticker):
    try:
        df = data[ticker].copy()
        if len(df) < 50: return None
        
        # Simple Bollinger Logic
        close = df['Close']
        sma20 = close.rolling(20).mean()
        std = close.rolling(20).std()
        bbu = sma20 + 2*std
        bbl = sma20 - 2*std
        
        c_price = float(close.iloc[-1])
        status = "NEUTRAL"; target = 0.0; stop = 0.0
        
        if c_price < float(bbl.iloc[-1]):
            status = "ACTION LONG"
            target = float(sma20.iloc[-1])
            stop = float(bbl.iloc[-1]) * 0.98
            
        elif c_price > float(bbu.iloc[-1]):
            status = "ACTION SHORT"
            target = float(sma20.iloc[-1])
            stop = float(bbu.iloc[-1]) * 1.02
            
        return {
            "Ticker": ticker,
            "Status": status,
            "Price": c_price,
            "Target": target,
            "Stop": stop
        }
    except: return None

# ==============================================================================
# MAIN ENGINE
# ==============================================================================

def main():
    tracker = TrackManager(HISTORY_FILE)
    
    # 1. Daten holen
    all_tickers = FOREX_PAIRS + DAX40_TICKERS + list(PRO_INDICES.keys())
    data = fetch_data(all_tickers)
    
    if data is None or data.empty:
        print("Data Download Failed")
        return

    # 2. Aktuelle Preise für Tracking extrahieren
    current_prices = {}
    for t in all_tickers:
        try:
            val = data[t]['Close'].iloc[-1]
            # Speichere sowohl Clean als auch Raw Ticker
            current_prices[t] = float(val)
            current_prices[t.replace("=X", "")] = float(val)
        except: pass
        
    # 3. History Update (Gestern prüfen)
    hist_df, update_logs = tracker.update_open_trades(current_prices)
    
    # 4. Neue Scans
    fx_results = []
    for pair in FOREX_PAIRS:
        res = analyze_forex(data, pair)
        if res and res['Priority'] == 1:
            fx_results.append(res)
            # Speichern ins Gedächtnis
            tracker.add_new_signal(res['Ticker'], "FX BUY" if "BUY" in res['Verdict'] else "FX SELL", 
                                   res['Price'], res['Target'], res['Stop'])
    
    stock_results = []
    for stock in DAX40_TICKERS:
        res = analyze_stock(data, stock)
        if res and "ACTION" in res['Status']:
            stock_results.append(res)
            # Speichern ins Gedächtnis
            tracker.add_new_signal(res['Ticker'], res['Status'], res['Price'], res['Target'], res['Stop'])

    # 5. Email Bauen
    generate_email(fx_results, stock_results, hist_df, update_logs)

def generate_email(fx, stocks, history, logs):
    now_str = datetime.now().strftime('%d.%m.%Y %H:%M')
    
    # AI Prompt (Bestes Signal)
    prompt_txt = ""
    best_sig = None
    if fx: best_sig = fx[0]
    elif stocks: best_sig = stocks[0]
    
    if best_sig:
        t_name = best_sig.get('Ticker')
        s_type = best_sig.get('Verdict', best_sig.get('Status'))
        prompt_txt = f"""*** AI BACKTEST PROMPT ***
Asset: {t_name} | Signal: {s_type}
Price: {best_sig['Price']:.4f} | Target: {best_sig['Target']:.4f}
StopLoss Idee: {best_sig['Stop']:.4f}
Aufgabe: Validiere dieses Mean Reversion Setup."""

    html = f"""<html><body>
    <h2>🚀 Market Report v6 (Tracking Active)</h2>
    <p>Time: {now_str}</p>
    
    <div style="background:#ecf0f1; padding:10px; border-left:5px solid #2ecc71;">
        <h3>📊 PERFORMANCE TRACKER</h3>
    """
    
    if logs:
        html += "<b>🆕 NEUE ERGEBNISSE (Seit letztem Scan):</b><br>"
        for l in logs: html += f"{l}<br>"
    else: html += "Keine Trade-Abschlüsse seit gestern.<br>"
    
    # Offene Trades zeigen
    open_trades = history[history['Status'] == "OPEN"]
    if not open_trades.empty:
        html += "<br><b>🔓 AKTIV OFFENE TRADES:</b><br><table border='1' cellspacing='0' cellpadding='5'>"
        html += "<tr><th>Date</th><th>Ticker</th><th>Type</th><th>Entry</th><th>Target</th></tr>"
        for _, row in open_trades.iterrows():
            html += f"<tr><td>{row['Date']}</td><td>{row['Ticker']}</td><td>{row['Type']}</td>"
            html += f"<td>{row['Entry']}</td><td>{row['Target']}</td></tr>"
        html += "</table>"
    else: html += "<br>Keine offenen Positionen."
    
    html += "</div>"

    # SIGNALE HEUTE
    if fx:
        html += "<h3>💱 FOREX SIGNALS (PRIO 1)</h3><table>"
        for r in fx:
             link = create_link(r['RawTicker'], r['Ticker'])
             html += f"<tr><td><b>{link}</b></td><td>{r['Verdict']}</td><td>Target: {r['Target']:.4f}</td></tr>"
        html += "</table>"
        
    if stocks:
        html += "<h3>📈 STOCK ACTIONS</h3><table>"
        for r in stocks:
             link = create_link(r['Ticker'], r['Ticker'])
             html += f"<tr><td><b>{link}</b></td><td>{r['Status']}</td><td>Target: {r['Target']:.2f}</td></tr>"
        html += "</table>"

    # Prompt
    if prompt_txt:
        html += f"""<br><hr><b>🤖 AI Prompt:</b><br>
        <textarea rows="5" style="width:100%">{prompt_txt}</textarea>"""

    html += "</body></html>"
    
    send_mail("Market Report v6 + Tracking", html)

def send_mail(subj, html):
    sender = os.environ.get("EMAIL_USER")
    pw = os.environ.get("EMAIL_PASS")
    target = os.environ.get("EMAIL_TARGET")
    
    if not sender: 
        print("Local Test - HTML saved to report.html")
        with open("report.html", "w", encoding="utf-8") as f: f.write(html)
        return

    msg = MIMEMultipart()
    msg['Subject'] = subj
    msg['From'] = sender
    msg['To'] = target
    msg.attach(MIMEText(html, 'html'))

    try:
        s = smtplib.SMTP("smtp-relay.brevo.com", 587)
        s.starttls()
        s.login(sender, pw)
        s.sendmail(sender, target, msg.as_string())
        s.quit()
        print("Email Sent.")
    except Exception as e: print(f"Mail Error: {e}")

if __name__ == "__main__":
    main()
