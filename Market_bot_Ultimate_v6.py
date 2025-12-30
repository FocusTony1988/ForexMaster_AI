import sys
import os
import smtplib
import warnings
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Unterdrücke Warnungen für saubereren Output
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    print("CRITICAL: Fehlende Pakete. Bitte installiere sie mit: pip install yfinance pandas numpy")
    sys.exit(1)

# =========================================
# 1. KONFIGURATION
# =========================================
HISTORY_FILE = "signal_history.csv"

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

FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", 
    "NZDUSD=X", "USDCAD=X", "USDCHF=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CHFJPY=X",
    "GC=F", "BTC-USD"
]

# ==============================================================================
# MODUL: TRACK MANAGER (DAS GEDÄCHTNIS)
# ==============================================================================
class TrackManager:
    def __init__(self, filename):
        self.filename = filename
        self.columns = ["Date", "Ticker", "Type", "Entry", "Target", "StopLoss", "Status", "CloseDate", "Result"]
        
    def load_history(self):
        if os.path.exists(self.filename):
            try:
                return pd.read_csv(self.filename)
            except Exception:
                return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(columns=self.columns)

    def save_history(self, df):
        df.to_csv(self.filename, index=False)

    def update_open_trades(self, current_prices):
        """Prüft offene Trades auf Zielerreichung"""
        df = self.load_history()
        if df.empty: 
            return df, []

        updates_log = []
        today = datetime.now().strftime('%Y-%m-%d')

        for idx, row in df.iterrows():
            if row['Status'] == "OPEN":
                ticker = row['Ticker']
                curr_price = None
                
                # Versuche Preis zu finden (manchmal fehlt =X in der CSV oder andersherum)
                if ticker in current_prices:
                    curr_price = current_prices[ticker]
                elif ticker + "=X" in current_prices:
                    curr_price = current_prices[ticker + "=X"]
                
                if curr_price:
                    # CHECK LOGIC
                    is_long = "LONG" in str(row['Type']).upper() or "BUY" in str(row['Type']).upper()
                    
                    target = float(row['Target'])
                    stop_loss = float(row['StopLoss'])
                    
                    result_type = None
                    
                    if is_long:
                        if curr_price >= target:
                            result_type = "WIN (TP)"
                        elif curr_price <= stop_loss:
                            result_type = "LOSS (SL)"
                    else: # SHORT
                        if curr_price <= target:
                            result_type = "WIN (TP)"
                        elif curr_price >= stop_loss:
                            result_type = "LOSS (SL)"
                    
                    if result_type:
                        df.at[idx, 'Status'] = "CLOSED"
                        df.at[idx, 'Result'] = result_type
                        df.at[idx, 'CloseDate'] = today
                        icon = "💰" if "WIN" in result_type else "❌"
                        updates_log.append(f"{icon} {result_type}: {ticker} (Exit: {curr_price:.4f})")
        
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
        # Pandas concat statt append (append ist deprecated)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_history(df)
        return True

# ==============================================================================
# MODUL: ANALYSE ENGINE
# ==============================================================================

def create_link(ticker, text):
    base_url = f"https://finance.yahoo.com/chart/{ticker}?interval=1d"
    return f'<a href="{base_url}" target="_blank" style="text-decoration:none; color:#2980b9;">{text}</a>'

def fetch_data(tickers):
    """Lädt Daten für alle Ticker herunter."""
    if not tickers:
        return None
    try:
        # Threads erhöhen die Geschwindigkeit
        data = yf.download(tickers, period="1y", group_by='ticker', progress=False, threads=True)
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def analyze_forex(data, ticker):
    try:
        # Zugriff auf Multi-Index Dataframes sicherstellen
        if ticker not in data.columns.levels[0]:
            return None
            
        df = data[ticker].copy()
        
        # Leere Zeilen entfernen
        df.dropna(inplace=True)
        
        if len(df) < 50: 
            return None
        
        # Indikatoren Berechnung
        close = df['Close']
        sma20 = close.rolling(20).mean()
        std = close.rolling(20).std()
        bbu = sma20 + 2*std
        bbl = sma20 - 2*std
        
        # Stochastic
        low_min = df['Low'].rolling(10).min()
        high_max = df['High'].rolling(10).max()
        # Division durch Null abfangen
        denom = high_max - low_min
        denom = denom.replace(0, 0.0001) 
        
        k = 100 * (close - low_min) / denom
        k = k.rolling(3).mean()
        d = k.rolling(3).mean()
        
        # Letzte Werte holen
        c_price = float(close.iloc[-1])
        c_k = float(k.iloc[-1])
        c_d = float(d.iloc[-1])
        p_k = float(k.iloc[-2])
        p_d = float(d.iloc[-2])
        
        verdict = "-"
        priority = 99
        stop = 0.0
        
        # Logic Limits
        bbl_val = float(bbl.iloc[-1])
        bbu_val = float(bbu.iloc[-1])

        # Buy Signal
        if c_price <= bbl_val or (c_price - bbl_val)/c_price < 0.002:
            if p_k <= p_d and c_k > c_d and c_k < 30:
                verdict = "★ BUY SIGNAL"
                priority = 1
                stop = bbl_val * 0.995 # SL etwas unter BB

        # Sell Signal
        if c_price >= bbu_val or (bbu_val - c_price)/c_price < 0.002:
            if p_k >= p_d and c_k < c_d and c_k > 70:
                verdict = "★ SELL SIGNAL"
                priority = 1
                stop = bbu_val * 1.005 # SL etwas über BB

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
    except Exception as e:
        # print(f"Error analyzing {ticker}: {e}") # Debugging
        return None

def analyze_stock(data, ticker):
    try:
        if ticker not in data.columns.levels[0]:
            return None

        df = data[ticker].copy()
        df.dropna(inplace=True)
        
        if len(df) < 50: 
            return None
        
        close = df['Close']
        sma20 = close.rolling(20).mean()
        std = close.rolling(20).std()
        bbu = sma20 + 2*std
        bbl = sma20 - 2*std
        
        c_price = float(close.iloc[-1])
        status = "NEUTRAL"
        target = 0.0
        stop = 0.0
        
        bbl_val = float(bbl.iloc[-1])
        bbu_val = float(bbu.iloc[-1])
        sma_val = float(sma20.iloc[-1])

        if c_price < bbl_val:
            status = "ACTION LONG"
            target = sma_val
            stop = bbl_val * 0.98
            
        elif c_price > bbu_val:
            status = "ACTION SHORT"
            target = sma_val
            stop = bbu_val * 1.02
            
        return {
            "Ticker": ticker,
            "Status": status,
            "Price": c_price,
            "Target": target,
            "Stop": stop
        }
    except Exception:
        return None

# ==============================================================================
# MAIN & EMAIL
# ==============================================================================

def send_mail(subj, html):
    sender = os.environ.get("EMAIL_USER")
    pw = os.environ.get("EMAIL_PASS")
    target = os.environ.get("EMAIL_TARGET")
    
    # Fallback für lokales Testen
    if not sender or not pw or not target: 
        print("INFO: Keine E-Mail Zugangsdaten (ENV) gefunden.")
        print("--> Speichere HTML lokal als 'report.html'")
        with open("report.html", "w", encoding="utf-8") as f:
            f.write(html)
        return

    msg = MIMEMultipart()
    msg['Subject'] = subj
    msg['From'] = sender
    msg['To'] = target
    msg.attach(MIMEText(html, 'html'))

    try:
        # Hinweis: Checke, ob dein Provider Port 587 oder 465 nutzt.
        s = smtplib.SMTP("smtp-relay.brevo.com", 587)
        s.starttls()
        s.login(sender, pw)
        s.sendmail(sender, target, msg.as_string())
        s.quit()
        print("✅ Email erfolgreich gesendet.")
    except Exception as e:
        print(f"❌ Mail Error: {e}")

def generate_email(fx, stocks, history, logs):
    now_str = datetime.now().strftime('%d.%m.%Y %H:%M')
    
    prompt_txt = ""
    best_sig = None
    if fx: best_sig = fx[0]
    elif stocks: best_sig = stocks[0]
    
    if best_sig:
        t_name = best_sig.get('Ticker')
        s_type = best_sig.get('Verdict', best_sig.get('Status'))
        prompt_txt = f"Asset: {t_name} | Signal: {s_type}\nPrice: {best_sig['Price']:.4f} | Target: {best_sig['Target']:.4f}\nCheck Mean Reversion."

    html = f"""<html><body style="font-family: Arial, sans-serif;">
    <h2>🚀 Market Report v6 (Tracking Active)</h2>
    <p>Time: {now_str}</p>
    
    <div style="background:#f9f9f9; padding:15px; border-left:5px solid #2ecc71; margin-bottom: 20px;">
        <h3>📊 PERFORMANCE TRACKER</h3>
    """
    
    if logs:
        html += "<b>🆕 NEUE ERGEBNISSE (Seit letztem Scan):</b><br><ul>"
        for l in logs: 
            html += f"<li>{l}</li>"
        html += "</ul>"
    else:
        html += "<p>Keine Trade-Abschlüsse seit dem letzten Scan.</p>"
    
    # Offene Trades zeigen
    open_trades = history[history['Status'] == "OPEN"]
    if not open_trades.empty:
        html += "<br><b>🔓 AKTIV OFFENE TRADES:</b><br><table border='1' cellspacing='0' cellpadding='5' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr style='background:#ddd;'><th>Date</th><th>Ticker</th><th>Type</th><th>Entry</th><th>Target</th><th>Stop</th></tr>"
        for _, row in open_trades.iterrows():
            html += f"<tr><td>{row['Date']}</td><td>{row['Ticker']}</td><td>{row['Type']}</td>"
            html += f"<td>{row['Entry']}</td><td>{row['Target']}</td><td>{row['StopLoss']}</td></tr>"
        html += "</table>"
    else:
        html += "<br><i>Keine offenen Positionen.</i>"
    
    html += "</div>"

    # SIGNALE HEUTE
    if fx:
        html += "<h3>💱 FOREX SIGNALS (PRIO 1)</h3><table border='1' cellpadding='5' style='border-collapse:collapse; width:100%;'>"
        html += "<tr style='background:#eee;'><th>Ticker</th><th>Signal</th><th>Target</th><th>Stoch</th></tr>"
        for r in fx:
             link = create_link(r['RawTicker'], r['Ticker'])
             html += f"<tr><td><b>{link}</b></td><td>{r['Verdict']}</td><td>{r['Target']:.4f}</td><td>{r['Stoch']}</td></tr>"
        html += "</table>"
        
    if stocks:
        html += "<h3>📈 STOCK ACTIONS</h3><table border='1' cellpadding='5' style='border-collapse:collapse; width:100%;'>"
        html += "<tr style='background:#eee;'><th>Ticker</th><th>Status</th><th>Price</th><th>Target</th></tr>"
        for r in stocks:
             link = create_link(r['Ticker'], r['Ticker'])
             html += f"<tr><td><b>{link}</b></td><td>{r['Status']}</td><td>{r['Price']:.2f}</td><td>{r['Target']:.2f}</td></tr>"
        html += "</table>"

    # Prompt Area
    if prompt_txt:
        html += f"""<br><hr><b>🤖 AI Quick Prompt:</b><br>
        <div style="background:#333; color:#fff; padding:10px; font-family:monospace;">{prompt_txt}</div>"""

    html += "</body></html>"
    
    send_mail("Market Report v6 + Tracking", html)

def main():
    print("--- Start Market Bot v6 ---")
    tracker = TrackManager(HISTORY_FILE)
    
    # 1. Daten holen
    all_tickers = FOREX_PAIRS + DAX40_TICKERS + list(PRO_INDICES.keys())
    print(f"Lade Daten für {len(all_tickers)} Ticker...")
    
    data = fetch_data(all_tickers)
    
    if data is None or data.empty:
        print("❌ Data Download Failed. Breche ab.")
        return

    # 2. Aktuelle Preise extrahieren (Für den Tracker)
    current_prices = {}
    
    # Iteriere sicher durch die Spalten (MultiIndex Handling)
    # yfinance liefert meist (Price, Ticker) als Spaltenstruktur
    try:
        # Wir schauen uns nur die 'Close' Spalte an
        closes = data['Close']
        for ticker in closes.columns:
            try:
                # Hole den letzten gültigen Wert (dropna)
                last_price = closes[ticker].dropna().iloc[-1]
                current_prices[ticker] = float(last_price)
                # Backup für Ticker ohne "=X" Suffix
                clean_ticker = ticker.replace("=X", "")
                if clean_ticker != ticker:
                    current_prices[clean_ticker] = float(last_price)
            except IndexError:
                continue
    except Exception as e:
        print(f"Fehler beim Preis-Mapping: {e}")

    # 3. History Update
    print("Prüfe offene Trades...")
    hist_df, update_logs = tracker.update_open_trades(current_prices)
    
    # 4. Neue Scans durchführen
    print("Analysiere Märkte...")
    fx_results = []
    for pair in FOREX_PAIRS:
        res = analyze_forex(data, pair)
        if res and res['Priority'] == 1:
            fx_results.append(res)
            tracker.add_new_signal(res['Ticker'], "FX BUY" if "BUY" in res['Verdict'] else "FX SELL", 
                                   res['Price'], res['Target'], res['Stop'])
    
    stock_results = []
    for stock in DAX40_TICKERS:
        res = analyze_stock(data, stock)
        if res and "ACTION" in res['Status']:
            stock_results.append(res)
            tracker.add_new_signal(res['Ticker'], res['Status'], res['Price'], res['Target'], res['Stop'])

    # 5. Report erstellen
    print(f"Gefunden: {len(fx_results)} Forex Signale, {len(stock_results)} Aktien Signale.")
    generate_email(fx_results, stock_results, hist_df, update_logs)
    print("--- Fertig ---")

if __name__ == "__main__":
    main()
